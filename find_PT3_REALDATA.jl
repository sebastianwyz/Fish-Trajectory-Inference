## -------------------------------------------------------
##
## Functions to track a fish with HMC
##
## February 15, 2024 -- Andreas Scheidegger
## andreas.scheidegger@eawag.ch
## -------------------------------------------------------

using Distributions
using Interpolations
using Plots
using Random
using DynamicHMC
using TransformVariables
using TransformedLogDensities: TransformedLogDensity
using LogDensityProblemsAD: ADgradient
using Optim             
import ForwardDiff

using LogDensityProblems
using LogDensityProblemsAD
import LogDensityProblems: logdensity, dimension, capabilities
using Pigeons
using Pigeons: HMC
using Pigeons: record_default
import Pigeons: initialization, sample_iid!
import TransformVariables: AbstractTransform
using MCMCChains
using StatsPlots



# -----------unction for coordinates
dist(c1, c2) = sqrt((c1.x - c2.x)^2 + (c1.y - c2.y)^2)
direction(c1, c2) = atan(c1.x - c2.x,  c1.y - c2.y)


# ----------- Bayesian problem
# p(X_t | X_{t_1}, X_{t-2}), movement model


function log_p_moveRW(c1, c2)
    σ = 10.0 #1
    logpdf(Normal(c2.x, σ), c1.x) +
        logpdf(Normal(c2.y, σ), c1.y)  #uniform
end

#=
function log_p_moveRW(c1, c2)
    σ = 0.3  # controlla quanto "stretta" è la preferenza per la distanza 3
    d = dist(c1, c2)
    logpdf(Normal(3.0, σ), d)
end
=#
# p(Y_t | X_t), observation model for accustic receivers
abstract type Sensor
end

struct Receiver <: Sensor
    x::Float64
    y::Float64
    dist::Float64  # receiving distance
    k::Float64     # smoothness
end

Receiver(c; dist=50.0, k=30.0) = Receiver(c.x, c.y, dist, k)


function log_prob_signal(signal, s::NamedTuple, device::Receiver)
    d = dist((x=device.x, y=device.y), s)

    d0 = device.dist
    #k = device.k
    k = 5
    prob_detect = 1 - 1/(1 + exp(-(d - d0)/k))

    if signal == :detect
        return log(prob_detect)
    end
    if signal == :nondetect
        return log(1 - prob_detect)
    end
    error("unkown signal: $(signal)")
end


# p(Y_t | X_t), observation model depth measurements
struct DepthGauge <: Sensor
end

#=
function get_depth(s, bathymetry)
    dy, dx = size(bathymetry)
    print("dy: ", dy, " dx: ", dx, "\n")
    bathymetry(dy - s.y, s.x)  / 100
    
end
=#


function get_depth_p(s, bathymetry, x_origin, y_origin, dx, dy)
    # x_origin, y_origin: coordinate reali del centro del pixel (1,1)
    # dx, dy: passo della griglia in x e y
    # s.x, s.y: coordinate reali

    # Calcola indici frazionari (col, row) per la matrice/interpolazione
    col = (s.x - x_origin) / dx + 1
    row = (s.y - y_origin) / dy + 1

    # Controlla che siano nel range
    #if row < 1 || row > size(bathymetry, 1) || col < 1 || col > size(bathymetry, 2)
    #    return -1.0
    #end
    return bathymetry(row, col)  # Interpolazione frazionaria
end



function get_depth(s, bathymetry, x_origin, y_origin, dx, dy)
    # x_origin, y_origin: coordinate reali del centro del pixel (1,1)
    # dx, dy: passo della griglia in x e y
    # s.x, s.y: coordinate reali

    # Calcola indici frazionari (col, row) per la matrice/interpolazione
    col = (s.x - x_origin) / dx + 1
    row = (s.y - y_origin) / dy + 1

    # Controlla che siano nel range
    #if row < 1 || row > size(bathymetry, 1) || col < 1 || col > size(bathymetry, 2)
    #    return -1.0
    #end
    return bathymetry(row, col)  # Interpolazione frazionaria
end
#=
function get_depth_p(s, bathymetry, bathy_orig, row_start, col_start)
    # s deve essere NamedTuple o struct con campi .x e .y
    idx_full = GeoArrays.indices(bathy_orig, (s.x, s.y))
    println("idx_full: ", idx_full)
    row_full, col_full = Tuple(idx_full)
    row = row_full - row_start + 1
    col = col_full - col_start + 1
    # Controlla che gli indici siano nel range del ritaglio
    if row < 1 || row > size(bathymetry, 1) || col < 1 || col > size(bathymetry, 2)
        return -1.0  # oppure NaN o altro valore per "fuori mappa"
    end
    return bathymetry(row, col) 
end
=#
#=
function get_depth_xy(x, y, itp)
    return itp(x, y)
end
=#
#=
function get_depth(s, bathymetry, bathy_orig, row_start, col_start)
    # s deve essere NamedTuple o struct con campi .x e .y
    idx_full = GeoArrays.indices(bathy_orig, (s.x, s.y))
    println("idx_full: ", idx_full)
    row_full, col_full = Tuple(idx_full)
    row = row_full - row_start + 1
    col = col_full - col_start + 1
    # Controlla che gli indici siano nel range del ritaglio
    if row < 1 || row > size(bathymetry, 1) || col < 1 || col > size(bathymetry, 2)
        return -1.0  # oppure NaN o altro valore per "fuori mappa"
    end
    return bathymetry(row, col) 
end

=#
#=
using GeoArrays

function get_depth(s, bathy)
    idx = GeoArrays.indices(bathy, (s.x, s.y))
    return bathy[idx]
end
=#
#=
function log_prob_signal(signal, s::NamedTuple, device::DepthGauge, map_int)

    # Depth at location s
    max_depth = get_depth(s, map_int)

    # not a very realsitc choise
    dist = Normal(max_depth, 2)   # 2
    logpdf(dist, signal)


end
=#

#=
function log_prob_signal(signal, s::NamedTuple, device::DepthGauge, map_int, bathy_orig, row_start, col_start)
    # Depth at location s
    max_depth = get_depth_p(s, map_int, bathy_orig, row_start, col_start)
    dist = Normal(max_depth, 2)
    logpdf(dist, signal)
end
=#


#=
function log_prob_signal(signal, s::NamedTuple, device::DepthGauge, map_int, bathy_orig, row_start, col_start)
    #max_depth = get_depth_p(s, map_int, bathy_orig, row_start, col_start)
    max_depth = get_depth_p(s, map_int, x_origin, y_origin, dx, dy)
    dist = Normal(max_depth, 2)
    logpdf(dist, signal)
end
=#

function log_prob_signal(signal, s::NamedTuple, device::DepthGauge, bathymetry_int, x_origin, y_origin, dx, dy)
    max_depth = get_depth_p(s, bathymetry_int, x_origin, y_origin, dx, dy)
    dist = Normal(max_depth, 2)
    return logpdf(dist, signal)
end

function log_prior(S::Vector{T}) where T <: NamedTuple  #The reference function
    lp = 0.0
    #μ = 3.0
    #σ = 5.0 
    #print("Log prior for ", length(S), " samples: ")
    for t in 2:length(S)
        d = dist(S[t], S[t-1])
        lp += log_p_moveRW(S[t], S[t-1])
    end
    return lp
end

#=
function log_posterior(S::Vector{T}, Ydepth, Yaccustic, bathymetry_int) where T <: NamedTuple
    tmax = length(S)
    lp   = zero(eltype(T))

    # eventual extra prior
    #for t in 1:tmax
    #    lp += log_p_environment(S[t], bathymetry_int)
    #end

    for t in 2:tmax
        lp += log_p_moveRW(S[t], S[t-1])
    end
    
    for (t, signal, device) in Ydepth
        t ≤ tmax || continue          
        lp += log_prob_signal(signal, S[t], device, bathymetry_int)
    end
    for (t, signal, device) in Yaccustic
        t ≤ tmax || continue
        lp += log_prob_signal(signal, S[t], device)
    end


    return lp
end
=#
#=
function log_posterior(S::Vector{T}, Ydepth, Yaccustic, bathymetry_int, bathy_orig, row_start, col_start) where T <: NamedTuple
    tmax = length(S)
    lp   = zero(eltype(T))

    for t in 2:tmax
        lp += log_p_moveRW(S[t], S[t-1])
    end
    
    for (t, signal, device) in Ydepth
        t ≤ tmax || continue          
        lp += log_prob_signal(signal, S[t], device, bathymetry_int, bathy_orig, row_start, col_start)
        #lp += log_prob_signal(signal, S[t], device, bathymetry_int, bathy_orig, row_start, col_start)
    end
    for (t, signal, device) in Yaccustic
        t ≤ tmax || continue
        lp += log_prob_signal(signal, S[t], device)
    end

    return lp
end
=#


function log_posterior(S::Vector{T}, Ydepth, Yaccustic, bathymetry_int, x_origin, y_origin, dx, dy) where T <: NamedTuple
    tmax = length(S)
    lp   = zero(eltype(T))

    for t in 2:tmax
        lp += log_p_moveRW(S[t], S[t-1])
    end

    for (t, signal, device) in Ydepth
        t ≤ tmax || continue
        lp += log_prob_signal(signal, S[t], device, bathymetry_int, x_origin, y_origin, dx, dy)
    end
    for (t, signal, device) in Yaccustic
        t ≤ tmax || continue
        lp += log_prob_signal(signal, S[t], device)
    end

    return lp
end

#------------Building trajectories
function build_Yaccustic_from_bridge(bridge, receivers)
    Yaccustic = Tuple{Int, Symbol, Receiver}[]
    for (t, point) in enumerate(bridge)
        for receiver in receivers
            d = sqrt((point.x - receiver.x)^2 + (point.y - receiver.y)^2)
            signal = d <= receiver.dist ? :detect : :nondetect
            push!(Yaccustic, (t, signal, receiver))
        end
    end
    return Yaccustic
end


function build_Yaccustic_from_trajectory(traj, receivers)
    Yaccustic = Tuple{Int, Symbol, Receiver}[]
    for (t, pos) in enumerate(traj)
        detected = false
        for receiver in receivers
            dx = pos.x - receiver.x
            dy = pos.y - receiver.y
            dist_to_receiver = sqrt(dx^2 + dy^2)
            if hasproperty(receiver, :dist) && dist_to_receiver ≤ receiver.dist
                push!(Yaccustic, (t+1, :detect, receiver))
                detected = true
            end
        end
        # Se almeno uno ha rilevato, non aggiungere :nondetect
        # Se vuoi aggiungere :nondetect SOLO se nessuno rileva, decommenta la riga sotto:
        # if !detected
        #     for receiver in receivers
        #         push!(Yaccustic, (t, :nondetect, receiver))
        #     end
        # end
    end
    return Yaccustic
end




"""
    simulateRW_free(tmax; s0, sigma = 30.0, rng = Random.GLOBAL_RNG)

Random-walk gaussiana in coordinate reali, **senza** limiti hard.
Restituisce `Vector{NamedTuple}` di punti `(x, y)`.
"""
function simulateRW_free(tmax; s0, sigma = 3.0, rng = Random.GLOBAL_RNG)
    traj = Vector{NamedTuple{(:x,:y),Tuple{Float64,Float64}}}(undef, tmax)
    traj[1] = s0
    for t in 2:tmax
        traj[t] = (
            x = traj[t-1].x + randn(rng)*sigma,
            y = traj[t-1].y + randn(rng)*sigma
        )
    end
    return traj
end


#=
using Random, GeoArrays

function simulateRW_idx(bathy, tmax;
                        s0_idx,             # può essere CartesianIndex o tupla
                        sigma_pix = 2,
                        rng = Random.GLOBAL_RNG)

    nrow, ncol = size(bathy)               # raster 2-D

    row = Vector{Int}(undef, tmax)
    col = Vector{Int}(undef, tmax)

    # accetta sia CartesianIndex che Tuple
    r0, c0 = isa(s0_idx, CartesianIndex) ? Tuple(s0_idx) : s0_idx
    row[1] = r0
    col[1] = c0

    for t in 2:tmax
        row[t] = clamp(row[t-1] + round(Int, randn(rng)*sigma_pix), 1, nrow)
        col[t] = clamp(col[t-1] + round(Int, randn(rng)*sigma_pix), 1, ncol)
    end

    return [(row=row[t], col=col[t]) for t in 1:tmax]
end
=#


#=
function simulateRW_s_init(tmax; s0, sigma = 3.0, rng = Random.GLOBAL_RNG,
                           xdim = (xmin, xmax), ydim = (ymin, ymax))

    x  = similar(fill(0.0, tmax))
    y  = similar(x)

    x[1] = s0.x;  y[1] = s0.y
    for t in 2:tmax
        # X
        val = x[t-1]
        repeat = 0
        while (val < xdim[1] || val > xdim[2]) && repeat < 1_000
            val = x[t-1] + randn(rng)*sigma
            repeat += 1
        end
        repeat == 1_000 && error("RW overflow: adjust σ or xdim")
        x[t] = val
        # Y (stessa logica)
        val = y[t-1]; repeat = 0
        while (val < ydim[1] || val > ydim[2]) && repeat < 1_000
            val = y[t-1] + randn(rng)*sigma
            repeat += 1
        end
        repeat == 1_000 && error("RW overflow: adjust σ or ydim")
        y[t] = val
    end
    return [(x=x[t], y=y[t]) for t in 1:tmax]
end

=#
#=
function simulateRW_s_init(tmax; s0=(x=100.0, y=100.0),
                           xdim=(1,400), ydim=(1,200),
                           sigma = 3.0 , rng  = Random.GLOBAL_RNG)
    #x = fill(-Inf, tmax)
    #y = fill(-Inf, tmax)
    x = zeros(tmax);  y = zeros(tmax)

    x[1] = s0.x
    y[1] = s0.y
    for t in 2:tmax
        while x[t] < xdim[1] || x[t] > xdim[2]
            Δx = randn(rng)*sigma
            x[t] = x[t-1] + Δx
        end
        while y[t] < ydim[1] || y[t] > ydim[2]
            Δy = randn(rng)*sigma
            y[t] = y[t-1] + Δy
        end
    end

    [(x=x[t], y=y[t]) for t in 1:tmax]
end 

=#

#=
function simulateRW_s_init(tmax; s0=(x=100.0, y=100.0),
                           xdim=(1,400), ydim=(1,200),
                           sigma = 40.0,
                           bathymetry_int=nothing)

    while true  # ciclo finché non trovi una traiettoria tutta in acqua
        x = fill(-Inf, tmax)
        y = fill(-Inf, tmax)
        x[1] = s0.x
        y[1] = s0.y
        valid = true
        for t in 2:tmax
            # genera x valido
            while x[t] < xdim[1] || x[t] > xdim[2]
                Δx = randn()*sigma
                x[t] = x[t-1] + Δx
            end
            # genera y valido
            while y[t] < ydim[1] || y[t] > ydim[2]
                Δy = randn()*sigma
                y[t] = y[t-1] + Δy
            end
            # controllo bathymetry: scarta se su terra (depth == -1)
            if bathymetry_int !== nothing
                depth = get_depth((x=x[t], y=y[t]), bathymetry_int)
                if depth == -1
                    valid = false
                    break
                end
            end
        end
        if valid
            return [(x=x[t], y=y[t]) for t in 1:tmax]
        end
        # altrimenti riprova da capo
    end
end
=#

function simulate_bridge(tmax; A, B, σ = 3, α = 0.7, bathymetry_int)
    x = zeros(tmax);  y = zeros(tmax)
    x[1] = A.x;       y[1] = A.y
    
    for t in 2:tmax
        τ = tmax - t + 1
        σ_eff = σ #* sqrt(τ / tmax)
        found = false
        b_prev = get_depth((x=x[t-1], y=y[t-1]), bathymetry_int)
        #b_prev = get_depth((x=x[t-1], y=y[t-1]), bathy)
        print("b_prev: ", b_prev, "\n")
        #print(b_prev, "\n")
        for _ in 1:40
            x_cand = x[t-1] + randn()*σ_eff + α * (B.x - x[t-1]) / τ
            y_cand = y[t-1] + randn()*σ_eff + α * (B.y - y[t-1]) / τ
            b_cand = get_depth((x=x_cand, y=y_cand), bathymetry_int)
            #b_cand = get_depth((x=x_cand, y=y_cand), bathy)
            # Constraint: the difference betw two consecutive depth measurements can't be too high
            if b_cand > 0 && abs(b_cand - b_prev) ≤ 50
                x[t], y[t] = x_cand, y_cand
                found = true
                break
            end
        end
        if !found
            #@info "No point found at iteration $t, function suspended."
            #print(x[1], y[1], "\n")
            return nothing  # esci subito se non trovi acqua
        end
    end
    [(x=x[t], y=y[t]) for t in 1:tmax]
end


function simulate_bridge_2(tmax; A, B, σ = 3, α = 0.7, bathymetry_int)
    x = zeros(tmax)
    y = zeros(tmax)
    max_step = 4.0  # Limite massimo del passo tra due punti
    x[1] = A.x
    y[1] = A.y
    x[end] = B.x
    y[end] = B.y

    for t in 2:(tmax-1)
        # Interpolazione lineare tra A e B
        frac = (t-1)/(tmax-1)
        x_target = (1-frac)*A.x + frac*B.x
        y_target = (1-frac)*A.y + frac*B.y

        # Passo random attorno al target
        dx = randn()*σ
        dy = randn()*σ

        # Proponi nuovo punto
        x_cand = x_target + dx
        y_cand = y_target + dy

        # Limita la lunghezza del passo rispetto al punto precedente
        step_len = sqrt((x_cand - x[t-1])^2 + (y_cand - y[t-1])^2)
        if step_len > max_step
            scale = max_step / step_len
            x_cand = x[t-1] + (x_cand - x[t-1]) * scale
            y_cand = y[t-1] + (y_cand - y[t-1]) * scale
        end

        # Se vuoi vincolare all'acqua:
        if bathymetry_int !== nothing
            b_cand = get_depth((x=x_cand, y=y_cand), bathymetry_int)
            #print("b_cand: ", b_cand, "\n")
            if b_cand <= 0
                # Se non è acqua, riprova (o copia il punto precedente)
                x_cand, y_cand = x[t-1], y[t-1]
            end
        end

        x[t] = x_cand
        y[t] = y_cand
        print("x: ",x[t]-x[t-1],"  y:",  y[t]-y[t-1], "d ", sqrt((x[t]-x[t-1])^2 + (y[t]-y[t-1])^2 ), "\n")
    end

    # Costruisci la traiettoria
    [(x=x[t], y=y[t]) for t in 1:tmax]
end
#------- HMC 
function infer_trajectories(Ydepth, Yaccustic, bathymetry_interpolated;
                            s_init, n_samples=100, tmax=100)

    isfinite(log_posterior(S, Ydepth, Yaccustic, bathymetry_int, x_origin, y_origin, dx, dy)) || error("Initial value has zero likelihood!")

    #asFishCoord(lo, hi) = TransformVariables.compose(as𝕀, x -> lo + (hi - lo) * x, y -> (y - lo) / (hi - lo))
    mapping = as(Array,
    as((x = asℝ,          # coordinata libera su ℝ
        y = asℝ)),tmax)


    v_init = inverse(mapping, s_init)

    pp = TransformedLogDensity(mapping,
                               s -> log_posterior(S, Ydepth, Yaccustic, bathymetry_int, x_origin, y_origin, dx, dy))

    ∇pp = ADgradient(:ForwardDiff, pp)
    # ∇pp = ADgradient(:ReverseDiff, pp, compile = Val(true))

    results = mcmc_with_warmup(Random.GLOBAL_RNG, ∇pp, n_samples,
                               initialization = (q = v_init, ),
                               reporter = ProgressMeterReport())

    # backtransform to tuples
    samples = transform.(mapping, eachcol(results.posterior_matrix))

    return samples

end



#------------- defining the target and reference distribution for pigeons
#Target:
#=
struct FishLogPotential{YD,YA,BI,M}
    Ydepth   ::Vector{YD}
    Yacc     ::Vector{YA}
    bathy_int::BI
    mapping  ::M
    v_init   ::Vector{Float64}
end

function (lp::FishLogPotential)(v::AbstractVector)
    S = TransformVariables.transform(lp.mapping, v)
    return log_posterior(S, lp.Ydepth, lp.Yacc, lp.bathy_int)
end

=#
#=
struct FishLogPotential{YD,YA,BI,BO,M}
    Ydepth    ::Vector{YD}
    Yacc      ::Vector{YA}
    bathy_int ::BI
    bathy_orig::BO
    row_start ::Int
    col_start ::Int
    mapping   ::M
    v_init    ::Vector{Float64}
end
=#

struct FishLogPotential{YD,YA,BI,M}
    Ydepth    ::Vector{YD}
    Yacc      ::Vector{YA}
    bathy_int ::BI
    x_origin  ::Float64
    y_origin  ::Float64
    dx        ::Float64
    dy        ::Float64
    mapping   ::M
    v_init    ::Vector{Float64}
end

function (lp::FishLogPotential)(v::AbstractVector)
    S = TransformVariables.transform(lp.mapping, v)
    return log_posterior(S, lp.Ydepth, lp.Yacc, lp.bathy_int, lp.x_origin, lp.y_origin, lp.dx, lp.dy)
end


#Reference:
struct FishPriorPotential{M,V}
    mapping :: M
    v_init  :: V
end

function (lp::FishPriorPotential)(v::AbstractVector)
    S = TransformVariables.transform(lp.mapping, v)
    return log_prior(S)
end


# ----------- functions for plotting
function add_trajectories!(plt, samples, t)
    for i in eachindex(samples)
        ss = samples[i]
        x = [ss[k].x for k in 1:min(t, length(ss))]
        y = [ss[k].y for k in 1:min(t, length(ss))]
        plot!(plt, x, y,
              alpha=0.3,
              # color=:red,
              legend=false,
              marker=:hex, markersize=0.9)
        scatter!(plt, x[end:end], y[end:end],
                 markersize=0.9)
    end
end


function make_circle(x, y, r)
    θ = LinRange(0, 2*π, 500)
    x .+ r*sin.(θ), y .+ r*cos.(θ)
end


function add_signal!(plt, Y, t)
    for i in eachindex(Y)
        ts, signal, device = Y[i]

        if ts == t && device isa Receiver
            col = signal == :detect ? :green : :red
            plot!(plt, make_circle(device.x, device.y, device.dist),
                  color=col, legend=false)
            scatter!([device.x], [device.y], color=col, markersize=1.5)
        end
    end
end

function plot_depth(Y, tmax)
    yy = []
    tt = []
    for i in eachindex(Y)
        t, signal, device = Y[i]
        if t <= tmax && device isa DepthGauge
            push!(tt, t)
            push!(yy, signal)
        end
    end

    scatter(tt, yy,  markersize=0.8, yflip=true,
            ylab=:depth, xlab="time", legend=false)
end




#---------------------------------------------------------------
#=# Soft barriers to avoid land / bounding box exits
σ(x) = 1 / (1 + exp(-x))         
function log_p_environment(s::NamedTuple, bathymetry_int;
                           xdim=(1.0,400.0), ydim=(1.0,200.0),
                           α=2.0, β=0.5)  # valori aumentati #era 1 e 0.25

    depth = get_depth(s, bathymetry_int)        
    p_water = σ(α * depth)                      # ≈0 su terra, ≈1 in acqua
    lp = log(p_water + eps())                  

    # --- penalità per uscire dal bounding box (4 log‑σ simmetrici)
    lp += log(σ(β * (s.x - xdim[1])) + eps())
    lp += log(σ(β * (xdim[2] - s.x)) + eps())
    lp += log(σ(β * (s.y - ydim[1])) + eps())
    lp += log(σ(β * (ydim[2] - s.y)) + eps())

    return lp          
end=#


#=function build_Yaccustic_conditional(bridge::Vector{P}, receivers_list::Vector{R}) where {P, R}
    # Yaccustic conterrà tuple di (Timestamp, Segnale, Oggetto Ricevitore)
    Yaccustic = Tuple{Int, Symbol, R}[]

    for (t, point) in enumerate(bridge) # 't' sarà l'indice (1, 2, 3...), 'point' l'elemento di bridge
        
        # Lista temporanea per conservare gli stati di tutti i ricevitori per l'istante 't'
        # Ogni elemento sarà una tupla: (segnale_calcolato, ricevitore_originale)
        statuses_this_instant = Tuple{Symbol, R}[]
        at_least_one_detection_this_instant = false

        # 1. Calcola lo stato per ogni ricevitore in questo istante 't'
        for receiver in receivers_list
            # Calcolo della distanza euclidea tra il pesce (point) e il ricevitore
            # Assicurati che 'point' abbia .x, .y e 'receiver' abbia .x, .y, .dist
            distance_squared = (point.x - receiver.x)^2 + (point.y - receiver.y)^2
            distance = sqrt(distance_squared)
            
            current_signal = distance <= receiver.dist ? :detect : :nondetect
            
            push!(statuses_this_instant, (current_signal, receiver))
            
            if current_signal == :detect
                at_least_one_detection_this_instant = true
            end
        end
        
        # 2. Se almeno un ricevitore ha rilevato il pesce, aggiungi tutti gli stati a Yaccustic
        if at_least_one_detection_this_instant
            for (signal_to_add, receiver_object) in statuses_this_instant
                push!(Yaccustic, (t, signal_to_add, receiver_object))
            end
        end
        # Se 'at_least_one_detection_this_instant' è false, non facciamo nulla per questo istante 't',
        # e Yaccustic non riceverà entry per questo 't'.
    end
    
    return Yaccustic
end

function build_Yaccustic_conditional_with_confirmed_formats(bridge::Vector{P}, receivers_list::Vector{R}) where {P, R}
    Yaccustic = Tuple{Int, Symbol, R}[] # R è il tipo del tuo oggetto Receiver

    for (t, point) in enumerate(bridge) # point è (x=..., y=...)
        statuses_this_instant = Tuple{Symbol, R}[]
        at_least_one_detection_this_instant = false

        for receiver in receivers_list
            distance_squared = (point.x - receiver.x)^2 + (point.y - receiver.y)^2

            distance = sqrt(distance_squared)
            println("distance: ", distance)
            
            current_signal = distance <= receiver.dist ? :detect : :nondetect # Usa receiver.dist
            
            push!(statuses_this_instant, (current_signal, receiver))
            
            if current_signal == :detect
                at_least_one_detection_this_instant = true
            end
        end
        
        if at_least_one_detection_this_instant
            for (signal_to_add, receiver_object) in statuses_this_instant
                push!(Yaccustic, (t, signal_to_add, receiver_object))
            end
        end
    end
    
    return Yaccustic
end=#



#=
function simulate_bridge(tmax; A, B, σ = 3.0, α = 0.7, bathymetry_int, Ydepth)
    x = zeros(tmax);  y = zeros(tmax)
    x[1] = A.x;       y[1] = A.y
    for t in 2:tmax
        τ = tmax - t + 1
        σ_eff = σ #* sqrt(τ / tmax)
        # Estrai profondità osservata al tempo t
        d_obs = Ydepth[t][2]
        print("d obs", d_obs , "\n")
        candidates = Tuple{Float64, Float64}[]
        diffs = Float64[]
        # Genera 10 candidati
        for _ in 1:20
            x_cand = x[t-1] + randn()*σ_eff + α * (B.x - x[t-1]) / τ
            y_cand = y[t-1] + randn()*σ_eff + α * (B.y - y[t-1]) / τ
            b_cand = get_depth((x=x_cand, y=y_cand), bathymetry_int)
            push!(candidates, (x_cand, y_cand))
            push!(diffs, abs(b_cand - d_obs))
            print("depth cand", b_cand , "\n")
        end
        # Scegli il candidato con differenza minima
        idx = argmin(diffs)
        x[t], y[t] = candidates[idx]
    end
    [(x=x[t], y=y[t]) for t in 1:tmax]
end=#

#=function simulate_bridge(tmax; A, B, σ = 3.0, α = 0.7, bathymetry_int)
    x = zeros(tmax);  y = zeros(tmax)
    x[1] = A.x;       y[1] = A.y
    for t in 2:tmax
        τ = tmax - t + 1
        σ_eff = σ #* sqrt(τ / tmax)
        found = false
        for _ in 1:40
            x_cand = x[t-1] + randn()*σ_eff + α * (B.x - x[t-1]) / τ
            y_cand = y[t-1] + randn()*σ_eff + α * (B.y - y[t-1]) / τ
            b_cand = get_depth((x=x_cand, y=y_cand), bathymetry_int)
            if b_cand > 0  # acqua
                x[t], y[t] = x_cand, y_cand
                found = true
                #print("depth cand", b_cand , "\n")
                break
            end
        end
        if !found
            @info "Nessun punto in acqua trovato al passo $t, funzione sospesa."
            return nothing  # esci subito se non trovi acqua
        end
    end
    [(x=x[t], y=y[t]) for t in 1:tmax]
end   =#
