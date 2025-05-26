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
import TransformVariables: AbstractTransform
using MCMCChains
using StatsPlots



# -----------unction for coordinates
dist(c1, c2) = sqrt((c1.x - c2.x)^2 + (c1.y - c2.y)^2)
direction(c1, c2) = atan(c1.x - c2.x,  c1.y - c2.y)


# ----------- Bayesian problem
# p(X_t | X_{t_1}, X_{t-2}), movement model
function log_p_moveRW(c1, c2)
    σ = 3.0 #1
    logpdf(Normal(c2.x, σ), c1.x) +
        logpdf(Normal(c2.y, σ), c1.y)  #uniform
end


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
    k = 10
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

function get_depth(s, bathymetry)
    dy, dx = size(bathymetry)
    bathymetry(dy - s.y, s.x)
end

function log_prob_signal(signal, s::NamedTuple, device::DepthGauge, map_int)

    # Depth at location s
    max_depth = get_depth(s, map_int)

    # not a very realsitc choise
    dist = Normal(max_depth, 0.5)   # 2
    logpdf(dist, signal)

    # -- truncated distributions to avoid land

    # Assumes the fish likes so swimm close to be sea floor
    # dist = Truncated(Exponential(10), 0, max_depth)
    # logpdf(dist, max_depth - signal)

    # dist = Truncated(Normal(max_depth, 20), 0, max_depth)
    # logpdf(dist, signal)

    # dist = Uniform(0, max_depth)
    # logpdf(dist, signal)

end


function log_prior(S::Vector{T}) where T <: NamedTuple  #The reference function
    lp = 0.0
    μ = 3.0
    σ = 5.0 
    for t in 2:length(S)
        d = dist(S[t], S[t-1])
        lp += logpdf(Normal(μ, σ), d)
    end
    return lp
end


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


function simulateRW_s_init(tmax; s0=(x=100.0, y=100.0),
                           xdim=(1,400), ydim=(1,200),
                           sigma = 40.0)

    x = fill(-Inf, tmax)
    y = fill(-Inf, tmax)
    x[1] = s0.x
    y[1] = s0.y
    for t in 2:tmax
        while x[t] < xdim[1] || x[t] > xdim[2]
            Δx = randn()*sigma
            x[t] = x[t-1] + Δx
        end
        while y[t] < ydim[1] || y[t] > ydim[2]
            Δy = randn()*sigma
            y[t] = y[t-1] + Δy
        end
    end

    [(x=x[t], y=y[t]) for t in 1:tmax]
end


function simulate_bridge(tmax; A, B, σ = 3.0, α = 0.7, bathymetry_int)
    x = zeros(tmax);  y = zeros(tmax)
    x[1] = A.x;       y[1] = A.y
    for t in 2:tmax
        τ = tmax - t + 1
        σ_eff = σ #* sqrt(τ / tmax)
        found = false
        b_prev = get_depth((x=x[t-1], y=y[t-1]), bathymetry_int)
        for _ in 1:40
            x_cand = x[t-1] + randn()*σ_eff + α * (B.x - x[t-1]) / τ
            y_cand = y[t-1] + randn()*σ_eff + α * (B.y - y[t-1]) / τ
            b_cand = get_depth((x=x_cand, y=y_cand), bathymetry_int)
            # Constraint: the difference betw two consecutive depth measurements can't be too high
            if b_cand > 0 && abs(b_cand - b_prev) ≤ 10
                x[t], y[t] = x_cand, y_cand
                found = true
                break
            end
        end
        if !found
            #@info "No point found at iteration $t, function suspended."
            return nothing  # esci subito se non trovi acqua
        end
    end
    [(x=x[t], y=y[t]) for t in 1:tmax]
end



#------- HMC 
function infer_trajectories(Ydepth, Yaccustic, bathymetry_interpolated;
                            s_init, n_samples=100, tmax=100)

    isfinite(log_posterior(s_init, Ydepth, Yaccustic, bathymetry_interpolated)) || error("Initial value has zero likelihood!")

    #asFishCoord(lo, hi) = TransformVariables.compose(as𝕀, x -> lo + (hi - lo) * x, y -> (y - lo) / (hi - lo))
    mapping = as(Array,
    as((x = asℝ,          # coordinata libera su ℝ
        y = asℝ)),tmax)


    v_init = inverse(mapping, s_init)

    pp = TransformedLogDensity(mapping,
                               s -> log_posterior(s, Ydepth, Yaccustic, bathymetry_interpolated))

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
