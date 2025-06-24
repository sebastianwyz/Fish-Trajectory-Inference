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
using Images
using Random
using TransformVariables
using Printf
# using TransformedLogDensities
# using DifferentiationInterface

# using Optim             
# using ForwardDiff

using LogDensityProblemsAD
import LogDensityProblems: logdensity, dimension, capabilities
using Pigeons
import Pigeons: initialization, sample_iid!
import TransformVariables: AbstractTransform
using MCMCChains
# using StatsPlots

# AutoEnzyme
using ADTypes
using Enzyme
using LogDensityProblems

# Displaying Diagnosis
#= using CSV
using DataFrames  =#


# -----------unction for coordinates
dist(c1, c2) = sqrt((c1.x - c2.x)^2 + (c1.y - c2.y)^2)
direction(c1, c2) = atan(c1.x - c2.x,  c1.y - c2.y)


# ----------- Bayesian problem
# p(X_t | X_{t_1}, X_{t-2}), movement model
#=
function log_p_moveRW(c1, c2)
    σ = 1.0 #1
    logpdf(Normal(c2.x, σ), c1.x) +
        logpdf(Normal(c2.y, σ), c1.y)  #uniform
end
=#

function log_p_moveRW(c1, c2)
    σ = 0.3  # controlla quanto "stretta" è la preferenza per la distanza 3
    d = dist(c1, c2)
    logpdf(Normal(3.0, σ), d)
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
    k = device.k
    # k = 5
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
    bathymetry(dy - s.y, s.x) / 100
end

function log_prob_signal(signal, s::NamedTuple, device::DepthGauge, map_int)

    # Depth at location s
    max_depth = get_depth(s, map_int)

    # not a very realsitc choise
    dist = Normal(max_depth, 15)   # 2
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
    #μ = 3.0
    #σ = 5.0 
    for t in 2:length(S)
        d = dist(S[t], S[t-1])
        lp += log_p_moveRW(S[t], S[t-1])
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
        #print(b_prev, "\n")
        for _ in 1:40
            x_cand = x[t-1] + randn()*σ_eff + α * (B.x - x[t-1]) / τ
            y_cand = y[t-1] + randn()*σ_eff + α * (B.y - y[t-1]) / τ
            b_cand = get_depth((x=x_cand, y=y_cand), bathymetry_int)
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
            if b_cand <= 0
                # Se non è acqua, riprova (o copia il punto precedente)
                x_cand, y_cand = x[t-1], y[t-1]
            end
        end

        x[t] = x_cand
        y[t] = y_cand
        # print("x: ",x[t]-x[t-1],"  y:",  y[t]-y[t-1], "d ", sqrt((x[t]-x[t-1])^2 + (y[t]-y[t-1])^2 ), "\n")
    end

    # Costruisci la traiettoria
    [(x=x[t], y=y[t]) for t in 1:tmax]
end

function simulate_unbiased_path(
    tmax::Int;
    rec1::Receiver,              
    rec2::Receiver,              
    σ_step::Float64 = 1.0,
    bathymetry_int::Any,
    max_retries::Int = 100,
    coarse_steps::Int = 50,
    min_endpoint_prob::Float64 = 0.5,
    noise_σ::Float64 = 0.5
)
    for _ in 1:max_retries
        # === Step 1: Generate main path with only positive x steps and over water ===
        x = Float64[]
        y = Float64[]
        push!(x, rec1.x)
        push!(y, rec1.y)

        valid = true
        for _ in 2:coarse_steps  # arbitrarily chosen max main path length
            success = false
            count=0
            for attempt in 1:30
                inner_count=0
                dx = abs(randn() * σ_step)  # x always increasing
                dy = randn() * σ_step

                x_new = x[end] + dx
                y_new = y[end] + dy

                b = get_depth((x = x_new, y = y_new), bathymetry_int)
                c = get_depth((x = x[end], y = y[end]), bathymetry_int)
                if (b > 0) #&& (abs(b-c)<=20)
                    push!(x, x_new)
                    push!(y, y_new)
                    success = true
                    break
                end
            end
            if !success
                valid = false
                break
            end
        end

        if !valid
            continue  # Retry
        end

        # === Step 2: Interpolate to tmax steps ===
        n_main = length(x)
        ts_main = range(1, n_main, length=n_main)
        ts_interp = range(1, n_main, length=tmax)

        x_interp = LinearInterpolation(ts_main, x, extrapolation_bc=Line())
        y_interp = LinearInterpolation(ts_main, y, extrapolation_bc=Line())
        interp_path = [(x = x_interp(t), y = y_interp(t)) for t in ts_interp]

        # === Step 3: Check detectability of final point ===
        final_point = interp_path[end]
        logp = log_prob_signal(:detect, final_point, rec2)
        prob = exp(logp)
        if prob < min_endpoint_prob
            continue
        end

        # === Step 4: Add noise to the interpolated path ===
        noisy_path = [(x = p.x + randn() * noise_σ, y = p.y + randn() * noise_σ) for p in interp_path]
        return noisy_path
    end

    return nothing  # failed after all retries
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

#########################  FishLogPotential  #########################
# — vecchia firma: ritorna un nuovo vettore —
function initialization(lp::FishLogPotential,
                        rng::AbstractRNG,
                        replica_index::Int)
    return copy(lp.v_init)        # oppure rand iniziale, ma dentro il supporto
end

# — nuova firma (≥ v0.4): riempie in-place un vettore esistente —
function initialization(lp::FishLogPotential,
                        rng::AbstractRNG,
                        v::AbstractVector)
    copyto!(v, lp.v_init)
    return v
end


#########################  FishPriorPotential  #########################
# ---------- vecchia firma ----------
function sample_iid!(lp::FishPriorPotential,
                     rng::AbstractRNG,
                     v::AbstractVector)
    tmax, s0, sigma = 100, (x=10.0,y=45.0), 3.0
    while true
        traj = simulateRW_s_init(tmax; s0=s0, sigma=sigma, rng=rng)
        isfinite(fish_lp(TransformVariables.inverse(lp.mapping, traj))) || continue
        copyto!(v, TransformVariables.inverse(lp.mapping, traj))
        return v
    end
end

# ---------- nuova firma (≥ v0.4) ----------
function sample_iid!(lp::FishPriorPotential,
                     replica,
                     shared)
    sample_iid!(lp, replica.rng, replica.state)
    return replica.state
end