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





function get_depth_at(bathy::GeoArray, interp, x::Float64, y::Float64)
    
    f_inv = inv(bathy.f)
    colrow = f_inv(SVector(x, y))
    row = colrow[1]
    col = colrow[2]
    return interp(row, col)
end




function log_prob_signal(signal, s::NamedTuple, device::DepthGauge, 
                        bathymetry_int, x_origin, y_origin, dx, dy, 
                        bathy, ex_itp)  # ← AGGIUNGI PARAMETRI
    max_depth = get_depth_at(bathy, ex_itp, s.x, s.y)
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



function log_posterior(S::Vector{T}, Ydepth, Yaccustic, bathymetry_int, 
                      x_origin, y_origin, dx, dy, bathy, ex_itp) where T <: NamedTuple
    tmax = length(S)
    lp   = zero(eltype(T))

    for t in 2:tmax
        lp += log_p_moveRW(S[t], S[t-1])
    end

    for (t, signal, device) in Ydepth
        t ≤ tmax || continue
        lp += log_prob_signal(signal, S[t], device, bathymetry_int, 
                             x_origin, y_origin, dx, dy, bathy, ex_itp)  # ← AGGIUNGI
    end
    for (t, signal, device) in Yaccustic
        t ≤ tmax || continue
        lp += log_prob_signal(signal, S[t], device)
    end

    return lp
end







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



struct FishPriorPotential{M,V}
    mapping :: M
    v_init  :: V
end

function (lp::FishPriorPotential)(v::AbstractVector)
    S = TransformVariables.transform(lp.mapping, v)
    return log_prior(S)
end





# Nel file find_PT3_REALDATA_jose.jl, modifica la struct (circa riga 500):
struct FishLogPotential{YD,YA,BI,M,B,E}
    Ydepth    ::Vector{YD}
    Yacc      ::Vector{YA}
    bathy_int ::BI
    x_origin  ::Float64
    y_origin  ::Float64
    dx        ::Float64
    dy        ::Float64
    mapping   ::M
    v_init    ::Vector{Float64}
    bathy     ::B          # ← AGGIUNGI
    ex_itp    ::E          # ← AGGIUNGI
end

function (lp::FishLogPotential)(v::AbstractVector)
    S = TransformVariables.transform(lp.mapping, v)
    return log_posterior(S, lp.Ydepth, lp.Yacc, lp.bathy_int, 
                        lp.x_origin, lp.y_origin, lp.dx, lp.dy,
                        lp.bathy, lp.ex_itp)  # ← PASSA ANCHE QUESTI
end


dimension(lp::FishLogPotential) = length(lp.v_init)
logdensity(lp::FishLogPotential, v::AbstractVector) = lp(v)
capabilities(::FishLogPotential) = LogDensityProblems.LogDensityOrder{0}()

dimension(lp::FishPriorPotential) = length(lp.v_init)
logdensity(lp::FishPriorPotential, v::AbstractVector) = lp(v)
capabilities(::FishPriorPotential) = LogDensityProblems.LogDensityOrder{0}()



function initialization(lp::FishLogPotential,
                        rng::AbstractRNG,
                        replica_index::Int)
    return copy(lp.v_init)        # oppure rand iniziale, ma dentro il supporto
end

function initialization(lp::FishLogPotential,
                        rng::AbstractRNG,
                        v::AbstractVector)
    copyto!(v, lp.v_init)
    return v
end


function sample_iid!(lp::FishPriorPotential,
                     rng::AbstractRNG,
                     v::AbstractVector;
                     tries = 300)

    s0, σ =  (x = 709_757.1116, y = 6.2677260356e6), 2.0

    for _ in 1:tries
        traj = simulateRW_free(tmax; s0 = s0, sigma = 2.0, rng = rng)
        copyto!(v, TransformVariables.inverse(lp.mapping, traj))
        return v
    end

    # fallback deterministico (mai su terra)
    traj = fill(s0, tmax + 1)
    @warn "sample_iid! fallback after $tries tentativi → path piatto"
    copyto!(v, TransformVariables.inverse(lp.mapping, traj))
    return v
end

function sample_iid!(lp::FishPriorPotential, replica, shared)
    sample_iid!(lp, replica.rng, replica.state)   # riusa la funzione sopra
    return replica.state
end

