## -------------------------------------------------------
##
## Find a fish with HMC
##
## February 15, 2024 -- Andreas Scheidegger
## andreas.scheidegger@eawag.ch
## -------------------------------------------------------

import Pkg


# prior for sigma  -> reference
# explorers
# way to parallelize

Pkg.activate(".")
Pkg.instantiate()

using Images: load, Gray, channelview
using Plots

include("find_PT3.jl")


# -----------
# Notes
#
# - N.B., this is prototype and is not working properly!!!
# - Meaning, even with simple simulated data it does not converge.
#
# Potential issues / todos:
#
# - land/no land results currently in a discontinuity in `log_prob_signal`, something HMC
#   does not like at all. We should smooth this somehow.
#   A simple unbounded likelihood (e.g. normal), however results in trajectories that are
#   often on land, or are even leaving the bounding box. Hence, we need a smarter formulation for that.
#
# - Use better initial values, i.e. optimize the posterior first. This alone may already
#   be a challenging.
#
# - Try Enzyme.jl or Mooncake.jl for faster Automatic differentiation
#
# - Test HMC with tempering:
#   - Pigeons.jl
#   - MCMCTempering.jl ?
#
# - Test Repelling-Attracting HMC?  https://github.com/sidv23/ra-hmc


#--------- basics
# Load bathymetry, negative values are treated as land:
bathymetry_map = channelview(Gray.(load("C:\\Users\\teresa i robert\\Desktop\\Physics of Data\\PoD Fish tracking\\Code\\fish_tracking_HMC\\HMC\\bathymetry_maps\\map_channel_mazemod2.jpg"))) * 100 .- 1;
#bathymetry_map = channelview(Gray.(load("C:\\Users\\teresa i robert\\Desktop\\Physics of Data\\PoD Fish tracking\\Code\\fish_tracking_HMC\\HMC\\bathymetry\\map_Firth_of_Lorn_200m_SEBA.tif"))) * 100 .- 1;

bathymetry_int = extrapolate(interpolate(bathymetry_map, BSpline(Linear())),-1.0);

receiver1 = Receiver((x=100, y=100), k=50.0, dist=30.0)
receiver2 = Receiver((x=300, y=100), dist=80.0)
receiver3 = Receiver((x=250, y=150), dist=50.0)

#--------- bridges
#Building bridges from receiver1 to receiver 2, which is a way to get all (if n_bridges, tmax and sigma are big enough to explore all the channels) the plausible 
#paths that the fish may have followed:
tmax = 200       




n_bridges = 400
bridges = []
for _ in 1:n_bridges
    bridge = simulate_bridge(tmax; A=receiver1, B=receiver2, σ=3.0, α=0.7, bathymetry_int=bathymetry_int)
    if bridge !== nothing
        push!(bridges, bridge)
    end
end

for _ in 1:n_bridges
    bridge = simulate_bridge(tmax; A=receiver1, B=receiver3, σ=3.0, α=0.7, bathymetry_int=bathymetry_int)
    if bridge !== nothing
        push!(bridges, bridge)
    end
end

#bridge = simulate_bridge(tmax; A=receiver1, B=receiver3, σ=3.0, α=0.7, bathymetry_int=bathymetry_int)
#push!(bridges, bridge)  

bridges = filter(!isnothing, bridges)  # keep just the valid trajectories
bridges_x = [[p.x for p in bridge] for bridge in bridges]
bridges_y = [[p.y for p in bridge] for bridge in bridges]


plt = heatmap(bathymetry_map[end:-1:1,:],
              xlim=(0, 400), ylim=(0, 200),
              color=:blues,
              legend=false,
              title="Brownian bridges betw receiver1 e receiver2")
for i in 1:length(bridges)
    plot!(plt, bridges_x[i], bridges_y[i], lw=2)
end
plot!(plt, make_circle(receiver1.x, receiver1.y, receiver1.dist), color=:red, label="Receiver 1")
plot!(plt, make_circle(receiver2.x, receiver2.y, receiver2.dist), color=:red, label="Receiver 1")

#s_init = bridges[1] #The trajectory that we try to optimize via PT
s_depth = bridges[1] #The trajectory from which we extract the depth data
s_init = bridges[end]
#Plotting the accepted bridges:
xs_d = [p.x for p in s_depth]
ys_d = [p.y for p in s_depth]

xs = [p.x for p in s_init]
ys = [p.y for p in s_init]

scatter!(plt, [receiver1.x, receiver2.x], [receiver1.y, receiver2.y], color=:red, label="Receivers")

display(plt)
plt = heatmap(bathymetry_map[end:-1:1,:],
              xlim=(0, 400), ylim=(0, 200),
              color=:blues,
              legend=false,
              title="Brownian bridges tra receiver1 e receiver2")
plot!(plt, xs, ys, lw=3, color=:red, label="chosen trajectory")
plot!(plt, xs_d, ys_d, lw=3, color=:blue, label="goal trajectory")

#--------- data
# Accustic signals:


receivers = [receiver1, receiver2]
Yaccustic = build_Yaccustic_from_trajectory(s_depth, receivers)

#Depth signal:
Ydepth = Tuple{Int, Float64, DepthGauge}[] 
depthgauge = DepthGauge()
sigma_noise=0.5


for (t, point) in enumerate(bridges[1])
    # Get the depth from the bathymetry
    d = get_depth((x=point.x, y=point.y), bathymetry_int)

    noisy_d = d + randn() * sigma_noise

    push!(Ydepth, (t+1, noisy_d, depthgauge))
end


tempi = [y[1] for y in Ydepth]
profondita = [y[2] for y in Ydepth]

# Plot tempo vs profondità
plot(tempi, profondita, xlabel="Tempo", ylabel="Profondità", label="Ydepth2", legend=:topright)



#--------- PT 
@info "Optimizing posterior for a MAP starting point…"              

#Setting the problem characteristics:
mapping = TransformVariables.as(Array, 
                                TransformVariables.as((x = TransformVariables.asℝ, y = TransformVariables.asℝ)),
                                tmax)
v_init = TransformVariables.inverse(mapping, s_init)


dimension(lp::FishLogPotential) = length(lp.v_init)
logdensity(lp::FishLogPotential, v::AbstractVector) = lp(v)
capabilities(::FishLogPotential) = LogDensityProblems.LogDensityOrder{0}()

dimension(lp::FishPriorPotential) = length(lp.v_init)
logdensity(lp::FishPriorPotential, v::AbstractVector) = lp(v)
capabilities(::FishPriorPotential) = LogDensityProblems.LogDensityOrder{0}()


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
    tmax, s0, sigma = 200, (x=100.0,y=100.0), 3.0
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


#=
LogDensityProblems.dimension(lp::FishLogPotential) = length(lp.v_init)

Pigeons.initialization(lp::FishLogPotential, rng::AbstractRNG, ::Int) = lp.v_init


#Actually running pigeons:
import Pkg
Pkg.status("PigeonsMCMCChainsExt")

@info "Running Pigeons parallel tempering…"
import Enzyme

#=
function sample_iid!(lp::FishPriorPotential, rng, v::AbstractVector)
    # scegli una traiettoria a caso dalle bridges disponibili
    bridge = rand(rng, bridges)
    v_sample = TransformVariables.inverse(lp.mapping, bridge)
    copyto!(v, v_sample)
    return v
end
=#
#=
function sample_iid!(lp::FishPriorPotential, rng, v::AbstractVector)
    # Genera una traiettoria random walk pura
    tmax = 200 # imposta il valore corretto
    s0 = (x = 100.0, y = 100.0) # punto di partenza
    sigma = 3.0 # o il valore che usi nel prior
    traj = simulateRW_s_init(tmax; s0=s0, sigma=sigma)
    v_sample = TransformVariables.inverse(lp.mapping, traj)
    copyto!(v, v_sample)
    return v
end
=#
function Pigeons.sample_iid!(lp::FishPriorPotential, replica, shared)
    rng   = replica.rng          # generatore casuale
    state = replica.state        # vettore in-place da riempire

    # -- genera un campione indipendente dal prior --------------------
    tmax  = 200
    s0    = (x = 100.0, y = 100.0)
    sigma = 3.0
    traj  = simulateRW_s_init(tmax; s0=s0, sigma=sigma, rng=rng)

    v_sample = TransformVariables.inverse(lp.mapping, traj)
    copyto!(state, v_sample)     # scrive sullo stato della replica
    return state                 # (facoltativo ma consigliato)
end
=#

fish_prior_lp = FishPriorPotential(mapping, v_init)
fish_lp = FishLogPotential(Ydepth, Yaccustic, bathymetry_int, mapping, v_init)

pt = pigeons(
    target            = fish_lp,               # log posterior to smaple from
    reference         = fish_prior_lp,         # reference distribution (that coincides with the distribution at beta=0)
    seed              = 1234,                  
    n_rounds          = 4,                     # up to 2^nround–1 scans
    n_chains          = 100,                    
    checkpoint        = false,                
    multithreaded     = true,                  
    #=explorer          = AutoMALA(
                           step_size            = 6.0,           # passo iniziale MALA
                           base_n_refresh       = 7,     #13        # passi base per esplorazione
                           exponent_n_refresh   = 0.5,           # scala con √dim
                           default_autodiff_backend = :ForwardDiff        # backend autodiff
                       ),=#
    record            = [traces; record_default()]  # registra traiettorie + diagnostica
)
pt_samples  = Chains(pt)         
#my_plot = StatsPlots.plot(pt_samples)
#StatsPlots.savefig(my_plot, "julia_posterior_densities_and_traces.svg");

cold_last_v = pt_samples.value[end, 1: 2*tmax] |> vec
cold_last_S = TransformVariables.transform(mapping, cold_last_v)

#Some plotting:
xs_pigeons = [p.x for p in cold_last_S] #extracting trajectories
ys_pigeons = [p.y for p in cold_last_S]

plt = heatmap(bathymetry_map[end:-1:1, :],
              xlim=(0, 400), ylim=(0, 200),
              color=:blues,
              title="Pigeons results")
plot!(plt, xs_pigeons, ys_pigeons, lw=3, color=:orange, label="Pigeons PT")
plot!(plt, xs, ys, lw=3, color=:red, label="starting trajectory")
plot!(plt, xs_d, ys_d, lw=3, color=:blue, label="goal trajectory")
plot!(plt, make_circle(receiver1.x, receiver1.y, receiver1.dist), color=:red, label="Receiver 1")
plot!(plt, make_circle(receiver2.x, receiver2.y, receiver2.dist), color=:red, label="Receiver 1")

scatter!(plt, [receiver1.x, receiver2.x], [receiver1.y, receiver2.y], color=:red, label="Receivers")
display(plt)



logposteriors = [logdensity(fish_lp, pt_samples.value[i, 1:2*tmax]) for i in 1:size(pt_samples.value, 1)]
plt_logpost = plot(logposteriors, xlabel="iteration", ylabel="Log-posterior", title="log-posterior evolution")

display(plt_logpost)


n_points = tmax
n_samples = size(pt_samples.value, 1)

for i in 1:n_samples
    v = pt_samples.value[i, 1:2*n_points]
    S = TransformVariables.transform(mapping, v)
    xs = [p.x for p in S]
    ys = [p.y for p in S]
    plot!(xs, ys, alpha=0.2, color=:orange, label=false)
end
heatmap!(bathymetry_map[end:-1:1, :], xlim=(0, 400), ylim=(0, 200), color=:blues, alpha=0.3, legend=false)



# Calcola i vettori nello spazio dei parametri
v_cold = pt_samples.value[end, 1:2*tmax]
v_init = TransformVariables.inverse(mapping, s_init)
v_depth = TransformVariables.inverse(mapping, s_depth)

# Calcola i log-posterior
logpost_cold = logdensity(fish_lp, v_cold)
logpost_init = logdensity(fish_lp, v_init)
logpost_depth = logdensity(fish_lp, v_depth)
logpost_init2 = log_posterior(s_init, Ydepth, Yaccustic, bathymetry_int)
logpost_depth2 = log_posterior(s_depth, Ydepth, Yaccustic, bathymetry_int)


println("Log-posterior cold chain: ", logpost_cold)
println("Log-posterior initial bridge: ", logpost_init)
println("Log-posterior depth bridge: ", logpost_depth)


print(cold_last_S)
YdepthPIGEONS = Tuple{Int, Float64, DepthGauge}[] 
depthgaugep = DepthGauge()
for (t, point) in enumerate(cold_last_S)
    # Get the depth from the bathymetry
    d = get_depth((x=point.x, y=point.y), bathymetry_int)
    push!(YdepthPIGEONS, (t+1,d, depthgaugep))
end


tempiP = [y[1] for y in YdepthPIGEONS]
profonditaP = [y[2] for y in YdepthPIGEONS]

plt = plot(tempi, profondita, xlabel="Tempo", ylabel="Profondità", label="Ydepth2", legend=:topright)
plot!(plt, tempiP, profonditaP, label="Ydepth PIGEONS")
display(plt)

#=
samples = infer_trajectories(
            Ydepth, Yaccustic, bathymetry_int;
            s_init = cold_last_S    ,    # proveniente dal PT
            #s_init = s_map,           # <‑‑ qui
            n_samples = 1_000,
            tmax = tmax)



# -----------
# 3) plot distribution

n_trajectories = 40          # Number of trajectories to plot
burnin = length(samples) ÷ 2
idx = rand(burnin:length(samples), n_trajectories)


# Estrai le coordinate della traiettoria iniziale
xs_init = [p.x for p in s_init]
ys_init = [p.y for p in s_init]

xs_map = [p.x for p in cold_last_S]
ys_map = [p.y for p in cold_last_S]

animation = @animate for t in 1:tmax
    plt = heatmap(bathymetry_map[end:-1:1,:],
                  xlim=(0, 400), ylim=(0, 200),
                  color=:blues,
                  title="t = $t")
    # Aggiungi la traiettoria iniziale (tutta, in ogni frame)
    plot!(plt, xs_init, ys_init, color=:orange, lw=2, label="Traiettoria iniziale")
    plot!(plt, xs_map, ys_map, color=:green, lw=2, label="Traiettoria map")

    add_trajectories!(plt, samples[idx], t)
    add_signal!(plt, Yaccustic, t);

    plt2 = plot_depth(Ydepth, tmax)
    vline!(plt2, [t])

    plot(plt, plt2, layout=(2,1))
end
gif(animation, "C:\\Users\\teresa i robert\\Desktop\\TEST\\TEST_2.gif", fps = 10)


# ---------


=#
