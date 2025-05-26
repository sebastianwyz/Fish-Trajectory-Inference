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

include("findfish_HMC3.jl")


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


# -----------
# 1) Define observation signals

# ---
# Load bathymetry

# negative values are treated as land
bathymetry_map = channelview(Gray.(load("C:\\Users\\teresa i robert\\Desktop\\Physics of Data\\PoD Fish tracking\\Code\\fish_tracking_HMC\\HMC\\bathymetry_maps\\map_channel_maze18.jpg"))) * 100 .- 1;

#bathymetry_map = channelview(Gray.(load("C:\\Users\\teresa i robert\\Desktop\\Physics of Data\\PoD Fish tracking\\Code\\fish_tracking_HMC\\HMC\\bathymetry\\map_Firth_of_Lorn_100m_SEBA.tif"))) * 100 .- 1;

# interpolate the bathymetry, so that gradients are defined
bathymetry_int = extrapolate(
    interpolate(bathymetry_map, BSpline(Linear())),
    -1.0);


# ---
# Observations are a vector of tuples:
# (timepoint, signal, sensor)

# Depth path
Ydepth = Tuple{Int, Float64, DepthGauge}[]
depthgauge = DepthGauge();
for t in 1:100
    if t < 30   #30
        d = 80 + randn()
    elseif t > 81
        d = 70 + randn()
    else
        d = 15 + randn()
    end
    push!(Ydepth, (t, d, depthgauge))
end

# Accustic signals
receiver1 = Receiver((x=100, y=100), k=50.0, dist=30.0)
receiver2 = Receiver((x=300, y=100), dist=50.0)

"""Yaccustic = Tuple{Int, Symbol, Receiver}[]

for t in 10:30
    push!(Yaccustic, (t, :detect, receiver1))
    push!(Yaccustic, (t, :nondetect, receiver2))
end
for t in 81:100
    push!(Yaccustic, (t, :nondetect, receiver1))
    push!(Yaccustic, (t, :detect, receiver2))
end

print(Yaccustic)"""

# -----------
# 2) Run inference

tmax = 100                            
#s_init = simulateRW_s_init(tmax, xdim=(90,100), ydim=(90,100))
n_bridges = 100
#bridges = [simulate_bridge(tmax; A=receiver1, B=receiver2, σ=2.0, α=0.7, bathymetry_int=bathymetry_int) for _ in 1:n_bridges]
bridges = []
for _ in 1:n_bridges
    bridge = simulate_bridge(tmax; A=receiver1, B=receiver2, σ=3.0, α=0.7, bathymetry_int=bathymetry_int)
    if bridge !== nothing
        push!(bridges, bridge)
        #break  # fermati appena trovi una traiettoria valida
    end
end
bridges = filter(!isnothing, bridges)  # tieni solo le traiettorie valide

bridges_x = [[p.x for p in bridge] for bridge in bridges]
bridges_y = [[p.y for p in bridge] for bridge in bridges]
# Plotta la batimetria e le traiettorie
plt = heatmap(bathymetry_map[end:-1:1,:],
              xlim=(0, 400), ylim=(0, 200),
              color=:blues,
              legend=false,
              title="Brownian bridges tra receiver1 e receiver2")

for i in 1:length(bridges)
    plot!(plt, bridges_x[i], bridges_y[i], lw=2)
end
s_init = bridges[1] #simulate_bridge(tmax; A=receiver1, B=receiver2, σ=2.0, α=0.7, bathymetry_int=bathymetry_int, Ydepth=Ydepth)
s_depth = bridges[2]

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
plot!(plt, xs, ys, lw=3, color=:red, label="bridge")
plot!(plt, xs_d, ys_d, lw=3, color=:blue, label="depth")
#s_init = simulate_bridge(tmax; A = receiver1, B = receiver2, σ = 2.0)
# Estrai tempo e profondità da Ydepth2


# ...existing code...

# ...existing code...

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

# Sostituisci la generazione manuale di Yaccustic con:
receivers = [receiver1, receiver2]
Yaccustic = build_Yaccustic_from_trajectory(s_init, receivers)

# ...existing code...

# ...existing code...
#Yaccustic    = build_Yaccustic_conditional(truth_bridge, receivers)
#print(Yaccustic)
Ydepth2 = Tuple{Int, Float64, DepthGauge}[]
depthgauge = DepthGauge()
for (t, point) in enumerate(bridges[2])
    d = get_depth((x=point.x, y=point.y), bathymetry_int)
    push!(Ydepth2, (t, d, depthgauge))
end

tempi = [y[1] for y in Ydepth2]
profondita = [y[2] for y in Ydepth2]

# Plot tempo vs profondità
plot(tempi, profondita, xlabel="Tempo", ylabel="Profondità", label="Ydepth2", legend=:topright)


# ---- nuovo: ottimizziamo
@info "Optimizing posterior for a MAP starting point…"
s_map = map_estimate(Ydepth2, Yaccustic, bathymetry_int;
                     s_init = s_init, tmax = tmax)
#########################  (2b) Exploration con PT  ###########################
using Pigeons                           # <-- NEW
using LogDensityProblemsAD
using MCMCChains                        # per convertire i tracciati PT

# stesso mapping usato dall’HMC
mapping = TransformVariables.as(Array,
            TransformVariables.as((x = TransformVariables.asℝ,
                                   y = TransformVariables.asℝ)),
            tmax)

v_init = TransformVariables.inverse(mapping, s_init)
#v_init = TransformVariables.inverse(mapping, s_random)

@show length(v_init)  # Deve essere 200

function log_prior(S::Vector{T}) where T <: NamedTuple
    lp = 0.0
    μ = 3.0  # passo atteso (puoi adattare)
    σ = 5.0   # deviazione standard del passo
    for t in 2:length(S)
        d = dist(S[t], S[t-1])
        lp += logpdf(Normal(μ, σ), d)
    end
    return lp
end

############################ FishLogPotential ##########################
using LogDensityProblems
using Pigeons
using Pigeons: HMC
using Random
using TransformVariables
import TransformVariables: AbstractTransform
using Pigeons: record_default

struct FishLogPotential{YD,YA,BI,M}
    Ydepth   ::Vector{YD}
    Yacc     ::Vector{YA}
    bathy_int::BI
    mapping  ::M
    v_init   ::Vector{Float64}
end


struct FishPriorPotential{M,V}
    mapping :: M
    v_init  :: V
end

import LogDensityProblems: logdensity, dimension, capabilities

dimension(lp::FishLogPotential) = length(lp.v_init)
logdensity(lp::FishLogPotential, v::AbstractVector) = lp(v)
capabilities(::FishLogPotential) = LogDensityProblems.LogDensityOrder{0}()

dimension(lp::FishPriorPotential) = length(lp.v_init)
logdensity(lp::FishPriorPotential, v::AbstractVector) = lp(v)
capabilities(::FishPriorPotential) = LogDensityProblems.LogDensityOrder{0}()

function (lp::FishPriorPotential)(v::AbstractVector)
    S = TransformVariables.transform(lp.mapping, v)
    return log_prior(S)
end

function (lp::FishLogPotential)(v::AbstractVector)
    S = TransformVariables.transform(lp.mapping, v)
    return log_posterior(S, lp.Ydepth, lp.Yacc, lp.bathy_int)
end

#dimensione del problema 
LogDensityProblems.dimension(lp::FishLogPotential) = length(lp.v_init)

# punto iniziale
Pigeons.initialization(lp::FishLogPotential,
                       rng::AbstractRNG,
                       ::Int) = lp.v_init
########################################################################

fish_prior_lp = FishPriorPotential(mapping, v_init)

fish_lp = FishLogPotential(Ydepth2, Yaccustic, bathymetry_int,
                           mapping, v_init)
function record_custom(pt::PT)
    samples = []  # Array per memorizzare i campioni
    function callback(state)
        push!(samples, state)  # Aggiungi lo stato corrente
    end
    return callback
end

#recorder = record_custom(pt)  # Crea il registratore personalizzato
import Pkg
Pkg.status(["Pigeons", "PigeonsMCMCChainsExt", "MCMCChains", "LogDensityProblems", "TransformVariables"])
@info "Running Pigeons parallel tempering…"
@show length(v_init)
@show TransformVariables.dimension(mapping)

using ADTypes


pt = pigeons(
    target            = fish_lp,               # distribuzione target
    reference         = fish_prior_lp,               # prior identico alla target
    seed              = 1234,                  # seme per la riproducibilità
    n_rounds          = 3,                     # fino a 2^5–1 = 31 scans iniziali
    n_chains          = 4,                    # numero di catene parallele
    n_chains_variational = 0,                  # disabilita la fase variational
    checkpoint        = false,                 # no checkpoint su disco
    multithreaded     = true,                  # sfrutta i thread locali
    explorer          = AutoMALA(
                           step_size            = 6.0,           # passo iniziale MALA
                           base_n_refresh       = 13,     #13        # passi base per esplorazione
                           exponent_n_refresh   = 0.5,           # scala con √dim
                           default_autodiff_backend = :ForwardDiff    # backend autodiff
                       ),
    record            = [traces; record_default()]  # registra traiettorie + diagnostica
)
pt_samples  = Chains(pt)              # MCMCChains.Chains
using StatsPlots

my_plot = StatsPlots.plot(pt_samples)
StatsPlots.savefig(my_plot, "julia_posterior_densities_and_traces.html");

pt_samples


cold_last_v = pt_samples.value[end, 1: 2*tmax] |> vec

@show length(cold_last_v)
@show TransformVariables.dimension(mapping)

cold_last_S = TransformVariables.transform(mapping, cold_last_v)

# Estrai le coordinate della traiettoria proposta da Pigeons
xs_pigeons = [p.x for p in cold_last_S]
ys_pigeons = [p.y for p in cold_last_S]

xs = [p.x for p in s_init]
ys = [p.y for p in s_init]

xs_d = [p.x for p in s_depth]
ys_d = [p.y for p in s_depth]
# Plotta la batimetria e la traiettoria Pigeons
plt = heatmap(bathymetry_map[end:-1:1, :],
              xlim=(0, 400), ylim=(0, 200),
              color=:blues,
              title="Traiettoria proposta da Pigeons")

# Plotta la traiettoria Pigeons
plot!(plt, xs_pigeons, ys_pigeons, lw=3, color=:orange, label="Pigeons PT")
plot!(plt, xs, ys, lw=3, color=:red, label="bridge")
plot!(plt, xs_d, ys_d, lw=3, color=:blue, label="depth")


# Plotta i ricevitori
scatter!(plt, [receiver1.x, receiver2.x], [receiver1.y, receiver2.y], color=:red, label="Receivers")

display(plt)
# opzionale: usa più campioni freddi come punti di partenza diversi
###########################################################################

logposteriors = [logdensity(fish_lp, pt_samples.value[i, 1:2*tmax]) for i in 1:size(pt_samples.value, 1)]
plot(logposteriors, xlabel="Iterazione", ylabel="Log-posterior", title="Evoluzione del log-posterior")






using StatsPlots

# Supponiamo che ogni sample sia un vettore di lunghezza 200 (100 punti x,y)
n_points = tmax
n_samples = size(pt_samples.value, 1)

# Estrai solo la catena fredda (di solito la prima o l'ultima, verifica con la doc di Pigeons)
# Qui assumiamo che la catena fredda sia la prima
for i in 1:n_samples
    v = pt_samples.value[i, 1:2*n_points]
    S = TransformVariables.transform(mapping, v)
    xs = [p.x for p in S]
    ys = [p.y for p in S]
    plot!(xs, ys, alpha=0.2, color=:orange, label=false)
end

# Plotta la batimetria sotto
heatmap!(bathymetry_map[end:-1:1, :], xlim=(0, 400), ylim=(0, 200), color=:blues, alpha=0.3, legend=false)




# HMC-NUTS sampler
samples = infer_trajectories(
            Ydepth2, Yaccustic, bathymetry_int;
            s_init = cold_last_S    ,    # proveniente dal PT
            #s_init = s_map,           # <‑‑ qui
            n_samples = 1_000,
            tmax = tmax)



# -----------
# 3) plot distribution

n_trajectories = 10          # Number of trajectories to plot
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

    plt2 = plot_depth(Ydepth2, tmax)
    vline!(plt2, [t])

    plot(plt, plt2, layout=(2,1))
end
gif(animation, "C:\\Users\\teresa i robert\\Desktop\\TEST\\TEST_realdata_1.gif", fps = 10)


# ---------

