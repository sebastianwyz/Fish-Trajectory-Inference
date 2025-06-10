import Pkg
using XLSX            # lettura xlsx
using DataFrames      # gestione tabelle
using Dates           # DateTime
using Missings        # gestione dei NA
#import Pigeons.Recorders: Receiver, DepthGauge
#Pkg.add("CSV")
import CSV
using CSV

Pkg.activate(".")
Pkg.instantiate()
#import GeoArrays
using GeoArrays
using Images: load, Gray, channelview
using Plots

include("find_PT3.jl")




spatial_resolution = 200                         # spatial resolution


# -- bathymetry
bathy_orig = GeoArrays.read("C:\\Users\\teresa i robert\\Desktop\\Physics of Data\\PoD Fish tracking\\Code\\fish_tracking_HMC\\HMC\\bathymetry\\map_Firth_of_Lorn_$(spatial_resolution)m.tif")

bathy = GeoArrays.read("C:\\Users\\teresa i robert\\Desktop\\Physics of Data\\PoD Fish tracking\\Code\\fish_tracking_HMC\\HMC\\bathymetry\\map_Firth_of_Lorn_$(spatial_resolution)m.tif")
#bathy = channelview(Gray.(load("C:\\Users\\teresa i robert\\Desktop\\Physics of Data\\PoD Fish tracking\\Code\\fish_tracking_HMC\\HMC\\bathymetry\\map_Firth_of_Lorn_200m.tif"))) * 100 .- 1;
#print(bathy)
#bathy = bathy[340:360, 430:450, :] # zoom in a bit
bathy = bathy[300:380, 410:470, :] # zoom in a bit

bathymetry_int = extrapolate(interpolate(bathy, BSpline(Linear())),-1.0);

aff = bathy.f
A = aff.linear      # 2x2 matrice (scaling e rotazione)
b = aff.translation # 2-element vector (origine, coordinate reali del pixel (1,1))

x_origin = b[1]     # coordinata x del centro del pixel (1,1)
y_origin = b[2]     # coordinata y del centro del pixel (1,1)
dx = A[1,1]         # passo in x (dimensione pixel)
dy = A[2,2]         # passo in y (dimensione pixel)

println("x_origin = ", x_origin)
println("y_origin = ", y_origin)
println("dx = ", dx)
println("dy = ", dy)

println(typeof(bathymetry_int))

# Supponiamo che s sia un NamedTuple o struct con campi .x e .y
s = (x=710799.3, y=6.267726e6)

#s = (x = 10, y = 10)
# bathymetry_int: oggetto di interpolazione
# x_origin, y_origin, dx, dy: già calcolati come sopra

depth = get_depth_p(s, bathymetry_int, x_origin, y_origin, dx, dy)
println("Profondità interpolata: ", depth)



x_test = 10.1
y_test = 20.0
#x_test = 709757.111649658
#y_test = 6.26772603565296e6
row_start = 340
col_start = 430

x = 709812.9206002407
y = 6.267736355257149e6
idx_full = GeoArrays.indices(bathy_orig, (x_test, y_test))
row_full, col_full = Tuple(idx_full)  # Converte il CartesianIndex in tupla
row = row_full - row_start + 1
col = col_full - col_start + 1
println("Profondità interpolata: ", bathymetry_int(row, col))

#depth = get_depth((x=710799.5, y=6.267726e6), bathymetry_int, bathy_orig, 340, 430)


using Interpolations
itp = interpolate(bathy, BSpline(Linear()), OnGrid())

#depth = get_depth((x=710798.1, y=6.267726e6), itp, bathy_orig, row_start, col_start)


x_test = 10.000000001
y_test = 10
#row, col = GeoArrays.indices(bathy, (x_test, y_test))
println("Profondità interpolata: ", bathymetry_int(x_test, y_test))

plt = heatmap(bathy,
              color = :blues,
              legend = false,
              title = "Posizioni dei receiver sulla batimetria")
#


depth_obs_df = CSV.read("C:\\Users\\teresa i robert\\Desktop\\Physics of Data\\PoD Fish tracking\\Code\\fish_tracking_HMC\\HMC\\observation\\depth.csv", DataFrame,
                        dateformat="yyyy-mm-dd H:M:S")
rename!(depth_obs_df, :Column1 => :time)

# N.B. original depth data are incompatible with the other observations!
depth_obs_df.depth[2041:2047] .= 205
depth_obs_df.depth[4842:4845] .= 205

depth_signals = depth_obs_df.depth
####################################################à
#start_idx = 10400
#end_idx = 10500
start_idx = 1
end_idx = 2500
# Filtra il DataFrame delle profondità
depth_obs_df = depth_obs_df[start_idx:end_idx, :]
depth_signals = depth_obs_df.depth
########################################################

#
##
moorings_df = CSV.read("C:\\Users\\teresa i robert\\Desktop\\Physics of Data\\PoD Fish tracking\\Code\\fish_tracking_HMC\\HMC\\observation\\moorings.csv", DataFrame)
#acoustic_pos = tuple.(moorings_df.receiver_x, moorings_df.receiver_y)
acoustic_pos = [(x=x, y=y) for (x, y) in zip(moorings_df.receiver_x, moorings_df.receiver_y)]
#print(rows)
print(acoustic_pos[1])

acoustic_obs_df = CSV.read("C:\\Users\\teresa i robert\\Desktop\\Physics of Data\\PoD Fish tracking\\Code\\fish_tracking_HMC\\HMC\\observation\\acoustics.csv", DataFrame,
                           dateformat="yyyy-mm-dd H:M:S", missingstring="NA")

                           print(acoustic_obs_df)
# each row contains the observation of a sensor
# '-1' stands for "no signal"
acoustic_array = coalesce.(Array(acoustic_obs_df[:,2:end]), -1)
###########################################################################
acoustic_array = acoustic_array[start_idx:end_idx, :]
for t in 1:size(acoustic_array, 1)
    for r in 1:size(acoustic_array, 2)
        if acoustic_array[t, r] == 1
            println("Tempo: $(t + start_idx - 1), Receiver: $r")
        end
    end
end




plt = heatmap(bathy,
              color = :blues,
              legend = false,
              title = "Posizioni dei receiver sulla batimetria")
xs = first.(acoustic_pos)      # prende il primo elemento di ogni tupla
ys = last.(acoustic_pos)  
scatter!(plt, xs, ys;
         color = :red,
         markersize = 4,
         label = "Receivers")
scatter!(plt, [709757.111649658], [6.26772603565296e6]; color = :green, markersize = 4, label = "Starting point")
#print(xs)
display(plt)





using Plots

# Supponiamo che acoustic_array sia una matrice (tempi x receiver)
times = Int[]
receivers = Int[]





for t in 1:size(acoustic_array, 1)
    for r in 1:size(acoustic_array, 2)
        if acoustic_array[t, r] == 1
            push!(times, t)
            push!(receivers, r)
        end
    end
end

plt = scatter(times, receivers;
    xlabel="Tempo (indice)",
    ylabel="Receiver (indice)",
    title="Attivazioni dei receiver nel tempo",
    legend=false,
    markersize=4,
    color=:red)

display(plt)




n_bridges = 100
bridges = []

start_point = (x=709757.111649658, y=6.26772603565296e6)
start_point = (709757.111649658, 6.26772603565296e6)
idx = GeoArrays.indices(bathy, start_point)
bathy[idx]

start_point = (70974.9526917059, 6.2677292579214815e6)
idx = GeoArrays.indices(bathy, start_point)


function build_receiver_sequence(acoustic_array,
                                 acoustic_pos;
                                 start_point::NamedTuple)

    n_time, n_receivers = size(acoustic_array)

    #### 1) Trova le prime attivazioni come prima ###############
    last_state = falses(n_receivers)
    events = Tuple{Int,Int}[]         # (tempo, idx_receiver)

    for t in 1:n_time, r in 1:n_receivers
        cur = acoustic_array[t, r] == 1
        if cur && !last_state[r]
            push!(events, (t, r))
        end
        last_state[r] = cur
    end

    #### 2) Costruisci sequenza e passi #########################
    if isempty(events)
        # → nessuna attivazione
        receiver_seq = [start_point, start_point]
        t_steps      = [max(1, n_time - 1)]

    elseif length(events) == 1
        # → una sola attivazione
        only_rec     = acoustic_pos[events[1][2]]
        receiver_seq = [only_rec, only_rec]
        # lunghezza tratta: dal primo evento alla fine della serie
        t_steps      = [max(1, n_time - events[1][1])]

    else
        # → caso normale
        receiver_seq = [acoustic_pos[r] for (_t, r) in events]
        t_steps      = [events[i+1][1] - events[i][1] for i in 1:length(events)-1]
    end

    return receiver_seq, t_steps
end

receiver_seq, t_steps =build_receiver_sequence(acoustic_array, acoustic_pos;
                            start_point = (x=709757.1, y=6.267726e6))

print("Receiver sequence: ", receiver_seq, "\n")
print("Time steps: ", t_steps, "\n")
println("Numero di receiver unici: ", length(unique(receiver_seq)))




function make_linear_traj(receiver_seq::Vector{<:NamedTuple},
                          t_steps::Vector{<:Integer};
                          noise_std::Real = 1.0)

    @assert length(receiver_seq) ≥ 2
    @assert length(t_steps)     == length(receiver_seq) - 1

    traj = Vector{NamedTuple{(:x,:y),Tuple{Float64,Float64}}}()

    for (seg, (A, B)) in enumerate(zip(receiver_seq[1:end-1],
                                       receiver_seq[2:end]))
        n = t_steps[seg]                        # quante suddivisioni
        for k in 0:n                           # 0 … n  → n+1 punti
            λ = n == 0 ? 1.0 : k / n           # parametro [0,1]
            x = (1-λ)*A.x + λ*B.x + randn()*noise_std
            y = (1-λ)*A.y + λ*B.y + randn()*noise_std
            # evita il duplicato del punto iniziale di ogni segmento,
            # tranne che per il primissimo
            if seg == 1 || k > 0
                push!(traj, (x=x, y=y))
            end
        end
    end
    return traj
end


print(t_steps)

traj = make_linear_traj(receiver_seq, t_steps; noise_std = 3.0)



plt = heatmap(bathy,
              color = :blues,
              legend = false,
              title = "Posizioni dei receiver sulla batimetria")

plot!(plt, traj)

scatter!(plt, first.(traj), last.(traj); m=:circle, ms=2, color=:black)
display(plt)

print(traj)





xs = [p.x for p in traj]
ys = [p.y for p in traj]
delta_x = diff(xs)                # vettore delle differenze tra x consecutivi
mean_delta_x = mean(abs.(delta_x))  # media dei valori assoluti delle differenze
print(xs)
println("Delta x medio: ", mean_delta_x)


s_init = traj


receivers = [
    Receiver(
        moorings_df.receiver_x[i],
        moorings_df.receiver_y[i],
        50.0,   # dist: puoi mettere il valore reale o uno standard
        30.0    # k: idem
    ) for i in 1:nrow(moorings_df)
]



Yaccustic = Tuple{Int, Symbol, Receiver}[]
for t in 1:size(acoustic_array, 1)
    detects = [acoustic_array[t, r] == 1 for r in 1:size(acoustic_array, 2)]
    if any(detects)
        for r in 1:size(acoustic_array, 2)
            stato = acoustic_array[t, r]
            if stato != -1
                signal = stato == 1 ? :detect : :nondetect
                push!(Yaccustic, (t, signal, receivers[r]))
            end
        end
    end
end



Ydepth = Tuple{Int, Float64, DepthGauge}[] 
depthgauge = DepthGauge()

for (t, d) in enumerate(depth_signals)
    push!(Ydepth, (t, d, depthgauge))
end

print(Ydepth)

tempi = [y[1] for y in Ydepth]
profondita = [y[2] for y in Ydepth]

# Plot tempo vs profondità
plot(tempi, profondita, xlabel="Tempo", ylabel="Profondità", label="Ydepth2", legend=:topright)
tmax = 100


#--------- PT 
@info "Optimizing posterior for a MAP starting point…"              

#Setting the problem characteristics:
mapping = TransformVariables.as(Array, 
                                TransformVariables.as((x = TransformVariables.asℝ, y = TransformVariables.asℝ)),
                                tmax)
v_init = TransformVariables.inverse(mapping, s_init)
print(v_init)

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


s0, σ =  (x = 709_757.1116, y = 6.2677260356e6), 2.0

traj = simulateRW_free(tmax; s0 = s0, sigma = 2.0, rng = Random.default_rng())


function sample_iid!(lp::FishPriorPotential,
                     rng::AbstractRNG,
                     v::AbstractVector;
                     tries = 300)

    s0, σ =  (x = 709_757.1116, y = 6.2677260356e6), 2.0

    for _ in 1:tries
        traj = simulateRW_free(tmax; s0 = s0, sigma = 2.0, rng = rng)

        #isfinite(lp.target(TransformVariables.inverse(lp.mapping, traj))) || continue


        #all_water(traj) || continue              # controllo O(tmax)
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


fish_prior_lp = FishPriorPotential(mapping, v_init)

fish_lp = FishLogPotential(
    Ydepth,
    Yaccustic,
    bathymetry_int,
    x_origin,
    y_origin,
    dx,
    dy,
    mapping,
    v_init
)
using Pigeons
using Pigeons: round_trip
#println(Pigeons.VERSION)
#using InfernenceReport

#=
@time fish_lp(v_init)
@time fish_prior_lp(v_init)
println(any(ismissing, depth_signals))
println(any(ismissing, acoustic_array))
println(extrema(depth_signals))
println(extrema(acoustic_array))
# ...existing code...

println("DEBUG: typeof(s_init) = ", typeof(s_init))
println("DEBUG: typeof(Ydepth) = ", typeof(Ydepth))
println("DEBUG: typeof(Yaccustic) = ", typeof(Yaccustic))

println("DEBUG: tmax = ", tmax)
println("DEBUG: mapping type = ", typeof(mapping))
println("DEBUG: v_init type = ", typeof(v_init), ", length = ", length(v_init))
println("DEBUG: s_init type = ", typeof(s_init), ", length = ", length(s_init))

println("DEBUG: Ydepth type = ", typeof(Ydepth), ", length = ", length(Ydepth))
for (i, y) in enumerate(Ydepth)
    println("  Ydepth[$i] = ", y, " | type: ", typeof(y))
end

println("DEBUG: Yaccustic type = ", typeof(Yaccustic), ", length = ", length(Yaccustic))
for (i, y) in enumerate(Yaccustic)
    println("  Yaccustic[$i] = ", y, " | type: ", typeof(y))
end

println("DEBUG: receivers type = ", typeof(receivers), ", length = ", length(receivers))
for (i, r) in enumerate(receivers)
    println("  receivers[$i] = ", r, " | type: ", typeof(r))
end

println("DEBUG: bathymetry_int type = ", typeof(bathymetry_int))
println("DEBUG: FishLogPotential type = ", typeof(fish_lp))
println("DEBUG: FishPriorPotential type = ", typeof(fish_prior_lp))



println("DEBUG: log_posterior(s_init, Ydepth, Yaccustic, bathymetry_int) = ", 
    #log_posterior(s_init, Ydepth, Yaccustic, bathymetry_int))
    log_posterior(s_init, Ydepth, Yaccustic, bathymetry_int, x_origin, y_origin, dx, dy))
# ...existing code...
#Pkg.add("BenchmarkTools")
#import BenchmarkTools
using BenchmarkTools
@btime fish_lp($v_init) 


v_tmp = similar(v_init)
print(v_init)
print(v_tmp)
v_tmp === v_init      # ⇒ false
v_tmp = similar(v_init)
#@time sample_iid!(fish_prior_lp, Random.default_rng(), v_tmp; tries = 300)
@time sample_iid!(fish_prior_lp, Random.default_rng(), v_tmp; tries = 2)
#@time sample_iid!(fish_prior_lp, Random.default_rng(), similar(v_init))
@btime fish_lp($v_init) 
=#
pt = pigeons(
    target            = fish_lp,               # log posterior to smaple from
    reference         = fish_prior_lp,         # reference distribution (that coincides with the distribution at beta=0)
    seed              = 1234,                  
    n_rounds          = 7,                     # up to 2^nround–1 scans
    n_chains          = 50,                    
    checkpoint        = false,                
    multithreaded     = false,    
    record            = [traces; record_default()]  # registra traiettorie + diagnostica
)


pt_samples  = Chains(pt)         

cold_last_v = pt_samples.value[end, 1: 2*tmax] |> vec
cold_last_S = TransformVariables.transform(mapping, cold_last_v)
names(Pigeons; all=true)


xs_pigeons = [p.x for p in cold_last_S] #extracting trajectories
ys_pigeons = [p.y for p in cold_last_S]



plt = heatmap(bathy,
              color = :blues,
              legend = false,
              title = "Posizioni dei receiver sulla batimetria")

for rcv in receivers
    plot!(plt, make_circle(rcv.x, rcv.y, 50), color=:red, lw=1, label=false)
end
plot!(plt, xs_pigeons, ys_pigeons, lw=3, color=:orange, label="Pigeons PT")
plot!(plt, xs, ys, lw=3, color=:red, label="starting trajectory")
#plot!(plt, xs_d, ys_d, lw=3, color=:blue, label="goal trajectory")
#plot!(plt, make_circle(receiver1.x, receiver1.y, receiver1.dist), color=:red, label="Receiver 1")
#plot!(plt, make_circle(receiver2.x, receiver2.y, receiver2.dist), color=:red, label="Receiver 1")

#display(plt)


logposteriors = [logdensity(fish_lp, pt_samples.value[i, 1:2*tmax]) for i in 1:size(pt_samples.value, 1)]
plt_logpost = plot(logposteriors, xlabel="iteration", ylabel="Log-posterior", title="log-posterior evolution")

display(plt_logpost)


n_points = tmax
n_samples = size(pt_samples.value, 1)

for i in 1:n_samples
    v = pt_samples.value[i, 1:2*n_points]
    S = TransformVariables.transform(mapping, v)
    xs_ss = [p.x for p in S]
    ys_ss = [p.y for p in S]
    plot!(xs_ss, ys_ss, alpha=0.2, color=:orange, label=false)
end
heatmap!(bathy,
              color = :blues,
              legend = false,
              title = "Posizioni dei receiver sulla batimetria")


#=
n_iter, n_param, n_chain = size(pt_samples)
idx_hot = n_chain  # la catena più calda

for i in 1:n_iter
    v = pt_samples[i, 1:2*n_points, idx_hot]
    S = TransformVariables.transform(mapping, v)
    xs = [p.x for p in S]
    ys = [p.y for p in S]
    plot!(xs, ys, alpha=0.2, color=:red, label=false)  # colore diverso per distinguerla
end
=#
# Calcola i vettori nello spazio dei parametri
v_cold = pt_samples.value[end, 1:2*tmax]
v_init = TransformVariables.inverse(mapping, s_init)
#v_depth = TransformVariables.inverse(mapping, s_depth)

# Calcola i log-posterior
logpost_cold = logdensity(fish_lp, v_cold)
logpost_init = logdensity(fish_lp, v_init)
#logpost_depth = logdensity(fish_lp, v_depth)
#logpost_init2 = log_posterior(s_init, Ydepth, Yaccustic, bathymetry_int)
#logpost_depth2 = log_posterior(s_depth, Ydepth, Yaccustic, bathymetry_int)

logpost_init2 = log_posterior(s_init, Ydepth, Yaccustic, bathymetry_int, x_origin, y_origin, dx, dy)
println("Log-posterior cold chain: ", logpost_cold)
println("Log-posterior initial bridge: ", logpost_init)
#println("Log-posterior depth bridge: ", logpost_depth)



print(cold_last_S)
YdepthPIGEONS = Tuple{Int, Float64, DepthGauge}[] 
depthgaugep = DepthGauge()
for (t, point) in enumerate(cold_last_S)
# Calcola la posizione frazionaria rispetto al ritaglio
    idx_full = GeoArrays.indices(bathy_orig, (point[1], point[2]))
    row_full, col_full = Tuple(idx_full)
    row = row_full - row_start + 1
    col = col_full - col_start + 1
    depth = bathymetry_int(row, col)
    #println("Profondità interpolata: ", bathymetry_int(row, col))
    println("t ", t, "DEBUG: point = ", point[1])
    #depth = get_depth((x=point[1], y=point[2]), bathymetry_int, bathy_orig, 340, 430)
    depth = get_depth_p((x=point[1], y=point[2]), bathymetry_int, x_origin, y_origin, dx, dy)

    println("DEBUG: depth = ", depth)
    push!(YdepthPIGEONS, (t, depth, depthgaugep))
end




print(YdepthPIGEONS)
tempiP = [y[1] for y in YdepthPIGEONS]
profonditaP = [y[2] for y in YdepthPIGEONS]

plt = plot(tempi, profondita, xlabel="Tempo", ylabel="Profondità", label="Ydepth2", legend=:topright)
plot!(plt, tempiP, profonditaP, label="Ydepth PIGEONS")
display(plt)





samples = infer_trajectories(
            Ydepth, Yaccustic, bathymetry_int;
            s_init = cold_last_S    ,    # proveniente dal PT
            #s_init = s_init,           # <‑‑ qui
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
                  xlim=(0, 150), ylim=(0, 150),
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



