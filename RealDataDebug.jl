import Pkg
using XLSX            # lettura xls
using DataFrames      # gestione tabelle
using Dates           # DateTime
using Missings        # gestione dei NA
import CSV
using CSV

Pkg.activate(".")
Pkg.instantiate()
using GeoArrays
using Images: load, Gray, channelview
using Plots
using StaticArrays: SVector
using CoordinateTransformations
using Plots
include("find_PT3_REALDATA.jl")




spatial_resolution = 200                         # spatial resolution



bathy = GeoArrays.read("C:\\Users\\teresa i robert\\Desktop\\TEST\\run_wahoo\\bathymetry\\map_Firth_of_Lorn_200m.tif")

plot(bathy)
arr = GeoArrays.values(bathy)
itp = interpolate(arr, BSpline(Linear()))
ex_itp = extrapolate(itp, -1.0)
bathymetry_int = extrapolate(interpolate(bathy, BSpline(Linear())),-1.0);


##############################################################################
depth_matrix = bathy.A
#print(depth_matrix)

aff = bathy.f
A = aff.linear      # 2x2 matrice (scaling e rotazione)
b = aff.translation # 2-element vector (origine, coordinate reali del pixel (1,1))

x_origin = b[1]     # coordinata x del centro del pixel (1,1)
y_origin = b[2]     # coordinata y del centro del pixel (1,1)
dx = A[1,1]         # passo in x (dimensione pixel)
dy = A[2,2]        # passo in y (dimensione pixel)

println("x_origin = ", x_origin)
println("y_origin = ", y_origin)
println("dx = ", dx)
println("dy = ", dy)



depth = get_depth_at(bathy,ex_itp, 709757.111649658, 6.26772603565296e6)

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
end_idx = 100

depth_obs_df = depth_obs_df[start_idx:end_idx, :]
depth_signals = depth_obs_df.depth
########################################################


moorings_df = CSV.read("C:\\Users\\teresa i robert\\Desktop\\Physics of Data\\PoD Fish tracking\\Code\\fish_tracking_HMC\\HMC\\observation\\moorings.csv", DataFrame)
acoustic_pos = [(x=x, y=y) for (x, y) in zip(moorings_df.receiver_x, moorings_df.receiver_y)]
print(acoustic_pos[1])

for point in acoustic_pos
    print(get_depth_at(bathy, ex_itp, point.x, point.y), "\n")
end
acoustic_obs_df = CSV.read("C:\\Users\\teresa i robert\\Desktop\\Physics of Data\\PoD Fish tracking\\Code\\fish_tracking_HMC\\HMC\\observation\\acoustics.csv", DataFrame,
                           dateformat="yyyy-mm-dd H:M:S", missingstring="NA")


acoustic_array = coalesce.(Array(acoustic_obs_df[:,2:end]), -1)


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
         markersize = 1,
         label = "Receivers")
         
scatter!(plt, [710057.121649658], [6.26772603565296e6]; color = :green, markersize = 40, label = "Starting point")
scatter!(plt, [709757.111649658], [6.26772603565296e6]; color = :green, markersize = 1, label = "Starting point")


x_min = 700000.0
x_max = 715000.0
y_min = 6.2550e6
y_max = 6.2720e6



start_point = (710057.121649658, 6.27100603565296e6)
start_point = (707500.121649658, 6.27000603565296e6)


get_depth_at(bathy, ex_itp, start_point[1], start_point[2])

println("Starting point: ", start_point[1])

plt = heatmap(bathy;
    color = :blues,
    legend = false,
    title = "Zoom batimetria",
    xlims = (x_min, x_max),
    ylims = (y_min, y_max)
)
scatter!(plt, xs, ys; color = :red, markersize = 4, label = "Receivers")
scatter!(plt, [710057.121649658], [6.26772603565296e6]; color = :green, markersize = 4, label = "Starting point")

scatter!(plt, [start_point[1]], [start_point[2]]; color = :yellow, markersize = 4, label = "Starting point")

display(plt)


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

start_point = (7100757.121649658, 6.26772603565296e6)
get_depth_at(bathy, ex_itp, start_point[1], start_point[2])



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



traj = make_linear_traj(receiver_seq, t_steps; noise_std = 3.0)



plt = heatmap(bathy,
              color = :blues,
              legend = false,
              title = "Posizioni dei receiver sulla batimetria")

plot!(plt, traj)

scatter!(plt, first.(traj), last.(traj); m=:circle, ms=2, color=:black)
display(plt)





xs = [p.x for p in traj]
ys = [p.y for p in traj]
delta_x = diff(xs)                # vettore delle differenze tra x consecutivi
mean_delta_x = mean(abs.(delta_x))  # media dei valori assoluti delle differenze
println("Delta x medio: ", mean_delta_x)


s_init = traj
for i in 1:length(s_init)
    print(get_depth_at(bathy, ex_itp, s_init[i].x, s_init[i].y), "\n ")
end
print(s_init[1])
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



s0, σ =  (x = 709_757.1116, y = 6.2677260356e6), 2.0

traj = simulateRW_free(tmax; s0 = s0, sigma = 2.0, rng = Random.default_rng())





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


pt = pigeons(
    target            = fish_lp,               # log posterior to smaple from
    reference         = fish_prior_lp,         # reference distribution (that coincides with the distribution at beta=0)
    seed              = 1234,                  
    n_rounds          = 6,                     # up to 2^nround–1 scans
    n_chains          = 10,                    
    checkpoint        = true,                
    multithreaded     = true,    
    #on = ChildProcess(n_local_mpi_processes = 2,
    #                    n_threads=2,
    #                    dependencies = [find_grid_path]),
    record            = [traces; record_default()]  # registra traiettorie + diagnostica
    #record        = [traces, online, round_trip, Pigeons.timing_extrema, Pigeons.allocation_extrema, index_process]

)

#pt = Pigeons.load(pt_results)
pt_samples  = Chains(pt)       

myplot3 = plot(pt.reduced_recorders.index_process)
savefig(myplot3, "index_process_plot.svg")

cold_last_v = pt_samples.value[end, 1: 2*tmax] |> vec
cold_last_S = TransformVariables.transform(mapping, cold_last_v)
names(Pigeons; all=true)


xs_pigeons = [p.x for p in cold_last_S] #extracting trajectories
ys_pigeons = [p.y for p in cold_last_S]


delta_x = diff(xs_pigeons)                # vettore delle differenze tra x consecutivi
mean_delta_x = mean(abs.(delta_x))  #


plt = heatmap(bathy,
              color = :blues,
              legend = false,
              title = "Posizioni dei receiver sulla batimetria",
              xlims = (x_min, x_max),
              ylims = (y_min, y_max))

for rcv in receivers
    plot!(plt, make_circle(rcv.x, rcv.y, 50), color=:red, lw=4, label=false)
end
plot!(plt, xs_pigeons, ys_pigeons, lw=3, color=:orange, label="Pigeons PT")
plot!(plt, xs, ys, lw=3, color=:red, label="starting trajectory")


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



v_cold = pt_samples.value[end, 1:2*tmax]
v_init = TransformVariables.inverse(mapping, s_init)

logpost_cold = logdensity(fish_lp, v_cold)
logpost_init = logdensity(fish_lp, v_init)

logpost_init2 = log_posterior(s_init, Ydepth, Yaccustic, bathymetry_int, x_origin, y_origin, dx, dy)
println("Log-posterior cold chain: ", logpost_cold)
println("Log-posterior initial bridge: ", logpost_init)



print(cold_last_S)
YdepthPIGEONS = Tuple{Int, Float64, DepthGauge}[] 
depthgaugep = DepthGauge()
for (t, point) in enumerate(cold_last_S)
    depth = get_depth_at(bathy, ex_itp, point[1], point[2])
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



