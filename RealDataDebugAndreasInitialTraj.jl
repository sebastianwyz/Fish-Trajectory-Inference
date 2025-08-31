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
include("find_PT3_REALDATA_August.jl")

using Starfish

spatial_resolution = 200                         # spatial resolution



bathy = GeoArrays.read("C:\\Users\\J\\Desktop\\map_Firth_of_Lorn_200m.tif")

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

depth_obs_df = CSV.read("C:\\Users\\J\\Desktop\\observation\\depth.csv", DataFrame,
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
end_idx = 23040
depth_obs_df = depth_obs_df[1:end, :]
depth_obs_df = depth_obs_df[start_idx:end_idx, :]

tmax=23040
########################################################


moorings_df = CSV.read("C:\\Users\\J\\Desktop\\observation\\moorings.csv", DataFrame)
acoustic_pos = [(x=x, y=y) for (x, y) in zip(moorings_df.receiver_x, moorings_df.receiver_y)]
print(acoustic_pos[1])

#Added this because is needed for initializing Andreas function
acoustic_pos_starfish = tuple.(moorings_df.receiver_x, moorings_df.receiver_y)


for point in acoustic_pos
    print(get_depth_at(bathy, ex_itp, point.x, point.y), "\n")
end
acoustic_obs_df = CSV.read("C:\\Users\\J\\Desktop\\observation\\acoustics.csv", DataFrame,
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

acoustic_signals = [acoustic_array[:, r] for r in 1:size(acoustic_array, 2)]


activated_receivers = Int[]
for t in 1:size(acoustic_array, 1)
    for r in 1:size(acoustic_array, 2)
        if acoustic_array[t, r] == 1
            push!(activated_receivers, r)
        end
    end
end
unique_receivers = unique(activated_receivers)
println("Unique activated receivers: ", unique_receivers)

for (i, rcv) in enumerate(acoustic_pos)
    depth = get_depth_at(bathy, ex_itp, rcv.x, rcv.y)
    println("Receiver $i at ($(rcv.x), $(rcv.y)) has depth $depth")
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

    return receiver_seq, t_steps, events
end

receiver_seq, t_steps, events = build_receiver_sequence(acoustic_array, acoustic_pos;
                            start_point = (x=709757.1, y=6.267726e6))

print("Receiver sequence: ", receiver_seq, "\n")
print("Time steps: ", t_steps, "\n")
println("Numero di receiver unici: ", length(unique(receiver_seq)))


#Using Andreas Trajectory Builder Function
res = find_shortest_trajectory(
    bathy,                # your GeoArrays bathymetry
    acoustic_signals,     # list of signals per receiver
    acoustic_pos_starfish,# tuple of receiver positions
    depth_signals,        # your depth observations
    goal_tol = 2,
    abstol = 10,
    reltol = 0.1
)


# Function to print NaN runs in the trajectory
function print_nan_runs(path)
    nan_runs = []
    current_run = Int[]
    for (i, p) in enumerate(path)
        if isnan(p[1]) || isnan(p[2])
            push!(current_run, i)
        else
            if !isempty(current_run)
                push!(nan_runs, copy(current_run))
                empty!(current_run)
            end
        end
    end
    # Catch a run at the end
    if !isempty(current_run)
        push!(nan_runs, current_run)
    end

    if isempty(nan_runs)
        println("No NaN runs found.")
    else
        for run in nan_runs
            println("NaN run from index $(first(run)) to $(last(run)), length = $(length(run))")
        end
    end
end

print_nan_runs(res.path)


# Using Interpolations
using Interpolations


# Function to fill NaN runs with spline interpolation
function fill_nan_runs_with_spline(path)
    filled_path = copy(path)
    n = length(path)
    nan_indices = findall(p -> isnan(p[1]) || isnan(p[2]), path)

    # Find runs of consecutive NaNs
    runs = []
    current_run = Int[]
    for idx in nan_indices
        if isempty(current_run) || idx == last(current_run) + 1
            push!(current_run, idx)
        else
            push!(runs, copy(current_run))
            empty!(current_run)
            push!(current_run, idx)
        end
    end
    if !isempty(current_run)
        push!(runs, current_run)
    end

    for run in runs
        before_idx = first(run) - 1
        after_idx = last(run) + 1
        # Only interpolate if both neighbors exist and are not NaN
        if before_idx >= 1 && after_idx <= n &&
           !isnan(path[before_idx][1]) && !isnan(path[after_idx][1])
            x0, y0 = path[before_idx]
            x1, y1 = path[after_idx]
            n_interp = length(run)
            # Use a range for interpolation
            xs = 0:(n_interp+1)
            ys_x = [x0, x1]
            ys_y = [y0, y1]
            # Linear interpolation for two points
            itp_x = LinearInterpolation(xs[[1, end]], ys_x, extrapolation_bc=Line())
            itp_y = LinearInterpolation(xs[[1, end]], ys_y, extrapolation_bc=Line())
            for (k, idx) in enumerate(run)
                filled_path[idx] = (itp_x(k), itp_y(k))
            end
        end
        # If at the edge or neighbors are NaN, you may want to skip or fill with nearest valid value
    end
    return filled_path
end

# Usage:
filled_path = fill_nan_runs_with_spline(res.path)

# Get the first and last activation times from events
first_idx = events[1][1]
last_idx  = events[end][1]

# Trim filled_path to only include points between first and last activation (inclusive)
trimmed_filled_path = filled_path[first_idx:last_idx]

println("Trimmed filled_path has ", length(trimmed_filled_path), " points.")

filled_path=trimmed_filled_path

xs = [p.x for p in acoustic_pos]
ys = [p.y for p in acoustic_pos]
xs_filled = [p[1] for p in filled_path]
ys_filled = [p[2] for p in filled_path]

plt = heatmap(bathy;
    color = :blues,
    legend = false,
    title = "Trajectory with NaN-filled interpolation",
    xlims = (x_min, x_max),
    ylims = (y_min, y_max)
)
scatter!(plt, xs, ys; color = :red, markersize = 4, label = "Receivers")
plot!(plt, xs_filled, ys_filled, lw=2, color=:purple, label="Interpolated trajectory")
display(plt)

print(print_nan_runs(filled_path))

# Check how many points in filled_path are on land (depth ≤ 0)
n_ground_filled = 0
ground_indices = Int[]
for (i, p) in enumerate(filled_path)
    depth = get_depth_at(bathy, ex_itp, p[1], p[2])
    if depth <= 0
        push!(ground_indices, i)
        n_ground_filled += 1
    end
end

println("Points on ground (depth ≤ 0) in filled_path: $n_ground_filled out of $(length(filled_path))")
println("Indices of ground points in filled_path: ", ground_indices)

#Checking and plotting Andreas trajectory without NaNs (is the trimmed trajectory=trajectory between first and last activation)
s_init = [(x=p[1], y=p[2]) for p in filled_path]

xs_path = [p[1] for p in filled_path]
ys_path = [p[2] for p in filled_path]

plt = heatmap(bathy;
    color = :blues,
    legend = false,
    title = "Starfish shortest trajectory",
    xlims = (x_min, x_max),
    ylims = (y_min, y_max)
)
scatter!(plt, xs, ys; color = :red, markersize = 4, label = "Receivers")
plot!(plt, xs_path, ys_path, lw=2, color=:orange, label="Starfish path")
display(plt)


#This was my function por generating the initial trajectory before Andreas one
#=
# Extract activation times from events (output of build_receiver_sequence)
activation_times = [t for (t, r) in events]

# Compute raw steps between activations
raw_steps = [activation_times[i+1] - activation_times[i] for i in 1:length(activation_times)-1]

# Rescale so sum(raw_steps) == tmax
scale = tmax / sum(raw_steps)
t_steps = [round(Int, s * scale) for s in raw_steps]

# Adjust last segment to ensure sum(t_steps) == tmax
t_steps[end] += tmax - sum(t_steps)

function make_water_only_traj(receiver_seq::Vector{<:NamedTuple},
                             t_steps::Vector{<:Integer};
                             bathy, ex_itp,
                             noise_std::Real = 1.0,
                             max_tries::Int = 100)
    traj = Vector{NamedTuple{(:x,:y),Tuple{Float64,Float64}}}()
    for (seg, (A, B)) in enumerate(zip(receiver_seq[1:end-1], receiver_seq[2:end]))
        n = t_steps[seg]
        prev = A
        for k in 1:n
            tries = 0
            while tries < max_tries
                λ = k / n
                # Propose a random walk step around the straight line
                x_prop = (1-λ)*A.x + λ*B.x + randn()*noise_std
                y_prop = (1-λ)*A.y + λ*B.y + randn()*noise_std
                depth = get_depth_at(bathy, ex_itp, x_prop, y_prop)
                if depth > 0
                    push!(traj, (x=x_prop, y=y_prop))
                    prev = (x=x_prop, y=y_prop)
                    break
                end
                tries += 1
            end
            if tries == max_tries
                @warn "Could not find water-only step for segment $seg, step $k. Using previous valid point."
                push!(traj, prev)
            end
        end
    end
    return traj
end

traj_water = make_water_only_traj(receiver_seq, t_steps; bathy=bathy, ex_itp=ex_itp, noise_std=3.0)

all_water = all(get_depth_at(bathy, ex_itp, p.x, p.y) > 0 for p in traj_water)
println("All trajectory points in water? ", all_water)

plt = heatmap(bathy;
    color = :blues,
    legend = false,
    title = "Water-only random walk trajectory",
    xlims = (x_min, x_max),
    ylims = (y_min, y_max)
)
scatter!(plt, xs, ys; color = :red, markersize = 4, label = "Receivers")
plot!(plt, [p.x for p in traj_water], [p.y for p in traj_water], lw=2, color=:black, label="Water-only trajectory")
display(plt)


for (i, steps) in enumerate(t_steps)
    println("Segment $i: t_steps = $steps")
end

function average_step_length(traj)
    n = length(traj)
    if n < 2
        return 0.0
    end
    step_lengths = [hypot(traj[i+1].x - traj[i].x, traj[i+1].y - traj[i].y) for i in 1:n-1]
    return mean(step_lengths)
end

println("Average step length: ", average_step_length(traj))

function maximum_step_length(traj)
    n = length(traj)
    if n < 2
        return 0.0
    end
    step_lengths = [hypot(traj[i+1].x - traj[i].x, traj[i+1].y - traj[i].y) for i in 1:n-1]
    return maximum(step_lengths)
end

println("Maximum step length: ", maximum_step_length(traj))

for (i, p) in enumerate(traj)
    depth = get_depth_at(bathy, ex_itp, p.x, p.y)
    println("Step $i: depth = $depth")
end


traj= traj_water
=#

#=
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
=#

traj=filled_path

# Check for NaNs in the trimmed trajectory
for (i, p) in enumerate(filled_path)
    if isnan(p[1]) || isnan(p[2])
        println("NaN found at index $i: ($p[1], $p[2])")
    end
end


xs = [p[1] for p in filled_path]
ys = [p[2] for p in filled_path]
delta_x = diff(xs)                # vettore delle differenze tra x consecutivi
mean_delta_x = mean(abs.(delta_x))  # media dei valori assoluti delle differenze
println("Delta x medio: ", mean_delta_x)


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



#--------- PT 
@info "Optimizing posterior for a MAP starting point…"              

# Select the segment or interval of interest from the original res.path 
segment_path = res.path[4573:10467]

# Fill NaN runs in this segment
filled_path = fill_nan_runs_with_spline(segment_path)

# (Optional) Convert to NamedTuples for downstream code
s_init = [(x=p[1], y=p[2]) for p in filled_path]

# Update tmax to match the new trajectory length
tmax = length(filled_path)

println("Using trajectory segment from 4573 to 10467 (length = $tmax)")

#Remove consecutive duplicates of the trajectories (this was needed because Pigeons seemed to give problem when there were that many duplicate points)
function remove_consecutive_duplicates(traj)
    if isempty(traj)
        return traj
    end
    cleaned = [traj[1]]
    for i in 2:length(traj)
        if traj[i] != traj[i-1]
            push!(cleaned, traj[i])
        end
    end
    return cleaned
end

# Usage:
s_init_clean = remove_consecutive_duplicates(s_init)
println("Removed $(length(s_init) - length(s_init_clean)) consecutive duplicates from s_init.")
#Here we obtained the 26 "Fundamental" Points that "summarize" the trajectory


#Now we connect each pair of these 26 points with a linear trajectory and add some Gaussian noise to them
#So we have a trajectory with 5895 non-equal points, the amount of points between each fundamental points was equal to the amount of equal points we had before 
"""
    make_noisy_linear_trajectory(ref_points, n_points; noise_std=3.0, bathy, ex_itp)

Creates a trajectory of `n_points` by interpolating along the piecewise-linear path
through `ref_points`, then adds Gaussian noise to each point.
"""
function make_noisy_linear_trajectory(ref_points, n_points; noise_std=3.0, bathy, ex_itp)
    seg_lengths = [hypot(ref_points[i+1].x - ref_points[i].x,
                        ref_points[i+1].y - ref_points[i].y)
                   for i in 1:length(ref_points)-1]
    total_length = sum(seg_lengths)
    points_per_seg = [round(Int, l/total_length * n_points) for l in seg_lengths]
    diff_points = n_points - sum(points_per_seg)
    if diff_points != 0
        idxs = sortperm(seg_lengths, rev=true)
        for i in 1:abs(diff_points)
            points_per_seg[idxs[i]] += sign(diff_points)
        end
    end

    traj = NamedTuple[]
    for (i, n_seg) in enumerate(points_per_seg)
        A, B = ref_points[i], ref_points[i+1]
        for k in 0:n_seg-1
            λ = n_seg == 1 ? 0.0 : k/(n_seg-1)
            x = (1-λ)*A.x + λ*B.x + randn()*noise_std
            y = (1-λ)*A.y + λ*B.y + randn()*noise_std
            depth = get_depth_at(bathy, ex_itp, x, y)
            if depth > 0
                push!(traj, (x=x, y=y))
            else
                push!(traj, (x=(1-λ)*A.x + λ*B.x, y=(1-λ)*A.y + λ*B.y))
            end
        end
    end
    if length(traj) > n_points
        traj = traj[1:n_points]
    elseif length(traj) < n_points
        while length(traj) < n_points
            push!(traj, traj[end])
        end
    end
    return traj
end

# --- Use s_init_clean as the reference for the noisy trajectory, s_init_clean is the 26 fundamental points trajectory ---
n_points = length(filled_path)
noise_std = 3.0
primordial_traj = make_noisy_linear_trajectory(s_init_clean, n_points; noise_std=noise_std, bathy=bathy, ex_itp=ex_itp)
println("Primordial noisy trajectory length: ", length(primordial_traj))

# Set and rename as your s_init for the next steps
s_init = primordial_traj

# Check for ground points in this new s_init
n_ground = 0
ground_indices = Int[]
for (i, p) in enumerate(s_init)
    depth = get_depth_at(bathy, ex_itp, p.x, p.y)
    if depth <= 0
        println("Ground point at index $i: ($(p.x), $(p.y)), depth = $depth")
        push!(ground_indices, i)
        n_ground += 1
    end
end
println("Total ground points in s_init: $n_ground out of $(length(s_init))")
println("Indices of ground points: ", ground_indices)

# --- Visualization ---
xs_init = [p.x for p in s_init]
ys_init = [p.y for p in s_init]
xs_prim = [p.x for p in s_init_clean]
ys_prim = [p.y for p in s_init_clean]

plt = heatmap(bathy;
    color = :blues,
    legend = false,
    title = "Noisy linear trajectory from s_init_clean",
    xlims = (x_min, x_max),
    ylims = (y_min, y_max)
)
scatter!(plt, xs_prim, ys_prim; color = :red, markersize = 3, label = "Primordial points")
scatter!(plt, xs_init, ys_init; color = :orange, markersize=1 , label = "Noisy trajectory")
display(plt)



# Use only the first 200 points of s_init for testing if Pigeons can run it, so we can confirm that is not a problem of the trajectory
s_init = s_init[1:200]
tmax = length(s_init)
println("Testing with first $tmax points of s_init")

#Now this is all as it was, nothing changed from here (at least by my hand)
#Setting the problem characteristics:
mapping = TransformVariables.as(Array, 
                                TransformVariables.as((x = TransformVariables.asℝ, y = TransformVariables.asℝ)),
                                tmax)
v_init = TransformVariables.inverse(mapping, s_init)
print(v_init)


s0, σ =  (x = 709_757.1116, y = 6.2677260356e6), 2.0


#traj = simulateRW_free(tmax; s0 = s0, sigma = 2.0, rng = Random.default_rng())


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

xs_init = [p.x for p in s_init]
ys_init = [p.y for p in s_init]

plt = heatmap(bathy;
    color = :blues,
    legend = false,
    title = "s_init points on bathymetry",
    xlims = (x_min, x_max),
    ylims = (y_min, y_max)
)
scatter!(plt, xs, ys; color = :red, markersize = 4, label = "Receivers")
scatter!(plt, xs_init, ys_init; color = :orange, markersize = 6, label = "s_init points")
display(plt)

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



