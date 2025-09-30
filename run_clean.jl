# Fish Tracking Analysis with Parallel Tempering
# Cleaned and optimized version

using Pkg
Pkg.activate(".")
Pkg.instantiate()

# Load required packages
using XLSX, DataFrames, Dates, Missings, CSV
using GeoArrays, Images, Plots, StaticArrays, CoordinateTransformations
using Interpolations, Statistics
using Starfish, Pigeons, TransformVariables, Random
using Pigeons: round_trip, traces, record_default

include("find_PT_clean.jl")

# ============================================================================
# CONFIGURATION PARAMETERS
# ============================================================================
const SPATIAL_RESOLUTION = 200
const START_IDX = 1
const END_IDX = 6000
const TMAX = END_IDX
const NOISE_STD = 3.0

# Spatial bounds for visualization
const X_MIN = 700000.0
const X_MAX = 715000.0
const Y_MIN = 6.2550e6
const Y_MAX = 6.2720e6

init_segment = 1
end_segment = 100


# File paths
const BATHY_PATH = "C:\\Users\\teresa i robert\\Desktop\\TEST\\run_wahoo\\bathymetry\\map_Firth_of_Lorn_200m.tif"
const DEPTH_CSV = "C:\\Users\\teresa i robert\\Desktop\\Physics of Data\\PoD Fish tracking\\Code\\fish_tracking_HMC\\HMC\\observation\\depth.csv"
const MOORINGS_CSV = "C:\\Users\\teresa i robert\\Desktop\\Physics of Data\\PoD Fish tracking\\Code\\fish_tracking_HMC\\HMC\\observation\\moorings.csv"
const ACOUSTICS_CSV = "C:\\Users\\teresa i robert\\Desktop\\Physics of Data\\PoD Fish tracking\\Code\\fish_tracking_HMC\\HMC\\observation\\acoustics.csv"

# ============================================================================
# DATA LOADING AND PREPROCESSING
# ============================================================================

"""Load and prepare bathymetry data with interpolation"""
function load_bathymetry(path::String)
    bathy = GeoArrays.read(path)
    arr = GeoArrays.values(bathy)
    itp = interpolate(arr, BSpline(Linear()))
    ex_itp = extrapolate(itp, -1.0)
    bathymetry_int = extrapolate(interpolate(bathy, BSpline(Linear())), -1.0)
    
    # Extract coordinate transformation parameters
    aff = bathy.f
    A = aff.linear
    b = aff.translation
    
    return (
        bathy = bathy,
        ex_itp = ex_itp,
        bathymetry_int = bathymetry_int,
        x_origin = b[1],
        y_origin = b[2],
        dx = A[1,1],
        dy = A[2,2]
    )
end

"""Load and clean depth observations"""
function load_depth_data(path::String, start_idx::Int, end_idx::Int)
    df = CSV.read(path, DataFrame, dateformat="yyyy-mm-dd H:M:S")
    rename!(df, :Column1 => :time)
    
    # Fix incompatible depth data
    df.depth[2041:2047] .= 205
    df.depth[4842:4845] .= 205
    
    return df[start_idx:end_idx, :]
end

"""Load acoustic receiver positions and observations"""
function load_acoustic_data(moorings_path::String, acoustics_path::String, start_idx::Int, end_idx::Int)
    moorings_df = CSV.read(moorings_path, DataFrame)
    acoustic_pos = [(x=x, y=y) for (x, y) in zip(moorings_df.receiver_x, moorings_df.receiver_y)]
    acoustic_pos_starfish = tuple.(moorings_df.receiver_x, moorings_df.receiver_y)
    
    acoustic_obs_df = CSV.read(acoustics_path, DataFrame, 
                               dateformat="yyyy-mm-dd H:M:S", missingstring="NA")
    acoustic_obs_df = acoustic_obs_df[start_idx:end_idx, :]
    acoustic_array = coalesce.(Array(acoustic_obs_df[:,2:end]), -1)
    acoustic_signals = [acoustic_array[:, r] for r in 1:size(acoustic_array, 2)]
    
    return (
        moorings_df = moorings_df,
        acoustic_pos = acoustic_pos,
        acoustic_pos_starfish = acoustic_pos_starfish,
        acoustic_array = acoustic_array,
        acoustic_signals = acoustic_signals
    )
end

# ============================================================================
# TRAJECTORY PROCESSING FUNCTIONS
# ============================================================================

"""Build receiver activation sequence from acoustic data"""
function build_receiver_sequence(acoustic_array, acoustic_pos; start_point::NamedTuple)
    n_time, n_receivers = size(acoustic_array)
    
    # Find first activations
    last_state = falses(n_receivers)
    events = Tuple{Int,Int}[]
    
    for t in 1:n_time, r in 1:n_receivers
        cur = acoustic_array[t, r] == 1
        if cur && !last_state[r]
            push!(events, (t, r))
        end
        last_state[r] = cur
    end
    
    # Build sequence and time steps
    if isempty(events)
        receiver_seq = [start_point, start_point]
        t_steps = [max(1, n_time - 1)]
    elseif length(events) == 1
        only_rec = acoustic_pos[events[1][2]]
        receiver_seq = [only_rec, only_rec]
        t_steps = [max(1, n_time - events[1][1])]
    else
        receiver_seq = [acoustic_pos[r] for (_t, r) in events]
        t_steps = [events[i+1][1] - events[i][1] for i in 1:length(events)-1]
    end
    
    return receiver_seq, t_steps, events
end

"""Fill NaN runs in trajectory using linear interpolation"""
function fill_nan_runs_with_spline(path)
    filled_path = copy(path)
    n = length(path)
    nan_indices = findall(p -> isnan(p[1]) || isnan(p[2]), path)
    
    if isempty(nan_indices)
        return filled_path
    end
    
    # Find consecutive NaN runs
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
    !isempty(current_run) && push!(runs, current_run)
    
    # Interpolate each run
    for run in runs
        before_idx = first(run) - 1
        after_idx = last(run) + 1
        
        if before_idx >= 1 && after_idx <= n &&
           !isnan(path[before_idx][1]) && !isnan(path[after_idx][1])
            x0, y0 = path[before_idx]
            x1, y1 = path[after_idx]
            n_interp = length(run)
            
            itp_x = LinearInterpolation([0, n_interp+1], [x0, x1], extrapolation_bc=Line())
            itp_y = LinearInterpolation([0, n_interp+1], [y0, y1], extrapolation_bc=Line())
            
            for (k, idx) in enumerate(run)
                filled_path[idx] = (itp_x(k), itp_y(k))
            end
        end
    end
    
    return filled_path
end

"""Remove consecutive duplicate points from trajectory"""
function remove_consecutive_duplicates(traj)
    isempty(traj) && return traj
    
    cleaned = [traj[1]]
    for i in 2:length(traj)
        if traj[i] != traj[i-1]
            push!(cleaned, traj[i])
        end
    end
    return cleaned
end

"""Create noisy linear trajectory through reference points"""
function make_noisy_linear_trajectory(ref_points, n_points; noise_std=3.0, bathy, ex_itp)
    # Calculate segment lengths and distribute points
    seg_lengths = [hypot(ref_points[i+1].x - ref_points[i].x,
                        ref_points[i+1].y - ref_points[i].y)
                   for i in 1:length(ref_points)-1]
    total_length = sum(seg_lengths)
    points_per_seg = [round(Int, l/total_length * n_points) for l in seg_lengths]
    
    # Adjust for rounding errors
    diff_points = n_points - sum(points_per_seg)
    if diff_points != 0
        idxs = sortperm(seg_lengths, rev=true)
        for i in 1:abs(diff_points)
            points_per_seg[idxs[i]] += sign(diff_points)
        end
    end
    
    # Generate trajectory
    traj = NamedTuple[]
    for (i, n_seg) in enumerate(points_per_seg)
        A, B = ref_points[i], ref_points[i+1]
        for k in 0:n_seg-1
            λ = n_seg == 1 ? 0.0 : k/(n_seg-1)
            x_base = (1-λ)*A.x + λ*B.x
            y_base = (1-λ)*A.y + λ*B.y
            x = x_base + randn()*noise_std
            y = y_base + randn()*noise_std
            
            # Keep point in water
            depth = get_depth_at(bathy, ex_itp, x, y)
            if depth > 0
                push!(traj, (x=x, y=y))
            else
                push!(traj, (x=x_base, y=y_base))
            end
        end
    end
    
    # Ensure exact length
    while length(traj) < n_points
        push!(traj, traj[end])
    end
    length(traj) > n_points && (traj = traj[1:n_points])
    
    return traj
end

# ============================================================================
# OBSERVATION PREPARATION
# ============================================================================

"""Prepare acoustic observations for Pigeons"""
function prepare_acoustic_observations(acoustic_array, moorings_df)
    receivers = [
        Receiver(
            moorings_df.receiver_x[i],
            moorings_df.receiver_y[i],
            50.0,   # detection distance
            30.0    # detection parameter k
        ) for i in 1:nrow(moorings_df)
    ]
    
    Yaccustic = Tuple{Int, Symbol, Receiver}[]
    for t in 1:size(acoustic_array, 1)
        for r in 1:size(acoustic_array, 2)
            stato = acoustic_array[t, r]
            if stato != -1
                signal = stato == 1 ? :detect : :nondetect
                push!(Yaccustic, (t, signal, receivers[r]))
            end
        end
    end
    
    return Yaccustic, receivers
end

"""Prepare depth observations for Pigeons"""
function prepare_depth_observations(depth_signals)
    Ydepth = Tuple{Int, Float64, DepthGauge}[]
    depthgauge = DepthGauge()
    
    for (t, d) in enumerate(depth_signals)
        push!(Ydepth, (t, d, depthgauge))
    end
    
    return Ydepth
end

# ============================================================================
# VISUALIZATION
# ============================================================================

"""Create bathymetry plot with receivers and trajectory"""
function plot_trajectory(bathy, receivers, trajectory; 
                        title="Fish Trajectory", xlims=(X_MIN, X_MAX), ylims=(Y_MIN, Y_MAX))
    plt = heatmap(bathy;
        color = :blues,
        legend = false,
        title = title,
        xlims = xlims,
        ylims = ylims
    )
    
    # Add receivers
    xs = [r.x for r in receivers]
    ys = [r.y for r in receivers]
    scatter!(plt, xs, ys; color = :red, markersize = 4, label = "Receivers")
    
    # Add trajectory
    xs_traj = [p.x for p in trajectory]
    ys_traj = [p.y for p in trajectory]
    plot!(plt, xs_traj, ys_traj, lw=2, color=:orange, label="Trajectory")
    
    return plt
end

# ============================================================================
# MAIN ANALYSIS PIPELINE
# ============================================================================

println("Loading bathymetry data...")
bathy_data = load_bathymetry(BATHY_PATH)

println("Loading observation data...")
depth_obs_df = load_depth_data(DEPTH_CSV, START_IDX, END_IDX)
acoustic_data = load_acoustic_data(MOORINGS_CSV, ACOUSTICS_CSV, START_IDX, END_IDX)

println("Building initial trajectory...")
# Build receiver sequence
receiver_seq, t_steps, events = build_receiver_sequence(
    acoustic_data.acoustic_array, 
    acoustic_data.acoustic_pos;
    start_point = (x=709757.1, y=6.267726e6)
)

# Generate shortest trajectory using Starfish
@info "Finding shortest trajectory with Starfish..."
res = find_shortest_trajectory(
    bathy_data.bathy,
    acoustic_data.acoustic_signals,
    acoustic_data.acoustic_pos_starfish,
    depth_obs_df.depth,
    goal_tol = 2
)


# Process trajectory segment
segment_path = res.path[START_IDX:END_IDX]  # Select specific segment

first_valid = findfirst(p -> !isnan(p[1]) && !isnan(p[2]), res.path)
last_valid = findlast(p -> !isnan(p[1]) && !isnan(p[2]), res.path)

# Usa questi indici per definire il segmento
segment_path = res.path[first_valid:last_valid]


filled_path = fill_nan_runs_with_spline(segment_path)
s_init = [(x=p[1], y=p[2]) for p in filled_path]
tmax = length(filled_path)

# Clean trajectory
s_init_clean = remove_consecutive_duplicates(s_init)
println("Removed $(length(s_init) - length(s_init_clean)) consecutive duplicates")

# Generate noisy trajectory
s_init = make_noisy_linear_trajectory(
    s_init_clean, 
    length(filled_path); 
    noise_std=NOISE_STD, 
    bathy=bathy_data.bathy, 
    ex_itp=bathy_data.ex_itp
)


# For testing: use only first 200 points
s_init = s_init[init_segment:end_segment]
tmax = length(s_init)
println("Using first $tmax points for Pigeons")

# Prepare observations
println("Preparing observations...")
Yaccustic, receivers = prepare_acoustic_observations(
    acoustic_data.acoustic_array, 
    acoustic_data.moorings_df
)


depth_obs_df = load_depth_data(DEPTH_CSV, init_segment, end_segment)
Ydepth = prepare_depth_observations(depth_obs_df.depth)

# Setup Pigeons
println("Setting up Pigeons parallel tempering...")
mapping = TransformVariables.as(
    Array, 
    TransformVariables.as((x = TransformVariables.asℝ, y = TransformVariables.asℝ)),
    tmax
)
v_init = TransformVariables.inverse(mapping, s_init)

# Create potentials
fish_prior_lp = FishPriorPotential(mapping, v_init)
fish_lp = FishLogPotential(
    Ydepth,
    Yaccustic,
    bathy_data.bathymetry_int,
    bathy_data.x_origin,
    bathy_data.y_origin,
    bathy_data.dx,
    bathy_data.dy,
    mapping,
    v_init,
    bathy_data.bathy,      # ← AGGIUNGI
    bathy_data.ex_itp      # ← AGGIUNGI
)

# Run Pigeons
@info "Running Pigeons parallel tempering..."
pt = pigeons(
    target = fish_lp,
    reference = fish_prior_lp,
    seed = 1234,
    n_rounds = 6,
    n_chains = 10,
    checkpoint = true,
    multithreaded = true,
    record = [traces; record_default()]
)

# Extract results
pt_samples = Chains(pt)
cold_last_v = pt_samples.value[end, 1:2*tmax] |> vec
cold_last_S = TransformVariables.transform(mapping, cold_last_v)

# Visualize results
println("Creating visualizations...")
plt = plot_trajectory(bathy_data.bathy, receivers, cold_last_S; 
                        title="Pigeons PT Result")
display(plt)

# Plot log-posterior evolution
logposteriors = [logdensity(fish_lp, pt_samples.value[i, 1:2*tmax]) 
                    for i in 1:size(pt_samples.value, 1)]
plt_logpost = plot(logposteriors, 
                    xlabel="Iteration", 
                    ylabel="Log-posterior", 
                    title="Log-posterior Evolution")
display(plt_logpost)

# Compare depths
tempi = 1:tmax
depth_observed = [Ydepth[t][2] for t in tempi]
depth_estimated = [get_depth_at(bathy_data.bathy, bathy_data.ex_itp, p.x, p.y) 
                    for p in cold_last_S]

plt_depth = plot(tempi, depth_observed, 
                xlabel="Time", 
                ylabel="Depth", 
                label="Observed", 
                legend=:topright)
plot!(plt_depth, tempi, depth_estimated, label="Pigeons Estimate")
display(plt_depth)


