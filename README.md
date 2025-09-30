Fish Tracking Analysis with Parallel Tempering
--------------
A Julia-based implementation for tracking fish movement using Bayesian inference
with Parallel Tempering (PT). The code combines acoustic receiver data, depth 
measurements, and bathymetric information to reconstruct fish trajectories in 
the Firth of Lorn.

OVERVIEW
--------------
This project uses Pigeons.jl for parallel tempering Monte Carlo sampling to 
infer fish trajectories from noisy observational data. The approach integrates:

  * Acoustic detections from underwater receivers
  * Depth measurements from tagged fish
  * Bathymetric constraints to ensure realistic paths

REQUIREMENTS
--------------

using XLSX, DataFrames, Dates, Missings, CSV
using GeoArrays, Images, Plots, StaticArrays, CoordinateTransformations
using Interpolations, Statistics
using Starfish, Pigeons, TransformVariables, Random
using Distributions, DynamicHMC, LogDensityProblems, LogDensityProblemsAD
using MCMCChains, StatsPlots

Data Files
----------
The code expects the following data files:

1. Bathymetry: GeoTIFF file with depth information
   Path: bathymetry/map_Firth_of_Lorn_200m.tif

2. Depth observations: CSV with timestamp and depth columns
   Path: observation/depth.csv
   Format: time, depth

3. Acoustic receivers: CSV with receiver positions
   Path: observation/moorings.csv
   Format: receiver_x, receiver_y

4. Acoustic detections: CSV with detection events
   Path: observation/acoustics.csv
   Format: time, receiver_1, receiver_2, ...
   Values: 1 (detection), 0 (no detection), NA (missing)

CONFIGURATION
--------------
Edit the configuration parameters in the main script:

const SPATIAL_RESOLUTION = 200      # Bathymetry resolution (meters)
const START_IDX = 1                 # Start index for data subset
const END_IDX = 6000                # End index for data subset
const NOISE_STD = 3.0               # Trajectory noise standard deviation

# Segment for PT analysis
init_segment = 1
end_segment = 100

Update file paths to match your directory structure:

const BATHY_PATH = "path/to/bathymetry.tif"
const DEPTH_CSV = "path/to/depth.csv"
const MOORINGS_CSV = "path/to/moorings.csv"
const ACOUSTICS_CSV = "path/to/acoustics.csv"

WORKFLOW
--------------
1. Data Loading and Preprocessing
----------------------------------
The pipeline loads and prepares all observational data:

bathy_data = load_bathymetry(BATHY_PATH)
depth_obs_df = load_depth_data(DEPTH_CSV, START_IDX, END_IDX)
acoustic_data = load_acoustic_data(MOORINGS_CSV, ACOUSTICS_CSV, START_IDX, END_IDX)

2. Initial Trajectory Generation
---------------------------------
Uses Starfish.jl to find a shortest-path trajectory through receiver activations:

res = find_shortest_trajectory(
    bathy_data.bathy,
    acoustic_data.acoustic_signals,
    acoustic_data.acoustic_pos_starfish,
    depth_obs_df.depth,
    goal_tol = 2
)

3. Trajectory Processing
-------------------------
Cleans and prepares the initial trajectory:
  - Fills NaN gaps using linear interpolation
  - Removes consecutive duplicate points
  - Adds realistic noise to create a smooth path

4. Parallel Tempering with Pigeons
-----------------------------------
Runs Bayesian inference using parallel tempering:

pt = pigeons(
    target = fish_lp,              # Posterior distribution
    reference = fish_prior_lp,     # Prior distribution (random walk)
    seed = 1234,
    n_rounds = 6,                  # Number of PT rounds
    n_chains = 10,                 # Number of parallel chains
    checkpoint = true,
    multithreaded = true
)



EXPLORER METHODS IN PIGEONS.JL
--------

Standard Sampler (Slice Sampling)
----------------------------------
The default explorer implements a slice sampler that operates without gradient 
information. It constructs proposals by:
  - Defining a slice at the current log-density level
  - Sampling uniformly from the region above this slice
  - Accepting proposals based on the slice criterion

ADVANTAGES:
  * Gradient-free: works with any target distribution, including 
    non-differentiable posteriors
  * Robust to poorly scaled parameters
  * No tuning of step sizes required
  * Effective for low-to-moderate dimensional problems

LIMITATIONS:
  * Does not exploit gradient information when available
  * Slower convergence in high-dimensional spaces
  * Less efficient for strongly correlated parameters
  * Scales poorly beyond ~50 dimensions

WHEN TO USE: 
Default choice for initial exploration, non-smooth targets, or when gradient 
computation is expensive or unavailable.

--------------------------------------------------------------------------------

MALA (Metropolis-Adjusted Langevin Algorithm)
----------------------------------------------
MALA incorporates first-order gradient information to guide proposals toward 
high-probability regions. The proposal mechanism follows a Langevin diffusion:

    θ' = θ + (ε²/2)∇log π(θ) + ε·Z,  Z ~ N(0, I)

where ε is a fixed step size and ∇log π(θ) is the gradient of the log-posterior.

ADVANTAGES:
  * Exploits gradient information for directed exploration
  * More efficient than random-walk samplers in smooth, high-dimensional spaces
  * Scales better to moderate dimensions (50-200)
  * Provides substantial speedup when gradients are cheap to compute

LIMITATIONS:
  * Requires manual step size tuning (ε parameter)
  * Performance highly sensitive to step size choice
  * Suboptimal step size can lead to low acceptance rates or slow mixing
  * Requires differentiable target distribution
  * No automatic adaptation to local geometry

WHEN TO USE: 
When gradients are available and computational cost is acceptable, particularly 
for moderately high-dimensional problems where manual tuning is feasible.

--------------------------------------------------------------------------------

AutoMALA (Automated MALA)
-------------------------
AutoMALA extends MALA with automatic step size adaptation using a preconditioner 
learned during sampling. The algorithm:
  - Adapts step sizes locally based on gradient magnitude and curvature
  - Learns a diagonal or full preconditioner matrix
  - Continuously tunes parameters to maintain target acceptance rates

ADVANTAGES:
  * Automatic tuning eliminates manual parameter selection
  * Adapts to local posterior geometry
  * Robust across different scales and parameter correlations
  * Maintains efficiency of gradient-based methods
  * Particularly effective for challenging, multi-scale posteriors
  * Reduces sensitivity to initial step size choice

LIMITATIONS:
  * Higher computational overhead per iteration than MALA
  * Adaptation phase requires burn-in period
  * May require more iterations in very high dimensions (>500)
  * Gradient computation still required
  * Additional memory for storing preconditioner

WHEN TO USE: 
Recommended default for gradient-based sampling. Especially valuable for:
  - Problems with parameters at different scales
  - Posteriors with varying curvature
  - When manual tuning is impractical
  - Production-level inference requiring robustness



5. Results Extraction and Visualization
----------------------------------------
Extracts the MAP (Maximum A Posteriori) trajectory and creates visualizations:
  - Trajectory overlay on bathymetry
  - Log-posterior evolution
  - Depth comparison (observed vs. estimated)


Prior Distribution
------------------
A random walk model with Gaussian increments:

log_p_moveRW(c1, c2) = logpdf(Normal(c2.x, σ), c1.x) + logpdf(Normal(c2.y, σ), c1.y)

Observation Models
------------------

Acoustic Receivers: Logistic detection probability based on distance

prob_detect = 1 - 1/(1 + exp(-(d - d0)/k))

  - d: distance from receiver
  - d0: detection range (default: 50m)
  - k: smoothness parameter (default: 30)

Depth Gauge: Gaussian measurement error

logpdf(Normal(max_depth, σ), observed_depth)

  - max_depth: bathymetric depth at fish position
  - σ: measurement noise (default: 2m)

Posterior Distribution
----------------------
Combines prior and likelihood:

log_posterior = log_prior + Σ log_prob_acoustic + Σ log_prob_depth


Trajectory Processing
---------------------
  - build_receiver_sequence(): Extracts activation sequence from acoustic data
  - fill_nan_runs_with_spline(): Interpolates missing trajectory segments
  - remove_consecutive_duplicates(): Cleans redundant points
  - make_noisy_linear_trajectory(): Generates smooth trajectory with noise

Observations
------------
  - prepare_acoustic_observations(): Formats acoustic data for inference
  - prepare_depth_observations(): Formats depth data for inference

Visualization
-------------
  - plot_trajectory(): Creates bathymetry map with receivers and path

OUTPUT
--------------
The code produces three main visualizations:

1. Trajectory Plot: Shows the estimated fish path overlaid on bathymetry 
   with receiver locations

2. Log-Posterior Evolution: Tracks convergence during sampling

3. Depth Comparison: Compares observed depths with estimated depths along 
   the trajectory

COMPUTATIONAL NOTES
--------------
  - The parallel tempering sampler can be computationally intensive
  - Consider starting with a small segment (e.g., 100-200 time points)
  - Use multithreaded = true to leverage multiple CPU cores
  - Checkpointing allows resuming interrupted runs


REFERENCES
--------------
  - Pigeons.jl: Parallel tempering framework (https://pigeons.run)
  - Starfish.jl: Shortest path trajectory finding
  - TransformVariables.jl: Parameter space transformations


