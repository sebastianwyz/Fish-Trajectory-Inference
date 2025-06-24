import Pkg

Pkg.activate(".")
Pkg.instantiate()

include("find_PT3.jl")

# - Try Enzyme.jl or Mooncake.jl for faster Automatic differentiation

bathymetry_map = channelview(Gray.(load("C:\\Users\\Usuario\\OneDrive - UNIVERSIDAD DE SEVILLA\\Escritorio\\SECOND HALF\\LABORATORY OF COMPUTATIONAL PHYSICS B\\FISH TRACKING\\LAST VERSION\\map_Firth_of_Lorn_200m.tif"))) * 100 .- 1;

bathymetry_int = extrapolate(interpolate(bathymetry_map, BSpline(Linear())),-1.0);

receiver1 = Receiver((x=10, y=45), k=30.0, dist=30.0)
receiver2 = Receiver((x=100, y=50), k=30.0, dist=30.0)
receiver3 = Receiver((x=110, y=140), k=30.0, dist=30.0)

#--------- bridges

tmax = 100       #100 for toy bathimetry

n_bridges = 10000
bridges = []
failed_traj=0
n=n_bridges/2
for _ in (1:(n_bridges/2)) 
    global failed_traj
    path = simulate_unbiased_path(
        tmax;
        rec1=receiver1,
        rec2=receiver2,
        σ_step=3.0,
        bathymetry_int=bathymetry_int,
        coarse_steps=50,  #70 for toy bathimetry
        min_endpoint_prob=0.5,
        noise_σ=0.5
    )
    
    if path !== nothing
        push!(bridges, path)
    else
        failed_traj=failed_traj+1
    end
end
print("Number of failed trajectories for horizontal bridges: $failed_traj, over $n")

failed_traj=0
for _ in ((n_bridges/2 +1) : n_bridges)
    global failed_traj
    path = simulate_unbiased_path(
        tmax;
        rec1=receiver1,
        rec2=receiver3,
        σ_step=3.0,
        bathymetry_int=bathymetry_int,
        coarse_steps=65,  #70 for toy bathimetry
        min_endpoint_prob=0.5,
        noise_σ=0.5
    )
    
    if path !== nothing
        push!(bridges, path)
    else
        failed_traj=failed_traj+1
    end
end
print("\nNumber of failed trajectories for oblique bridges: $failed_traj, over $n")

bridges = filter(!isnothing, bridges)  # keep just the valid trajectories
bridges_x = [[p.x for p in bridge] for bridge in bridges]
bridges_y = [[p.y for p in bridge] for bridge in bridges]

s_depth = bridges[1] #The trajectory from which we extract the depth data
s_init = bridges[end] #The trajectory that we try to optimize via PT 


xs_d = [p.x for p in s_depth]
ys_d = [p.y for p in s_depth]

xs = [p.x for p in s_init]
ys = [p.y for p in s_init]

#--------- data
# Accustic signals:

receivers = [receiver1, receiver2, receiver3]
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


fish_prior_lp = FishPriorPotential(mapping, v_init)
fish_lp = FishLogPotential(Ydepth, Yaccustic, bathymetry_int, mapping, v_init)


# ----------------- AutoMALA Grid Search -----------------
step_sizes = [2.5]
base_n_refreshes = [15, 30]
exponent_n_refreshes = [0.5, 1]


results_mala = []

for step_size in step_sizes, base_n_refresh in base_n_refreshes, exponent_n_refresh in exponent_n_refreshes
    println("AutoMALA: step_size=$step_size, base_n_refresh=$base_n_refresh, exponent_n_refresh=$exponent_n_refresh")
    elapsed_time = @elapsed begin
        pt = pigeons(
            target            = fish_lp,
            reference         = fish_prior_lp,
            seed              = 1234,
            n_rounds          = 7,
            n_chains          = 5,
            checkpoint        = true,
            multithreaded     = true,
            explorer          = AutoMALA(
                step_size = step_size,
                base_n_refresh = base_n_refresh,
                exponent_n_refresh = exponent_n_refresh,
                default_autodiff_backend = AutoEnzyme()
            ),
            record            = [traces; record_default()]
        )
        pt_samples  = Chains(pt)
    end

    # Evaluate log-posterior for the last sample
    v_cold = pt_samples.value[end, 1:2*tmax]
    logpost_cold = logdensity(fish_lp, v_cold)

    push!(results_mala, (
        step_size = step_size,
        base_n_refresh = base_n_refresh,
        exponent_n_refresh = exponent_n_refresh,
        elapsed_time = elapsed_time,
        logpost_cold = logpost_cold,
        pt_samples = pt_samples # Save samples for plotting
    ))
end

# Find best trajectory for AutoMALA
best_result_mala = nothing
best_logpost_mala = -Inf
for res in results_mala
    pt_samples = res.pt_samples
    logposteriors = [logdensity(fish_lp, pt_samples.value[i, 1:2*tmax]) for i in 1:size(pt_samples.value, 1)]
    idx = argmax(logposteriors)
    if logposteriors[idx] > best_logpost_mala
        global best_logpost_mala = logposteriors[idx]
        global best_result_mala = (res, idx)
    end
end

# Plot AutoMALA results
if best_result_mala !== nothing
    res, idx = best_result_mala
    pt_samples = res.pt_samples
    v_best = pt_samples.value[idx, 1:2*tmax] |> vec
    best_trajectory = TransformVariables.transform(mapping, v_best)
    xs_best = [p.x for p in best_trajectory]
    ys_best = [p.y for p in best_trajectory]

    plt = heatmap(bathymetry_map[end:-1:1, :],
                  xlim=(0, 400), ylim=(0, 300),
                  color=:blues,
                  title="AutoMALA: All Trajectories")
    n_samples = size(pt_samples.value, 1)
    for i in 1:n_samples
        v = pt_samples.value[i, 1:2*tmax]
        if all(v .== v_best)
            continue
        end
        S = TransformVariables.transform(mapping, v)
        xs_ss = [p.x for p in S]
        ys_ss = [p.y for p in S]
        plot!(plt, xs_ss, ys_ss, alpha=0.15, color=:black, label=false)
    end
    plot!(plt, xs_best, ys_best, lw=3, color=:orange, label="Best Pigeons Trajectory")
    plot!(plt, xs, ys, lw=3, color=:red, label="Initial Trajectory")
    plot!(plt, xs_d, ys_d, lw=3, color=:blue, label="Data Trajectory")
    plot!(plt, make_circle(receiver1.x, receiver1.y, receiver1.dist), color=:red, label=false)
    plot!(plt, make_circle(receiver2.x, receiver2.y, receiver2.dist), color=:red, label=false)
    plot!(plt, make_circle(receiver3.x, receiver3.y, receiver3.dist), color=:red, label=false)
    scatter!(plt, [receiver1.x, receiver2.x, receiver3.x], [receiver1.y, receiver2.y, receiver3.y], color=:red, label=false)
    display(plt)
end

# Print AutoMALA results table
v_init_vec = TransformVariables.inverse(mapping, s_init)
v_depth_vec = TransformVariables.inverse(mapping, s_depth)
logpost_init = logdensity(fish_lp, v_init_vec)
logpost_depth = logdensity(fish_lp, v_depth_vec)

println("\nAutoMALA Results:")
println(rpad("step_size",10) *
        rpad("base_n_refresh",15) *
        rpad("exponent_n_refresh",20) *
        rpad("elapsed_time",15) *
        rpad("logpost_cold",15) *
        rpad("logpost_init",15) *
        "logpost_depth")
println("-"^100)
for res in results_mala
    println(
        rpad(string(res.step_size),10) *
        rpad(string(res.base_n_refresh),15) *
        rpad(string(res.exponent_n_refresh),20) *
        rpad(@sprintf("%.3f", res.elapsed_time),15) *
        rpad(@sprintf("%.3f", res.logpost_cold),15) *
        rpad(@sprintf("%.3f", logpost_init),15) *
        @sprintf("%.3f", logpost_depth)
    )
end
if best_result_mala !== nothing
    res, _ = best_result_mala
    println("\nBest AutoMALA: step_size=$(res.step_size), base_n_refresh=$(res.base_n_refresh), exponent_n_refresh=$(res.exponent_n_refresh)")
    println("With logpost_cold = $(res.logpost_cold)")
end

# ----------------- SliceSampler Grid Search -----------------
n_passes_list = [1, 2, 10]
max_iter_list = [50, 100]

results_slice = []

for n_passes in n_passes_list, max_iter in max_iter_list
    println("SliceSampler: n_passes=$n_passes, max_iter=$max_iter")
    elapsed_time = @elapsed begin
        pt = pigeons(
            target            = fish_lp,
            reference         = fish_prior_lp,
            seed              = 1234,
            n_rounds          = 7,
            n_chains          = 5,
            checkpoint        = true,
            multithreaded     = true,
            explorer          = SliceSampler(
                n_passes = n_passes,
                max_iter = max_iter
            ),
            record            = [traces; record_default()]
        )
        pt_samples  = Chains(pt)
    end

    v_cold = pt_samples.value[end, 1:2*tmax]
    logpost_cold = logdensity(fish_lp, v_cold)

    push!(results_slice, (
        n_passes = n_passes,
        max_iter = max_iter,
        elapsed_time = elapsed_time,
        logpost_cold = logpost_cold,
        pt_samples = pt_samples
    ))
end

# Find best trajectory for SliceSampler
best_result_slice = nothing
best_logpost_slice = -Inf
for res in results_slice
    pt_samples = res.pt_samples
    logposteriors = [logdensity(fish_lp, pt_samples.value[i, 1:2*tmax]) for i in 1:size(pt_samples.value, 1)]
    idx = argmax(logposteriors)
    if logposteriors[idx] > best_logpost_slice
        global best_logpost_slice = logposteriors[idx]
        global best_result_slice = (res, idx)
    end
end

# Plot SliceSampler results
if best_result_slice !== nothing
    res, idx = best_result_slice
    pt_samples = res.pt_samples
    v_best = pt_samples.value[idx, 1:2*tmax] |> vec
    best_trajectory = TransformVariables.transform(mapping, v_best)
    xs_best = [p.x for p in best_trajectory]
    ys_best = [p.y for p in best_trajectory]

    plt = heatmap(bathymetry_map[end:-1:1, :],
                  xlim=(0, 400), ylim=(0, 300),
                  color=:blues,
                  title="SliceSampler: All Trajectories")
    n_samples = size(pt_samples.value, 1)
    for i in 1:n_samples
        v = pt_samples.value[i, 1:2*tmax]
        if all(v .== v_best)
            continue
        end
        S = TransformVariables.transform(mapping, v)
        xs_ss = [p.x for p in S]
        ys_ss = [p.y for p in S]
        plot!(plt, xs_ss, ys_ss, alpha=0.15, color=:black, label=false)
    end
    plot!(plt, xs_best, ys_best, lw=3, color=:orange, label="Best Pigeons Trajectory")
    plot!(plt, xs, ys, lw=3, color=:red, label="Initial Trajectory")
    plot!(plt, xs_d, ys_d, lw=3, color=:blue, label="Data Trajectory")
    plot!(plt, make_circle(receiver1.x, receiver1.y, receiver1.dist), color=:red, label=false)
    plot!(plt, make_circle(receiver2.x, receiver2.y, receiver2.dist), color=:red, label=false)
    plot!(plt, make_circle(receiver3.x, receiver3.y, receiver3.dist), color=:red, label=false)
    scatter!(plt, [receiver1.x, receiver2.x, receiver3.x], [receiver1.y, receiver2.y, receiver3.y], color=:red, label=false)
    display(plt)
end

# Print SliceSampler results table
println("\nSliceSampler Results:")
println(rpad("n_passes",10) *
        rpad("max_iter",15) *
        rpad("elapsed_time",15) *
        rpad("logpost_cold",15) *
        rpad("logpost_init",15) *
        "logpost_depth")
println("-"^80)
for res in results_slice
    println(
        rpad(string(res.n_passes),10) *
        rpad(string(res.max_iter),15) *
        rpad(@sprintf("%.3f", res.elapsed_time),15) *
        rpad(@sprintf("%.3f", res.logpost_cold),15) *
        rpad(@sprintf("%.3f", logpost_init),15) *
        @sprintf("%.3f", logpost_depth)
    )
end
if best_result_slice !== nothing
    res, _ = best_result_slice
    println("\nBest SliceSampler: n_passes=$(res.n_passes), max_iter=$(res.max_iter)")
    println("With logpost_cold = $(res.logpost_cold)")
end