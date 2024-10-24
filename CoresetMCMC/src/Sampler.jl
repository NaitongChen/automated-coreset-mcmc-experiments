# general API for adaptive sampling algorithms

function sample!(kernel::AbstractKernel, model::AbstractModel, cv::AbstractLogProbEstimator, 
                    n_samples::Int64, rng::AbstractRNG; init_val::Any = nothing)
    state = init!(rng, model; init_val)
    return sample!(kernel, state, model, cv, n_samples)
end

function sample!(alg::AbstractAlgorithm, model::AbstractModel, cv::CoresetLogProbEstimator, 
                    n_samples::Int64, rng::AbstractRNG, test_func::Function; init_vals::Any = nothing)
    metaState = MetaState()
    for i in [1:alg.replicas;]
        if isnothing(init_vals)
            push!(metaState.states, init!(Xoshiro(abs(rand(rng, Int))), model))
        else    
            push!(metaState.states, init!(Xoshiro(abs(rand(rng, Int))), model; init_val = init_vals[i]))
        end
    end
    return sample!(alg, metaState, model, cv, n_samples, test_func)
end

function sample!(alg::CoresetMCMC, metaState::AbstractMetaState, model::AbstractModel, 
                    cv::CoresetLogProbEstimator, n_samples::Int64, test_func::Function)
    θs = Vector{typeof(metaState.states[1].θ)}(undef, 0)
    # reset = false
    mean_curr = nothing
    weights = Vector{Vector{Float64}}(undef, 0)
    overall_steps = alg.train_iter * alg.delay + Int(ceil(n_samples / alg.replicas))
    cumulative_lp_evals = Array{Int64}(undef, overall_steps+1)
    cumulative_grad_lp_evals = Array{Int64}(undef, overall_steps+1)
    cumulative_hess_lp_evals = Array{Int64}(undef, overall_steps+1)
    cumulative_time = Array{Float64}(undef, overall_steps+1)
    metrics = Vector{Float64}(undef, 0)
    p = Progress(overall_steps)
    generate_showvalues(x) = () -> [(:metric,test_func(x))]
    for i=0:overall_steps
        t = nothing
        try
            t = @elapsed step!(alg, metaState, model, cv, i, mean_curr)
        catch e
            println("log_potential == -Inf")
            break
        end
        cumulative_time[i+1] = (i == 0 ? t : cumulative_time[i]+t)
        cumulative_lp_evals[i+1] = mapreduce(x -> x.lp_evals, +, metaState.states)
        cumulative_grad_lp_evals[i+1] = mapreduce(x -> x.grad_lp_evals, +, metaState.states)
        cumulative_hess_lp_evals[i+1] = mapreduce(x -> x.hess_lp_evals, +, metaState.states)

        for j in [1:alg.replicas;]
            push!(θs, copy(metaState.states[j].θ))
        end
        push!(weights, copy(cv.weights))
        if i == 0
            mean_curr = mean(@view(θs[Int((i+1)*(alg.replicas/2))+1:end]))
        else
            mean_curr = (((i)*(alg.replicas/2)) .* mean_curr - sum(@view(θs[Int((i)*(alg.replicas/2))+1:Int((i+1)*(alg.replicas/2))])) + sum(@view(θs[end-alg.replicas+1:end])))/((i+1)*alg.replicas/2)
        end

        push!(metrics, test_func(mean_curr))
        next!(p; showvalues = generate_showvalues(mean_curr))
    end
    return θs, cumulative_lp_evals, cumulative_grad_lp_evals, cumulative_hess_lp_evals, cumulative_time, weights, metrics
end