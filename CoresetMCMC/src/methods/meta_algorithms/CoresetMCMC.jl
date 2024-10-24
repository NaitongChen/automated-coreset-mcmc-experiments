@with_kw mutable struct CoresetMCMC <: AbstractAlgorithm
    kernel::AbstractKernel
    replicas::Int64 = 2
    delay::Int64 = 1
    train_iter::Int64 = 20000
    optimizer::AbstractOptimizer
    cv_zero::Union{Nothing, ZeroLogProbEstimator} = nothing
    proj_n::Int64 = 1000
    sum_to_N::Bool = true
    non_neg::Bool = false
    proj_mat::Union{Nothing, AbstractMatrix} = nothing
    init_mix::Bool = true

    lps_trace::AbstractArray = []
    num_warmup::Int64 = 10000
    final_warmup_iter::Int64 = -1
    still_warming_up::Bool = true
    test_pass2_ratio::AbstractArray = []
    ratios_lp::AbstractArray = []
    lps::AbstractArray = []
    stepsizes::AbstractArray = []
end

function step!(alg::CoresetMCMC, metaState::AbstractMetaState, model::AbstractModel, cv::CoresetLogProbEstimator, iter::Int64, mean_curr::Union{Nothing, AbstractArray})
    if (alg.sum_to_N && alg.non_neg) || (!alg.sum_to_N && !alg.non_neg)
        error("one and only one constraint can be applied")
    end

    if isodd(alg.replicas)
        error("only even number of replicas supported")
    end
    
    if iter == 0
        if alg.num_warmup == 0
            alg.still_warming_up = false
        end
        init_estimator!(metaState.states[1], model, cv)
        init_optimizer!(alg.optimizer, cv)
        alg.proj_mat = I(cv.N) - (1/cv.N) * ones(cv.N, cv.N)
        for i in 1:alg.replicas
            push!(alg.lps_trace, zeros(0))
            push!(alg.lps, zeros(0))
            push!(alg.ratios_lp, zeros(0))
        end
    end

    if isnothing(alg.cv_zero)
        alg.cv_zero = ZeroLogProbEstimator(N = alg.proj_n)
        # only need to update once if projection dimension is equal to N
        if alg.proj_n == model.N
            update_estimator!(metaState.states[1], model, alg.cv_zero, nothing, nothing, [1:model.N;])
        end
    end

    if alg.init_mix && alg.num_warmup > 0 && iter <= alg.num_warmup && alg.still_warming_up
        Threads.@threads for i=1:alg.replicas
            _, cached_lp = step!(alg.kernel, metaState.states[i], model, cv, iter)
            push!(alg.lps[i], cached_lp)
        end

        if iter >= 3*length(metaState.states[1].θ) && iter % 30 == 0
            for i in 1:alg.replicas
                c2 = compute_termination_statistic(alg.lps[i])
                push!(alg.ratios_lp[i], c2)
            end

            mmm = median([alg.ratios_lp[i][end] for i=1:alg.replicas])
            if mmm <= 0.5
                push!(alg.test_pass2_ratio, iter)
                alg.final_warmup_iter = iter
                alg.still_warming_up = false
                println("done warming up, # of iteration = " * string(alg.final_warmup_iter))
            end
        end
    else
        Threads.@threads for i=1:alg.replicas
            _, cached_lp = step!(alg.kernel, metaState.states[i], model, cv, iter)
            push!(alg.lps_trace[i], cached_lp)
        end
    end

    # update the coreset weights
    if iter > 0 && iter <= alg.train_iter * alg.delay && iter % alg.delay == 0 && !alg.still_warming_up
        g = est_gradient(metaState, model, cv, alg.cv_zero, alg.optimizer)
        if alg.final_warmup_iter >= 0
            iter = iter - alg.final_warmup_iter + 1
        end
        gstep, stepsize = gradient_step(alg.optimizer, g, cv.weights, iter; reset=false)
        push!(alg.stepsizes, stepsize)
        gradient_update(alg, cv, gstep)
    end
end

function est_gradient(metaState::AbstractMetaState, model::AbstractModel, cv::CoresetLogProbEstimator, cv_zero::ZeroLogProbEstimator, optimizer::AbstractOptimizer)
    g = project(metaState, model, cv)
    proj_sum = project_sum(metaState, model, cv_zero, cv)
    h = proj_sum .- (g' * cv.weights)
    grd = -g*h/(length(metaState.states)-1)
    if typeof(optimizer) <: Union{ADAM}
        if optimizer.newton_correction
            grd = ((vec(sum(g.^2, dims=2))/(length(metaState.states)-1)).^-1) .* grd
        end
    end
    return grd
end

function gradient_update(alg::CoresetMCMC, cv::CoresetLogProbEstimator, gstep::AbstractVector)
    if alg.non_neg
        cv.weights -= gstep
        cv.weights = max.(0., cv.weights)
    elseif alg.sum_to_N
        cv.weights -= (alg.proj_mat * gstep)
    end

    if sum(isinf.(cv.weights)) > 0
        error("infinite weight")
    end
end

function compute_termination_statistic(seq::AbstractArray)
    sample_size_lp = Int(ceil(length(seq) / 3))
    s1_lp = seq[end-(2*sample_size_lp)+1:end-sample_size_lp]
    s2_lp = seq[end-sample_size_lp+1:end]

    m1_lp = mean(s1_lp)
    m2_lp = mean(s2_lp)

    X1_lp = hcat(ones(length(s1_lp)), [1:length(s1_lp);])
    X2_lp = hcat(ones(length(s2_lp)), [1:length(s2_lp);])
    coef1_lp = X1_lp \ s1_lp
    coef2_lp = X2_lp \ s2_lp

    residual1_lp = s1_lp - X1_lp*coef1_lp
    se1_lp = sqrt(residual1_lp' * residual1_lp / (sample_size_lp - 2))

    residual2_lp = s2_lp - X2_lp*coef2_lp
    se2_lp = sqrt(residual2_lp' * residual2_lp / (sample_size_lp - 2))

    return abs((m1_lp - m2_lp)/max(se1_lp, se2_lp))    
end