function joint_log_potential(state::AbstractState, r::AbstractVector, 
								model::AbstractModel, cv::AbstractLogProbEstimator)
	return log_potential(state, model, cv) - 0.5*norm(r)^2
end

function project(metaState::AbstractMetaState, model::AbstractModel, cv::SizeBasedLogProbEstimator)
    proj = zeros(cv.N, length(metaState.states))
    Threads.@threads for i in [1:length(metaState.states);]
		proj[:, i] = log_likelihood_array(metaState.states[i], model, cv)
    end
    proj .-=  mean(proj, dims=2)
    return proj
end

function project(metaState::AbstractMetaState, model::AbstractModel, cv::SizeBasedLogProbEstimator, θs::AbstractArray)
    θ0 = copy(metaState.states[1].θ)
	proj = zeros(cv.N, length(θs))
    for i in [1:length(θs);]
		metaState.states[1].θ = θs[i]
		proj[:, i] = log_likelihood_array(metaState.states[1], model, cv)
    end
    proj .-=  mean(proj, dims=2)
	metaState.states[1].θ = θ0
    return proj
end

function update_scan(metaState::AbstractMetaState, model::AbstractModel, cv_zero::SizeBasedLogProbEstimator)
	if cv_zero.N != model.N
		if isnothing(cv_zero.inds_set)
			cv_zero.inds_set = [1:model.N;]
			cv_zero.total_size = model.N
			cv_zero.current_location = 1
			cv_zero.inds_length = cv_zero.N
		end

		if cv_zero.current_location + cv_zero.inds_length - 1 <= cv_zero.total_size
			inds = cv_zero.inds_set[cv_zero.current_location:(cv_zero.current_location + cv_zero.inds_length - 1)]
			if cv_zero.current_location + cv_zero.inds_length <= cv_zero.total_size
				cv_zero.current_location = cv_zero.current_location + cv_zero.inds_length
			else
				cv_zero.current_location = 1
			end
		else
			inds = cv_zero.inds_set[cv_zero.current_location:end]
			l = length(inds)
			inds = vcat(cv_zero.inds_set[1:(cv_zero.inds_length - l)], inds)
			cv_zero.current_location = cv_zero.inds_length - l + 1
		end

		update_estimator!(metaState.states[1], model, cv_zero, nothing, nothing, inds)
	end
end

function project_sum(metaState::AbstractMetaState, model::AbstractModel, cv_zero::SizeBasedLogProbEstimator, cv::CoresetLogProbEstimator)
    proj = zeros(cv_zero.N, length(metaState.states))
	update_scan(metaState, model, cv_zero)
    Threads.@threads for i in [1:length(metaState.states);]
		proj[:, i] = (model.N / cv_zero.N) * log_likelihood_array(metaState.states[i], model, cv_zero)
    end
    proj .-=  mean(proj, dims=2)
    return vec(sum(proj, dims=1))
end

function project_sum(metaState::AbstractMetaState, model::AbstractModel, cv::SizeBasedLogProbEstimator, θs::AbstractArray)
    θ0 = copy(metaState.states[1].θ)
	proj = zeros(model.N, length(θs))
    for i in [1:length(θs);]
		metaState.states[1].θ = θs[i]
		proj[:, i] = log_likelihood_array(metaState.states[1], model, cv)
    end
    proj .-=  mean(proj, dims=2)
	metaState.states[1].θ = θ0
    return vec(sum(proj, dims=1))
end

function log_logistic(a::Real)
    return -log1pexp(-a)
end

function ordered(a::AbstractArray, coef::Real)
    return coef .* reverse(cumprod(vcat(1.0, logistic.(a))))
end

function neg_sigmoid(x::Real)
    return -1.0/(1.0 + exp(-x))
end

function log_sigmoid(x::Real)
    if x < -300
        return x
    else
        return -log1p(exp(-x))
    end
end