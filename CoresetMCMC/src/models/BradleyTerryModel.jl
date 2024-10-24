@with_kw struct BradleyTerryModel <: AbstractModel
    N::Int64
    d::Int64
    dataset::AbstractArray
    datamat::AbstractArray
    σ2::Float64
    sampler::Union{Nothing, Function} = nothing
end
function init!(rng::AbstractRNG, model::BradleyTerryModel; init_val::Any = nothing)
    if isnothing(init_val)
        θ0 = randn(rng, model.d)
    else
        θ0 = init_val
    end
    return State(θ = θ0, rng = rng)
end

log_prior(θ, model::BradleyTerryModel) = -0.5dot(θ, θ)/model.σ2

grad_log_prior(θ, model::BradleyTerryModel) = error("not implemented")
hess_log_prior(θ, model::BradleyTerryModel) = error("not implemented")

function log_likelihood(x, θ, model::BradleyTerryModel)
    difference = (θ[Int(x[2])] - θ[Int(x[1])])/400
    logp = -log1p(exp(difference))
    log1minusp = -log1p(exp(-difference))
    return x[3] * logp + (1-x[3]) * log1minusp
end

grad_log_likelihood(x, θ, model::BradleyTerryModel) = error("not implemented")
hess_log_likelihood(x, θ, model::BradleyTerryModel) = error("not implemented")
data_grad_log_likelihood(x, θ, model::BradleyTerryModel) = error("not implemented")
data_hess_log_likelihood(x, θ, model::BradleyTerryModel) = error("not implemented")
grad_data_grad_log_likelihood(x, θ, model::BradleyTerryModel) = error("not implemented")
hess_data_grad_log_likelihood(x, θ, model::BradleyTerryModel) = error("not implemented")
grad_data_hess_log_likelihood(x, θ, model::BradleyTerryModel) = error("not implemented")
hess_data_hess_log_likelihood(x, θ, model::BradleyTerryModel) = error("not implemented")
grad_log_potential(state::AbstractState, model::BradleyTerryModel, cv::CoresetLogProbEstimator) = error("not implemented")
log_potential(state::AbstractState, model::BradleyTerryModel, cv::ZeroLogProbEstimator) = error("not implemented")

function log_likelihood_array(state::AbstractState, model::BradleyTerryModel, cv::SizeBasedLogProbEstimator)
    state.lp_evals += cv.N
    θ1s = state.θ[Int.(@view(cv.sub_dataset[:,1]))]
    θ2s = state.θ[Int.(@view(cv.sub_dataset[:,2]))]
    ys = @view(cv.sub_dataset[:,3])
    differences = (θ2s .- θ1s) ./ 400
    logps = -log1p.(exp.(differences))
    log1minusps = -log1p.(exp.(-differences))
    return ys .* logps .+ (1 .- ys) .* log1minusps
end

# function log_likelihood_array(state::AbstractState, model::BradleyTerryModel, cv::SizeBasedLogProbEstimator)
#     state.lp_evals += cv.N
#     θ1s = state.θ[Int.(@view(cv.sub_dataset[:,1]))]
#     θ2s = state.θ[Int.(@view(cv.sub_dataset[:,2]))]
#     ys = @view(cv.sub_dataset[:,3])
#     ps = logistic.(θ1s .- θ2s)
#     oneminusps = logistic.(θ2s .- θ1s)
#     return ys .* log.(ps) .+ (1 .- ys) .* log.(oneminusps)
# end