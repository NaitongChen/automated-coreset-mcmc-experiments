@with_kw struct GaussianLocationModel <: AbstractModel
    N::Int64
    dataset::AbstractArray
    datamat::AbstractArray
    σ_likelihood::Float64
    σ_prior::Float64
    μ_prior::Vector{Float64}
    sampler::Union{Nothing, Function} = nothing
end
function init!(rng::AbstractRNG, model::GaussianLocationModel; init_val::Any = nothing)
    if isnothing(init_val)
        θ0 = randn(rng, length(model.μ_prior))
        # θ0 = randn(length(model.μ_prior)) .+ [1:length(model.μ_prior);]
        # θ0 = 1.0 .* [1:length(model.μ_prior);]
    else
        θ0 = init_val
    end
    return State(θ = θ0, rng = rng)
end
log_prior(θ, model::GaussianLocationModel) = -0.5dot(θ-model.μ_prior, θ-model.μ_prior)/model.σ_prior^2
grad_log_prior(θ, model::GaussianLocationModel) = -(θ-model.μ_prior)/model.σ_prior^2
hess_log_prior(θ, model::GaussianLocationModel) = -I/model.σ_prior^2
log_likelihood(x, θ, model::GaussianLocationModel) = -0.5dot(x-θ, x-θ)/model.σ_likelihood^2
grad_log_likelihood(x, θ, model::GaussianLocationModel) = -(θ-x)/model.σ_likelihood^2
hess_log_likelihood(x, θ, model::GaussianLocationModel) = -I/model.σ_likelihood^2
data_grad_log_likelihood(x, θ, model::GaussianLocationModel) = -(x-θ)/model.σ_likelihood^2
data_hess_log_likelihood(x, θ, model::GaussianLocationModel) = -I/model.σ_likelihood^2
grad_data_grad_log_likelihood(x, θ, model::GaussianLocationModel) = I/model.σ_likelihood^2
hess_data_grad_log_likelihood(x, θ, model::GaussianLocationModel) = [[zeros(length(θ)) for i=1:length(θ)] for j=1:length(θ)]
grad_data_hess_log_likelihood(x, θ, model::GaussianLocationModel) = [[zeros(length(θ)) for i=1:length(θ)] for j=1:length(θ)]
hess_data_hess_log_likelihood(x, θ, model::GaussianLocationModel) = [[[zeros(length(θ)) for i=1:length(θ)] for j=1:length(θ)] for k=1:length(θ)]

function grad_log_potential(state::AbstractState, model::GaussianLocationModel, cv::CoresetLogProbEstimator)
    state.grad_lp_evals += cv.N
    return (grad_log_prior(state.θ, model) + 
            ((cv.sub_dataset .- state.θ') ./ model.σ_likelihood^2)' * cv.weights)
end

function log_potential(state::AbstractState, model::GaussianLocationModel, cv::ZeroLogProbEstimator)
    state.lp_evals += cv.N
    return (log_prior(state.θ, model) + 
            (model.N ./ cv.N) * (-0.5/model.σ_likelihood^2) * 
            sum(sum(abs2,(reduce(hcat, @view(model.dataset[cv.inds]))' .- state.θ');dims=2)))
end

function log_likelihood_array(state::AbstractState, model::GaussianLocationModel, cv::SizeBasedLogProbEstimator)
    state.lp_evals += cv.N
    return (-0.5/model.σ_likelihood^2) * sum(abs2,(reduce(hcat, @view(model.dataset[cv.inds]))' .- state.θ');dims=2)
end