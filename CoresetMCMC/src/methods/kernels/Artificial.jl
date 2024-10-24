@with_kw struct Artificial <: AbstractKernel
    β::Float64 = 0.8
    D::Union{Nothing, Int64} = nothing
end

function step!(kernel::Artificial, state::AbstractState, model::AbstractModel, cv::AbstractLogProbEstimator, iter::Int64)
    update_estimator!(state, model, cv, nothing, nothing, nothing)
    μ = vec(sum(cv.sub_dataset' .* cv.weights', dims=2) ./ (1+sum(cv.weights)))
    state.θ = sqrt(kernel.β) * state.θ + sqrt(1 - kernel.β) * (sqrt(1/(sum(cv.weights)+1)) .* randn(kernel.D) + μ) + (1 - sqrt(kernel.β) - sqrt(1 - kernel.β)) * μ
    lp = log_potential(state, model, cv)
    return state.θ, lp
end