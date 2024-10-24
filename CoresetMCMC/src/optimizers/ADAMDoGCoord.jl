@with_kw mutable struct ADAMDoGCoord <: AbstractOptimizer
    r::Float64 = 1
    max_dist::Union{Nothing, AbstractArray} = nothing
    grd_sum::Union{Nothing, AbstractArray} = nothing
    x0::Union{Nothing, AbstractArray} = nothing
    m::Union{Nothing, AbstractArray} = nothing
    β1::Float64 = 0.9
    β2::Float64 = 0.999
    ϵ::Float64 = 10^-8
end

function init_optimizer!(optimizer::ADAMDoGCoord, cv::AbstractLogProbEstimator)
    optimizer.max_dist = zeros(cv.N)
    optimizer.grd_sum = zeros(cv.N)
    optimizer.m = zeros(cv.N)
    optimizer.x0 = deepcopy(cv.weights)
end

function gradient_step(optimizer::ADAMDoGCoord, grd::AbstractArray, curr::AbstractArray, iter::Int64; reset=false)
    optimizer.grd_sum = optimizer.β2 * optimizer.grd_sum .+ (1-optimizer.β2) * grd.^2
    grd_sum_hat = optimizer.grd_sum / (1-optimizer.β2^iter)

    optimizer.m .= optimizer.β1 * optimizer.m + (1-optimizer.β1) * grd
    m_hat = optimizer.m / (1-optimizer.β1^iter)

    if iter == 1
        return (optimizer.r ./ (sqrt.(iter .* grd_sum_hat) .+ optimizer.ϵ)) .* m_hat, (optimizer.r ./ (sqrt.(iter .* grd_sum_hat) .+ optimizer.ϵ))
    else
        optimizer.max_dist = optimizer.β1 * optimizer.max_dist + (1-optimizer.β1) * max.(abs.(curr - optimizer.x0), optimizer.max_dist)
        max_dist_hat = optimizer.max_dist / (1-optimizer.β1^iter)
        return (max_dist_hat ./ (sqrt.(iter .* grd_sum_hat) .+ optimizer.ϵ)) .* m_hat, (max_dist_hat ./ (sqrt.(iter .* grd_sum_hat) .+ optimizer.ϵ))
    end
end