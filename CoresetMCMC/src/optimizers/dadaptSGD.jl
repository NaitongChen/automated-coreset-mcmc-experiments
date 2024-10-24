@with_kw mutable struct dadaptSGD <: AbstractOptimizer
    dotsum::Union{Float64, Nothing} = nothing
    γ::Float64 = 1
    G::Union{Float64, Nothing} = nothing
    s::Union{Nothing, AbstractArray} = nothing
    d::Float64 = 10^-6
end

function init_optimizer!(optimizer::dadaptSGD, cv::AbstractLogProbEstimator)
    optimizer.s = zeros(cv.N)
end

function gradient_step(optimizer::dadaptSGD, grd::AbstractArray, curr::AbstractArray, iter::Int64; reset=false)
    if iter == 1
        optimizer.G = norm(grd)
        optimizer.dotsum = (optimizer.d * optimizer.γ / optimizer.G) * (grd' * optimizer.s)
    end

    λk = optimizer.d * optimizer.γ / optimizer.G # λ_k
    ds = λk * (grd' * optimizer.s)
    optimizer.s = optimizer.s + λk .* grd # s_k+1
    optimizer.d = max(optimizer.d, 2 * optimizer.dotsum / norm(optimizer.s)) # d_k+1
    optimizer.dotsum = optimizer.dotsum + ds

    return λk .* grd, λk
end