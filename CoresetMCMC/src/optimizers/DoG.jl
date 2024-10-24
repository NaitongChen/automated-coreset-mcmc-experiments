@with_kw mutable struct DoG <: AbstractOptimizer
    r::Float64 = 1
    max_dist::Float64 = 0.
    grd_sum::Float64 = 0.
    x0::Union{Nothing, AbstractArray} = nothing
end

function init_optimizer!(optimizer::DoG, cv::AbstractLogProbEstimator)
    optimizer.x0 = deepcopy(cv.weights)
end

function gradient_step(optimizer::DoG, grd::AbstractArray, curr::AbstractArray, iter::Int64; reset=false)
    optimizer.grd_sum = optimizer.grd_sum + grd' * grd
    if iter == 1
        return (optimizer.r / sqrt(optimizer.grd_sum)) .* grd, (optimizer.r / sqrt(optimizer.grd_sum))
    else
        optimizer.max_dist = max(norm(curr - optimizer.x0), optimizer.max_dist)
        return (optimizer.max_dist / sqrt(optimizer.grd_sum)) .* grd, (optimizer.max_dist / sqrt(optimizer.grd_sum))
    end
end