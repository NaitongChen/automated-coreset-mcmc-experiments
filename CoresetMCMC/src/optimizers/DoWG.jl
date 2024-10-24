@with_kw mutable struct DoWG <: AbstractOptimizer
    rbar::Float64 = 50
    grd_sum::Float64 = 0.
    x0::Union{Nothing, AbstractArray} = nothing
end

function init_optimizer!(optimizer::DoWG, cv::AbstractLogProbEstimator)
    optimizer.x0 = deepcopy(cv.weights)
end

function gradient_step(optimizer::DoWG, grd::AbstractArray, curr::AbstractArray, iter::Int64; reset=false)
    optimizer.rbar = max(norm(curr - optimizer.x0), optimizer.rbar)
    optimizer.grd_sum = optimizer.grd_sum + optimizer.rbar^2 * (grd' * grd)
    return (optimizer.rbar^2 / sqrt(optimizer.grd_sum)) .* grd, (optimizer.rbar^2 / sqrt(optimizer.grd_sum))
end