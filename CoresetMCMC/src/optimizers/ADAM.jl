@with_kw mutable struct ADAM <: AbstractOptimizer
    α::Function = t -> 0.001
    β1::Float64 = 0.9
    β2::Float64 = 0.999
    ϵ::Float64 = 10^-8
    t::Int64 = 0
    m::Union{Nothing, Vector{Float64}} = nothing
    v::Union{Nothing, Vector{Float64}} = nothing
    newton_correction::Bool = false
end

function init_optimizer!(optimizer::ADAM, cv::AbstractLogProbEstimator)
    optimizer.m = zeros(cv.N)
    optimizer.v = zeros(cv.N)
end

function gradient_step(optimizer::ADAM, grd::AbstractArray, curr::AbstractArray, iter::Int64; reset=false)
    optimizer.t += 1
    @unpack α, β1, β2, ϵ, t = optimizer
    optimizer.m .= β1 * optimizer.m + (1-β1)*grd
    optimizer.v .= β2 * optimizer.v + (1-β2)*grd.^2
    m̂ = optimizer.m/(1-β1^t)
    v̂ = optimizer.v/(1-β2^t) 
    return α(t) * m̂./(sqrt.(v̂) .+ ϵ), α(t) ./ (sqrt.(v̂) .+ ϵ)
end