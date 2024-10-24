@with_kw mutable struct prodigyADAM <: AbstractOptimizer
    α::Union{Function, Nothing, ParameterSchedulers.CosAnneal{Float64, Int64}} = nothing
    β1::Float64 = 0.9
    β2::Float64 = 0.999
    ϵ::Float64 = 10^-8
    t::Int64 = 0
    d::Float64 = 10^-6
    s::Union{Nothing, Vector{Float64}} = nothing
    r::Float64 = 0
    m::Union{Nothing, Vector{Float64}} = nothing
    v::Union{Nothing, Vector{Float64}} = nothing
    x0::Union{Nothing, AbstractArray} = nothing
    ca::Bool = false
end

function init_optimizer!(optimizer::prodigyADAM, cv::AbstractLogProbEstimator)
    optimizer.m = zeros(cv.N)
    optimizer.v = zeros(cv.N)
    optimizer.s = zeros(cv.N)
    optimizer.x0 = deepcopy(cv.weights)
    if optimizer.ca
        optimizer.α = CosAnneal(l0 = 0.0, l1 = 1.0, period = 100)
    else
        optimizer.α = t -> 1.
    end
end

function gradient_step(optimizer::prodigyADAM, grd::AbstractArray, curr::AbstractArray, iter::Int64; reset=false)
    optimizer.t += 1
    @unpack α, β1, β2, ϵ, t = optimizer
    optimizer.m .= β1 * optimizer.m + (1-β1)*optimizer.d*grd
    optimizer.v .= β2 * optimizer.v + (1-β2)*(optimizer.d^2)*grd.^2

    # learning rate update
    optimizer.r = sqrt(β2) * optimizer.r + (1 - sqrt(β2)) * α(t) * optimizer.d^2 * (grd' * (optimizer.x0 - curr))
    optimizer.s .= sqrt(β2) * optimizer.s + (1 - sqrt(β2)) * α(t) * optimizer.d^2 * grd
    dhat = optimizer.r / norm(optimizer.s, 1)
    dk = deepcopy(optimizer.d)
    optimizer.d = max(dhat, optimizer.d)

    return α(t) * dk * optimizer.m ./ (sqrt.(optimizer.v) .+ dk*ϵ), α(t) * dk ./ (sqrt.(optimizer.v) .+ dk*ϵ)
end