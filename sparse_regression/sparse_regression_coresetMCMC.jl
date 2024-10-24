using CSV
using DataFrames
include("../CoresetMCMC/src/CoresetMCMC.jl")
using Random
using JLD2
using Statistics
using ProgressMeter
using Tables
include("../util.jl")

# seed / iter / coresetSize / learningRate / optimizer / initMix
function main(args)
    println("Running sampler with $(args)")
    
    N = 50000
    D = 10

    if isfile("../data/sparse_regression_50000.jld")
        dataset = load("../data/sparse_regression_50000.jld")["data"]
    else
        rng_data_gen = Xoshiro(2024)
        dataset = Vector{Vector{Float64}}(undef, 0)
        beta = zeros(D)
        beta[D-5+1:D] .= 5
        for i=1:N
            x = randn(rng_data_gen, D)
            y = beta' * x + 25 * randn(rng_data_gen)
            push!(dataset, vcat(x,y))
        end
        save("sparse_regression_50000.jld", "data", dataset)
        data = JLD2.load("../data/sparse_regression_50000.jld")["data"]
    end
    
    # Create the model
    println("Initializing model")
    datamat = Matrix(reduce(hcat, data)')
    model = SparseRegressionModel(length(data), D, data, datamat, 0.1, 0.1, 1, 10, 1/10, nothing)

    μp = JLD2.load("stan/summary_stats.jld", "μp")
    
    dΣp = JLD2.load("stan/summary_stats.jld", "dΣp")

    @assert length(args) == 6 "Error: script has 6 mandatory cmd line args"

    # Initialize the rng
    println("Initializing RNG")
    rng = Xoshiro(parse(Int, args[1]))
    println("Initializing sampler")
    kernel = prepare_kernel(N, 2, args)
    println("Initializing coreset")
    cv = prepare_cv(model, args, rng)

    rng_init = Xoshiro(2024)
    init_vals = []
    for i in [1:kernel.replicas;]
        θ0 = zeros(2*model.d + 1)
        X = model.datamat[:,1:end-1]
        y = model.datamat[:,end]
        θ0[1:model.d] = (X' * X) \ (X' * y) + randn(rng_init, model.d)
        θ0[(model.d+1):(2*model.d)] .= 1. .+ randn(rng_init, model.d)
        θ0[end] = var(X * θ0[1:model.d] - y) + randn(rng_init)
        push!(init_vals, θ0)
    end
    
    println("Running sampler")
    _, lp_evals, _, _, times, weights, metrics = CoresetMCMC.sample!(kernel, model, cv, 0, rng, x -> compute_metric_cont(x, μp, dΣp), init_vals=init_vals)

    CSV.write("results/sparse_regression_coresetMCMC_metric_" * args[3] * "_" *args[4] * "_" * args[1] * "_" * args[5] * "_" * args[6] * ".csv",  Tables.table(metrics), writeheader=false)
    CSV.write("results/sparse_regression_coresetMCMC_lpeval_" * args[3] * "_" *args[4] * "_" * args[1] * "_" * args[5] * "_" * args[6] * ".csv",  Tables.table(lp_evals), writeheader=false)
    if args[1] == "1000"
        JLD2.save("results/sparse_regression_burnin" * args[3] * ".jld", "test_pass2_ratio", kernel.test_pass2_ratio,
                                                                            "lps", kernel.lps,
                                                                            "mmms", kernel.mmms,
                                                                            "gnorms", kernel.gnorms,
                                                                            "iter_checks", kernel.iter_checks)
    end
end

function prepare_kernel(N::Int64, replicas::Int64, args::AbstractArray)
    # tuning settings
    proj_n = parse(Int, args[3])
    num_warmup = parse.(Int, args[6])

    # Create the algorithm
    if args[5] == "ADAM"
        sizes = parse.(Float64, split(args[4], "_"))
        if length(sizes) > 1
            kernel = CoresetMCMC(kernel = GibbsSR(), replicas = replicas, optimizer=ADAM(α = t -> sizes[1]/(t^sizes[2]), newton_correction=false), delay = 1, train_iter = parse(Int, args[2]), proj_n = proj_n, sum_to_N=false, non_neg=true, num_warmup=num_warmup)
        else
            kernel = CoresetMCMC(kernel = GibbsSR(), replicas = replicas, optimizer=ADAM(α = t -> sizes[1], newton_correction=false), delay = 1, train_iter = parse(Int, args[2]), proj_n = proj_n, sum_to_N=false, non_neg=true, num_warmup=num_warmup)
        end
    elseif args[5] == "DoG"
        sizes = parse.(Float64, split(args[4], "_"))
        kernel = CoresetMCMC(kernel = GibbsSR(), replicas = replicas, optimizer=DoG(r = sizes[1]), delay = 1, train_iter = parse(Int, args[2]), proj_n = proj_n, sum_to_N=false, non_neg=true, num_warmup=num_warmup)
    elseif args[5] == "DoWG"
        sizes = parse.(Float64, split(args[4], "_"))
        kernel = CoresetMCMC(kernel = GibbsSR(), replicas = replicas, optimizer=DoWG(rbar = sizes[1]), delay = 1, train_iter = parse(Int, args[2]), proj_n = proj_n, sum_to_N=false, non_neg=true, num_warmup=num_warmup)
    elseif args[5] == "ADAMDoGCoord"
        sizes = parse.(Float64, split(args[4], "_"))
        kernel = CoresetMCMC(kernel = GibbsSR(), replicas = replicas, optimizer=ADAMDoGCoord(r = sizes[1]), delay = 1, train_iter = parse(Int, args[2]), proj_n = proj_n, sum_to_N=false, non_neg=true, num_warmup=num_warmup)
    elseif args[5] == "ProdigyADAM_ca"
        sizes = parse.(Float64, split(args[4], "_"))
        kernel = CoresetMCMC(kernel = GibbsSR(), replicas = replicas, optimizer=prodigyADAM(ca=true, d=sizes[1]), delay = 1, train_iter = parse(Int, args[2]), proj_n = proj_n, sum_to_N=false, non_neg=true, num_warmup=num_warmup)
    end
    return kernel
end

function prepare_cv(model::CoresetMCMC.AbstractModel, args::AbstractArray, rng::AbstractRNG)
    cv = CoresetLogProbEstimator(N = parse(Int, args[3]))
    cv.inds = sample(rng, [1:model.N;], cv.N; replace = false)
    cv.sub_dataset = @view(model.datamat[cv.inds,:])
    cv.weights = (model.N / cv.N) * ones(cv.N)

    return cv
end