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

    N = 1_0000
    D = 20

    if isfile("../data/gaussian_data_1M_20D.jld")
        dataset = load("../data/gaussian_data_1M_20D.jld")["data"]
    else
        rng_data_gen = Xoshiro(2024)
        dataset = [randn(rng_data_gen, D) for i=1:N]
        save("gaussian_data_1M_20D.jld", "data", dataset)
    end
    datamat = Matrix(reduce(hcat, dataset)')
    
    # Create the model
    println("Initializing model")
    model = CoresetMCMCSampler.GaussianLocationModel(length(dataset), dataset, datamat, 1.0, 1.0, fill(10.0, D), nothing)

    μp = sum(dataset) / (1+N)
    Σp = diagm((1/(1+N)) * ones(D))
    dΣp = diag(Σp)

    @assert length(args) == 6 "Error: script has 6 mandatory cmd line args"

    # Initialize the rng
    println("Initializing RNG")
    rng = Xoshiro(parse(Int, args[1]))
    println("Initializing sampler")
    kernel = prepare_kernel(D, 2, args)
    println("Initializing coreset")
    cv = prepare_cv(model, args, rng)

    rng_init = Xoshiro(2024)
    init_vals = []
    for i in 1:kernel.replicas
        push!(init_vals, sqrt(1/(sum(cv.weights)+1)) .* randn(rng_init, D) + vec(sum(cv.sub_dataset' .* cv.weights', dims=2) ./ (1+sum(cv.weights))))
    end

    println("Running sampler")
    _, lp_evals, _, _, times, weights, metrics = CoresetMCMCSampler.sample!(kernel, model, cv, 0, rng, x -> compute_metric(x, μp, dΣp), init_vals=init_vals)

    CSV.write("results/gaussian_location_coresetMCMC_metric_" * args[3] * "_" *args[4] * "_" * args[1] * "_" * args[5] * "_" * args[6] * ".csv",  Tables.table(metrics), writeheader=false)
    CSV.write("results/gaussian_location_coresetMCMC_lpeval_" * args[3] * "_" *args[4] * "_" * args[1] * "_" * args[5] * "_" * args[6] * ".csv",  Tables.table(lp_evals), writeheader=false)
    if args[1] == "1000"
        JLD2.save("results/gaussian_location_burnin" * args[3] * ".jld", "test_pass2_ratio", kernel.test_pass2_ratio,
                                                                            "lps", kernel.lps,
                                                                            "mmms", kernel.mmms,
                                                                            "gnorms", kernel.gnorms,
                                                                            "iter_checks", kernel.iter_checks)
    end
end

function prepare_kernel(D::Int64, replicas::Int64, args::AbstractArray)
    # tuning settings
    proj_n = parse(Int, args[3])
    num_warmup = parse.(Int, args[6])

    # Create the algorithm
    if args[5] == "ADAM"
        sizes = parse.(Float64, split(args[4], "_"))
        if length(sizes) > 1
            kernel = CoresetMCMCSampler.CoresetMCMC(kernel = CoresetMCMCSampler.Artificial(D=D), replicas = replicas, optimizer=CoresetMCMCSampler.ADAM(α = t -> sizes[1]/(t^sizes[2]), newton_correction=false), delay = 1, train_iter = parse(Int, args[2]), proj_n = proj_n, sum_to_N=false, non_neg=true, num_warmup=num_warmup)
        else
            kernel = CoresetMCMCSampler.CoresetMCMC(kernel = CoresetMCMCSampler.Artificial(D=D), replicas = replicas, optimizer=CoresetMCMCSampler.ADAM(α = t -> sizes[1], newton_correction=false), delay = 1, train_iter = parse(Int, args[2]), proj_n = proj_n, sum_to_N=false, non_neg=true, num_warmup=num_warmup)
        end
    elseif args[5] == "DoG"
        sizes = parse.(Float64, split(args[4], "_"))
        kernel = CoresetMCMCSampler.CoresetMCMC(kernel = CoresetMCMCSampler.Artificial(D=D), replicas = replicas, optimizer=CoresetMCMCSampler.DoG(r = sizes[1]), delay = 1, train_iter = parse(Int, args[2]), proj_n = proj_n, sum_to_N=false, non_neg=true, num_warmup=num_warmup)
    elseif args[5] == "DoWG"
        sizes = parse.(Float64, split(args[4], "_"))
        kernel = CoresetMCMCSampler.CoresetMCMC(kernel = CoresetMCMCSampler.Artificial(D=D), replicas = replicas, optimizer=CoresetMCMCSampler.DoWG(rbar = sizes[1]), delay = 1, train_iter = parse(Int, args[2]), proj_n = proj_n, sum_to_N=false, non_neg=true, num_warmup=num_warmup)
    elseif args[5] == "ADAMDoGCoord"
        sizes = parse.(Float64, split(args[4], "_"))
        kernel = CoresetMCMCSampler.CoresetMCMC(kernel = CoresetMCMCSampler.Artificial(D=D), replicas = replicas, optimizer=CoresetMCMCSampler.ADAMDoGCoord(r = sizes[1]), delay = 1, train_iter = parse(Int, args[2]), proj_n = proj_n, sum_to_N=false, non_neg=true, num_warmup=num_warmup)
    elseif args[5] == "ProdigyADAM_ca"
        sizes = parse.(Float64, split(args[4], "_"))
        kernel = CoresetMCMCSampler.CoresetMCMC(kernel = CoresetMCMCSampler.Artificial(D=D), replicas = replicas, optimizer=CoresetMCMCSampler.prodigyADAM(ca=true, d=sizes[1]), delay = 1, train_iter = parse(Int, args[2]), proj_n = proj_n, sum_to_N=false, non_neg=true, num_warmup=num_warmup)
    end
    return kernel
end

function prepare_cv(model::CoresetMCMCSampler.AbstractModel, args::AbstractArray, rng::AbstractRNG)
    cv = CoresetMCMCSampler.CoresetLogProbEstimator(N = parse(Int, args[3]))
    cv.inds = sample(rng, [1:model.N;], cv.N; replace = false)
    cv.sub_dataset = @view(model.datamat[cv.inds,:])
    cv.weights = (model.N / cv.N) * ones(cv.N)

    return cv
end