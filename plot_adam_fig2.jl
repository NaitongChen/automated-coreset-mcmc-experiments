using CSV
using DataFrames
include("../CoresetMCMC/src/CoresetMCMC.jl")
using Random
using JLD2
using Statistics
using Plots
using Glob
using ColorBrewer
include("util.jl")
include("plotting_util.jl")

function main()
    step_sizes = ["0.001_0.5", "0.01_0.5", "0.1_0.5", "1_0.5", "10_0.5", "100_0.5"]
    alg = ["ADAM"]
    coreset_sizes = ["100", "500", "1000"]
    examples = ["gaussian_location", "sparse_regression", "linear_regression", "logistic_regression", "poisson_regression", "bradley_terry"]
    b=1
    markershape = [:circle, :rect, :diamond]
    colors = ColorBrewer.palette("Set1",8)[[2,3,4,5,8,7]]

    labels = []
    metrics = []

    for d in [1:length(examples);]
        labels_e = []
        metrics_e = []
        # metrics_e_n = []
        for c in [1:length(coreset_sizes);]
            l_step_vec = NaN .* zeros(length(step_sizes))

            for a in [1:length(step_sizes);]
                l_m = nothing
                try
                    ll = filter(x->startswith(x, examples[d] * "_coresetMCMC_metric_" * coreset_sizes[c] * "_" * step_sizes[a] * "_"), readdir(pwd() * "/" * examples[d] * "/results"))
                    l = filter(x->endswith(x, "_" * alg[b] * "_0.csv"), ll)

                    l_v = Vector{Vector{Float64}}(undef, 0)
                    for i in [1:length(l);] push!(l_v, vec(Matrix(CSV.read(examples[d]*"/results/"*l[i], DataFrame)))) end

                    l_m = Matrix(reduce(hcat, l_v)')
                catch y
                    println(examples[d] * "_coresetMCMC_metric_" * coreset_sizes[c] * "_" * step_sizes[a] * "_")
                    println("_" * alg[b] * "_0.csv")
                    @warn("missing file")
                end

                if !isnothing(l_m)
                    l_step_vec[a] = get_medians(l_m)[end]
                end
            end

            if sum(isnan.(l_step_vec)) < 7
                push!(metrics_e, l_step_vec)
                # push!(metrics_e_n, l_step_vec ./ minimum(l_step_vec))
                push!(labels_e, examples[d] * "_" * coreset_sizes[c])
            end
        end
        push!(labels, labels_e)
        push!(metrics, metrics_e)
        # push!(metrics_n, metrics_e_n)
    end

    shown_labels = ["Gaussian location", "sparse regression", "linear regression", "logistic regression", "Poisson regression", "Bradley-Terry"]
    shown_corese_sizes = ["M=100", "M=500", "M=1000"]

    plot()
    for i in [1:size(labels,1);] # examples
        for j in [1:size(labels[i],1);] # coreset sizes
            if i == 1 && j == 1
                plot(metrics[i][j], yscale=:log10, seriescolor=colors[i], label=false, legend = false #=:outerbottom=#, xticks=([1:6;], ["0.001", "0.01", "0.1", "1", "10", "100"]), 
                    markershape=markershape[j], markercolor=:match, linewidth=1, markerstrokewidth=0, #=linestyle=:dash,=# legend_column=3, guidefontsize=20, tickfontsize=15, 
                    formatter=:plain, legendfontsize=10, margin=(5.0, :mm) #=(10.0, :mm)=#)
            else
                plot!(metrics[i][j], seriescolor=colors[i], label=false, markershape=markershape[j], markercolor=:match, linewidth=1, markerstrokewidth=0 #=, linestyle=:dash=#)
            end
        end
    end
    yticks!(10. .^[-4:1:5;])
    ylims!((10^-4.5, 10^5.5))
    xlabel!("Learning Rate")
    ylabel!("Avg. Squared z-score")
    savefig("plots/adam_shrink.png")

    JLD2.save("adam_shrink.jld", "labels", labels, "metrics", metrics)
end

function ADAM_best()
    dat = JLD2.load("adam_shrink.jld")
    metrics = dat["metrics"]
    labels = dat["labels"]

    lrs = []

    for i in [1:size(labels,1);] # examples
        lr = []
        for j in [1:size(labels[i],1);] # coreset sizes
            push!(lr, argmin(metrics[i][j]))
        end
        push!(lrs, lr)
    end

    JLD2.save("adam_best_lrs.jld", "labels", labels, "lrs", lrs)
end