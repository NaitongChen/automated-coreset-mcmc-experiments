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

function main_adam()
    step_sizes = ["0.001", "0.01", "0.1", "1", "10", "100"]
    alg = ["ADAM"]
    coreset_sizes = ["100", "500", "1000"]
    examples = ["gaussian_location", "sparse_regression", "linear_regression", "logistic_regression", "poisson_regression", "bradley_terry"]
    b=1
    markershape = [:circle, :rect, :diamond]
    colors = ColorBrewer.palette("Set1",8)[[2,3,4,5,8,7]]

    labels = []
    metrics = []
    metrics_n = []

    ADAMDoGCoord_m = JLD2.load("ADAMDoGCoord_mix.jld", "metrics")

    for d in [1:length(examples);]
        labels_e = []
        metrics_e = []
        metrics_e_n = []
        for c in [1:length(coreset_sizes);]
            ADAMDoGCoord_best = ADAMDoGCoord_m[d][c][1]

            l_step_vec = NaN .* zeros(length(step_sizes))

            for a in [1:length(step_sizes);]
                l_m = nothing
                try
                    ll = filter(x->startswith(x, examples[d] * "_coresetMCMC_metric_" * coreset_sizes[c] * "_" * step_sizes[a] * "_0.5_"), readdir(pwd() * "/" * examples[d] * "/results"))
                    l = filter(x->endswith(x, "_" * alg[b] * "_0.csv"), ll)
                    
                    l_v = Vector{Vector{Float64}}(undef, 0)
                    for i in [1:length(l);] push!(l_v, vec(Matrix(CSV.read(examples[d]*"/results/"*l[i], DataFrame)))) end
                    l_m = Matrix(reduce(hcat, l_v)')
                catch y
                    @warn("missing file")
                end

                if !isnothing(l_m)
                    l_step_vec[a] = get_medians(l_m)[end]
                end
            end

            if sum(isnan.(l_step_vec)) < 7
                push!(metrics_e, l_step_vec)
                push!(metrics_e_n, l_step_vec ./ ADAMDoGCoord_best)
                push!(labels_e, examples[d] * "_" * coreset_sizes[c])
            end
        end
        push!(labels, labels_e)
        push!(metrics, metrics_e)
        push!(metrics_n, metrics_e_n)
    end

    plot()
    hline([1], color=:black, linewidth=2, guidefontsize=20, tickfontsize=15, formatter=:plain, legendfontsize=10, margin=(5.0, :mm))
    for i in [1:size(labels,1);] # examples
        for j in [1:size(labels[i],1);] # coreset sizes
            if i == 1 && j == 1
                plot!(metrics_n[i][j], yscale=:log10, seriescolor=colors[i], label=labels[i][j], legend = false, xticks=([1:6;], ["0.001", "0.01", "0.1", "1", "10", "100"]), markershape=markershape[j], markercolor=:match, linewidth=1, markerstrokewidth=0, linestyle=:dash)
            else
                plot!(metrics_n[i][j], seriescolor=colors[i], label=labels[i][j], markershape=markershape[j], markercolor=:match, linewidth=1, markerstrokewidth=0, linestyle=:dash)
            end
        end
    end
    yticks!(10. .^[-2:1:5;])
    ylims!((10^-2.5, 10^5.5))
    xlims!((0.8,6.2))
    xlabel!("Learning Rate")
    ylabel!("Rel. Avg. Sq. z-score")
    savefig("plots/ADAMDoGCoord_normalized_adam_reverse.png")

    # JLD2.save("ADAMDoGCoord_mix.jld", "labels", labels, "metrics", metrics)
end

function main_adam_mix()
    step_sizes = ["0.001", "0.01", "0.1", "1", "10", "100"]
    alg = ["ADAM"]
    coreset_sizes = ["100", "500", "1000"]
    examples = ["gaussian_location", "sparse_regression", "linear_regression", "logistic_regression", "poisson_regression", "bradley_terry"]
    b=1
    markershape = [:circle, :rect, :diamond]
    colors = ColorBrewer.palette("Set1",8)[[2,3,4,5,8,7]]

    labels = []
    metrics = []
    metrics_n = []

    ADAMDoGCoord_m = JLD2.load("ADAMDoGCoord_mix.jld", "metrics")

    for d in [1:length(examples);]
        labels_e = []
        metrics_e = []
        metrics_e_n = []
        for c in [1:length(coreset_sizes);]
            ADAMDoGCoord_best = ADAMDoGCoord_m[d][c][1]

            l_step_vec = NaN .* zeros(length(step_sizes))

            for a in [1:length(step_sizes);]
                l_m = nothing
                try
                    ll = filter(x->startswith(x, examples[d] * "_coresetMCMC_metric_" * coreset_sizes[c] * "_" * step_sizes[a] * "_0.5_"), readdir(pwd() * "/" * examples[d] * "/results"))
                    l = filter(x->endswith(x, "_" * alg[b] * "_10000.csv"), ll)
                    
                    l_v = Vector{Vector{Float64}}(undef, 0)
                    for i in [1:length(l);] push!(l_v, vec(Matrix(CSV.read(examples[d]*"/results/"*l[i], DataFrame)))) end
                    l_m = Matrix(reduce(hcat, l_v)')
                catch y
                    @warn("missing file")
                end

                if !isnothing(l_m)
                    l_step_vec[a] = get_medians(l_m)[end]
                end
            end

            if sum(isnan.(l_step_vec)) < 7
                push!(metrics_e, l_step_vec)
                push!(metrics_e_n, l_step_vec ./ ADAMDoGCoord_best)
                push!(labels_e, examples[d] * "_" * coreset_sizes[c])
            end
        end
        push!(labels, labels_e)
        push!(metrics, metrics_e)
        push!(metrics_n, metrics_e_n)
    end

    plot()
    hline([1], color=:black, linewidth=2, guidefontsize=20, tickfontsize=15, formatter=:plain, legendfontsize=10, margin=(5.0, :mm))
    for i in [1:size(labels,1);] # examples
        for j in [1:size(labels[i],1);] # coreset sizes
            if i == 1 && j == 1
                plot!(metrics_n[i][j], yscale=:log10, seriescolor=colors[i], label=labels[i][j], legend = false, xticks=([1:6;], ["0.001", "0.01", "0.1", "1", "10", "100"]), markershape=markershape[j], markercolor=:match, linewidth=1, markerstrokewidth=0, linestyle=:dash)
            else
                plot!(metrics_n[i][j], seriescolor=colors[i], label=labels[i][j], markershape=markershape[j], markercolor=:match, linewidth=1, markerstrokewidth=0, linestyle=:dash)
            end
        end
    end
    yticks!(10. .^[-2:1:5;])
    ylims!((10^-2.5, 10^5.5))
    xlims!((0.8,6.2))
    xlabel!("Learning Rate")
    ylabel!("Rel. Avg. Sq. z-score")
    savefig("plots/ADAMDoGCoord_normalized_adam_reverse_mix.png")

    # JLD2.save("ADAMDoGCoord_mix.jld", "labels", labels, "metrics", metrics)
end

function main_DoG()
    step_sizes = ["0.001", "0.01", "0.1", "1", "10"]
    alg = ["DoG"]
    coreset_sizes = ["100", "500", "1000"]
    examples = ["gaussian_location", "sparse_regression", "linear_regression", "logistic_regression", "poisson_regression", "bradley_terry"]
    b=1
    markershape = [:circle, :rect, :diamond]
    colors = ColorBrewer.palette("Set1",8)[[2,3,4,5,8,7]]

    labels = []
    metrics = []
    metrics_n = []

    ADAMDoGCoord_m = JLD2.load("ADAMDoGCoord_mix.jld", "metrics")

    for d in [1:length(examples);]
        labels_e = []
        metrics_e = []
        metrics_e_n = []
        for c in [1:length(coreset_sizes);]
            ADAMDoGCoord_best = ADAMDoGCoord_m[d][c][1]

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
                    @warn("missing file")
                end

                if !isnothing(l_m)
                    l_step_vec[a] = get_medians(l_m)[end]
                end
            end

            if sum(isnan.(l_step_vec)) < 7
                push!(metrics_e, l_step_vec)
                push!(metrics_e_n, l_step_vec ./ ADAMDoGCoord_best)
                push!(labels_e, examples[d] * "_" * coreset_sizes[c])
            end
        end
        push!(labels, labels_e)
        push!(metrics, metrics_e)
        push!(metrics_n, metrics_e_n)
    end

    plot()
    hline([1], color=:black, linewidth=2, guidefontsize=20, tickfontsize=15, formatter=:plain, legendfontsize=10, margin=(5.0, :mm))
    for i in [1:size(labels,1);] # examples
        for j in [1:size(labels[i],1);] # coreset sizes
            if i == 1 && j == 1
                plot!(metrics_n[i][j], yscale=:log10, seriescolor=colors[i], label=labels[i][j], legend = false, xticks=([1:5;], ["0.001", "0.01", "0.1", "1", "10"]), markershape=markershape[j], markercolor=:match, linewidth=1, markerstrokewidth=0, linestyle=:dash)
            else
                plot!(metrics_n[i][j], seriescolor=colors[i], label=labels[i][j], markershape=markershape[j], markercolor=:match, linewidth=1, markerstrokewidth=0, linestyle=:dash)
            end
        end
    end
    yticks!(10. .^[-1:1:4;])
    ylims!((10^-1.5, 10^4.5))
    xlims!((0.8,5.2))
    xlabel!("Initial Parameter r")
    ylabel!("Rel. Avg. Sq. z-score")
    savefig("plots/ADAMDoGCoord_normalized_DoG_reverse.png")

    # JLD2.save("ADAMDoGCoord_mix.jld", "labels", labels, "metrics", metrics)
end

function main_dog_mix()
    step_sizes = ["0.001", "0.01", "0.1", "1", "10"]
    alg = ["DoG"]
    coreset_sizes = ["100", "500", "1000"]
    examples = ["gaussian_location", "sparse_regression", "linear_regression", "logistic_regression", "poisson_regression", "bradley_terry"]
    b=1
    markershape = [:circle, :rect, :diamond]
    colors = ColorBrewer.palette("Set1",8)[[2,3,4,5,8,7]]

    labels = []
    metrics = []
    metrics_n = []

    ADAMDoGCoord_m = JLD2.load("ADAMDoGCoord_mix.jld", "metrics")

    for d in [1:length(examples);]
        labels_e = []
        metrics_e = []
        metrics_e_n = []
        for c in [1:length(coreset_sizes);]
            ADAMDoGCoord_best = ADAMDoGCoord_m[d][c][1]

            l_step_vec = NaN .* zeros(length(step_sizes))

            for a in [1:length(step_sizes);]
                l_m = nothing
                try
                    ll = filter(x->startswith(x, examples[d] * "_coresetMCMC_metric_" * coreset_sizes[c] * "_" * step_sizes[a] * "_"), readdir(pwd() * "/" * examples[d] * "/results"))
                    l = filter(x->endswith(x, "_" * alg[b] * "_10000.csv"), ll)
                    
                    l_v = Vector{Vector{Float64}}(undef, 0)
                    for i in [1:length(l);] push!(l_v, vec(Matrix(CSV.read(examples[d]*"/results/"*l[i], DataFrame)))) end
                    l_m = Matrix(reduce(hcat, l_v)')
                catch y
                    println(examples[d] * "_coresetMCMC_metric_" * coreset_sizes[c] * "_" * step_sizes[a] * "_")
                    @warn("missing file")
                end

                if !isnothing(l_m)
                    l_step_vec[a] = get_medians(l_m)[end]
                end
            end

            if sum(isnan.(l_step_vec)) < 7
                push!(metrics_e, l_step_vec)
                push!(metrics_e_n, l_step_vec ./ ADAMDoGCoord_best)
                push!(labels_e, examples[d] * "_" * coreset_sizes[c])
            end
        end
        push!(labels, labels_e)
        push!(metrics, metrics_e)
        push!(metrics_n, metrics_e_n)
    end

    plot()
    hline([1], color=:black, linewidth=2, guidefontsize=20, tickfontsize=15, formatter=:plain, legendfontsize=10, margin=(5.0, :mm))
    for i in [1:size(labels,1);] # examples
        for j in [1:size(labels[i],1);] # coreset sizes
            if i == 1 && j == 1
                plot!(metrics_n[i][j], yscale=:log10, seriescolor=colors[i], label=labels[i][j], legend = false, xticks=([1:5;], ["0.001", "0.01", "0.1", "1", "10"]), markershape=markershape[j], markercolor=:match, linewidth=1, markerstrokewidth=0, linestyle=:dash)
            else
                plot!(metrics_n[i][j], seriescolor=colors[i], label=labels[i][j], markershape=markershape[j], markercolor=:match, linewidth=1, markerstrokewidth=0, linestyle=:dash)
            end
        end
    end
    yticks!(10. .^[-1:1:3;])
    ylims!((10^-1.5, 10^3.5))
    xlims!((0.8,5.2))
    xlabel!("Initial Parameter r")
    ylabel!("Rel. Avg. Sq. z-score")
    savefig("plots/ADAMDoGCoord_normalized_DoG_reverse_mix.png")

    # JLD2.save("ADAMDoGCoord_mix.jld", "labels", labels, "metrics", metrics)
end

function main_DoWG()
    step_sizes = ["0.001", "0.01", "0.1", "1", "10", "100"]
    alg = ["DoWG"]
    coreset_sizes = ["100", "500", "1000"]
    examples = ["gaussian_location", "sparse_regression", "linear_regression", "logistic_regression", "poisson_regression", "bradley_terry"]
    b=1
    markershape = [:circle, :rect, :diamond]
    colors = ColorBrewer.palette("Set1",8)[[2,3,4,5,8,7]]

    labels = []
    metrics = []
    metrics_n = []

    ADAMDoGCoord_m = JLD2.load("ADAMDoGCoord_mix.jld", "metrics")

    for d in [1:length(examples);]
        labels_e = []
        metrics_e = []
        metrics_e_n = []
        for c in [1:length(coreset_sizes);]
            ADAMDoGCoord_best = ADAMDoGCoord_m[d][c][1]

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
                    @warn("missing file")
                end

                if !isnothing(l_m)
                    l_step_vec[a] = get_medians(l_m)[end]
                end
            end

            if sum(isnan.(l_step_vec)) < 7
                push!(metrics_e, l_step_vec)
                push!(metrics_e_n, l_step_vec ./ ADAMDoGCoord_best)
                push!(labels_e, examples[d] * "_" * coreset_sizes[c])
            end
        end
        push!(labels, labels_e)
        push!(metrics, metrics_e)
        push!(metrics_n, metrics_e_n)
    end

    plot()
    hline([1], color=:black, linewidth=2, guidefontsize=20, tickfontsize=15, formatter=:plain, legendfontsize=10, margin=(5.0, :mm))
    for i in [1:size(labels,1);] # examples
        for j in [1:size(labels[i],1);] # coreset sizes
            if i == 1 && j == 1
                plot!(metrics_n[i][j], yscale=:log10, seriescolor=colors[i], label=labels[i][j], legend = false, xticks=([1:5;], ["0.001", "0.01", "0.1", "1", "10"]), markershape=markershape[j], markercolor=:match, linewidth=1, markerstrokewidth=0, linestyle=:dash)
            else
                plot!(metrics_n[i][j], seriescolor=colors[i], label=labels[i][j], markershape=markershape[j], markercolor=:match, linewidth=1, markerstrokewidth=0, linestyle=:dash)
            end
        end
    end
    yticks!(10. .^[-1:1:4;])
    ylims!((10^-1.5, 10^4.5))
    xlims!((0.8,5.2))
    xlabel!("Initial Parameter r")
    ylabel!("Rel. Avg. Sq. z-score")
    savefig("plots/ADAMDoGCoord_normalized_DoWG_reverse.png")

    # JLD2.save("ADAMDoGCoord_mix.jld", "labels", labels, "metrics", metrics)
end

function main_DoWG_mix()
    step_sizes = ["0.001", "0.01", "0.1", "1", "10"]
    alg = ["DoWG"]
    coreset_sizes = ["100", "500", "1000"]
    examples = ["gaussian_location", "sparse_regression", "linear_regression", "logistic_regression", "poisson_regression", "bradley_terry"]
    b=1
    markershape = [:circle, :rect, :diamond]
    colors = ColorBrewer.palette("Set1",8)[[2,3,4,5,8,7]]

    labels = []
    metrics = []
    metrics_n = []

    ADAMDoGCoord_m = JLD2.load("ADAMDoGCoord_mix.jld", "metrics")

    for d in [1:length(examples);]
        labels_e = []
        metrics_e = []
        metrics_e_n = []
        for c in [1:length(coreset_sizes);]
            ADAMDoGCoord_best = ADAMDoGCoord_m[d][c][1]

            l_step_vec = NaN .* zeros(length(step_sizes))

            for a in [1:length(step_sizes);]
                l_m = nothing
                try
                    ll = filter(x->startswith(x, examples[d] * "_coresetMCMC_metric_" * coreset_sizes[c] * "_" * step_sizes[a] * "_"), readdir(pwd() * "/" * examples[d] * "/results"))
                    l = filter(x->endswith(x, "_" * alg[b] * "_10000.csv"), ll)
                    
                    l_v = Vector{Vector{Float64}}(undef, 0)
                    for i in [1:length(l);] push!(l_v, vec(Matrix(CSV.read(examples[d]*"/results/"*l[i], DataFrame)))) end
                    l_m = Matrix(reduce(hcat, l_v)')
                catch y
                    println(examples[d] * "_coresetMCMC_metric_" * coreset_sizes[c] * "_" * step_sizes[a] * "_")
                    @warn("missing file")
                end

                if !isnothing(l_m)
                    l_step_vec[a] = get_medians(l_m)[end]
                end
            end

            if sum(isnan.(l_step_vec)) < 7
                push!(metrics_e, l_step_vec)
                push!(metrics_e_n, l_step_vec ./ ADAMDoGCoord_best)
                push!(labels_e, examples[d] * "_" * coreset_sizes[c])
            end
        end
        push!(labels, labels_e)
        push!(metrics, metrics_e)
        push!(metrics_n, metrics_e_n)
    end

    plot()
    hline([1], color=:black, linewidth=2, guidefontsize=20, tickfontsize=15, formatter=:plain, legendfontsize=10, margin=(5.0, :mm))
    for i in [1:size(labels,1);] # examples
        for j in [1:size(labels[i],1);] # coreset sizes
            if i == 1 && j == 1
                plot!(metrics_n[i][j], yscale=:log10, seriescolor=colors[i], label=labels[i][j], legend = false, xticks=([1:5;], ["0.001", "0.01", "0.1", "1", "10"]), markershape=markershape[j], markercolor=:match, linewidth=1, markerstrokewidth=0, linestyle=:dash)
            else
                plot!(metrics_n[i][j], seriescolor=colors[i], label=labels[i][j], markershape=markershape[j], markercolor=:match, linewidth=1, markerstrokewidth=0, linestyle=:dash)
            end
        end
    end
    yticks!(10. .^[-1:1:3;])
    ylims!((10^-1.5, 10^3.5))
    xlims!((0.8,5.2))
    xlabel!("Initial Parameter r")
    ylabel!("Rel. Avg. Sq. z-score")
    savefig("plots/ADAMDoGCoord_normalized_DoWG_reverse_mix.png")

    # JLD2.save("ADAMDoGCoord_mix.jld", "labels", labels, "metrics", metrics)
end

function main_prodigyADAM_mix()
    step_sizes = ["0.001", "0.01", "0.1", "1", "10"]
    alg = ["ProdigyADAM_ca"]
    coreset_sizes = ["100", "500", "1000"]
    examples = ["gaussian_location", "sparse_regression", "linear_regression", "logistic_regression", "poisson_regression", "bradley_terry"]
    b=1
    markershape = [:circle, :rect, :diamond]
    colors = ColorBrewer.palette("Set1",8)[[2,3,4,5,8,7]]

    labels = []
    metrics = []
    metrics_n = []

    ADAMDoGCoord_m = JLD2.load("ADAMDoGCoord_mix.jld", "metrics")

    for d in [1:length(examples);]
        labels_e = []
        metrics_e = []
        metrics_e_n = []
        for c in [1:length(coreset_sizes);]
            ADAMDoGCoord_best = ADAMDoGCoord_m[d][c][1]

            l_step_vec = NaN .* zeros(length(step_sizes))

            for a in [1:length(step_sizes);]
                l_m = nothing
                try
                    ll = filter(x->startswith(x, examples[d] * "_coresetMCMC_metric_" * coreset_sizes[c] * "_" * step_sizes[a] * "_"), readdir(pwd() * "/" * examples[d] * "/results/oct10"))
                    l = filter(x->endswith(x, "_" * alg[b] * "_10000.csv"), ll)
                    
                    l_v = Vector{Vector{Float64}}(undef, 0)
                    for i in [1:length(l);] push!(l_v, vec(Matrix(CSV.read(examples[d]*"/results/oct10/"*l[i], DataFrame)))) end
                    l_m = Matrix(reduce(hcat, l_v)')
                catch y
                    println(examples[d] * "_coresetMCMC_metric_" * coreset_sizes[c] * "_" * step_sizes[a] * "_")
                    @warn("missing file")
                end

                if !isnothing(l_m)
                    l_step_vec[a] = get_medians(l_m)[end]
                end
            end

            if sum(isnan.(l_step_vec)) < 7
                push!(metrics_e, l_step_vec)
                push!(metrics_e_n, l_step_vec ./ ADAMDoGCoord_best)
                push!(labels_e, examples[d] * "_" * coreset_sizes[c])
            end
        end
        push!(labels, labels_e)
        push!(metrics, metrics_e)
        push!(metrics_n, metrics_e_n)
    end

    plot()
    hline([1], color=:black, linewidth=2, guidefontsize=20, tickfontsize=15, formatter=:plain, legendfontsize=10, margin=(5.0, :mm))
    for i in [1:size(labels,1);] # examples
        for j in [1:size(labels[i],1);] # coreset sizes
            if i == 1 && j == 1
                plot!(metrics_n[i][j], yscale=:log10, seriescolor=colors[i], label=labels[i][j], legend = false, xticks=([1:5;], ["0.001", "0.01", "0.1", "1", "10"]), markershape=markershape[j], markercolor=:match, linewidth=1, markerstrokewidth=0, linestyle=:dash)
            else
                plot!(metrics_n[i][j], seriescolor=colors[i], label=labels[i][j], markershape=markershape[j], markercolor=:match, linewidth=1, markerstrokewidth=0, linestyle=:dash)
            end
        end
    end
    yticks!(10. .^[-1:1:5;])
    ylims!((10^-1.5, 10^5.5))
    xlims!((0.8,5.2))
    xlabel!("Initial Parameter d")
    ylabel!("Rel. Avg. Sq. z-score")
    savefig("plots/ADAMDoGCoord_normalized_padam_reverse_mix.png")

    # JLD2.save("ADAMDoGCoord_mix.jld", "labels", labels, "metrics", metrics)
end

function main_prodigyADAM()
    step_sizes = ["0.001", "0.01", "0.1", "1", "10"]
    alg = ["ProdigyADAM_ca"]
    coreset_sizes = ["100", "500", "1000"]
    examples = ["gaussian_location", "sparse_regression", "linear_regression", "logistic_regression", "poisson_regression", "bradley_terry"]
    b=1
    markershape = [:circle, :rect, :diamond]
    colors = ColorBrewer.palette("Set1",8)[[2,3,4,5,8,7]]

    labels = []
    metrics = []
    metrics_n = []

    ADAMDoGCoord_m = JLD2.load("ADAMDoGCoord_mix.jld", "metrics")

    for d in [1:length(examples);]
        labels_e = []
        metrics_e = []
        metrics_e_n = []
        for c in [1:length(coreset_sizes);]
            ADAMDoGCoord_best = ADAMDoGCoord_m[d][c][1]

            l_step_vec = NaN .* zeros(length(step_sizes))

            for a in [1:length(step_sizes);]
                l_m = nothing
                try
                    ll = filter(x->startswith(x, examples[d] * "_coresetMCMC_metric_" * coreset_sizes[c] * "_" * step_sizes[a] * "_"), readdir(pwd() * "/" * examples[d] * "/results/oct10")) 
                    l = filter(x->endswith(x, "_" * alg[b] * "_0.csv"), ll)
                    
                    l_v = Vector{Vector{Float64}}(undef, 0)
                    for i in [1:length(l);] push!(l_v, vec(Matrix(CSV.read(examples[d]*"/results/oct10/"*l[i], DataFrame)))) end
                    l_m = Matrix(reduce(hcat, l_v)')
                catch y
                    println(examples[d] * "_coresetMCMC_metric_" * coreset_sizes[c] * "_" * step_sizes[a] * "_")
                    @warn("missing file")
                end

                if !isnothing(l_m)
                    l_step_vec[a] = get_medians(l_m)[end]
                end
            end

            if sum(isnan.(l_step_vec)) < 7
                push!(metrics_e, l_step_vec)
                push!(metrics_e_n, l_step_vec ./ ADAMDoGCoord_best)
                push!(labels_e, examples[d] * "_" * coreset_sizes[c])
            end
        end
        push!(labels, labels_e)
        push!(metrics, metrics_e)
        push!(metrics_n, metrics_e_n)
    end

    plot()
    hline([1], color=:black, linewidth=2, guidefontsize=20, tickfontsize=15, formatter=:plain, legendfontsize=10, margin=(5.0, :mm))
    for i in [1:size(labels,1);] # examples
        for j in [1:size(labels[i],1);] # coreset sizes
            if i == 1 && j == 1
                plot!(metrics_n[i][j], yscale=:log10, seriescolor=colors[i], label=labels[i][j], legend = false, xticks=([1:5;], ["0.001", "0.01", "0.1", "1", "10"]), markershape=markershape[j], markercolor=:match, linewidth=1, markerstrokewidth=0, linestyle=:dash)
            else
                plot!(metrics_n[i][j], seriescolor=colors[i], label=labels[i][j], markershape=markershape[j], markercolor=:match, linewidth=1, markerstrokewidth=0, linestyle=:dash)
            end
        end
    end
    yticks!(10. .^[-1:1:5;])
    ylims!((10^-1.5, 10^5.5))
    xlims!((0.8,5.2))
    xlabel!("Initial Parameter d")
    ylabel!("Rel. Avg. Sq. z-score")
    savefig("plots/ADAMDoGCoord_normalized_padam_reverse.png")

    # JLD2.save("ADAMDoGCoord_mix.jld", "labels", labels, "metrics", metrics)
end