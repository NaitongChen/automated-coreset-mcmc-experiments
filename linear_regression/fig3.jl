using CSV
using DataFrames
include("../CoresetMCMC/src/CoresetMCMC.jl")
using Random
using JLD2
using Statistics
using Plots
using Glob
using ColorBrewer
include("../util.jl")
include("../plotting_util.jl")

function main_DoG()
    step_sizes = ["0.001", "0.01", "0.1", "1", "10"]
    ADAM_lr = ["0.001_0.5", "0.01_0.5", "0.1_0.5", "1_0.5", "10_0.5", "100_0.5"]
    alg = ["DoG"]
    coreset_sizes = ["100", "500", "1000"]
    
    adam_best_lrs = JLD2.load("../adam_best_lrs.jld", "lrs")[3]
    adam_best_labels = JLD2.load("../adam_best_lrs.jld", "labels")[3]

    b = 1

    for c in [1:length(coreset_sizes);]
        labels_mix = []
        labels_no_mix = []
        
        metrics_mix = []
        metrics_no_mix = []

        adam_no_mix = []

        for a in [1:length(step_sizes);]
            l_mix_m = nothing
            l_no_mix_m = nothing

            try
                ll = filter(x->startswith(x,"linear_regression_coresetMCMC_metric_" * coreset_sizes[c] * "_" * step_sizes[a] * "_"), readdir(pwd() * "/results"))
                l_mix = filter(x->endswith(x, "_" * alg[b] * "_10000.csv"), ll)
                ll = filter(x->startswith(x,"linear_regression_coresetMCMC_metric_" * coreset_sizes[c] * "_" * step_sizes[a] * "_"), readdir(pwd() * "/results"))
                l_no_mix = filter(x->endswith(x, "_" * alg[b] * "_0.csv"), ll)

                l_mix_v = Vector{Vector{Float64}}(undef, 0)
                l_no_mix_v = Vector{Vector{Float64}}(undef, 0)
                
                for i in [1:length(l_mix);] push!(l_mix_v, vec(Matrix(CSV.read("results/"*l_mix[i], DataFrame)))) end
                for i in [1:length(l_no_mix);] push!(l_no_mix_v, vec(Matrix(CSV.read("results/"*l_no_mix[i], DataFrame)))) end

                l_mix_m = Matrix(reduce(hcat, l_mix_v)')
                l_no_mix_m = Matrix(reduce(hcat, l_no_mix_v)')
            catch y
                @warn("missing file")
                println("linear_regression_coresetMCMC_metric_" * coreset_sizes[c] * "_" * step_sizes[a] * "_")
            end

            if !isnothing(l_mix_m)
                push!(metrics_mix, get_medians(l_mix_m))
                push!(labels_mix, step_sizes[a]*"_"*"warmup")
            end
            if !isnothing(l_no_mix_m)
                push!(metrics_no_mix, get_medians(l_no_mix_m))
                push!(labels_no_mix, step_sizes[a])
            end

            if a == 1
                p_no_mix_m = nothing

                try
                    pp = filter(x->startswith(x,"linear_regression_coresetMCMC_metric_" * coreset_sizes[c] * "_" * ADAM_lr[adam_best_lrs[c]] * "_"), readdir(pwd() * "/results"))
                    p_no_mix = filter(x->endswith(x, "_" * "ADAM" * "_0.csv"), pp)

                    p_no_mix_v = Vector{Vector{Float64}}(undef, 0)
                    for i in [1:length(p_no_mix);] push!(p_no_mix_v, vec(Matrix(CSV.read("results/"*p_no_mix[i], DataFrame)))) end

                    p_no_mix_m = Matrix(reduce(hcat, p_no_mix_v)')
                catch y
                    @warn("missing file")
                    println("linear_regression_coresetMCMC_metric_" * coreset_sizes[c] * "_" * ADAM_lr[adam_best_lrs[c]] * "_")
                end

                if !isnothing(p_no_mix_m)
                    push!(adam_no_mix, get_medians(p_no_mix_m))
                end
            end
        end

        plot()
        plot(adam_no_mix[1], linecolor = ColorBrewer.palette("Greens",5)[5], label="optimally-tuned ADAM", linewidth = 1,
                guidefontsize=20, tickfontsize=15, formatter=:plain, legendfontsize=10, margin=(5.0, :mm))

        for i in [1:size(labels_no_mix,1);]
            if i == 1
                plot!(metrics_no_mix[i], yscale=:log10, xscale=:log10, linecolor = ColorBrewer.palette("Blues",7)[i+1], label=labels_no_mix[i], legend = :bottomleft, linewidth = 1)
            else
                plot!(metrics_no_mix[i], linecolor = ColorBrewer.palette("Blues",7)[i+1], label=labels_no_mix[i], linewidth = 1)
            end
        end

        yticks!(10. .^[-2:1:9;])
        ylims!((10^-2.5, 10^9.5))
        xticks!(10. .^[0:1:6;])
        xlabel!("Iteration")
        ylabel!("Avg. Squared z-score")
        savefig("plots/trace/linear_regression_coresetMCMC_metrics_no_mix_" * coreset_sizes[c] * "_" * alg[b] * ".png")
    end
end

function main_DoWG()
    step_sizes = ["0.001", "0.01", "0.1", "1", "10"]
    ADAM_lr = ["0.001_0.5", "0.01_0.5", "0.1_0.5", "1_0.5", "10_0.5", "100_0.5"]
    alg = ["DoWG"]
    coreset_sizes = ["100", "500", "1000"]
    
    adam_best_lrs = JLD2.load("../adam_best_lrs.jld", "lrs")[3]
    adam_best_labels = JLD2.load("../adam_best_lrs.jld", "labels")[3]

    b = 1

    for c in [1:length(coreset_sizes);]
        labels_mix = []
        labels_no_mix = []
        
        metrics_mix = []
        metrics_no_mix = []

        adam_no_mix = []

        for a in [1:length(step_sizes);]
            l_mix_m = nothing
            l_no_mix_m = nothing

            try
                ll = filter(x->startswith(x,"linear_regression_coresetMCMC_metric_" * coreset_sizes[c] * "_" * step_sizes[a] * "_"), readdir(pwd() * "/results"))
                l_mix = filter(x->endswith(x, "_" * alg[b] * "_10000.csv"), ll)
                ll = filter(x->startswith(x,"linear_regression_coresetMCMC_metric_" * coreset_sizes[c] * "_" * step_sizes[a] * "_"), readdir(pwd() * "/results"))
                l_no_mix = filter(x->endswith(x, "_" * alg[b] * "_0.csv"), ll)

                l_mix_v = Vector{Vector{Float64}}(undef, 0)
                l_no_mix_v = Vector{Vector{Float64}}(undef, 0)
                
                for i in [1:length(l_mix);] push!(l_mix_v, vec(Matrix(CSV.read("results/"*l_mix[i], DataFrame)))) end
                for i in [1:length(l_no_mix);] push!(l_no_mix_v, vec(Matrix(CSV.read("results/"*l_no_mix[i], DataFrame)))) end

                l_mix_m = Matrix(reduce(hcat, l_mix_v)')
                l_no_mix_m = Matrix(reduce(hcat, l_no_mix_v)')
            catch y
                @warn("missing file")
            end

            if !isnothing(l_mix_m)
                push!(metrics_mix, get_medians(l_mix_m))
                push!(labels_mix, step_sizes[a]*"_"*"warmup")
            end
            if !isnothing(l_no_mix_m)
                push!(metrics_no_mix, get_medians(l_no_mix_m))
                push!(labels_no_mix, step_sizes[a])
            end

            if a == 1
                p_no_mix_m = nothing

                try
                    pp = filter(x->startswith(x,"linear_regression_coresetMCMC_metric_" * coreset_sizes[c] * "_" * ADAM_lr[adam_best_lrs[c]] * "_"), readdir(pwd() * "/results"))
                    p_no_mix = filter(x->endswith(x, "_" * "ADAM" * "_0.csv"), pp)

                    p_no_mix_v = Vector{Vector{Float64}}(undef, 0)
                    for i in [1:length(p_no_mix);] push!(p_no_mix_v, vec(Matrix(CSV.read("results/"*p_no_mix[i], DataFrame)))) end

                    p_no_mix_m = Matrix(reduce(hcat, p_no_mix_v)')
                catch y
                    @warn("missing file")
                    println("linear_regression_coresetMCMC_metric_" * coreset_sizes[c] * "_" * ADAM_lr[adam_best_lrs[c]] * "_")
                end

                if !isnothing(p_no_mix_m)
                    push!(adam_no_mix, get_medians(p_no_mix_m))
                end
            end
        end

        plot()
        plot(adam_no_mix[1], linecolor = ColorBrewer.palette("Greens",5)[5], label="optimally-tuned ADAM", linewidth = 1,
                guidefontsize=20, tickfontsize=15, formatter=:plain, legendfontsize=10, margin=(5.0, :mm))
        for i in [1:size(labels_no_mix,1);]
            if i == 1
                plot!(metrics_no_mix[i], yscale=:log10, xscale=:log10, linecolor = ColorBrewer.palette("Blues",7)[i+1], label=labels_no_mix[i], legend = :bottomleft, linewidth = 1)
            else
                plot!(metrics_no_mix[i], linecolor = ColorBrewer.palette("Blues",7)[i+1], label=labels_no_mix[i], linewidth = 1)
            end
        end

        yticks!(10. .^[-2:1:9;])
        ylims!((10^-2.5, 10^9.5))
        xticks!(10. .^[0:1:6;])
        xlabel!("Iteration")
        ylabel!("Avg. Squared z-score")
        savefig("plots/trace/linear_regression_coresetMCMC_metrics_no_mix_" * coreset_sizes[c] * "_" * alg[b] * ".png")
    end
end

function main_dadaptSGD()
    step_sizes = ["0.001", "0.01", "0.1", "1", "10"]
    ADAM_lr = ["0.001_0.5", "0.01_0.5", "0.1_0.5", "1_0.5", "10_0.5", "100_0.5"]
    alg = ["dadaptSGD"]
    coreset_sizes = ["100", "500", "1000"]
    
    adam_best_lrs = JLD2.load("../adam_best_lrs.jld", "lrs")[3]
    adam_best_labels = JLD2.load("../adam_best_lrs.jld", "labels")[3]

    b = 1

    for c in [1:length(coreset_sizes);]
        labels_no_mix = []
        metrics_no_mix = []
        adam_no_mix = []

        for a in [1:1;]
            l_no_mix_m = nothing

            try
                ll = filter(x->startswith(x,"linear_regression_coresetMCMC_metric_" * coreset_sizes[c] * "_" * step_sizes[a] * "_"), readdir(pwd() * "/results"))
                l_no_mix = filter(x->endswith(x, "_" * alg[b] * "_0.csv"), ll)

                l_no_mix_v = Vector{Vector{Float64}}(undef, 0)
                for i in [1:length(l_no_mix);] 
                    trace = vec(Matrix(CSV.read("results/"*l_no_mix[i], DataFrame)))
                    if length(trace) == 200000
                        push!(l_no_mix_v, trace) 
                    end
                end
                
                l_no_mix_m = Matrix(reduce(hcat, l_no_mix_v)')
            catch y
                @warn("missing file")
            end

            if !isnothing(l_no_mix_m)
                push!(metrics_no_mix, get_medians(l_no_mix_m))
                push!(labels_no_mix, step_sizes[a])
            end

            if a == 1
                p_no_mix_m = nothing

                try
                    pp = filter(x->startswith(x,"linear_regression_coresetMCMC_metric_" * coreset_sizes[c] * "_" * ADAM_lr[adam_best_lrs[c]] * "_"), readdir(pwd() * "/results"))
                    p_no_mix = filter(x->endswith(x, "_" * "ADAM" * "_0.csv"), pp)

                    p_no_mix_v = Vector{Vector{Float64}}(undef, 0)
                    for i in [1:length(p_no_mix);] push!(p_no_mix_v, vec(Matrix(CSV.read("results/"*p_no_mix[i], DataFrame)))) end

                    p_no_mix_m = Matrix(reduce(hcat, p_no_mix_v)')
                catch y
                    @warn("missing file")
                    println("linear_regression_coresetMCMC_metric_" * coreset_sizes[c] * "_" * ADAM_lr[adam_best_lrs[c]] * "_")
                end

                if !isnothing(p_no_mix_m)
                    push!(adam_no_mix, get_medians(p_no_mix_m))
                end
            end
        end

        plot()
        plot(adam_no_mix[1], linecolor = ColorBrewer.palette("Greens",5)[5], label="optimally-tuned ADAM", linewidth = 1,
                guidefontsize=20, tickfontsize=15, formatter=:plain, legendfontsize=10, margin=(5.0, :mm))
        for i in [1:1;]
            plot!(metrics_no_mix[i], yscale=:log10, xscale=:log10, linecolor = ColorBrewer.palette("Blues",7)[7], label="d-adaptation SGD", legend = :bottomleft, linewidth = 1)
        end

        yticks!(10. .^[-2:1:9;])
        ylims!((10^-2.5, 10^9.5))
        xticks!(10. .^[0:1:6;])
        xlabel!("Iteration")
        ylabel!("Avg. Squared z-score")
        savefig("plots/trace/linear_regression_coresetMCMC_metrics_no_mix_" * coreset_sizes[c] * "_" * alg[b] * ".png")
    end
end

function main_prodigyADAM()
    step_sizes = ["0.001", "0.01", "0.1", "1", "10"]
    ADAM_lr = ["0.001_0.5", "0.01_0.5", "0.1_0.5", "1_0.5", "10_0.5", "100_0.5"]
    alg = ["ProdigyADAM_ca"]
    coreset_sizes = ["100", "500", "1000"]
    
    adam_best_lrs = JLD2.load("../adam_best_lrs.jld", "lrs")[3]
    adam_best_labels = JLD2.load("../adam_best_lrs.jld", "labels")[3]

    b = 1

    for c in [1:length(coreset_sizes);]
        labels_no_mix = []
        metrics_no_mix = []
        adam_no_mix = []

        for a in [1:1;]
            l_no_mix_m = nothing

            try
                ll = filter(x->startswith(x,"linear_regression_coresetMCMC_metric_" * coreset_sizes[c] * "_" * step_sizes[a] * "_"), readdir(pwd() * "/results"))
                l_no_mix = filter(x->endswith(x, "_" * alg[b] * "_0.csv"), ll)

                l_no_mix_v = Vector{Vector{Float64}}(undef, 0)
                for i in [1:length(l_no_mix);] 
                    trace = vec(Matrix(CSV.read("results/"*l_no_mix[i], DataFrame)))
                    if length(trace) == 200000
                        push!(l_no_mix_v, trace) 
                    end
                end
                
                l_no_mix_m = Matrix(reduce(hcat, l_no_mix_v)')
            catch y
                @warn("missing file")
            end

            if !isnothing(l_no_mix_m)
                push!(metrics_no_mix, get_medians(l_no_mix_m))
                push!(labels_no_mix, step_sizes[a])
            end

            if a == 1
                p_no_mix_m = nothing

                try
                    pp = filter(x->startswith(x,"linear_regression_coresetMCMC_metric_" * coreset_sizes[c] * "_" * ADAM_lr[adam_best_lrs[c]] * "_"), readdir(pwd() * "/results"))
                    p_no_mix = filter(x->endswith(x, "_" * "ADAM" * "_0.csv"), pp)

                    p_no_mix_v = Vector{Vector{Float64}}(undef, 0)
                    for i in [1:length(p_no_mix);] push!(p_no_mix_v, vec(Matrix(CSV.read("results/"*p_no_mix[i], DataFrame)))) end

                    p_no_mix_m = Matrix(reduce(hcat, p_no_mix_v)')
                catch y
                    @warn("missing file")
                    println("linear_regression_coresetMCMC_metric_" * coreset_sizes[c] * "_" * ADAM_lr[adam_best_lrs[c]] * "_")
                end

                if !isnothing(p_no_mix_m)
                    push!(adam_no_mix, get_medians(p_no_mix_m))
                end
            end
        end

        plot()
        plot(adam_no_mix[1], linecolor = ColorBrewer.palette("Greens",5)[5], label="optimally-tuned ADAM", linewidth = 1,
                guidefontsize=20, tickfontsize=15, formatter=:plain, legendfontsize=10, margin=(5.0, :mm))
        for i in [1:1;]
            plot!(metrics_no_mix[i], yscale=:log10, xscale=:log10, linecolor = ColorBrewer.palette("Blues",7)[7], label="Prodigy ADAM", legend = :bottomleft, linewidth = 1)
        end

        yticks!(10. .^[-2:1:9;])
        ylims!((10^-2.5, 10^9.5))
        xticks!(10. .^[0:1:6;])
        xlabel!("Iteration")
        ylabel!("Avg. Squared z-score")
        savefig("plots/trace/linear_regression_coresetMCMC_metrics_no_mix_" * coreset_sizes[c] * "_" * alg[b] * ".png")
    end
end