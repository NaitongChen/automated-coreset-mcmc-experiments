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

function main_gaussian_location()
    for c in ["100", "500", "1000"]
        dat = JLD2.load("results/gaussian_location_burnin" * c *".jld")
        gnorms = dat["gnorms"]
        mmms = dat["mmms"]
        iter_checks = dat["iter_checks"]
        pass = dat["test_pass2_ratio"][1]

        p = plot(xlabel = "Iteration", margin=(5.0, :mm))
        pt = twinx(p)
        plot!(p, gnorms, yscale=:log10, color=ColorBrewer.palette("Blues",5)[3], ylabel="Gradient Norm", legend=false, guidefontsize=20, tickfontsize=15, formatter=:plain, legendfontsize=10)
        hline!(pt, [0.5], yscale=:identity, color=ColorBrewer.palette("Oranges",5)[3], legend=false, linewidth=2)
        plot!(pt, iter_checks, mmms, yscale=:identity, color=ColorBrewer.palette("Greens",5)[5], ylabel="Hot-Start Test Statistic", legend=false, guidefontsize=20, tickfontsize=15, formatter=:plain, legendfontsize=10)
        savefig("plots/gaussian_location_burnin" * c * ".png")
    end
end

function main_linear_regression()
    for c in ["100", "500", "1000"]
        dat = JLD2.load("results/lin_reg_burnin" * c * ".jld")
        gnorms = dat["gnorms"]
        mmms = dat["mmms"]
        iter_checks = dat["iter_checks"]
        pass = dat["test_pass2_ratio"][1]

        p = plot(xlabel = "Iteration", margin=(5.0, :mm))
        pt = twinx(p)
        plot!(p, gnorms, yscale=:log10, color=ColorBrewer.palette("Blues",5)[3], ylabel="Gradient Norm", guidefontsize=20, tickfontsize=15, formatter=:plain, legendfontsize=10, legend=false, yticks=10. .^[-2:1:6;])
        plot!(pt, iter_checks, NaN.*mmms, label = "gradient norm", color=ColorBrewer.palette("Blues",5)[3], grid=false, legend=:topright)
        plot!(pt, iter_checks, mmms, yscale=:identity, color=ColorBrewer.palette("Greens",5)[5], ylabel="Hot-Start Test Statistic", guidefontsize=20, tickfontsize=15, formatter=:plain, legendfontsize=10, label="burn-in test stat", yticks=[0,5,10,15,20])
        hline!(pt, [0.5], yscale=:identity, color=ColorBrewer.palette("Oranges",5)[3], linewidth=2, label="hot-start test stat = 0.5")
        savefig("plots/lin_reg_burnin" * c * ".png")
    end
end

function main_logistic_regression()
    for c in ["100", "500", "1000"]
        dat = JLD2.load("results/log_reg_burnin" * c * ".jld")
        gnorms = dat["gnorms"]
        mmms = dat["mmms"]
        iter_checks = dat["iter_checks"]
        pass = dat["test_pass2_ratio"][1]

        p = plot(xlabel = "Iteration", margin=(5.0, :mm))
        pt = twinx(p)
        plot!(p, gnorms, yscale=:log10, color=ColorBrewer.palette("Blues",5)[3], ylabel="Gradient Norm", legend=false, guidefontsize=20, tickfontsize=15, formatter=:plain, legendfontsize=10, yticks=10. .^[-2:1:6;])
        hline!(pt, [0.5], yscale=:identity, color=ColorBrewer.palette("Oranges",5)[3], legend=false, linewidth=2)
        plot!(pt, iter_checks, mmms, yscale=:identity, color=ColorBrewer.palette("Greens",5)[5], ylabel="Hot-Start Test Statistic", legend=false, guidefontsize=20, tickfontsize=15, formatter=:plain, legendfontsize=10, yticks=[0,2,4,6,8,10,12,14])
        savefig("plots/log_reg_burnin" * c * ".png")
    end
end

function main_poisson_regression()
    for c in ["100", "500", "1000"]
        dat = JLD2.load("results/poi_reg_burnin" * c * ".jld")
        gnorms = dat["gnorms"]
        mmms = dat["mmms"]
        iter_checks = dat["iter_checks"]
        pass = dat["test_pass2_ratio"][1]

        p = plot(xlabel = "Iteration", margin=(5.0, :mm))
        pt = twinx(p)
        plot!(p, gnorms, yscale=:log10, color=ColorBrewer.palette("Blues",5)[3], ylabel="Gradient Norm", legend=false, guidefontsize=20, tickfontsize=15, formatter=:plain, legendfontsize=10, yticks=10. .^[0:5:10;])
        hline!(pt, [0.5], yscale=:identity, color=ColorBrewer.palette("Oranges",5)[3], legend=false, linewidth=2)
        plot!(pt, iter_checks, mmms, yscale=:identity, color=ColorBrewer.palette("Greens",5)[5], ylabel="Hot-Start Test Statistic", legend=false, guidefontsize=20, tickfontsize=15, formatter=:plain, legendfontsize=10, yticks=[0,0.5,1,1.5,2,2.5,3])
        savefig("plots/poi_reg_burnin" * c * ".png")
    end
end

function bradley_terry_regression()
    for c in ["100", "500", "1000"]
        dat = JLD2.load("results/bradley_terry_burnin" * c * ".jld")
        gnorms = dat["gnorms"]
        mmms = dat["mmms"]
        iter_checks = dat["iter_checks"]
        pass = dat["test_pass2_ratio"][1]

        p = plot(xlabel = "Iteration", margin=(5.0, :mm))
        pt = twinx(p)
        plot!(p, gnorms, yscale=:log10, color=ColorBrewer.palette("Blues",5)[3], ylabel="Gradient Norm", legend=false, guidefontsize=20, tickfontsize=15, formatter=:plain, legendfontsize=10)
        hline!(pt, [0.5], yscale=:identity, color=ColorBrewer.palette("Oranges",5)[3], legend=false, linewidth=2)
        plot!(pt, iter_checks, mmms, yscale=:identity, color=ColorBrewer.palette("Greens",5)[5], ylabel="Hot-Start Test Statistic", legend=false, guidefontsize=20, tickfontsize=15, formatter=:plain, legendfontsize=10)
        savefig("plots/bradley_terry_burnin" * c * ".png")
    end
end

function sparse_regression_regression()
    for c in ["100", "500", "1000"]
        dat = JLD2.load("results/sparse_reg_burnin" * c * ".jld")
        gnorms = dat["gnorms"]
        mmms = dat["mmms"]
        iter_checks = dat["iter_checks"]
        pass = dat["test_pass2_ratio"][1]

        p = plot(xlabel = "Iteration", margin=(5.0, :mm))
        pt = twinx(p)
        plot!(p, gnorms, yscale=:log10, color=ColorBrewer.palette("Blues",5)[3], ylabel="Gradient Norm", legend=false, guidefontsize=20, tickfontsize=15, formatter=:plain, legendfontsize=10, yticks=10. .^[-3:1:5;])
        hline!(pt, [0.5], yscale=:identity, color=ColorBrewer.palette("Oranges",5)[3], legend=false, linewidth=2)
        plot!(pt, iter_checks, mmms, yscale=:identity, color=ColorBrewer.palette("Greens",5)[5], ylabel="Hot-Start Test Statistic", legend=false, guidefontsize=20, tickfontsize=15, formatter=:plain, legendfontsize=10, yticks=[0,0.1,0.2,0.3,0.4,0.5])
        savefig("plots/sparse_reg_burnin" * c * ".png")
    end
end