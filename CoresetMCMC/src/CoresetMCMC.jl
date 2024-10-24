module CoresetMCMCSampler

using Plots
using Statistics
using StatsBase
using Distributions
using Random
using LinearAlgebra
using Parameters
using ProgressMeter
using Infiltrator
using ForwardDiff
using Zygote
using PDMats
using LogExpFunctions
using Distances
using HypothesisTests
using GLM
using DataFrames
using CSV
using ParameterSchedulers

abstract type AbstractModel end
abstract type AbstractState end
abstract type AbstractMetaState end
abstract type AbstractKernel end
abstract type AbstractAlgorithm end
abstract type AbstractOptimizer end
abstract type AbstractLogProbEstimator end
abstract type SizeBasedLogProbEstimator <: AbstractLogProbEstimator end
abstract type QualityBasedLogProbEstimator <: AbstractLogProbEstimator end

# logProbEstimators
include("estimators/ZeroLogProbEstimator.jl")
export ZeroLogProbEstimator
include("estimators/CoresetLogProbEstimator.jl")
export CoresetLogProbEstimator

# kernels
include("methods/kernels/Artificial.jl")
export Artificial
include("methods/kernels/SliceSamplerMD.jl")
export SliceSamplerMD
include("methods/kernels/GibbsSR.jl")
export GibbsSR

# meta-algorithms
include("methods/meta_algorithms/CoresetMCMC.jl")
export CoresetMCMC

# models
include("models/GaussianLocationModel.jl")
export GaussianLocationModel
include("models/LinearRegressionModel.jl")
export LinearRegressionModel
include("models/LogisticRegressionModel.jl")
export LogisticRegressionModel
include("models/PoissonRegressionModel.jl")
export PoissonRegressionModel
include("models/SparseRegressionModel.jl")
export SparseRegressionModel
include("models/BradleyTerryModel.jl")
export BradleyTerryModel
include("models/log_potentials.jl")

# optimizers
include("optimizers/dadaptSGD.jl")
export dadaptSGD
include("optimizers/ADAM.jl")
export ADAM
include("optimizers/DoG.jl")
export DoG
include("optimizers/DoWG.jl")
export DoWG
include("optimizers/prodigyADAM.jl")
export prodigyADAM
include("optimizers/ADAMDoGCoord.jl")
export ADAMDoGCoord

# sampler state tracker
include("States.jl")
export State
export MetaState

# sampler API
include("Sampler.jl")
export sample!

# utilities
include("utilities.jl")

end # module CoresetMCMC