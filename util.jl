using LinearAlgebra
using StatsBase

function log_reg_stratified_sampling(model, M, rng)
    ind_seq = [1:model.N ;]
    positives = ind_seq[model.datamat[:,end] .== 1.]
    negatives = ind_seq[model.datamat[:,end] .== 0.]

    count_positives = size(positives, 1)

    # take 50% positive, 50% negative (if possible)
    n_pos = min(Int(ceil(M / 2.)), count_positives)
    n_neg = M - n_pos

    inds_pos = sort(sample(rng, positives, n_pos, replace = false))
    inds_neg = sort(sample(rng, negatives, n_neg, replace = false))

    inds = sort(vcat(inds_pos, inds_neg))

    return inds
end

function compute_metric(batch_mean, μp, dΣp)
    return mean((batch_mean - μp).^2 ./ dΣp)
end