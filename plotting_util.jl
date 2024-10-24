using StatsBase, LinearAlgebra, Interpolations
using LinearAlgebra, Random, Distributions

function get_medians(dat)
    n = size(dat,2)
    median_dat = vec(median(dat, dims=1))

    for i in 1:n
        dat_remove_inf = (dat[:,i])[iszero.(isinf.(dat[:,i]))]
        dat_remove_nan = (dat_remove_inf)[iszero.(isnan.(dat_remove_inf))]
        median_dat[i] = median(dat_remove_nan)
    end

    return median_dat
end