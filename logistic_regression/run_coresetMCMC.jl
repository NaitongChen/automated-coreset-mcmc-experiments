include("logistic_regression_coresetMCMC.jl")

# ARGS[1] = seed
# ARGS[2] = optimizer

lr = ["0.001", "0.01", "0.1", "1", "10", "100"]
lr_shrinking = ["0.001_0.5", "0.01_0.5", "0.1_0.5", "1_0.5", "10_0.5", "100_0.5"]
coreset_sizes = ["100", "500", "1000"]

coreset_size = coreset_sizes[parse(Int64, ARGS[1]) % length(coreset_sizes) + 1]

if ARGS[2] == "ADAM" # learningRate / shrinking / initMix
    for i in [1:length(lr);]
        # seed / iter / coresetSize / learningRate / optimizer / initMix
        main([ARGS[1], "200000", coreset_size, lr_shrinking[i], ARGS[2], "10000"])
        GC.gc()
        main([ARGS[1], "200000", coreset_size, lr_shrinking[i], ARGS[2], "0"])
        GC.gc()
        main(["1000", "10000", coreset_size, lr_shrinking[i], ARGS[2], "10000"])
        GC.gc()
    end
elseif ARGS[2] == "ProdigyADAM_ca"
    for i in [1:length(lr);]
        # seed / iter / coresetSize / learningRate / optimizer / initMix
        main([ARGS[1], "200000", coreset_size, lr[i], ARGS[2], "10000"])
        GC.gc()
        main([ARGS[1], "200000", coreset_size, lr[i], ARGS[2], "0"])
        GC.gc()
    end
else # learningRate / initMix
    for i in [1:length(lr);]
        # seed / iter / coresetSize / learningRate / optimizer / initMix
        main([ARGS[1], "200000", coreset_size, lr[i], ARGS[2], "10000"])
        GC.gc()
        main([ARGS[1], "200000", coreset_size, lr[i], ARGS[2], "0"])
        GC.gc()
    end
end