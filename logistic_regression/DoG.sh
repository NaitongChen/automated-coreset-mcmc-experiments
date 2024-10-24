for i in $(seq 1 30);
do
    julia run_coresetMCMC.jl i DoG
done