sudo /usr/local/cuda/bin/ncu --set full \
    --section SpeedOfLight_HierarchicalSingleRooflineChart \
    --section MemoryWorkloadAnalysis \
    --section Occupancy \
    --kernel-name {KERNEL_NAME} \
    --launch-count 1 \
    --print-summary per-kernel \
    {PATH_TO_EXECUTABLE} {ARGS}

example: 
sudo /usr/local/cuda/bin/ncu --set full \
    --section SpeedOfLight_HierarchicalSingleRooflineChart \
    --section MemoryWorkloadAnalysis \
    --section Occupancy \
    --kernel-name naive_sgemv_kernel \
    --launch-count 1 \
    --print-summary per-kernel \
    ./gemv.o naive
