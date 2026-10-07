#include <cuda_fp16.h>
#include <driver_types.h>

#pragma once

__global__ void relu_kernel(half *matd, half *resd, uint M, uint N) {
  uint idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < M * N) {
    resd[idx] = __hmax(0, matd[idx]);
  }
}

void run_relu(half *matd, half *resd, uint M, uint N, cudaStream_t stream) {

  dim3 block_size(1024);
  dim3 grid_size(((M * N) + block_size.x - 1) / block_size.x);

  relu_kernel<<<grid_size, block_size, 0, stream>>>(matd, resd, M, N);
}