#include <cuda_fp16.h>

#pragma once

__global__ void transpose_kernel(half *matd, half *resd, uint M, uint N) {
  uint idx = blockIdx.x * blockDim.x + threadIdx.x;

  uint row = idx / N;
  uint col = idx % N;

  if (idx >= M * N) {
    return;
  }

  resd[col * M + row] = matd[row * N + col];
}

void run_transpose(half *matd, half *resd, uint M, uint N,
                   cudaStream_t stream) {
  dim3 block_size(1024);
  dim3 grid_size(((M * N) + block_size.x - 1) / block_size.x);

  transpose_kernel<<<grid_size, block_size, 0, stream>>>(matd, resd, M, N);
}