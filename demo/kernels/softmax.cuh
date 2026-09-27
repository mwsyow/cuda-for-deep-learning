#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <driver_types.h>
#include <vector_types.h>

__global__ void naive_softmax_kernel(half *matd, half *resd, uint M, uint N) {
  // Calculate row index for this thread
  uint row = blockDim.x * blockIdx.x + threadIdx.x;

  // Bounds check
  if (row < M) {
    // Step 1: Find maximum value in the row (for numerical stability)
    half m = -1 * INFINITY;

    // Step 2: Compute sum of exponentials (shifted by max for stability)
    half L = 0.0f;

    // First pass: find maximum
    for (uint col = 0; col < N; col++) {
      uint i = row * N + col;
      m = __hmax(m, matd[i]);
    }

    // Second pass: compute sum of exponentials
    for (uint col = 0; col < N; col++) {
      uint i = row * N + col;
      L += hexp(matd[i] - m);
    }

    // Third pass: compute softmax probabilities
    for (uint col = 0; col < N; col++) {
      uint i = row * N + col;
      resd[i] = hexp(matd[i] - m) / L;
    }
  }
}

void run_softmax(half *matd, half *resd, uint M, uint N, cudaStream_t stream) {
  // Configure kernel launch parameters
  dim3 block_size(1024); // Maximum threads per block
  dim3 grid_size((M + block_size.x - 1) / block_size.x); // One block per row

  // Launch naive softmax kernel
  naive_softmax_kernel<<<grid_size, block_size, 0, stream>>>(matd, resd, M, N);
}