// Softmax Naive Implementation
// Based in part on Maharshi Pandya's CUDA optimization blog (Apache-2.0
// license) https://github.com/Maharshi-Pandya/cuda-mode-resource-stream

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <vector_types.h>

// Ceiling division macro
#ifndef CEIL_DIV
#define CEIL_DIV(x, y) (((x) + (y) - 1) / (y))
#endif

/**
 * Naive softmax kernel implementation
 * This kernel implements a basic softmax operation on a matrix of size (M, N)
 * The softmax operation is performed on the last dimension (columns) of the
 * matrix
 *
 * Performance characteristics:
 * - One thread processes one entire row
 * - Only parallelizes over rows, not exploiting full GPU parallelism
 * - Sequential computation within each thread for max, sum, and normalization
 * - This is the baseline implementation for comparison with optimized versions
 *
 * @param matd Input matrix (M×N, device memory)
 * @param resd Output matrix (M×N, device memory)
 * @param M Number of rows (batch size)
 * @param N Number of columns (feature dimension)
 */
__global__ void naive_softmax_kernel(float *__restrict__ matd,
                                     float *__restrict__ resd, int M, int N) {
  // Calculate row index for this thread
  int row = blockDim.x * blockIdx.x + threadIdx.x;

  // Bounds check
  if (row < M) {
    // Step 1: Find maximum value in the row (for numerical stability)
    float m = -1 * INFINITY;

    // Step 2: Compute sum of exponentials (shifted by max for stability)
    float L = 0.0f;

    // First pass: find maximum
    for (int col = 0; col < N; col++) {
      int i = row * N + col;
      m = max(m, matd[i]);
    }

    // Second pass: compute sum of exponentials
    for (int col = 0; col < N; col++) {
      int i = row * N + col;
      L += expf(matd[i] - m);
    }

    // Third pass: compute softmax probabilities
    for (int col = 0; col < N; col++) {
      int i = row * N + col;
      resd[i] = expf(matd[i] - m) / L;
    }
  }
}

/**
 * Launcher function for naive softmax kernel
 * Configures and launches the baseline softmax implementation
 *
 * @param matd Input matrix (M×N, device memory)
 * @param resd Output matrix (M×N, device memory)
 * @param M Number of rows (batch size)
 * @param N Number of columns (feature dimension)
 */
void run_naive_softmax(float *__restrict__ matd, float *__restrict__ resd,
                       int M, int N) {
  // Configure kernel launch parameters
  dim3 block_size(1024);                     // Maximum threads per block
  dim3 grid_size(CEIL_DIV(M, block_size.x)); // One block per row

  // Launch naive softmax kernel
  naive_softmax_kernel<<<grid_size, block_size>>>(matd, resd, M, N);
}

__global__ void online_softmax_kernel(float *__restrict__ matd,
                                      float *__restrict__ resd, int M, int N) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;

  if (row >= M)
    return;

  float running_max = -INFINITY;
  float running_norm_sum = 0.0f;
  for (int col = 0; col < N; ++col) {
    float cur_val = matd[row * N + col];

    if (cur_val > running_max) {
      // Rescale running norm sum relative to new max without restarting
      // computation compute e^(old_max - new_max)
      running_norm_sum *= expf(running_max - cur_val);

      // Set new max
      running_max = cur_val;
    }

    // Add exponential of current value (shifted by current max)
    running_norm_sum += expf(cur_val - running_max);
  }

  for (int col = 0; col < N; ++col) {
    resd[row * N + col] =
        expf(matd[row * N + col] - running_max) / running_norm_sum;
  }
}

void run_online_softmax(float *__restrict__ matd, float *__restrict__ resd,
                        int M, int N) {
  // Configure kernel launch parameters
  dim3 block_size(1024); // Maximum threads per block
  dim3 grid_size(M);     // One block per row

  // Launch naive softmax kernel
  online_softmax_kernel<<<grid_size, block_size>>>(matd, resd, M, N);
}

__global__ void shared_softmax_kernel(float *__restrict__ matd,
                                      float *__restrict__ resd, int M, int N) {
  // 1024 because we know that max threads per thread block is 1024
  __shared__ float smem[1024];

  // BLOCK PER ROW APPROACH
  int bid = blockIdx.x;
  int tid = threadIdx.x;

  // Phase 1: Do online softmax on tid with blockDim.x stride
  float local_max = -INFINITY;
  float local_norm = 0.0f;
  for (int col = tid; col < N; col += blockDim.x) {
    float cur = matd[bid * N + col];
    if (cur > local_max) {
      local_norm *= expf(local_max - cur);
      local_max = cur;
    }
    local_norm += expf(cur - local_max);
  }

  // Phase 2: Each thread saves its local_max to smem.
  smem[tid] = local_max;
  __syncthreads();

  // Phase 3: max reduction on shared memory
  // NOTE: By the end of this process tread 0 would have the final row max.
  for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
    // makesure minimum threads to do the work, since every step we reduce the
    // number of computing threads by half.
    if (tid < stride) {
      smem[tid] = fmax(smem[tid], smem[tid + stride]);
    }
    __syncthreads();
  }

  // Phase 5: Each thread loads final row max.
  float row_max = smem[0];
  __syncthreads();

  // Phase 6: Each thread calculates its scaled norm with final row max from
  local_norm *= expf(local_max - row_max);
  smem[tid] = local_norm;
  __syncthreads();

  // Phase 7: Sum reduction on shared memory
  // NOTE: By the end of this process tread 0 would have the final row norm.
  // >>= means right-shift assign operation, essentially equal to /=2.
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    // makesure minimum threads to do the work, since every step we reduce the
    // number of computing threads by half.
    if (tid < stride) {
      smem[tid] += smem[tid + stride];
    }
    __syncthreads();
  }

  // Phase 9: Each thread loads final row norm.
  float row_norm = smem[0];
  __syncthreads();

  // Phase 10: Each calculates the end per entry softmax of tid with stride
  // blockDim.x.
  for (int col = tid; col < N; col += blockDim.x) {
    resd[bid * N + col] = expf(matd[bid * N + col] - row_max) / row_norm;
  }
}

void run_shared_softmax(float *__restrict__ matd, float *__restrict__ resd,
                        int M, int N) {
  // Configure kernel launch parameters
  dim3 block_size(1024); // Maximum threads per block
  dim3 grid_size(M);     // One block per row

  int shared_mem_size = block_size.x * sizeof(float);
  // Launch naive softmax kernel
  shared_softmax_kernel<<<grid_size, block_size, shared_mem_size>>>(matd, resd,
                                                                    M, N);
}

__device__ __forceinline__ float warpReduceSum(float partial_sum) {
  for (int offset = warpSize / 2; offset > 0; offset /= 2) {
    partial_sum += __shfl_down_sync(0xffffffff, partial_sum, offset);
  }
  return partial_sum;
}
__device__ __forceinline__ float warpReduceMax(float local_max) {
  for (int offset = warpSize / 2; offset > 0; offset /= 2) {
    local_max =
        fmaxf(local_max, __shfl_down_sync(0xffffffff, local_max, offset));
  }
  return local_max;
};

__global__ void shfl_softmax_kernel(float *__restrict__ matd,
                                    float *__restrict__ resd, int M, int N) {
  // Only contains blockDim.x / warpSize elements which is the number of warps.
  extern __shared__ float smem[];

  int bid = blockIdx.x;
  int tid = threadIdx.x;

  float local_max = -INFINITY;
  float local_norm = 0.0f;
  for (int col = tid; col < N; col += blockDim.x) {
    float cur_val = matd[bid * N + col];
    if (cur_val > local_max) {
      local_norm *= expf(local_max - cur_val);
      local_max = cur_val;
    }
    local_norm += expf(cur_val - local_max);
  }

  // The first thread of each warp has the warp-level max.
  float reduced_max = warpReduceMax(local_max);

  if (tid % warpSize == 0) {
    smem[tid / warpSize] = reduced_max;
  }
  __syncthreads();

  if (blockDim.x > warpSize) {
    float warp_max = -INFINITY;
    if (tid < CEIL_DIV(blockDim.x, warpSize)) {
      warp_max = smem[tid];
    }
    __syncthreads();

    warp_max = warpReduceMax(warp_max);

    if (tid == 0) {
      smem[0] = warp_max;
    }
    __syncthreads();
  }

  float row_max = smem[0];
  __syncthreads();

  local_norm *= expf(local_max - row_max);
  local_norm = warpReduceSum(local_norm);

  if (tid % warpSize == 0) {
    smem[tid / warpSize] = local_norm;
  }
  __syncthreads();

  if (blockDim.x > warpSize) {
    float warp_norm = 0.0f;
    if (tid < CEIL_DIV(blockDim.x, warpSize)) {
      warp_norm = smem[tid];
    }
    __syncthreads();

    warp_norm = warpReduceSum(warp_norm);

    if (tid == 0) {
      smem[0] = warp_norm;
    }
    __syncthreads();
  }

  float row_norm = smem[0];
  __syncthreads();

  for (int col = tid; col < N; col += blockDim.x) {
    resd[bid * N + col] = expf(matd[bid * N + col] - row_max) / row_norm;
  }
}

void run_shfl_softmax(float *__restrict__ matd, float *__restrict__ resd, int M,
                      int N) {
  // Configure kernel launch parameters
  dim3 block_size(1024); // Maximum threads per block
  dim3 grid_size(M);     // One block per row

  int warp_size = 32;
  int shared_mem_size = CEIL_DIV(block_size.x, warp_size) * sizeof(float);
  // Launch naive softmax kernel
  shfl_softmax_kernel<<<grid_size, block_size, shared_mem_size>>>(matd, resd, M,
                                                                  N);
}

__global__ void vectorized_softmax_kernel(float *__restrict__ matd,
                                          float *__restrict__ resd, int M,
                                          int N) {
  // Only contains blockDim.x / warpSize elements which is the number of warps.
  extern __shared__ float smem[];

  int bid = blockIdx.x;
  int tid = threadIdx.x;

  float4 *mat = reinterpret_cast<float4 *>(matd + bid * N);
  float4 *res = reinterpret_cast<float4 *>(resd + bid * N);

  float local_max = -INFINITY;
  float local_norm = 0.0f;
  for (int col = tid; col < N / 4; col += blockDim.x) {
    float4 cur_val = mat[col];
    float cur_max = -INFINITY;

    cur_max = fmaxf(cur_max, cur_val.x);
    cur_max = fmaxf(cur_max, cur_val.y);
    cur_max = fmaxf(cur_max, cur_val.z);
    cur_max = fmaxf(cur_max, cur_val.w);
    if (cur_max > local_max) {
      local_norm *= expf(local_max - cur_max);
      local_max = cur_max;
    }
    local_norm += expf(cur_val.x - local_max);
    local_norm += expf(cur_val.y - local_max);
    local_norm += expf(cur_val.z - local_max);
    local_norm += expf(cur_val.w - local_max);
  }

  // The first thread of each warp has the warp-level max.
  float reduced_max = warpReduceMax(local_max);

  if (tid % warpSize == 0) {
    smem[tid / warpSize] = reduced_max;
  }
  __syncthreads();

  if (blockDim.x > warpSize) {
    float warp_max = -INFINITY;
    if (tid < CEIL_DIV(blockDim.x, warpSize)) {
      warp_max = smem[tid];
    }
    __syncthreads();

    warp_max = warpReduceMax(warp_max);

    if (tid == 0) {
      smem[0] = warp_max;
    }
    __syncthreads();
  }

  float row_max = smem[0];
  __syncthreads();

  local_norm *= expf(local_max - row_max);
  local_norm = warpReduceSum(local_norm);

  if (tid % warpSize == 0) {
    smem[tid / warpSize] = local_norm;
  }
  __syncthreads();

  if (blockDim.x > warpSize) {
    float warp_norm = 0.0f;
    if (tid < CEIL_DIV(blockDim.x, warpSize)) {
      warp_norm = smem[tid];
    }
    __syncthreads();

    warp_norm = warpReduceSum(warp_norm);

    if (tid == 0) {
      smem[0] = warp_norm;
    }
    __syncthreads();
  }

  float row_norm = smem[0];
  __syncthreads();

  for (int col = tid; col < N / 4; col += blockDim.x) {
    float4 cur_val = mat[col];
    cur_val.x = expf(cur_val.x - row_max) / row_norm;
    cur_val.y = expf(cur_val.y - row_max) / row_norm;
    cur_val.z = expf(cur_val.z - row_max) / row_norm;
    cur_val.w = expf(cur_val.w - row_max) / row_norm;

    res[col] = cur_val;
  }

  //----------------------------------------
  // TODO: need to check whether the tail mechanism from the book implementation
  // actually work when N mod 4 != 0
  //----------------------------------------
}

void run_vectorized_softmax(float *__restrict__ matd, float *__restrict__ resd,
                            int M, int N) {
  // Configure kernel launch parameters
  dim3 block_size(1024); // Maximum threads per block
  dim3 grid_size(M);     // One block per row

  int warp_size = 32;
  int shared_mem_size = CEIL_DIV(block_size.x, warp_size) * sizeof(float);
  // Launch naive softmax kernel
  vectorized_softmax_kernel<<<grid_size, block_size, shared_mem_size>>>(
      matd, resd, M, N);
}

int main(int argc, char **argv) {
  if (argc != 2) {
    std::fprintf(stderr, "Usage: %s naive|online|shared|shfl|vectorized\n",
                 argv[0]);
    return 1;
  }

  constexpr int M = 16384;
  constexpr int N = 8192;

  float *matrix;
  float *result;

  cudaMalloc(&matrix, M * N * sizeof(float));
  cudaMalloc(&result, M * N * sizeof(float));

  cudaMemset(matrix, 0, M * N * sizeof(float));

  if (strcmp(argv[1], "naive") == 0) {
    run_naive_softmax(matrix, result, M, N);
  } else if (strcmp(argv[1], "online") == 0) {
    run_online_softmax(matrix, result, M, N);
  } else if (strcmp(argv[1], "shared") == 0) {
    run_shared_softmax(matrix, result, M, N);
  } else if (strcmp(argv[1], "shfl") == 0) {
    run_shfl_softmax(matrix, result, M, N);
  } else if (strcmp(argv[1], "vectorized") == 0) {
    run_vectorized_softmax(matrix, result, M, N);
  } else {
    std::fprintf(stderr, "Unknown kernel: %s\n", argv[1]);
    return 1;
  }

  cudaDeviceSynchronize();

  cudaFree(matrix);
  cudaFree(result);
}