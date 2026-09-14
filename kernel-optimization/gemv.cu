#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <stdlib.h>

// Ceiling division macro
#ifndef CEIL_DIV
#define CEIL_DIV(x, y) (((x) + (y) - 1) / (y))
#endif

/**
 * CUDA error checking macro
 * Checks CUDA function calls for errors and exits on failure
 */
#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,         \
              cudaGetErrorString(err));                                        \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

/**
 * Naive SGEMM kernel for matrix-vector multiplication
 * Computes: result = matrix * vector
 *
 * Performance characteristics:
 * - Each thread calculates one element of the output vector
 * - The row index is calculated using block index and thread index
 * - Uses linearized indexing
 * - Memory accesses are not coalesced (poor performance)
 *
 * @param matd Input matrix (M×N, device memory)
 * @param vecd Input vector (N, device memory)
 * @param resd Output vector (M, device memory)
 * @param M Number of rows in matrix and size of output vector
 * @param N Number of columns in matrix and size of input vector
 */
__global__ void naive_sgemv_kernel(float *__restrict__ matd,
                                   float *__restrict__ vecd,
                                   float *__restrict__ resd, int M, int N) {
  // Calculate global thread index across all blocks
  int row = blockDim.x * blockIdx.x + threadIdx.x;

  // Bounds check to ensure we don't access out-of-range elements
  if (row < M) {
    float sum = 0.0f;
    // Compute dot product of matrix row and input vector
    for (int col = 0; col < N; col++) {
      sum += matd[row * N + col] * vecd[col];
    }
    // Store result in output vector
    resd[row] = sum;
  }
}

/**
 * CUDA wrapper function for naive SGEMM kernel
 * Launches the kernel with appropriate grid and block dimensions
 *
 * @param matd Input matrix (M×N, device memory)
 * @param vecd Input vector (N, device memory)
 * @param resd Output vector (M, device memory)
 * @param M Number of rows in matrix and size of output vector
 * @param N Number of columns in matrix and size of input vector
 */
void run_naive_sgemv(float *__restrict__ matd, float *__restrict__ vecd,
                     float *__restrict__ resd, int M, int N) {
  // Configure kernel launch parameters
  dim3 block_size(1024);                     // Maximum threads per block
  dim3 grid_size(CEIL_DIV(M, block_size.x)); // Number of blocks needed

  // Launch CUDA kernel
  naive_sgemv_kernel<<<grid_size, block_size>>>(matd, vecd, resd, M, N);
}

__device__ __forceinline__ float warpReduceSum(float partial_sum) {
  // Each thread in a warp access itself and thread in lanes that are `offset`
  // higher than itself. Sequentially reducing offset to halves
  for (int offset = 16; offset > 0; offset /= 2) {
    partial_sum += __shfl_down_sync(0xffffffff, partial_sum, offset);
  }
  return partial_sum;
}

__global__ void coalesced_warp_sgmev_kernel(float *__restrict__ matd,
                                            float *__restrict__ vecd,
                                            float *__restrict__ resd, int M,
                                            int N) {
  // Launch each thread block for each M output elements.
  int bid = blockIdx.x;
  int tid = threadIdx.x;

  // Each thread computes elements on a stride of blockDim.x and start at idx ==
  // tid
  float partial_sum = 0.0f;
  for (int col = tid; col < N; col += blockDim.x) {
    partial_sum += matd[bid * N + col] * vecd[col];
  }

  // get the result from thread 0 which basically computes the total sums
  float sum = warpReduceSum(partial_sum);
  if (tid == 0) {
    resd[bid] = sum;
  }
}

void run_coalesced_warp_sgmev(float *__restrict__ matd,
                              float *__restrict__ vecd,
                              float *__restrict__ resd, int M, int N) {
  // Configure kernel launch parameters
  dim3 block_size(32); // Maximum threads per block
  dim3 grid_size(M);   // Number of blocks needed

  // Launch CUDA kernel
  coalesced_warp_sgmev_kernel<<<grid_size, block_size>>>(matd, vecd, resd, M,
                                                         N);
}

__global__ void coalesced_warpblock_sgmev_kernel(float *__restrict__ matd,
                                                 float *__restrict__ vecd,
                                                 float *__restrict__ resd,
                                                 int M, int N) {
  // Dynamically allocated array, but should have length of number of warps.
  extern __shared__ float smem[];

  int bid = blockIdx.x;
  int tid = threadIdx.x;

  if (bid >= M) {
    return;
  }

  float partial_sum = 0.0f;
  for (int col = tid; col < N; col += blockDim.x) {
    partial_sum += matd[bid * N + col] + vecd[col];
  }

  //------------------------------------------------
  // BLOCK-LEVEL REDUCE SUM
  //------------------------------------------------
  // Phase 1: Calculate partial sum of each warp.
  float warp_sum = warpReduceSum(partial_sum);

  int warp_size = 32;
  // Phase 2: Each first thread of a warp writes to shared memory.
  // NOTE: Bank conflict happen within warp. This case only one thread per warp
  // access the shared memory.
  if (tid % warp_size == 0)
    smem[tid / warp_size] = warp_sum;
  __syncthreads();

  // Phase 3: First num_warp threads on the first warp reads from shared memory
  // and rest gets 0.
  // NOTE: There is no bank conflict since consecutive threads reads consecutive
  // address in shared memory
  float block_partial_sum = 0.0f;
  if (tid < CEIL_DIV(blockDim.x, warp_size)) {
    block_partial_sum = smem[tid];
  }
  __syncthreads();

  // Phase 4: Do final warp-level reduce sum. Makesure only the first warp
  // compute this part.
  if (tid / warp_size == 0) {
    float total_sum = warpReduceSum(block_partial_sum);

    if (tid == 0) {
      resd[bid] = total_sum;
    }
  }
  //------------------------------------------------
}

void run_coalesced_warpblock_sgmev(float *__restrict__ matd,
                                   float *__restrict__ vecd,
                                   float *__restrict__ resd, int M, int N) {
  // Configure kernel launch parameters
  dim3 block_size(256); // Maximum threads per block
  dim3 grid_size(M);    // Number of blocks needed

  int warp_size = 32;
  size_t shared_mem_size = CEIL_DIV(block_size.x, warp_size) * sizeof(float);

  // Launch CUDA kernel
  coalesced_warpblock_sgmev_kernel<<<grid_size, block_size, shared_mem_size>>>(
      matd, vecd, resd, M, N);
}

__global__ void vectorized_sgemv_kernel(float *__restrict__ matd,
                                        float *__restrict__ vecd,
                                        float *__restrict__ resd, int M,
                                        int N) {
  // Dynamically allocated array, but should have length of number of warps.
  extern __shared__ float smem[];

  int bid = blockIdx.x;
  int tid = threadIdx.x;

  if (bid >= M)
    return;

  //------------------------------------------------
  // VECTORIZATION
  //------------------------------------------------

  int n_float4s = N / 4;

  // These lines tells the compiler to reinterpret the pointer from float to
  // float4. However with the caveat that the data should aligned. In general it
  // should follow that:
  //   (matd + bid * N * 4) mod 16 == 0
  //   => N mod 4 == 0
  // vec follows analogously.
  float4 *mat_row = reinterpret_cast<float4 *>(matd + bid * N);
  float4 *vec = reinterpret_cast<float4 *>(vecd);

  // Now each thread loads 4 floats instead of 1. Effectively reduce number of
  // transactions.
  float partial_sum = 0.0f;
  for (int col = tid; col < n_float4s; col += blockDim.x) {
    float4 mat_val = mat_row[col];
    float4 vec_val = vec[col];

    partial_sum += (mat_val.x * vec_val.x) + (mat_val.y * vec_val.y) +
                   (mat_val.z * vec_val.z) + (mat_val.w * vec_val.w);
  }

  //------------------------------------------------
  // BLOCK-LEVEL REDUCE SUM
  //------------------------------------------------
  // Phase 1: Calculate partial sum of each warp.
  float warp_sum = warpReduceSum(partial_sum);

  int warp_size = 32;
  // Phase 2: Each first thread of a warp writes to shared memory.
  // NOTE: Bank conflict happen within warp. This case only one thread per warp
  // access the shared memory.
  if (tid % warp_size == 0)
    smem[tid / warp_size] = warp_sum;
  __syncthreads();

  // Phase 3: First num_warp threads on the first warp reads from shared memory
  // and rest gets 0.
  // NOTE: There is no bank conflict since consecutive threads reads consecutive
  // address in shared memory
  float block_partial_sum = 0.0f;
  if (tid < CEIL_DIV(blockDim.x, warp_size)) {
    block_partial_sum = smem[tid];
  }
  __syncthreads();

  // Phase 4: Do final warp-level reduce sum. Makesure only the first warp
  // compute this part.
  if (tid / warp_size == 0) {
    float total_sum = warpReduceSum(block_partial_sum);

    if (tid == 0) {
      resd[bid] = total_sum;
    }
  }
  //------------------------------------------------
}

void run_vectorized_sgemv(float *__restrict__ matd, float *__restrict__ vecd,
                          float *__restrict__ resd, int M, int N) {
  // Configure kernel launch parameters
  dim3 block_size(256); // Maximum threads per block
  dim3 grid_size(M);    // Number of blocks needed

  int warp_size = 32;
  size_t shared_mem_size = CEIL_DIV(block_size.x, warp_size) * sizeof(float);

  // Launch CUDA kernel
  vectorized_sgemv_kernel<<<grid_size, block_size, shared_mem_size>>>(
      matd, vecd, resd, M, N);
}

void run_cublas_sgemv(float *__restrict__ matd, float *__restrict__ vecd,
                      float *__restrict__ resd, int M, int N) {
  // Static handle for reuse across function calls
  // Avoids overhead of creating/destroying handle (~0.4ms per call)
  static cublasHandle_t handle = nullptr;

  // Create handle on first call only
  if (handle == nullptr) {
    cublasCreate(&handle);
  }

  // cuBLAS SGEMM parameters
  // CUBLAS_OP_T: Transpose matrix A (matd)
  // Alpha = 1.0, Beta = 0.0: result = 1.0 * A^T * x + 0.0 * result
  float alpha = 1.0f, beta = 0.0f;
  cublasSgemv(handle, CUBLAS_OP_T, N, M, &alpha, matd, N, vecd, 1, &beta, resd,
              1);
}

int main(int argc, char **argv) {
  if (argc != 2) {
    std::fprintf(stderr, "Usage: %s naive|warp|warpblock|vectorized|cublas\n",
                 argv[0]);
    return 1;
  }

  constexpr int M = 16384;
  constexpr int N = 8192;

  float *matrix;
  float *vector;
  float *result;

  cudaMalloc(&matrix, M * N * sizeof(float));
  cudaMalloc(&vector, N * sizeof(float));
  cudaMalloc(&result, M * sizeof(float));

  cudaMemset(matrix, 0, M * N * sizeof(float));
  cudaMemset(vector, 0, N * sizeof(float));

  if (strcmp(argv[1], "naive") == 0) {
    run_naive_sgemv(matrix, vector, result, M, N);
  } else if (strcmp(argv[1], "warp") == 0) {
    run_coalesced_warp_sgmev(matrix, vector, result, M, N);
  } else if (strcmp(argv[1], "warpblock") == 0) {
    run_coalesced_warpblock_sgmev(matrix, vector, result, M, N);
  } else if (strcmp(argv[1], "vectorized") == 0) {
    run_vectorized_sgemv(matrix, vector, result, M, N);
  } else if (strcmp(argv[1], "cublas") == 0) {
    run_cublas_sgemv(matrix, vector, result, M, N);
  } else {
    std::fprintf(stderr, "Unknown kernel: %s\n", argv[1]);
    return 1;
  }

  cudaDeviceSynchronize();

  cudaFree(matrix);
  cudaFree(vector);
  cudaFree(result);
}
