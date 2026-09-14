#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

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

__device__ __forceinline__ float warpReduceSum(float partial_sum) {
  for (int offset = warpSize / 2; offset > 0; offset /= 2) {
    partial_sum += __shfl_down_sync(0xffffffff, partial_sum, offset);
  }
  return partial_sum;
}

__global__ void
warp_layernorm_kernel(float *__restrict__ out, float *__restrict__ mean,
                      float *__restrict__ rstd, const float *__restrict__ inp,
                      const float *__restrict__ weight,
                      const float *__restrict__ bias, int N, int C) {
  // WARP PER ROW IMPLEMENTATION
  // Here each warp computes each row. But in a single kernel launch we can have
  // multiple warps. think of it like packing num_warps (per thread) rows
  // into a unit (r_k, r_k+1, ..., r_k+n-1) that a thread block responsible for.
  // Each warp i of that thread block responsible for its corresponding row
  // r_k+i.

  // Directly maps to i on r_k+i notation above
  int warp_id = threadIdx.x / warpSize;
  // Id of each thread relative to its warp.
  int warp_lane = threadIdx.x % warpSize;
  int num_warps = (blockDim.x + warpSize - 1) / warpSize;

  // This computes the global index of each warp.
  // Maps to indexing on N dimension.
  int idx = blockIdx.x * num_warps + warp_id;

  if (idx > N)
    return;

  // REMINDER: each block consists of num_warps * warpSize threads. Each warp
  // will compute each row.
  const float *x = inp + idx * C;

  // Calculate sum with stride of warpSize to cover all columns.
  float sum = 0.0f;
  for (int col = warp_lane; col < C; col += warpSize) {
    sum += x[col];
  }

  // After covering the whole column, reduce the sum such that thread 0 gets the
  // final sum.
  sum = warpReduceSum(sum);

  // This functions let one thread reads a register value from a specific lane
  // of the same warp, described by offset.
  // In this case all threads in a warp will read the value stored in lane 0,
  // which is just the first thread of every warp.
  float row_mean = __shfl_sync(0xffffffff, sum, 0) / C;

  // Only the first thread of each warp updates it respective row.
  if (warp_lane == 0 && mean != nullptr) {
    mean[idx] = row_mean;
  }

  // Calculate sum with stride of warpSize to cover all columns.
  sum = 0.0f;
  for (int col = warp_lane; col < C; col += warpSize) {
    float cur = x[col];
    sum += (cur - row_mean) * (cur - row_mean);
  }

  // After covering the whole column, reduce the sum such that thread 0 gets the
  // final sum.
  sum = warpReduceSum(sum);

  // rsqrtf is reciprocal so its computing 1/sqrt(...)
  float row_var = rsqrtf(__shfl_sync(0xffffffff, sum, 0) / C + 1e-5f);

  if (warp_lane == 0 && rstd != nullptr) {
    rstd[idx] = row_var;
  }

  float *y = out + idx * C;
  for (int col = warp_lane; col < C; col += warpSize) {
    y[col] = weight[col] * (x[col] - row_mean) * row_var + bias[col];
  }
}

void run_warp_layernorm(float *out, float *mean, float *rstd, const float *inp,
                        const float *weight, const float *bias, int N, int C) {
  int warp_size = 32;
  int warps_per_block = 4;
  dim3 block_size(warps_per_block * warp_size);
  dim3 grid_size((N + warps_per_block - 1) / warps_per_block);

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  CUDA_CHECK(cudaEventRecord(start));
  warp_layernorm_kernel<<<grid_size, block_size>>>(out, mean, rstd, inp, weight,
                                                   bias, N, C);
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));

  float ms;
  CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
}