#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <sys/types.h>
#include <vector>

// Ceiling division macro
#define CEIL_DIV(x, y) (((x) + (y) - 1) / (y))
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

typedef __half fp16;

__global__ void naive_kernel(int M, int N, int K, fp16 *A, fp16 *B, fp16 *C) {
  // Calculate output element position (x, y) for this thread
  const uint x = blockIdx.x * blockDim.x + threadIdx.x;
  const uint y = blockIdx.y * blockDim.y + threadIdx.y;

  // Bounds check to prevent out-of-bounds access
  if (x < M && y < N) {
    // Initialize accumulator for dot product
    fp16 tmp = __float2half(0.0f);

    // Compute dot product: sum over k of A[x][k] * B[k][y]
    for (int i = 0; i < K; ++i) {
      // Accumulate: tmp += A[x][i] * B[i][y]
      // Note: A accessed row-wise (coalesced), B accessed column-wise (not
      // coalesced)
      tmp = __hadd(tmp, __hmul(A[x * K + i], B[i * N + y]));
    }
    // Write result to output matrix
    C[x * N + y] = tmp;
  }
}

void run_naive(int M, int N, int K, fp16 *A, fp16 *B, fp16 *C) {
  // 2D thread block: 32×32 = 1024 threads per block
  dim3 blockDim(32, 32);
  // Grid dimensions: ceil(M/32) × ceil(N/32) blocks
  dim3 gridDim((M + 31) / 32, (N + 31) / 32);
  naive_kernel<<<gridDim, blockDim>>>(M, N, K, A, B, C);
}

template <const uint BLOCKSIZE>
__global__ void gmem_coalesce_kernel(int M, int N, int K, fp16 *A, fp16 *B,
                                     fp16 *C) {
  // Map thread index to output element position
  // Ensures consecutive threads access consecutive columns for coalescing
  const int cRow = blockIdx.x * BLOCKSIZE + (threadIdx.x / BLOCKSIZE);
  const int cCol = blockIdx.y * BLOCKSIZE + (threadIdx.x % BLOCKSIZE);

  // Bounds check
  if (cRow < M && cCol < N) {
    // Initialize accumulator
    fp16 tmp = __float2half(0.0f);

    // Compute dot product - same as naive, but with better memory access
    // pattern
    for (int i = 0; i < K; ++i) {
      // Accumulate: tmp += A[cRow][i] * B[i][cCol]
      tmp = __hadd(tmp, __hmul(A[cRow * K + i], B[i * N + cCol]));
    }
    // Write result
    C[cRow * N + cCol] = tmp;
  }
}

void run_gmem_coalesce(int M, int N, int K, fp16 *A, fp16 *B, fp16 *C) {
  const uint BLOCKSIZE = 32;
  // 1D thread block: BLOCKSIZE×BLOCKSIZE threads
  dim3 blockDim(BLOCKSIZE * BLOCKSIZE);
  // Grid dimensions: ceil(M/BLOCKSIZE) × ceil(N/BLOCKSIZE) blocks
  dim3 gridDim((M + BLOCKSIZE - 1) / BLOCKSIZE,
               (N + BLOCKSIZE - 1) / BLOCKSIZE);
  gmem_coalesce_kernel<BLOCKSIZE><<<gridDim, blockDim>>>(M, N, K, A, B, C);
}

template <const int BLOCKSIZE>
__global__ void smem_blocking_kernel(int M, int N, int K, fp16 *A, fp16 *B,
                                     fp16 *C) {
  // THREADBLOCK PER OUTPUT TILE IMPLEMENTATION.
  // Each thread computes each dot product which results in each element of the
  // output tile.
  // A tile is defined as tA: (BM, BK) and tB: (BK, BN) -> tC: (BM, BN).
  // Assume that BM = BN = BK = BLOCKSIZE.
  __shared__ fp16 tA[BLOCKSIZE * BLOCKSIZE];
  __shared__ fp16 tB[BLOCKSIZE * BLOCKSIZE];

  int blockRow = blockIdx.x;
  int blockCol = blockIdx.y;

  int threadRow = threadIdx.x / BLOCKSIZE;
  int threadCol = threadIdx.x % BLOCKSIZE;

  // Step 1: set base pointers of A, B and C
  // A: (M, K), here BLOCKSIZE means bm.
  A += blockRow * BLOCKSIZE * K;
  // B: (K, N), here BLOCKSIZE means BN.
  B += blockCol * BLOCKSIZE;
  // C: (M, N), here first BLOCKSIZE means bm and second means BN.
  C += blockRow * BLOCKSIZE * N + blockCol * BLOCKSIZE;

  // Main iteration that shifts to the next tile.
  // BLOCKSIZE means bk, so shifting every bk elements.
  fp16 res = __float2half(0.0f);
  for (int idx = 0; idx < K; idx += BLOCKSIZE) {
    // Load tile to shared memory.
    // Since BM = BN = BK = BLOCKSIZE hence we can use all threads in the thread
    // block to load the elements at once.
    tA[threadRow * BLOCKSIZE + threadCol] = A[threadRow * K + threadCol];
    tB[threadRow * BLOCKSIZE + threadCol] = B[threadRow * N + threadCol];
    __syncthreads();

    // compute partial dot products on BK axis.
    for (int bk = 0; bk < BLOCKSIZE; ++bk) {
      res = __hadd(res, __hmul(tA[threadRow * BLOCKSIZE + bk],
                               tB[bk * BLOCKSIZE + threadCol]));
    }
    __syncthreads();

    // Shifts to the next tile.
    A += BLOCKSIZE;
    B += BLOCKSIZE * N;
  }

  C[threadRow * N + threadCol] = res;
}

void run_smem_blocking(int M, int N, int K, fp16 *A, fp16 *B, fp16 *C) {
  const uint BLOCKSIZE = 32;
  // 1D thread block: BLOCKSIZE×BLOCKSIZE threads
  dim3 blockDim(BLOCKSIZE * BLOCKSIZE);
  // Grid dimensions: one block per output tile
  dim3 gridDim(CEIL_DIV(M, BLOCKSIZE), CEIL_DIV(N, BLOCKSIZE));
  smem_blocking_kernel<BLOCKSIZE><<<gridDim, blockDim>>>(M, N, K, A, B, C);
}

template <const int BM, const int BN, const int BK, const int TM>
__global__ void blocktiling_1d_kernel(int M, int N, int K, fp16 *A, fp16 *B,
                                      fp16 *C) {
  // THREADBLOCK PER TILE, but now with bigger thread and a thread responsible
  // for multiple rows on the output tile.

  __shared__ fp16 tA[BM * BK];
  __shared__ fp16 tB[BK * BN];

  const uint blockRow = blockIdx.x;
  const uint blockCol = blockIdx.y;

  // These indices are used to write to output tile.
  int threadRow = threadIdx.x / BN;
  int threadCol = threadIdx.x % BN;

  // Shift base pointers
  A += blockRow * BM * K;
  B += blockCol * BN;
  C += blockRow * BM * N + blockCol * BN;

  // Initialize result array per thread on register
  fp16 threadResults[TM];
  for (int idx = 0; idx < TM; ++idx) {
    threadResults[idx] = __float2half(0.0f);
  }

  // These indices are used to load from global memory.
  int tARow = threadIdx.x / BK;
  int tACol = threadIdx.x % BK;
  int tBRow = threadIdx.x / BN;
  int tBCol = threadIdx.x % BN;
  for (int idx = 0; idx < K; idx += BK) {
    // Load from global memory
    tA[tARow * BK + tACol] = A[tARow * K + tACol];
    tB[tBRow * BN + tBCol] = B[tBRow * N + tBCol];
    __syncthreads();

    for (uint bk = 0; bk < BK; ++bk) {
      fp16 b = tB[bk * BN + threadCol];
      for (uint tm = 0; tm < TM; ++tm) {
        threadResults[tm] = __hadd(
            threadResults[tm], __hmul(tA[(threadRow * TM + tm) * BK + bk], b));
      }
    }
    __syncthreads();

    A += BK;
    B += BK * N;
  }

  for (uint tm = 0; tm < TM; ++tm) {
    C[(threadRow * TM + tm) * N + threadCol] = threadResults[tm];
  }
}

void run_blocktiling_1d(int M, int N, int K, fp16 *A, fp16 *B, fp16 *C,
                        int *DB = nullptr) {
  const uint BK = 8;  // K-tile size
  const uint TM = 8;  // Register blocking: threads compute 8 output elements
  const uint BM = 64; // M-tile size (larger than previous kernels)
  const uint BN = 64; // N-tile size (larger than previous kernels)

  // Grid dimensions: one block per output tile
  dim3 gridDim(CEIL_DIV(M, BM), CEIL_DIV(N, BN));
  // Block size: (BM * BN) / TM threads per block
  dim3 blockDim((BM * BN) / TM);
  blocktiling_1d_kernel<BM, BN, BK, TM>
      <<<gridDim, blockDim>>>(M, N, K, A, B, C);
}

template <const uint BM, const uint BN, const uint BK, const uint TM,
          const uint TN>
//__launch_bounds__ used to bound the (maximum threads per block, minimum num
// blocks) during compile time to control things such as number of registers per
// threads.
__global__ void __launch_bounds__((BM * BN) / (TM * TN), 1)
    blocktiling_2d_kernel(int M, int N, int K, fp16 *A, fp16 *B, fp16 *C) {
  // THREADBLOCK PER TILE, only (BM/TM)*(BN/TN) threads, where (BM/TM)
  // represents thread row and (BN/TN) represents thread col in a coordinate
  // system.
  int blockRow = blockIdx.x;
  int blockCol = blockIdx.y;

  // Shifts the base pointers for A, B and C
  A += blockRow * BM * K;
  B += blockCol * BN;
  C += blockRow * BM * N + blockCol * BN;

  // Three things are needed per thread:
  // 1. array of results: (TM * TN)
  // 2. array of cache from A: (TM)
  // 3. array of cache from B: (TN)
  // as private registers
  fp16 cacheA[TM];
  fp16 cacheB[TN];
  fp16 threadResults[TM * TN];
  for (int i = 0; i < TM * TN; ++i) {
    threadResults[i] = __float2half(0.0f);
  }

  // Declare smem for dot product calculation
  __shared__ fp16 tA[BM * BK];
  __shared__ fp16 tB[BN * BK];

  int numThreads = (BM * BN) / (TM * TN);

  // Global threads indices
  int threadRow = threadIdx.x / (BN / TN);
  int threadCol = threadIdx.x % (BN / TN);

  // Each iteration:
  // 1. load values from global memory to shared memory
  // 2. load values from shared memory to registers
  // 3. do dot product
  // 4. shift to the next tile until K is reached
  for (int tileIdx = 0; tileIdx < K; tileIdx += BK) {
    // Since with this setup number of threads are lower than tile dimension so
    // a thread needs to load multiple values.
    for (int stride = threadIdx.x; stride < BM * BK; stride += numThreads) {
      int col = stride % BK;
      int row = stride / BK;
      tA[row * BK + col] = A[row * K + col];
    }
    for (int stride = threadIdx.x; stride < BN * BK; stride += numThreads) {
      int col = stride % BN;
      int row = stride / BN;
      tB[row * BN + col] = B[row * N + col];
    }
    __syncthreads();

    for (int bk = 0; bk < BK; ++bk) {

      for (int i = 0; i < TM; ++i) {
        cacheA[i] = tA[(threadRow * TM + i) * BK + bk];
      }
      for (int i = 0; i < TN; ++i) {
        cacheB[i] = tB[bk * BN + (threadCol * TN) + i];
      }

      for (int tm = 0; tm < TM; ++tm) {
        for (int tn = 0; tn < TN; ++tn) {
          threadResults[tm * TN + tn] = __hadd(threadResults[tm * TN + tn],
                                               __hmul(cacheA[tm], cacheB[tn]));
        }
      }
    }
    __syncthreads();

    A += BK;
    B += BK * N;
  }

  for (int tm = 0; tm < TM; ++tm) {
    for (int tn = 0; tn < TN; ++tn) {
      C[(threadRow * TM + tm) * N + (threadCol * TN) + tn] =
          threadResults[tm * TN + tn];
    }
  }
}

void run_blocktiling_2d(int M, int N, int K, fp16 *A, fp16 *B, fp16 *C) {
  const uint BM = 64; // M-tile size
  const uint BN = 64; // N-tile size
  const uint BK = 8;  // K-tile size
  const uint TM = 8;  // Register blocking: threads compute 8 rows
  const uint TN = 8;  // Register blocking: threads compute 8 columns

  // Grid dimensions: one block per output tile
  dim3 gridDim(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
  // Block size: (BM * BN) / (TM * TN) threads per block
  dim3 blockDim((BM * BN) / (TM * TN));
  blocktiling_2d_kernel<BM, BN, BK, TM, TN>
      <<<gridDim, blockDim>>>(M, N, K, A, B, C);
}

template <const uint BM, const uint BN, const uint BK, const uint TM,
          const uint TN>
__global__ void __launch_bounds__((BM * BN) / (TM * TN), 1)
    vectorized_kernel(int M, int N, int K, fp16 *A, fp16 *B, fp16 *C) {}
/*
KEY TAKEAWAY: Indexing is just learning how to assign which thread to access a
certain location in memory.
*/
int main(int argc, char **argv) {
  if (argc < 2 || argc > 3 ||
      (argc == 3 && std::strcmp(argv[2], "check") != 0)) {
    std::fprintf(stderr,
                 "Usage: %s "
                 "naive|coalesce|smem_blocking|blocktiling_1d|blocktiling_2d "
                 "[check]\n",
                 argv[0]);
    return 1;
  }

  const bool checkCorrectness = argc == 3;
  // Keep profiling dimensions large. Check mode uses a practical full CPU
  // reference with a non-square, tile-aligned shape.
  const int M = checkCorrectness ? 128 : 4096;
  const int N = checkCorrectness ? 192 : 4096;
  const int K = checkCorrectness ? 64 : 4096;

  fp16 *A;
  fp16 *B;
  fp16 *C;

  CUDA_CHECK(cudaMalloc(&A, M * K * sizeof(fp16)));
  CUDA_CHECK(cudaMalloc(&B, K * N * sizeof(fp16)));
  CUDA_CHECK(cudaMalloc(&C, M * N * sizeof(fp16)));

  std::vector<fp16> hostA;
  std::vector<fp16> hostB;
  if (checkCorrectness) {
    hostA.resize(M * K);
    hostB.resize(K * N);
    for (int row = 0; row < M; ++row) {
      for (int col = 0; col < K; ++col) {
        hostA[row * K + col] =
            __float2half(((row * 3 + col * 5) % 11 - 5) * 0.0625f);
      }
    }
    for (int row = 0; row < K; ++row) {
      for (int col = 0; col < N; ++col) {
        hostB[row * N + col] =
            __float2half(((row * 7 + col * 2) % 13 - 6) * 0.0625f);
      }
    }
    CUDA_CHECK(cudaMemcpy(A, hostA.data(), M * K * sizeof(fp16),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(B, hostB.data(), K * N * sizeof(fp16),
                          cudaMemcpyHostToDevice));
  } else {
    CUDA_CHECK(cudaMemset(A, 0, M * K * sizeof(fp16)));
    CUDA_CHECK(cudaMemset(B, 0, K * N * sizeof(fp16)));
  }
  CUDA_CHECK(cudaMemset(C, 0, M * N * sizeof(fp16)));

  if (strcmp(argv[1], "naive") == 0) {
    run_naive(M, N, K, A, B, C);
  } else if (strcmp(argv[1], "coalesce") == 0) {
    run_gmem_coalesce(M, N, K, A, B, C);
  } else if (strcmp(argv[1], "smem_blocking") == 0) {
    run_smem_blocking(M, N, K, A, B, C);
  } else if (strcmp(argv[1], "blocktiling_1d") == 0) {
    run_blocktiling_1d(M, N, K, A, B, C);
  } else if (strcmp(argv[1], "blocktiling_2d") == 0) {
    run_blocktiling_2d(M, N, K, A, B, C);
  } else {
    std::fprintf(stderr, "Unknown kernel: %s\n", argv[1]);
    return 1;
  }

  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  bool passed = true;
  if (checkCorrectness) {
    std::vector<fp16> hostC(M * N);
    CUDA_CHECK(cudaMemcpy(hostC.data(), C, M * N * sizeof(fp16),
                          cudaMemcpyDeviceToHost));

    constexpr float absoluteTolerance = 0.05f;
    constexpr float relativeTolerance = 0.05f;
    float maxAbsoluteError = 0.0f;
    int mismatchCount = 0;
    for (int row = 0; row < M; ++row) {
      for (int col = 0; col < N; ++col) {
        float expected = 0.0f;
        for (int k = 0; k < K; ++k) {
          expected += __half2float(hostA[row * K + k]) *
                      __half2float(hostB[k * N + col]);
        }
        const float actual = __half2float(hostC[row * N + col]);
        const float absoluteError = std::fabs(actual - expected);
        const float allowedError =
            absoluteTolerance + relativeTolerance * std::fabs(expected);
        maxAbsoluteError = std::fmax(maxAbsoluteError, absoluteError);
        if (!std::isfinite(actual) || absoluteError > allowedError) {
          if (mismatchCount < 10) {
            std::fprintf(stderr,
                         "Mismatch C[%d,%d]: GPU=%g CPU=%g abs_error=%g\n", row,
                         col, actual, expected, absoluteError);
          }
          ++mismatchCount;
        }
      }
    }
    passed = mismatchCount == 0;
    std::printf("CPU correctness check: %s (max abs error %g, %d mismatches)\n",
                passed ? "PASS" : "FAIL", maxAbsoluteError, mismatchCount);
  }

  CUDA_CHECK(cudaFree(A));
  CUDA_CHECK(cudaFree(B));
  CUDA_CHECK(cudaFree(C));
  return passed ? 0 : 2;
}
