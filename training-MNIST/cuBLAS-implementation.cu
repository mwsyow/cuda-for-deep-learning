#include <cstdio>
#include <ctime>
#include <cublas_v2.h>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t error = call;                                                  \
    if (error != cudaSuccess) {                                                \
      fprintf(stderr, "CUDA error at %s:%d: %s (%d)\n", __FILE__, __LINE__,    \
              cudaGetErrorString(error), error);                               \
      cudaDeviceReset();                                                       \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

#define CUBLAS_CHECK(call)                                                     \
  do {                                                                         \
    cublasStatus_t status = call;                                              \
    if (status != CUBLAS_STATUS_SUCCESS) {                                     \
      fprintf(stderr, "cuBLAS error at %s:%d: %d\n", __FILE__, __LINE__,       \
              status);                                                         \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

double get_time_diff(struct timespec start, struct timespec end) {
  return (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
}

void init_matrix_host(float *A, int row, int col) {
  for (int r = 0; r < row; ++r) {
    for (int c = 0; c < col; ++c) {
      A[r * col + c] = (r * col + c) / (float)(r * col + col);
    }
  }
}

void gemm_cpu(const float *A, const float *B, float *C, int m_A, int n_B,
              int K_shared_dim) {

  for (int m{0}; m < m_A; ++m) {
    for (int n{0}; n < n_B; ++n) {
      float sum{0};
      for (int k{0}; k < K_shared_dim; ++k) {
        sum = sum + (A[(m * K_shared_dim) + k] * B[(k * n_B) + n]);
      }
      C[(m * n_B) + n] = sum;
    }
  }
}

__global__ void matmul_a_b_kernel(float *A, float *B, float *C, int m_A,
                                  int n_B, int k_shared) {
  // A: (m, k_shared) @ B: (k_shared, n) -> (m, n)
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;

  if (row < m_A && col < n_B) {
    float sum = 0.0f;
    for (int k = 0; k < k_shared; ++k) {
      sum += A[row * k_shared + k] * B[k * n_B + col];
    }

    C[row * n_B + col] = sum;
  }
}

int main() {

  int m = 5000;
  int n = 7000;
  int k = 3000;

  float *h_A = (float *)malloc(m * k * sizeof(float));
  float *h_B = (float *)malloc(k * n * sizeof(float));
  float *h_C = (float *)malloc(m * n * sizeof(float));
  init_matrix_host(h_A, m, k);
  init_matrix_host(h_B, k, n);
  float *d_A, *d_B, *d_C;
  CUDA_CHECK(cudaMalloc(&d_A, m * k * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_B, k * n * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_C, m * n * sizeof(float)));

  CUDA_CHECK(
      cudaMemcpy(d_A, h_A, m * k * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(
      cudaMemcpy(d_B, h_B, k * n * sizeof(float), cudaMemcpyHostToDevice));
  cudaEvent_t d_start, d_end;
  CUDA_CHECK(cudaEventCreate(&d_start));
  CUDA_CHECK(cudaEventCreate(&d_end));
  //   timespec h_start, h_end;
  float ms = 0.0;

  //   clock_gettime(CLOCK_MONOTONIC, &h_start);
  //   gemm_cpu(h_A, h_B, h_C, m, n, k);
  //   clock_gettime(CLOCK_MONOTONIC, &h_end);
  //   double h_time = get_time_diff(h_start, h_end);
  //   printf("CPU matmul time:     %6.3fs\n", h_time);

  dim3 threadsPerBlock(32, 32);
  dim3 gridDim((n + threadsPerBlock.x - 1) / threadsPerBlock.x,
               (m + threadsPerBlock.y - 1) / threadsPerBlock.y);

  float sum_ms = 0.0f;
  for (int i = 0; i < 100; ++i) {
    CUDA_CHECK(cudaEventRecord(d_start));
    matmul_a_b_kernel<<<gridDim, threadsPerBlock>>>(d_A, d_B, d_C, m, n, k);
    CUDA_CHECK(cudaEventRecord(d_end));
    CUDA_CHECK(cudaEventSynchronize(d_end));
    CUDA_CHECK(cudaEventElapsedTime(&ms, d_start, d_end));
    sum_ms += ms;
  }
  printf("naive CUDA matmul time:     %6.7fs\n", sum_ms / 1000 / 100);

  cublasHandle_t cublas_handle;
  float alpha = 1.0f;
  float beta = 0.0f;

  CUBLAS_CHECK(cublasCreate(&cublas_handle));
  CUBLAS_CHECK(cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k,
                           &alpha, d_B, n, d_A, k, &beta, d_C, n));
  CUDA_CHECK(cudaDeviceSynchronize());

  sum_ms = 0.0f;
  for (int i = 0; i < 100; ++i) {
    CUDA_CHECK(cudaEventRecord(d_start));
    CUBLAS_CHECK(cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k,
                             &alpha, d_B, n, d_A, k, &beta, d_C, n));
    CUDA_CHECK(cudaEventRecord(d_end));
    CUDA_CHECK(cudaEventSynchronize(d_end));
    CUDA_CHECK(cudaEventElapsedTime(&ms, d_start, d_end));
    sum_ms += ms;
  }
  printf("cuBLAS matmul time:     %6.7fs\n", sum_ms / 1000 / 100);

  CUDA_CHECK(cudaEventDestroy(d_start));
  CUDA_CHECK(cudaEventDestroy(d_end));
  CUBLAS_CHECK(cublasDestroy(cublas_handle));
  free(h_A);
  free(h_B);
  free(h_C);
  CUDA_CHECK(cudaFree(d_A));
  CUDA_CHECK(cudaFree(d_B));
  CUDA_CHECK(cudaFree(d_C));

  return 0;
}