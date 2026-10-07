#include <cublas_v2.h>
#include <iostream>
#include <mpi.h>

#pragma once

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t error = call;                                                  \
    if (error != cudaSuccess) {                                                \
      std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << ":"      \
                << cudaGetErrorString(error) << std::endl;                     \
      MPI_Abort(MPI_COMM_WORLD, 1);                                            \
    }                                                                          \
  } while (0);

#define CUBLAS_CHECK(call)                                                     \
  do {                                                                         \
    cublasStatus_t status = call;                                              \
    if (status != CUBLAS_STATUS_SUCCESS) {                                     \
      std::cerr << "CUBLAS error at " << __FILE__ << ":" << __LINE__ << ": "   \
                << cublasGetStatusString(status) << std::endl;                 \
      MPI_Abort(MPI_COMM_WORLD, 1);                                            \
    }                                                                          \
  } while (0)
;

#define NCCL_CHECK(call)                                                       \
  do {                                                                         \
    ncclResult_t result = call;                                                \
    if (result != ncclSuccess) {                                               \
      std::cerr << "NCCL error at " << __FILE__ << ":" << __LINE__ << ": "     \
                << ncclGetErrorString(result) << std::endl;                    \
      MPI_Abort(MPI_COMM_WORLD, 1);                                            \
    }                                                                          \
  } while (0);

struct TimeStats {
  float X_w1;
  float X_relu;
  float X_w2;
  float fflayer;
  float all_reduce_fflayer;
  float total;
};

// Aggregate throughput in GFLOP/s, derived from the final (rank-maximum,
// run-averaged) TimeStats. The all-reduce performs no counted FLOPs.
struct GflopsStats {
  float X_w1;
  float X_relu;
  float X_w2;
  float fflayer;
  float total;
};

struct DeviceContext {
  cudaStream_t stream;
  cublasHandle_t handle;
};