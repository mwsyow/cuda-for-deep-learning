#include <iostream>

#pragma once

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t error = call;                                                  \
    if (error != cudaSuccess) {                                                \
      std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << ":"      \
                << cudaGetErrorString(error) << std::endl;                     \
    }                                                                          \
  } while (0);

#define CUBLAS_CHECK(call)                                                     \
  do {                                                                         \
    cublasStatus_t status = call;                                              \
    if (status != CUBLAS_STATUS_SUCCESS) {                                     \
      std::cerr << "CUBLAS error at " << __FILE__ << ":" << __LINE__ << ": "   \
                << cublasGetStatusString(status) << std::endl;                 \
    }                                                                          \
  } while (0);

#define NCCL_CHECK(call)                                                       \
  do {                                                                         \
    ncclResult_t result = call;                                                \
    if (result != ncclSuccess) {                                               \
      std::cerr << "NCCL error at " << __FILE__ << ":" << __LINE__ << ": "     \
                << ncclGetErrorString(result) << std::endl;                    \
    }                                                                          \
  } while (0);