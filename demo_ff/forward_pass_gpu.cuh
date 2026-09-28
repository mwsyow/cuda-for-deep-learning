#include "common.cuh"
#include "feed_forward.cuh"
#include "kernels/relu.cuh"
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <nccl.h>

#pragma once

template <typename Func>
void timed(const DeviceContext &context, const cudaEvent_t &start,
           const cudaEvent_t &stop, float &elapsedTime, const Func func) {
  CUDA_CHECK(cudaEventRecord(start, context.stream));
  func();
  CUDA_CHECK(cudaEventRecord(stop, context.stream));
  CUDA_CHECK(cudaEventSynchronize(stop));
  CUDA_CHECK(cudaEventElapsedTime(&elapsedTime, start, stop));
}

void forward_pass_gpu(const DeviceContext &context, const FFLayer<half> &layer,
                      const half *X, half *Res, const uint seq_len,
                      TimeStats &stats) {
  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  half alpha = 1.0f;
  half beta = 0.0f;

  timed(context, start, stop, stats.X_w1, [&]() {
    CUBLAS_CHECK(cublasHgemm(context.handle, CUBLAS_OP_N, CUBLAS_OP_N,
                             layer.hidden_dim, seq_len, layer.in_dim, &alpha,
                             layer.W1, layer.hidden_dim, X, layer.in_dim, &beta,
                             layer.X_w1, layer.hidden_dim));
  });

  timed(context, start, stop, stats.X_relu, [&]() {
    run_relu(layer.X_w1, layer.X_relu, seq_len, layer.hidden_dim,
             context.stream);
  });

  timed(context, start, stop, stats.X_w2, [&]() {
    CUBLAS_CHECK(cublasHgemm(context.handle, CUBLAS_OP_N, CUBLAS_OP_N,
                             layer.in_dim, seq_len, layer.hidden_dim, &alpha,
                             layer.W2, layer.in_dim, layer.X_relu,
                             layer.hidden_dim, &beta, Res, layer.in_dim));
  });
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
}

void run_ff_layer_single_gpu(const DeviceContext &context,
                             const FFLayer<half> &layer, const half *X,
                             half *Res, const uint seq_len, TimeStats &stats) {
  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  timed(context, start, stop, stats.fflayer,
        [&]() { forward_pass_gpu(context, layer, X, Res, seq_len, stats); });
  stats.total = stats.fflayer;
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
}

void run_ff_layer_multi_gpus(const DeviceContext &context,
                             const FFLayer<half> &layer, const half *X,
                             half *Res, const uint seq_len, TimeStats &stats,
                             ncclComm_t ncclComm) {

  cudaEvent_t start, stop, total_start, total_stop;
  CUDA_CHECK(cudaEventCreate(&total_start));
  CUDA_CHECK(cudaEventCreate(&total_stop));
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  timed(context, total_start, total_stop, stats.total, [&]() {
    timed(context, start, stop, stats.fflayer,
          [&]() { forward_pass_gpu(context, layer, X, Res, seq_len, stats); });
    timed(context, start, stop, stats.all_reduce_fflayer, [&]() {
      NCCL_CHECK(ncclAllReduce(Res, Res, seq_len * layer.in_dim, ncclHalf,
                               ncclSum, ncclComm, context.stream));
    });
  });

  CUDA_CHECK(cudaEventDestroy(total_start));
  CUDA_CHECK(cudaEventDestroy(total_stop));
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
}