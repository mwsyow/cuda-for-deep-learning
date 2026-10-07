#include "common.cuh"
#include "kernels/relu.cuh"
#include "kernels/softmax.cuh"
#include "transformer.cuh"
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

void run_attention_layer_gpu(const DeviceContext &context,
                             const TransformerBlock<half> &transformer,
                             const half *X, half *Res, const uint seq_len,
                             TimeStats &stats) {
  cudaEvent_t total_start, total_stop, start, stop;
  CUDA_CHECK(cudaEventCreate(&total_start));
  CUDA_CHECK(cudaEventCreate(&total_stop));
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  half alpha = __float2half(1.0f);
  half beta = __float2half(0.0f);

  CUDA_CHECK(cudaEventRecord(total_start, context.stream));

  timed(context, start, stop, stats.X_q, [&]() {
    CUBLAS_CHECK(cublasHgemm(
        context.handle, CUBLAS_OP_N, CUBLAS_OP_N, transformer.sa_hidden_dim,
        seq_len, transformer.in_dim, &alpha, transformer.salayer->Q,
        transformer.sa_hidden_dim, X, transformer.in_dim, &beta,
        transformer.salayer->X_q, transformer.sa_hidden_dim));
  });
  timed(context, start, stop, stats.X_k, [&]() {
    CUBLAS_CHECK(cublasHgemm(
        context.handle, CUBLAS_OP_N, CUBLAS_OP_N, transformer.sa_hidden_dim,
        seq_len, transformer.in_dim, &alpha, transformer.salayer->K,
        transformer.sa_hidden_dim, X, transformer.in_dim, &beta,
        transformer.salayer->X_k, transformer.sa_hidden_dim));
  });
  timed(context, start, stop, stats.X_v, [&]() {
    CUBLAS_CHECK(cublasHgemm(
        context.handle, CUBLAS_OP_N, CUBLAS_OP_N, transformer.sa_hidden_dim,
        seq_len, transformer.in_dim, &alpha, transformer.salayer->V,
        transformer.sa_hidden_dim, X, transformer.in_dim, &beta,
        transformer.salayer->X_v, transformer.sa_hidden_dim));
  });

  half scale = __float2half(rsqrtf(transformer.sa_hidden_dim));
  timed(context, start, stop, stats.attention_scores, [&]() {
    CUBLAS_CHECK(
        cublasHgemm(context.handle, CUBLAS_OP_T, CUBLAS_OP_N, seq_len, seq_len,
                    transformer.sa_hidden_dim, &scale, transformer.salayer->X_k,
                    transformer.sa_hidden_dim, transformer.salayer->X_q,
                    transformer.sa_hidden_dim, &beta,
                    transformer.salayer->attention_scores, seq_len));
  });

  timed(context, start, stop, stats.attention_weights, [&]() {
    run_softmax(transformer.salayer->attention_scores,
                transformer.salayer->attention_weights, seq_len, seq_len,
                context.stream);
  });

  timed(context, start, stop, stats.Z, [&]() {
    CUBLAS_CHECK(cublasHgemm(
        context.handle, CUBLAS_OP_N, CUBLAS_OP_N, transformer.sa_hidden_dim,
        seq_len, seq_len, &alpha, transformer.salayer->X_v,
        transformer.sa_hidden_dim, transformer.salayer->attention_weights,
        seq_len, &beta, transformer.salayer->Z, transformer.sa_hidden_dim));
  });

  timed(context, start, stop, stats.attention_result, [&]() {
    CUBLAS_CHECK(cublasHgemm(
        context.handle, CUBLAS_OP_N, CUBLAS_OP_N, transformer.in_dim, seq_len,
        transformer.sa_hidden_dim, &alpha, transformer.salayer->O,
        transformer.in_dim, transformer.salayer->Z, transformer.sa_hidden_dim,
        &beta, Res, transformer.in_dim));
  });

  CUDA_CHECK(cudaEventRecord(total_stop, context.stream));
  CUDA_CHECK(cudaEventSynchronize(total_stop));
  CUDA_CHECK(cudaEventElapsedTime(&stats.salayer, total_start, total_stop));

  CUDA_CHECK(cudaEventDestroy(total_start));
  CUDA_CHECK(cudaEventDestroy(total_stop));
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
}

void run_feed_forward_layer_gpu(const DeviceContext &context,
                                const TransformerBlock<half> &transformer,
                                const half *X, half *Res, const uint seq_len,
                                TimeStats &stats) {
  cudaEvent_t total_start, total_stop, start, stop;
  CUDA_CHECK(cudaEventCreate(&total_start));
  CUDA_CHECK(cudaEventCreate(&total_stop));
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  half alpha = 1.0f;
  half beta = 0.0f;

  CUDA_CHECK(cudaEventRecord(total_start, context.stream));

  timed(context, start, stop, stats.X_w1, [&]() {
    CUBLAS_CHECK(cublasHgemm(
        context.handle, CUBLAS_OP_N, CUBLAS_OP_N, transformer.ff_hidden_dim,
        seq_len, transformer.in_dim, &alpha, transformer.fflayer->W1,
        transformer.ff_hidden_dim, X, transformer.in_dim, &beta,
        transformer.fflayer->X_w1, transformer.ff_hidden_dim));
  });

  timed(context, start, stop, stats.X_relu, [&]() {
    run_relu(transformer.fflayer->X_w1, transformer.fflayer->X_relu, seq_len,
             transformer.ff_hidden_dim, context.stream);
  });

  timed(context, start, stop, stats.X_w2, [&]() {
    CUBLAS_CHECK(cublasHgemm(
        context.handle, CUBLAS_OP_N, CUBLAS_OP_N, transformer.in_dim, seq_len,
        transformer.ff_hidden_dim, &alpha, transformer.fflayer->W2,
        transformer.in_dim, transformer.fflayer->X_relu,
        transformer.ff_hidden_dim, &beta, Res, transformer.in_dim));
  });

  CUDA_CHECK(cudaEventRecord(total_stop, context.stream));
  CUDA_CHECK(cudaEventSynchronize(total_stop));
  CUDA_CHECK(cudaEventElapsedTime(&stats.fflayer, total_start, total_stop));

  CUDA_CHECK(cudaEventDestroy(total_start));
  CUDA_CHECK(cudaEventDestroy(total_stop));
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
}

void run_transformer_block_single_gpu(const DeviceContext &context,
                                      const TransformerBlock<half> &transformer,
                                      const half *X, half *Res,
                                      const uint seq_len, TimeStats &stats) {

  cudaEvent_t total_start, total_stop;
  CUDA_CHECK(cudaEventCreate(&total_start));
  CUDA_CHECK(cudaEventCreate(&total_stop));
  CUDA_CHECK(cudaEventRecord(total_start, context.stream));

  run_attention_layer_gpu(context, transformer, X, transformer.salayer->Res,
                          seq_len, stats);
  run_feed_forward_layer_gpu(context, transformer, transformer.salayer->Res,
                             Res, seq_len, stats);

  CUDA_CHECK(cudaEventRecord(total_stop, context.stream));
  CUDA_CHECK(cudaEventSynchronize(total_stop));
  CUDA_CHECK(cudaEventElapsedTime(&stats.total, total_start, total_stop));

  CUDA_CHECK(cudaEventDestroy(total_start));
  CUDA_CHECK(cudaEventDestroy(total_stop));
}

void run_transformer_block_multi_gpus(const DeviceContext &context,
                                      const TransformerBlock<half> &transformer,
                                      const half *X, half *Res,
                                      const uint seq_len, TimeStats &stats,
                                      ncclComm_t ncclComm) {

  cudaEvent_t start, stop, total_start, total_stop;
  CUDA_CHECK(cudaEventCreate(&total_start));
  CUDA_CHECK(cudaEventCreate(&total_stop));
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  timed(context, start, stop, stats.total, [&]() {
    run_attention_layer_gpu(context, transformer, X, transformer.salayer->Res,
                            seq_len, stats);
    timed(context, start, stop, stats.all_reduce_salayer, [&]() {
      NCCL_CHECK(ncclAllReduce(transformer.salayer->Res,
                               transformer.salayer->Res,
                               seq_len * transformer.in_dim, ncclHalf, ncclSum,
                               ncclComm, context.stream));
    });
    run_feed_forward_layer_gpu(context, transformer, transformer.salayer->Res,
                               Res, seq_len, stats);
    timed(context, start, stop, stats.all_reduce_fflayer, [&]() {
      NCCL_CHECK(ncclAllReduce(Res, Res, seq_len * transformer.in_dim, ncclHalf,
                               ncclSum, ncclComm, context.stream));
    });
  });

  CUDA_CHECK(cudaEventDestroy(total_start));
  CUDA_CHECK(cudaEventDestroy(total_stop));
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
}