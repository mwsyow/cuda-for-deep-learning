#include "common.cuh"
#include "kernels/transpose.cuh"
#include <cassert>
#include <cstddef>
#include <cstdlib>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <mpi.h>
#include <nccl.h>
#include <random>
#include <vector>

#pragma once

template <typename T> struct FFLayer {
  T *W1;
  T *W2;

  // Caches
  T *X_w1;
  T *X_relu;
};

template <typename T> struct SelfAttentionLayer {
  T *Q;
  T *K;
  T *V;
  T *O;

  // Caches
  T *X_q;
  T *X_k;
  T *X_v;
  T *attention_scores;
  T *attention_weights;
  T *Z;
  T *Res;
};

template <typename T> struct TransformerBlock {
  FFLayer<T> *fflayer;
  SelfAttentionLayer<T> *salayer;

  uint in_dim;
  uint sa_hidden_dim;
  uint ff_hidden_dim;
};

template <typename T> T from_float(float x) { return static_cast<T>(x); }

template <> half from_float<half>(float x) { return __float2half(x); }

template <typename T>
void init_transformer_block(TransformerBlock<T> &transformer, uint in_dim,
                            uint hidden_dim, uint seq_len, ulong seed) {
  transformer.in_dim = in_dim;
  transformer.sa_hidden_dim = hidden_dim;
  transformer.ff_hidden_dim = 4 * hidden_dim;

  std::mt19937 gen(seed);
  std::uniform_real_distribution<float> distribution(-1.0f, 1.0f);

  transformer.salayer = new SelfAttentionLayer<T>{};
  transformer.fflayer = new FFLayer<T>{};

  transformer.salayer->Q =
      static_cast<T *>(std::malloc(in_dim * hidden_dim * sizeof(T)));
  transformer.salayer->K =
      static_cast<T *>(std::malloc(in_dim * hidden_dim * sizeof(T)));
  transformer.salayer->V =
      static_cast<T *>(std::malloc(in_dim * hidden_dim * sizeof(T)));
  transformer.salayer->O =
      static_cast<T *>(std::malloc(hidden_dim * in_dim * sizeof(T)));

  transformer.salayer->X_q = static_cast<float *>(
      std::malloc(seq_len * transformer.sa_hidden_dim * sizeof(float)));
  transformer.salayer->X_k = static_cast<float *>(
      std::malloc(seq_len * transformer.sa_hidden_dim * sizeof(float)));
  transformer.salayer->X_v = static_cast<float *>(
      std::malloc(seq_len * transformer.sa_hidden_dim * sizeof(float)));
  transformer.salayer->attention_scores =
      static_cast<float *>(std::malloc(seq_len * seq_len * sizeof(float)));
  transformer.salayer->attention_weights =
      static_cast<float *>(std::malloc(seq_len * seq_len * sizeof(float)));
  transformer.salayer->Z = static_cast<float *>(
      std::malloc(seq_len * transformer.sa_hidden_dim * sizeof(float)));
  transformer.salayer->Res = static_cast<float *>(
      std::malloc(seq_len * transformer.in_dim * sizeof(float)));

  for (uint i = 0; i < transformer.in_dim * transformer.sa_hidden_dim; i++)
    transformer.salayer->Q[i] = from_float<T>(distribution(gen));
  for (uint i = 0; i < transformer.in_dim * transformer.sa_hidden_dim; i++)
    transformer.salayer->K[i] = from_float<T>(distribution(gen));
  for (uint i = 0; i < transformer.in_dim * transformer.sa_hidden_dim; i++)
    transformer.salayer->V[i] = from_float<T>(distribution(gen));
  for (uint i = 0; i < transformer.sa_hidden_dim * transformer.in_dim; i++)
    transformer.salayer->O[i] = from_float<T>(distribution(gen));

  transformer.fflayer->W1 = static_cast<T *>(
      std::malloc(in_dim * transformer.ff_hidden_dim * sizeof(T)));
  transformer.fflayer->W2 = static_cast<T *>(
      std::malloc(transformer.ff_hidden_dim * in_dim * sizeof(T)));

  transformer.fflayer->X_w1 = static_cast<float *>(
      malloc(seq_len * transformer.ff_hidden_dim * sizeof(float)));
  transformer.fflayer->X_relu = static_cast<float *>(
      malloc(transformer.ff_hidden_dim * transformer.in_dim * sizeof(float)));

  for (uint i = 0; i < transformer.in_dim * transformer.ff_hidden_dim; i++)
    transformer.fflayer->W1[i] = from_float<T>(distribution(gen));
  for (uint i = 0; i < transformer.ff_hidden_dim * transformer.in_dim; i++)
    transformer.fflayer->W2[i] = from_float<T>(distribution(gen));
}

void weights_to_half(float *src, half *dst, uint size) {
  for (int i = 0; i < size; i++) {
    dst[i] = from_float<half>(src[i]);
  }
}

void to_device(TransformerBlock<float> &host_transformer,
               TransformerBlock<half> &device_transformer, uint seq_len) {

  device_transformer.in_dim = host_transformer.in_dim;
  device_transformer.sa_hidden_dim = host_transformer.sa_hidden_dim;
  device_transformer.ff_hidden_dim = host_transformer.ff_hidden_dim;

  device_transformer.salayer = new SelfAttentionLayer<half>{};
  device_transformer.fflayer = new FFLayer<half>{};

  uint salayer_weight_dim =
      device_transformer.in_dim * device_transformer.sa_hidden_dim;
  uint fflayer_weight_dim =
      4 * device_transformer.in_dim * device_transformer.in_dim;

  size_t salayer_weight_bytes = salayer_weight_dim * sizeof(half);
  size_t fflayer_weight_bytes = fflayer_weight_dim * sizeof(half);

  CUDA_CHECK(cudaMalloc(&device_transformer.salayer->Q, salayer_weight_bytes));
  CUDA_CHECK(cudaMalloc(&device_transformer.salayer->K, salayer_weight_bytes));
  CUDA_CHECK(cudaMalloc(&device_transformer.salayer->V, salayer_weight_bytes));
  CUDA_CHECK(cudaMalloc(&device_transformer.salayer->O, salayer_weight_bytes));
  CUDA_CHECK(
      cudaMalloc(&device_transformer.salayer->X_q,
                 seq_len * device_transformer.sa_hidden_dim * sizeof(half)));
  CUDA_CHECK(
      cudaMalloc(&device_transformer.salayer->X_k,
                 seq_len * device_transformer.sa_hidden_dim * sizeof(half)));
  CUDA_CHECK(
      cudaMalloc(&device_transformer.salayer->X_v,
                 seq_len * device_transformer.sa_hidden_dim * sizeof(half)));
  CUDA_CHECK(cudaMalloc(&device_transformer.salayer->attention_scores,
                        seq_len * seq_len * sizeof(half)));
  CUDA_CHECK(cudaMalloc(&device_transformer.salayer->attention_weights,
                        seq_len * seq_len * sizeof(half)));
  CUDA_CHECK(
      cudaMalloc(&device_transformer.salayer->Z,
                 seq_len * device_transformer.sa_hidden_dim * sizeof(half)));
  CUDA_CHECK(cudaMalloc(&device_transformer.salayer->Res,
                        seq_len * device_transformer.in_dim * sizeof(half)));

  CUDA_CHECK(cudaMalloc(&device_transformer.fflayer->W1, fflayer_weight_bytes));
  CUDA_CHECK(cudaMalloc(&device_transformer.fflayer->W2, fflayer_weight_bytes));
  CUDA_CHECK(
      cudaMalloc(&device_transformer.fflayer->X_w1,
                 seq_len * device_transformer.ff_hidden_dim * sizeof(half)));
  CUDA_CHECK(
      cudaMalloc(&device_transformer.fflayer->X_relu,
                 seq_len * device_transformer.ff_hidden_dim * sizeof(half)));

  std::vector<half> Q(salayer_weight_dim);
  weights_to_half(host_transformer.salayer->Q, Q.data(), Q.size());
  std::vector<half> K(salayer_weight_dim);
  weights_to_half(host_transformer.salayer->K, K.data(), K.size());
  std::vector<half> V(salayer_weight_dim);
  weights_to_half(host_transformer.salayer->V, V.data(), V.size());
  std::vector<half> O(salayer_weight_dim);
  weights_to_half(host_transformer.salayer->O, O.data(), O.size());
  std::vector<half> W1(fflayer_weight_dim);
  weights_to_half(host_transformer.fflayer->W1, W1.data(), W1.size());
  std::vector<half> W2(fflayer_weight_dim);
  weights_to_half(host_transformer.fflayer->W2, W2.data(), W2.size());

  CUDA_CHECK(cudaMemcpy(device_transformer.salayer->Q, Q.data(),
                        salayer_weight_bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(device_transformer.salayer->K, K.data(),
                        salayer_weight_bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(device_transformer.salayer->V, V.data(),
                        salayer_weight_bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(device_transformer.salayer->O, O.data(),
                        salayer_weight_bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(device_transformer.fflayer->W1, W1.data(),
                        fflayer_weight_bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(device_transformer.fflayer->W2, W2.data(),
                        fflayer_weight_bytes, cudaMemcpyHostToDevice));
}
template <typename T>
void to_host(TransformerBlock<T> &device_transformer,
             TransformerBlock<T> &host_transformer, uint seq_len) {

  host_transformer.in_dim = device_transformer.in_dim;
  host_transformer.sa_hidden_dim = device_transformer.sa_hidden_dim;
  host_transformer.ff_hidden_dim = device_transformer.ff_hidden_dim;

  host_transformer.salayer = new SelfAttentionLayer<T>{};
  host_transformer.fflayer = new FFLayer<T>{};

  uint salayer_weight_dim =
      device_transformer.in_dim * device_transformer.sa_hidden_dim;
  uint fflayer_weight_dim =
      4 * device_transformer.in_dim * device_transformer.in_dim;

  size_t salayer_weight_bytes = salayer_weight_dim * sizeof(T);
  size_t fflayer_weight_bytes = fflayer_weight_dim * sizeof(T);

  host_transformer.salayer->Q =
      static_cast<T *>(std::malloc(salayer_weight_bytes));
  host_transformer.salayer->K =
      static_cast<T *>(std::malloc(salayer_weight_bytes));
  host_transformer.salayer->V =
      static_cast<T *>(std::malloc(salayer_weight_bytes));
  host_transformer.salayer->O =
      static_cast<T *>(std::malloc(salayer_weight_bytes));

  host_transformer.salayer->X_q = static_cast<T *>(
      std::malloc(seq_len * host_transformer.sa_hidden_dim * sizeof(T)));
  host_transformer.salayer->X_k = static_cast<T *>(
      std::malloc(seq_len * host_transformer.sa_hidden_dim * sizeof(T)));
  host_transformer.salayer->X_v = static_cast<T *>(
      std::malloc(seq_len * host_transformer.sa_hidden_dim * sizeof(T)));
  host_transformer.salayer->attention_scores =
      static_cast<T *>(std::malloc(seq_len * seq_len * sizeof(T)));
  host_transformer.salayer->attention_weights =
      static_cast<T *>(std::malloc(seq_len * seq_len * sizeof(T)));
  host_transformer.salayer->Z = static_cast<T *>(
      std::malloc(seq_len * host_transformer.sa_hidden_dim * sizeof(T)));
  host_transformer.salayer->Res = static_cast<T *>(
      std::malloc(seq_len * host_transformer.in_dim * sizeof(T)));

  host_transformer.fflayer->W1 =
      static_cast<T *>(std::malloc(fflayer_weight_bytes));
  host_transformer.fflayer->W2 =
      static_cast<T *>(std::malloc(fflayer_weight_bytes));

  host_transformer.fflayer->X_w1 = static_cast<T *>(
      std::malloc(seq_len * host_transformer.ff_hidden_dim * sizeof(T)));
  host_transformer.fflayer->X_relu = static_cast<T *>(
      std::malloc(seq_len * host_transformer.ff_hidden_dim * sizeof(T)));

  CUDA_CHECK(cudaMemcpy(host_transformer.salayer->Q,
                        device_transformer.salayer->Q, salayer_weight_bytes,
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(host_transformer.salayer->K,
                        device_transformer.salayer->K, salayer_weight_bytes,
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(host_transformer.salayer->V,
                        device_transformer.salayer->V, salayer_weight_bytes,
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(host_transformer.salayer->O,
                        device_transformer.salayer->O, salayer_weight_bytes,
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(host_transformer.salayer->X_q,
                        device_transformer.salayer->X_q,
                        seq_len * device_transformer.sa_hidden_dim * sizeof(T),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(host_transformer.salayer->X_k,
                        device_transformer.salayer->X_k,
                        seq_len * device_transformer.sa_hidden_dim * sizeof(T),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(host_transformer.salayer->X_v,
                        device_transformer.salayer->X_v,
                        seq_len * device_transformer.sa_hidden_dim * sizeof(T),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(host_transformer.salayer->attention_scores,
                        device_transformer.salayer->attention_scores,
                        seq_len * seq_len * sizeof(T), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(host_transformer.salayer->attention_weights,
                        device_transformer.salayer->attention_weights,
                        seq_len * seq_len * sizeof(T), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(host_transformer.salayer->Z,
                        device_transformer.salayer->Z,
                        seq_len * device_transformer.sa_hidden_dim * sizeof(T),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(
      host_transformer.salayer->Res, device_transformer.salayer->Res,
      seq_len * device_transformer.in_dim * sizeof(T), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(host_transformer.fflayer->W1,
                        device_transformer.fflayer->W1, fflayer_weight_bytes,
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(host_transformer.fflayer->W2,
                        device_transformer.fflayer->W2, fflayer_weight_bytes,
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(host_transformer.fflayer->X_w1,
                        device_transformer.fflayer->X_w1,
                        seq_len * device_transformer.ff_hidden_dim * sizeof(T),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(host_transformer.fflayer->X_relu,
                        device_transformer.fflayer->X_relu,
                        seq_len * device_transformer.ff_hidden_dim * sizeof(T),
                        cudaMemcpyDeviceToHost));
}

void scatter_transformer(TransformerBlock<half> &full_transformer,
                         TransformerBlock<half> &shard_transformer,
                         const int world_size, const int rank,
                         const uint seq_len, ncclComm_t ncclComm,
                         cudaStream_t stream) {
  //-------------------------------------------------------------------------------
  // Allocate transposed weights and broadcast sharded dimension
  //-------------------------------------------------------------------------------
  uint shard_dims[3];
  half *Q_T, *K_T, *V_T, *W1_T;
  half *Q_T_shard, *K_T_shard, *V_T_shard, *W1_T_shard;
  if (rank == 0) {
    assert(full_transformer.ff_hidden_dim % world_size == 0);
    assert(full_transformer.sa_hidden_dim % world_size == 0);
    shard_dims[0] = full_transformer.in_dim;
    shard_dims[1] = full_transformer.sa_hidden_dim / world_size;
    shard_dims[2] = full_transformer.ff_hidden_dim / world_size;

    CUDA_CHECK(cudaMalloc(&Q_T, full_transformer.in_dim *
                                    full_transformer.sa_hidden_dim *
                                    sizeof(half)));
    CUDA_CHECK(cudaMalloc(&K_T, full_transformer.in_dim *
                                    full_transformer.sa_hidden_dim *
                                    sizeof(half)));
    CUDA_CHECK(cudaMalloc(&V_T, full_transformer.in_dim *
                                    full_transformer.sa_hidden_dim *
                                    sizeof(half)));
    CUDA_CHECK(cudaMalloc(&W1_T, full_transformer.in_dim *
                                     full_transformer.ff_hidden_dim *
                                     sizeof(half)));

    run_transpose(full_transformer.salayer->Q, Q_T, full_transformer.in_dim,
                  full_transformer.sa_hidden_dim, stream);
    run_transpose(full_transformer.salayer->K, K_T, full_transformer.in_dim,
                  full_transformer.sa_hidden_dim, stream);
    run_transpose(full_transformer.salayer->V, V_T, full_transformer.in_dim,
                  full_transformer.sa_hidden_dim, stream);
    run_transpose(full_transformer.fflayer->W1, W1_T, full_transformer.in_dim,
                  full_transformer.ff_hidden_dim, stream);
  }
  MPI_Bcast(shard_dims, sizeof(shard_dims), MPI_BYTE, 0, MPI_COMM_WORLD);

  //-------------------------------------------------------------------------------
  // Instantiate sharded transformer
  //-------------------------------------------------------------------------------
  shard_transformer.in_dim = shard_dims[0];
  shard_transformer.sa_hidden_dim = shard_dims[1];
  shard_transformer.ff_hidden_dim = shard_dims[2];

  shard_transformer.salayer = new SelfAttentionLayer<half>{};
  shard_transformer.fflayer = new FFLayer<half>{};

  uint salayer_weight_dim =
      shard_transformer.in_dim * shard_transformer.sa_hidden_dim;
  uint fflayer_weight_dim =
      4 * shard_transformer.in_dim * shard_transformer.in_dim;

  size_t salayer_weight_bytes = salayer_weight_dim * sizeof(half);
  size_t fflayer_weight_bytes = fflayer_weight_dim * sizeof(half);
  CUDA_CHECK(cudaMalloc(&shard_transformer.salayer->Q, salayer_weight_bytes));
  CUDA_CHECK(cudaMalloc(&shard_transformer.salayer->K, salayer_weight_bytes));
  CUDA_CHECK(cudaMalloc(&shard_transformer.salayer->V, salayer_weight_bytes));
  CUDA_CHECK(cudaMalloc(&shard_transformer.salayer->O, salayer_weight_bytes));
  CUDA_CHECK(
      cudaMalloc(&shard_transformer.salayer->X_q,
                 seq_len * shard_transformer.sa_hidden_dim * sizeof(half)));
  CUDA_CHECK(
      cudaMalloc(&shard_transformer.salayer->X_k,
                 seq_len * shard_transformer.sa_hidden_dim * sizeof(half)));
  CUDA_CHECK(
      cudaMalloc(&shard_transformer.salayer->X_v,
                 seq_len * shard_transformer.sa_hidden_dim * sizeof(half)));
  CUDA_CHECK(cudaMalloc(&shard_transformer.salayer->attention_scores,
                        seq_len * seq_len * sizeof(half)));
  CUDA_CHECK(cudaMalloc(&shard_transformer.salayer->attention_weights,
                        seq_len * seq_len * sizeof(half)));
  CUDA_CHECK(
      cudaMalloc(&shard_transformer.salayer->Z,
                 seq_len * shard_transformer.sa_hidden_dim * sizeof(half)));
  CUDA_CHECK(cudaMalloc(&shard_transformer.salayer->Res,
                        seq_len * shard_transformer.in_dim * sizeof(half)));

  CUDA_CHECK(cudaMalloc(&shard_transformer.fflayer->W1, fflayer_weight_bytes));
  CUDA_CHECK(cudaMalloc(&shard_transformer.fflayer->W2, fflayer_weight_bytes));
  CUDA_CHECK(
      cudaMalloc(&shard_transformer.fflayer->X_w1,
                 seq_len * shard_transformer.ff_hidden_dim * sizeof(half)));
  CUDA_CHECK(
      cudaMalloc(&shard_transformer.fflayer->X_relu,
                 seq_len * shard_transformer.ff_hidden_dim * sizeof(half)));

  CUDA_CHECK(cudaMalloc(&Q_T_shard, shard_transformer.in_dim *
                                        shard_transformer.sa_hidden_dim *
                                        sizeof(half)));
  CUDA_CHECK(cudaMalloc(&K_T_shard, shard_transformer.in_dim *
                                        shard_transformer.sa_hidden_dim *
                                        sizeof(half)));
  CUDA_CHECK(cudaMalloc(&V_T_shard, shard_transformer.in_dim *
                                        shard_transformer.sa_hidden_dim *
                                        sizeof(half)));
  CUDA_CHECK(cudaMalloc(&W1_T_shard, shard_transformer.in_dim *
                                         shard_transformer.ff_hidden_dim *
                                         sizeof(half)));

  //-------------------------------------------------------------------------------
  // Scatter the weights to all ranks
  //-------------------------------------------------------------------------------
  NCCL_CHECK(
      ncclScatter(Q_T, Q_T_shard,
                  shard_transformer.in_dim * shard_transformer.sa_hidden_dim,
                  ncclHalf, 0, ncclComm, stream));
  NCCL_CHECK(
      ncclScatter(K_T, K_T_shard,
                  shard_transformer.in_dim * shard_transformer.sa_hidden_dim,
                  ncclHalf, 0, ncclComm, stream));
  NCCL_CHECK(
      ncclScatter(V_T, V_T_shard,
                  shard_transformer.in_dim * shard_transformer.sa_hidden_dim,
                  ncclHalf, 0, ncclComm, stream));
  NCCL_CHECK(
      ncclScatter(full_transformer.salayer->O, shard_transformer.salayer->O,
                  shard_transformer.in_dim * shard_transformer.sa_hidden_dim,
                  ncclHalf, 0, ncclComm, stream));
  NCCL_CHECK(
      ncclScatter(W1_T, W1_T_shard,
                  shard_transformer.in_dim * shard_transformer.ff_hidden_dim,
                  ncclHalf, 0, ncclComm, stream));
  NCCL_CHECK(
      ncclScatter(full_transformer.fflayer->W2, shard_transformer.fflayer->W2,
                  shard_transformer.in_dim * shard_transformer.ff_hidden_dim,
                  ncclHalf, 0, ncclComm, stream));

  run_transpose(Q_T_shard, shard_transformer.salayer->Q,
                shard_transformer.sa_hidden_dim, shard_transformer.in_dim,
                stream);
  run_transpose(K_T_shard, shard_transformer.salayer->K,
                shard_transformer.sa_hidden_dim, shard_transformer.in_dim,
                stream);
  run_transpose(V_T_shard, shard_transformer.salayer->V,
                shard_transformer.sa_hidden_dim, shard_transformer.in_dim,
                stream);
  run_transpose(W1_T_shard, shard_transformer.fflayer->W1,
                shard_transformer.ff_hidden_dim, shard_transformer.in_dim,
                stream);

  //-------------------------------------------------------------------------------
  // Free temporary memory
  //-------------------------------------------------------------------------------
  if (rank == 0) {
    CUDA_CHECK(cudaFree(Q_T));
    CUDA_CHECK(cudaFree(K_T));
    CUDA_CHECK(cudaFree(V_T));
    CUDA_CHECK(cudaFree(W1_T));
  }
  CUDA_CHECK(cudaFree(Q_T_shard));
  CUDA_CHECK(cudaFree(K_T_shard));
  CUDA_CHECK(cudaFree(V_T_shard));
  CUDA_CHECK(cudaFree(W1_T_shard));
}

void shard_to_full(const TransformerBlock<half> &shard_transformer,
                   TransformerBlock<half> &full_transformer, const uint seq_len,
                   const int rank, ncclComm_t ncclComm, cudaStream_t stream) {
  half *X_q_T_shard, *X_k_T_shard, *X_v_T_shard;
  half *X_q_T, *X_k_T, *X_v_T;

  CUDA_CHECK(cudaMalloc(
      &X_q_T_shard, seq_len * shard_transformer.sa_hidden_dim * sizeof(half)));
  CUDA_CHECK(cudaMalloc(
      &X_k_T_shard, seq_len * shard_transformer.sa_hidden_dim * sizeof(half)));
  CUDA_CHECK(cudaMalloc(
      &X_v_T_shard, seq_len * shard_transformer.sa_hidden_dim * sizeof(half)));

  CUDA_CHECK(cudaMalloc(&X_q_T, seq_len * full_transformer.sa_hidden_dim *
                                    sizeof(half)));
  CUDA_CHECK(cudaMalloc(&X_k_T, seq_len * full_transformer.sa_hidden_dim *
                                    sizeof(half)));
  CUDA_CHECK(cudaMalloc(&X_v_T, seq_len * full_transformer.sa_hidden_dim *
                                    sizeof(half)));

  run_transpose(shard_transformer.salayer->X_q, X_q_T_shard, seq_len,
                shard_transformer.sa_hidden_dim, stream);
  run_transpose(shard_transformer.salayer->X_k, X_k_T_shard, seq_len,
                shard_transformer.sa_hidden_dim, stream);
  run_transpose(shard_transformer.salayer->X_v, X_v_T_shard, seq_len,
                shard_transformer.sa_hidden_dim, stream);

  NCCL_CHECK(ncclGather(X_q_T_shard, X_q_T,
                        seq_len * shard_transformer.sa_hidden_dim, ncclHalf, 0,
                        ncclComm, stream));
  NCCL_CHECK(ncclGather(X_k_T_shard, X_k_T,
                        seq_len * shard_transformer.sa_hidden_dim, ncclHalf, 0,
                        ncclComm, stream));
  NCCL_CHECK(ncclGather(X_v_T_shard, X_v_T,
                        seq_len * shard_transformer.sa_hidden_dim, ncclHalf, 0,
                        ncclComm, stream));

  if (rank == 0) {
    run_transpose(X_q_T, full_transformer.salayer->X_q,
                  full_transformer.sa_hidden_dim, seq_len, stream);
    run_transpose(X_k_T, full_transformer.salayer->X_k,
                  full_transformer.sa_hidden_dim, seq_len, stream);
    run_transpose(X_v_T, full_transformer.salayer->X_v,
                  full_transformer.sa_hidden_dim, seq_len, stream);
  }
}

template <typename T>
void free_host_transformer_block(TransformerBlock<T> &transformer) {
  std::free(transformer.salayer->Q);
  std::free(transformer.salayer->K);
  std::free(transformer.salayer->V);
  std::free(transformer.salayer->O);
  std::free(transformer.salayer->X_q);
  std::free(transformer.salayer->X_k);
  std::free(transformer.salayer->X_v);
  std::free(transformer.salayer->attention_scores);
  std::free(transformer.salayer->attention_weights);
  std::free(transformer.salayer->Z);
  std::free(transformer.salayer->Res);

  std::free(transformer.fflayer->W1);
  std::free(transformer.fflayer->W2);
  std::free(transformer.fflayer->X_w1);
  std::free(transformer.fflayer->X_relu);

  delete transformer.salayer;
  delete transformer.fflayer;
}

template <typename T>
void free_device_transformer_block(TransformerBlock<T> &transformer) {
  CUDA_CHECK(cudaFree(transformer.salayer->Q));
  CUDA_CHECK(cudaFree(transformer.salayer->K));
  CUDA_CHECK(cudaFree(transformer.salayer->V));
  CUDA_CHECK(cudaFree(transformer.salayer->O));
  CUDA_CHECK(cudaFree(transformer.salayer->X_q));
  CUDA_CHECK(cudaFree(transformer.salayer->X_k));
  CUDA_CHECK(cudaFree(transformer.salayer->X_v));
  CUDA_CHECK(cudaFree(transformer.salayer->attention_scores));
  CUDA_CHECK(cudaFree(transformer.salayer->attention_weights));
  CUDA_CHECK(cudaFree(transformer.salayer->Z));
  CUDA_CHECK(cudaFree(transformer.salayer->Res));

  CUDA_CHECK(cudaFree(transformer.fflayer->W1));
  CUDA_CHECK(cudaFree(transformer.fflayer->W2));
  CUDA_CHECK(cudaFree(transformer.fflayer->X_w1));
  CUDA_CHECK(cudaFree(transformer.fflayer->X_relu));

  delete transformer.salayer;
  delete transformer.fflayer;
}