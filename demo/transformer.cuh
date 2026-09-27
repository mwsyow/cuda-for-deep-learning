#include "common.cuh"

#include <cstddef>
#include <cstdlib>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>
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

void to_device(TransformerBlock<float> &host_model,
               TransformerBlock<half> &device_model, uint seq_len) {

  device_model.in_dim = host_model.in_dim;
  device_model.sa_hidden_dim = host_model.sa_hidden_dim;
  device_model.ff_hidden_dim = host_model.ff_hidden_dim;

  device_model.salayer = new SelfAttentionLayer<half>{};
  device_model.fflayer = new FFLayer<half>{};

  uint salayer_weight_dim = device_model.in_dim * device_model.sa_hidden_dim;
  uint fflayer_weight_dim = 4 * device_model.in_dim * device_model.in_dim;

  size_t salayer_weight_bytes = salayer_weight_dim * sizeof(half);
  size_t fflayer_weight_bytes = fflayer_weight_dim * sizeof(half);

  CUDA_CHECK(cudaMalloc(&device_model.salayer->Q, salayer_weight_bytes));
  CUDA_CHECK(cudaMalloc(&device_model.salayer->K, salayer_weight_bytes));
  CUDA_CHECK(cudaMalloc(&device_model.salayer->V, salayer_weight_bytes));
  CUDA_CHECK(cudaMalloc(&device_model.salayer->O, salayer_weight_bytes));
  CUDA_CHECK(cudaMalloc(&device_model.salayer->X_q,
                        seq_len * device_model.sa_hidden_dim * sizeof(half)));
  CUDA_CHECK(cudaMalloc(&device_model.salayer->X_k,
                        seq_len * device_model.sa_hidden_dim * sizeof(half)));
  CUDA_CHECK(cudaMalloc(&device_model.salayer->X_v,
                        seq_len * device_model.sa_hidden_dim * sizeof(half)));
  CUDA_CHECK(cudaMalloc(&device_model.salayer->attention_scores,
                        seq_len * seq_len * sizeof(half)));
  CUDA_CHECK(cudaMalloc(&device_model.salayer->attention_weights,
                        seq_len * seq_len * sizeof(half)));
  CUDA_CHECK(cudaMalloc(&device_model.salayer->Z,
                        seq_len * device_model.sa_hidden_dim * sizeof(half)));
  CUDA_CHECK(cudaMalloc(&device_model.salayer->Res,
                        seq_len * device_model.in_dim * sizeof(half)));

  CUDA_CHECK(cudaMalloc(&device_model.fflayer->W1, fflayer_weight_bytes));
  CUDA_CHECK(cudaMalloc(&device_model.fflayer->W2, fflayer_weight_bytes));
  CUDA_CHECK(cudaMalloc(&device_model.fflayer->X_w1,
                        seq_len * device_model.ff_hidden_dim * sizeof(half)));
  CUDA_CHECK(cudaMalloc(&device_model.fflayer->X_relu,
                        seq_len * device_model.ff_hidden_dim * sizeof(half)));

  std::vector<half> Q(salayer_weight_dim);
  weights_to_half(host_model.salayer->Q, Q.data(), Q.size());
  std::vector<half> K(salayer_weight_dim);
  weights_to_half(host_model.salayer->K, K.data(), K.size());
  std::vector<half> V(salayer_weight_dim);
  weights_to_half(host_model.salayer->V, V.data(), V.size());
  std::vector<half> O(salayer_weight_dim);
  weights_to_half(host_model.salayer->O, O.data(), O.size());
  std::vector<half> W1(fflayer_weight_dim);
  weights_to_half(host_model.fflayer->W1, W1.data(), W1.size());
  std::vector<half> W2(fflayer_weight_dim);
  weights_to_half(host_model.fflayer->W2, W2.data(), W2.size());

  CUDA_CHECK(cudaMemcpy(device_model.salayer->Q, Q.data(), salayer_weight_bytes,
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(device_model.salayer->K, K.data(), salayer_weight_bytes,
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(device_model.salayer->V, V.data(), salayer_weight_bytes,
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(device_model.salayer->O, O.data(), salayer_weight_bytes,
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(device_model.fflayer->W1, W1.data(),
                        fflayer_weight_bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(device_model.fflayer->W2, W2.data(),
                        fflayer_weight_bytes, cudaMemcpyHostToDevice));
}
template <typename T>
void to_host(TransformerBlock<T> &device_model, TransformerBlock<T> &host_model,
             uint seq_len) {

  host_model.in_dim = device_model.in_dim;
  host_model.sa_hidden_dim = device_model.sa_hidden_dim;
  host_model.ff_hidden_dim = device_model.ff_hidden_dim;

  host_model.salayer = new SelfAttentionLayer<T>{};
  host_model.fflayer = new FFLayer<T>{};

  uint salayer_weight_dim = device_model.in_dim * device_model.sa_hidden_dim;
  uint fflayer_weight_dim = 4 * device_model.in_dim * device_model.in_dim;

  size_t salayer_weight_bytes = salayer_weight_dim * sizeof(T);
  size_t fflayer_weight_bytes = fflayer_weight_dim * sizeof(T);

  host_model.salayer->Q = static_cast<T *>(std::malloc(salayer_weight_bytes));
  host_model.salayer->K = static_cast<T *>(std::malloc(salayer_weight_bytes));
  host_model.salayer->V = static_cast<T *>(std::malloc(salayer_weight_bytes));
  host_model.salayer->O = static_cast<T *>(std::malloc(salayer_weight_bytes));

  host_model.salayer->X_q = static_cast<T *>(
      std::malloc(seq_len * host_model.sa_hidden_dim * sizeof(T)));
  host_model.salayer->X_k = static_cast<T *>(
      std::malloc(seq_len * host_model.sa_hidden_dim * sizeof(T)));
  host_model.salayer->X_v = static_cast<T *>(
      std::malloc(seq_len * host_model.sa_hidden_dim * sizeof(T)));
  host_model.salayer->attention_scores =
      static_cast<T *>(std::malloc(seq_len * seq_len * sizeof(T)));
  host_model.salayer->attention_weights =
      static_cast<T *>(std::malloc(seq_len * seq_len * sizeof(T)));
  host_model.salayer->Z = static_cast<T *>(
      std::malloc(seq_len * host_model.sa_hidden_dim * sizeof(T)));
  host_model.salayer->Res =
      static_cast<T *>(std::malloc(seq_len * host_model.in_dim * sizeof(T)));

  host_model.fflayer->W1 = static_cast<T *>(std::malloc(fflayer_weight_bytes));
  host_model.fflayer->W2 = static_cast<T *>(std::malloc(fflayer_weight_bytes));

  host_model.fflayer->X_w1 = static_cast<T *>(
      std::malloc(seq_len * host_model.ff_hidden_dim * sizeof(T)));
  host_model.fflayer->X_relu = static_cast<T *>(
      std::malloc(seq_len * host_model.ff_hidden_dim * sizeof(T)));

  CUDA_CHECK(cudaMemcpy(host_model.salayer->Q, device_model.salayer->Q,
                        salayer_weight_bytes, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(host_model.salayer->K, device_model.salayer->K,
                        salayer_weight_bytes, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(host_model.salayer->V, device_model.salayer->V,
                        salayer_weight_bytes, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(host_model.salayer->O, device_model.salayer->O,
                        salayer_weight_bytes, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(host_model.salayer->X_q, device_model.salayer->X_q,
                        seq_len * device_model.sa_hidden_dim * sizeof(T),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(host_model.salayer->X_k, device_model.salayer->X_k,
                        seq_len * device_model.sa_hidden_dim * sizeof(T),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(host_model.salayer->X_v, device_model.salayer->X_v,
                        seq_len * device_model.sa_hidden_dim * sizeof(T),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(host_model.salayer->attention_scores,
                        device_model.salayer->attention_scores,
                        seq_len * seq_len * sizeof(T), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(host_model.salayer->attention_weights,
                        device_model.salayer->attention_weights,
                        seq_len * seq_len * sizeof(T), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(host_model.salayer->Z, device_model.salayer->Z,
                        seq_len * device_model.sa_hidden_dim * sizeof(T),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(host_model.salayer->Res, device_model.salayer->Res,
                        seq_len * device_model.in_dim * sizeof(T),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(host_model.fflayer->W1, device_model.fflayer->W1,
                        fflayer_weight_bytes, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(host_model.fflayer->W2, device_model.fflayer->W2,
                        fflayer_weight_bytes, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(host_model.fflayer->X_w1, device_model.fflayer->X_w1,
                        seq_len * device_model.ff_hidden_dim * sizeof(T),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(host_model.fflayer->X_relu,
                        device_model.fflayer->X_relu,
                        seq_len * device_model.ff_hidden_dim * sizeof(T),
                        cudaMemcpyDeviceToHost));
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