#include "common.cuh"
#include "kernels/transpose.cuh"
#include <cassert>
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

  uint hidden_dim;
  uint in_dim;
};

template <typename T> T from_float(float x) { return static_cast<T>(x); }

template <> half from_float<half>(float x) { return __float2half(x); }

template <typename T>
void init_ff_layer(FFLayer<T> &ff_layer, uint in_dim, uint hidden_dim,
                   uint seq_len, ulong seed) {
  ff_layer.in_dim = in_dim;
  ff_layer.hidden_dim = hidden_dim;

  std::mt19937 gen(seed);
  std::uniform_real_distribution<float> distribution(-1.0f, 1.0f);

  ff_layer.W1 =
      static_cast<T *>(std::malloc(in_dim * ff_layer.hidden_dim * sizeof(T)));
  ff_layer.W2 =
      static_cast<T *>(std::malloc(ff_layer.hidden_dim * in_dim * sizeof(T)));

  ff_layer.X_w1 = static_cast<float *>(
      malloc(seq_len * ff_layer.hidden_dim * sizeof(float)));
  ff_layer.X_relu = static_cast<float *>(
      malloc(ff_layer.hidden_dim * ff_layer.in_dim * sizeof(float)));

  for (uint i = 0; i < ff_layer.in_dim * ff_layer.hidden_dim; i++)
    ff_layer.W1[i] = from_float<T>(distribution(gen));
  for (uint i = 0; i < ff_layer.hidden_dim * ff_layer.in_dim; i++)
    ff_layer.W2[i] = from_float<T>(distribution(gen));
}

void weights_to_half(float *src, half *dst, uint size) {
  for (int i = 0; i < size; i++) {
    dst[i] = from_float<half>(src[i]);
  }
}

void to_device(FFLayer<float> &host, FFLayer<half> &device, uint seq_len) {

  device.in_dim = host.in_dim;
  device.hidden_dim = host.hidden_dim;

  CUDA_CHECK(
      cudaMalloc(&device.W1, device.hidden_dim * device.in_dim * sizeof(half)));
  CUDA_CHECK(
      cudaMalloc(&device.W2, device.hidden_dim * device.in_dim * sizeof(half)));
  CUDA_CHECK(
      cudaMalloc(&device.X_w1, seq_len * device.hidden_dim * sizeof(half)));
  CUDA_CHECK(
      cudaMalloc(&device.X_relu, seq_len * device.hidden_dim * sizeof(half)));

  std::vector<half> W1(device.hidden_dim * device.in_dim);
  weights_to_half(host.W1, W1.data(), W1.size());
  std::vector<half> W2(device.hidden_dim * device.in_dim);
  weights_to_half(host.W2, W2.data(), W2.size());

  CUDA_CHECK(cudaMemcpy(device.W1, W1.data(),
                        device.hidden_dim * device.in_dim * sizeof(half),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(device.W2, W2.data(),
                        device.hidden_dim * device.in_dim * sizeof(half),
                        cudaMemcpyHostToDevice));
}

template <typename T>
void to_host(FFLayer<T> &device, FFLayer<T> &host, uint seq_len) {

  host.in_dim = device.in_dim;
  host.hidden_dim = device.hidden_dim;

  host.W1 = static_cast<T *>(
      std::malloc(device.hidden_dim * device.in_dim * sizeof(half)));
  host.W2 = static_cast<T *>(
      std::malloc(device.hidden_dim * device.in_dim * sizeof(half)));

  host.X_w1 =
      static_cast<T *>(std::malloc(seq_len * host.hidden_dim * sizeof(T)));
  host.X_relu =
      static_cast<T *>(std::malloc(seq_len * host.hidden_dim * sizeof(T)));

  CUDA_CHECK(cudaMemcpy(host.W1, device.W1,
                        device.hidden_dim * device.in_dim * sizeof(half),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(host.W2, device.W2,
                        device.hidden_dim * device.in_dim * sizeof(half),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(host.X_w1, device.X_w1,
                        seq_len * device.hidden_dim * sizeof(T),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(host.X_relu, device.X_relu,
                        seq_len * device.hidden_dim * sizeof(T),
                        cudaMemcpyDeviceToHost));
}

void scatter_ff_layer(FFLayer<half> &full_ff, FFLayer<half> &shard_ff,
                      const int world_size, const int rank, const uint seq_len,
                      ncclComm_t ncclComm, cudaStream_t stream) {
  //-------------------------------------------------------------------------------
  // Allocate transposed weights and broadcast sharded dimension
  //-------------------------------------------------------------------------------
  uint shard_dims[2];
  half *W1_T, *W1_T_shard;
  if (rank == 0) {
    assert(full_ff.hidden_dim % world_size == 0);
    shard_dims[0] = full_ff.in_dim;
    shard_dims[1] = full_ff.hidden_dim / world_size;

    CUDA_CHECK(
        cudaMalloc(&W1_T, full_ff.in_dim * full_ff.hidden_dim * sizeof(half)));

    run_transpose(full_ff.W1, W1_T, full_ff.in_dim, full_ff.hidden_dim, stream);
  }
  MPI_Bcast(shard_dims, sizeof(shard_dims), MPI_BYTE, 0, MPI_COMM_WORLD);

  //-------------------------------------------------------------------------------
  // Instantiate sharded layer
  //-------------------------------------------------------------------------------
  shard_ff.in_dim = shard_dims[0];
  shard_ff.hidden_dim = shard_dims[1];

  CUDA_CHECK(cudaMalloc(&shard_ff.W1,
                        shard_ff.hidden_dim * shard_ff.in_dim * sizeof(half)));
  CUDA_CHECK(cudaMalloc(&shard_ff.W2,
                        shard_ff.hidden_dim * shard_ff.in_dim * sizeof(half)));
  CUDA_CHECK(
      cudaMalloc(&shard_ff.X_w1, seq_len * shard_ff.hidden_dim * sizeof(half)));
  CUDA_CHECK(cudaMalloc(&shard_ff.X_relu,
                        seq_len * shard_ff.hidden_dim * sizeof(half)));
  CUDA_CHECK(cudaMalloc(&W1_T_shard,
                        shard_ff.in_dim * shard_ff.hidden_dim * sizeof(half)));

  //-------------------------------------------------------------------------------
  // Scatter the weights to all ranks
  //-------------------------------------------------------------------------------
  NCCL_CHECK(ncclScatter(W1_T, W1_T_shard,
                         shard_ff.in_dim * shard_ff.hidden_dim, ncclHalf, 0,
                         ncclComm, stream));
  NCCL_CHECK(ncclScatter(full_ff.W2, shard_ff.W2,
                         shard_ff.in_dim * shard_ff.hidden_dim, ncclHalf, 0,
                         ncclComm, stream));

  run_transpose(W1_T_shard, shard_ff.W1, shard_ff.hidden_dim, shard_ff.in_dim,
                stream);

  //-------------------------------------------------------------------------------
  // Free temporary memory
  //-------------------------------------------------------------------------------
  if (rank == 0) {
    CUDA_CHECK(cudaFree(W1_T));
  }
  CUDA_CHECK(cudaFree(W1_T_shard));
}

void unsharding_intermediate_results(const FFLayer<half> &shard_ff,
                                     FFLayer<half> &full_ff, const uint seq_len,
                                     const int rank, ncclComm_t ncclComm,
                                     cudaStream_t stream) {
  half *X_w1_T_shard, *X_relu_T_shard, *X_w1_T, *X_relu_T;

  CUDA_CHECK(
      cudaMalloc(&X_w1_T_shard, seq_len * shard_ff.hidden_dim * sizeof(half)));
  CUDA_CHECK(cudaMalloc(&X_relu_T_shard,
                        seq_len * shard_ff.hidden_dim * sizeof(half)));

  CUDA_CHECK(cudaMalloc(&X_w1_T, seq_len * full_ff.hidden_dim * sizeof(half)));
  CUDA_CHECK(
      cudaMalloc(&X_relu_T, seq_len * full_ff.hidden_dim * sizeof(half)));

  run_transpose(shard_ff.X_w1, X_w1_T_shard, seq_len, shard_ff.hidden_dim,
                stream);
  run_transpose(shard_ff.X_relu, X_relu_T_shard, seq_len, shard_ff.hidden_dim,
                stream);

  NCCL_CHECK(ncclGather(X_w1_T_shard, X_w1_T, seq_len * shard_ff.hidden_dim,
                        ncclHalf, 0, ncclComm, stream));
  NCCL_CHECK(ncclGather(X_relu_T_shard, X_relu_T, seq_len * shard_ff.hidden_dim,
                        ncclHalf, 0, ncclComm, stream));

  if (rank == 0) {
    run_transpose(X_w1_T, full_ff.X_w1, full_ff.hidden_dim, seq_len, stream);
    run_transpose(X_relu_T, full_ff.X_relu, full_ff.hidden_dim, seq_len,
                  stream);
  }

  CUDA_CHECK(cudaFree(X_w1_T_shard));
  CUDA_CHECK(cudaFree(X_w1_T));
  CUDA_CHECK(cudaFree(X_relu_T_shard));
  CUDA_CHECK(cudaFree(X_relu_T));
}

template <typename T> void free_host_ff_layer(FFLayer<T> &ff) {
  std::free(ff.W1);
  std::free(ff.W2);
  std::free(ff.X_w1);
  std::free(ff.X_relu);
}

template <typename T> void free_device_ff_layer(FFLayer<T> &ff) {
  CUDA_CHECK(cudaFree(ff.W1));
  CUDA_CHECK(cudaFree(ff.W2));
  CUDA_CHECK(cudaFree(ff.X_w1));
  CUDA_CHECK(cudaFree(ff.X_relu));
}