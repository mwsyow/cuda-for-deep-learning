#include "feed_forward.cuh"
#include <cmath>

#pragma once

void gemm_cpu(const float *A, const float *B, float *C, uint M, uint N, uint K,
              float alpha, float beta) {

  for (uint m{0}; m < M; ++m) {
    for (uint n{0}; n < N; ++n) {
      float sum{0};
      for (uint k{0}; k < K; ++k) {
        sum = sum + alpha * (A[(m * K) + k] * B[(k * N) + n]);
      }
      C[(m * N) + n] = sum + beta * C[(m * N) + n];
    }
  }
}

void transpose_cpu(float *in, float *out, uint num_rows, uint num_columns) {
  for (uint c{0}; c < num_columns; ++c) {
    for (uint r{0}; r < num_rows; ++r) {
      out[c * num_rows + r] = in[r * num_columns + c];
    }
  }
}

void relu_cpu(float *in, float *out, uint size) {
  for (int i = 0; i < size; i++) {
    out[i] = std::fmax(in[i], 0.0f);
  }
}

void run_ff_layer_cpu(FFLayer<float> &layer, float *X, float *Res,
                      uint seq_len) {
  float alpha = 1.0f;
  float beta = 0.0f;

  gemm_cpu(X, layer.W1, layer.X_w1, seq_len, layer.hidden_dim, layer.in_dim,
           alpha, beta);

  relu_cpu(layer.X_w1, layer.X_relu, seq_len * layer.hidden_dim);

  gemm_cpu(layer.X_relu, layer.W2, Res, seq_len, layer.in_dim, layer.hidden_dim,
           alpha, beta);
}