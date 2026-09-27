#include "transformer.cuh"
#include <cmath>
#include <vector>

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

void softmax_cpu(const float *in, float *out, uint num_rows, uint num_columns) {
  for (uint row{0}; row < num_rows; ++row) {
    // Calculation is done in 4 steps

    // STEP 1: find max for numerical stability
    // such that exponent function doesn't blow up
    float max = in[row * num_columns];
    for (uint col{0}; col < num_columns; ++col) {
      float cur = in[row * num_columns + col];
      if (cur > max) {
        max = cur;
      }
    }

    // STEP 2 & 3: compute exp(in - max) & sum exponential
    float sum{0};
    for (uint col{0}; col < num_columns; ++col) {
      uint idx = row * num_columns + col;
      out[idx] = expf(in[idx] - max);
      sum = sum + out[idx];
    }

    // STEP 4: Normalize
    for (uint col{0}; col < num_columns; ++col) {
      out[row * num_columns + col] = out[row * num_columns + col] / sum;
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

void run_attention_layer_cpu(TransformerBlock<float> &transformer, float *X,
                             float *Res, uint seq_len) {
  float alpha = 1.0f;
  float beta = 0.0f;

  gemm_cpu(X, transformer.salayer->Q, transformer.salayer->X_q, seq_len,
           transformer.sa_hidden_dim, transformer.in_dim, alpha, beta);
  gemm_cpu(X, transformer.salayer->K, transformer.salayer->X_k, seq_len,
           transformer.sa_hidden_dim, transformer.in_dim, alpha, beta);
  gemm_cpu(X, transformer.salayer->V, transformer.salayer->X_v, seq_len,
           transformer.sa_hidden_dim, transformer.in_dim, alpha, beta);

  std::vector<float> X_k_T(transformer.sa_hidden_dim * seq_len);
  transpose_cpu(transformer.salayer->X_k, X_k_T.data(), seq_len,
                transformer.sa_hidden_dim);

  float scale = rsqrtf(transformer.sa_hidden_dim);
  gemm_cpu(transformer.salayer->X_q, X_k_T.data(),
           transformer.salayer->attention_scores, seq_len, seq_len,
           transformer.sa_hidden_dim, scale, beta);

  softmax_cpu(transformer.salayer->attention_scores,
              transformer.salayer->attention_weights, seq_len, seq_len);

  gemm_cpu(transformer.salayer->attention_weights, transformer.salayer->X_v,
           transformer.salayer->Z, seq_len, transformer.sa_hidden_dim, seq_len,
           alpha, beta);

  gemm_cpu(transformer.salayer->Z, transformer.salayer->O, Res, seq_len,
           transformer.in_dim, transformer.sa_hidden_dim, alpha, beta);
}

void run_feed_forward_layer_cpu(TransformerBlock<float> &transformer, float *X,
                                float *Res, uint seq_len) {
  float alpha = 1.0f;
  float beta = 0.0f;
  uint hidden_dim = 4 * transformer.in_dim;

  gemm_cpu(X, transformer.fflayer->W1, transformer.fflayer->X_w1, seq_len,
           hidden_dim, transformer.in_dim, alpha, beta);

  relu_cpu(transformer.fflayer->X_w1, transformer.fflayer->X_relu,
           seq_len * hidden_dim);

  gemm_cpu(transformer.fflayer->X_relu, transformer.fflayer->W2, Res, seq_len,
           transformer.in_dim, hidden_dim, alpha, beta);
}

void run_transformer_block_cpu(TransformerBlock<float> &transformer, float *X,
                               float *Res, uint seq_len) {

  run_attention_layer_cpu(transformer, X, transformer.salayer->Res, seq_len);

  run_feed_forward_layer_cpu(transformer, transformer.salayer->Res, Res,
                             seq_len);
}