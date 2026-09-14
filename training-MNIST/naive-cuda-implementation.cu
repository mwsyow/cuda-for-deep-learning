
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef struct {
  double data_loading; // Time spent loading data from host to device
  double fwd_matmul1;  // Time for first matrix multiplication (input -> hidden)
  double fwd_bias1;    // Time for first bias addition
  double fwd_relu;     // Time for ReLU activation
  double
      fwd_matmul2;  // Time for second matrix multiplication (hidden -> output)
  double fwd_bias2; // Time for second bias addition
  double fwd_softmax;     // Time for softmax activation
  double cross_entropy;   // Time for cross-entropy loss computation
  double bwd_output_grad; // Time for output gradient computation
  double bwd_matmul2;     // Time for backward matrix multiplication
  double bwd_bias2;       // Time for backward bias gradient
  double bwd_relu;        // Time for ReLU backward pass
  double bwd_matmul1;     // Time for backward matrix multiplication
  double bwd_bias1;       // Time for backward bias gradient
  double weight_updates;  // Time for weight updates
  double total_time;      // Total training time
} TimingStats;

double get_time_diff(struct timespec start, struct timespec end) {
  return (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
}

typedef struct {
  float *w1, *w2, *b1, *b2;
  float *grad_w1, *grad_w2, *grad_b1, *grad_b2;
} NeuralNetwork;

void load_data(const char *filename, float *data, int size) {
  FILE *file = fopen(filename, "rb");
  if (file == NULL) {
    fprintf(stderr, "Error opening file: %s\n", filename);
    exit(1);
  }
  size_t read_size = fread(data, sizeof(float), size, file);
  if (read_size != size) {
    fprintf(stderr, "Error reading data: expected %d elements, got %zu\n", size,
            read_size);
    exit(1);
  }
  fclose(file);
}

void load_labels(const char *filename, int *labels, int size) {
  FILE *file = fopen(filename, "rb");
  if (file == NULL) {
    fprintf(stderr, "Error opening file: %s\n", filename);
    exit(1);
  }
  size_t read_size = fread(labels, sizeof(int), size, file);
  if (read_size != size) {
    fprintf(stderr, "Error reading labels: expected %d elements, got %zu\n",
            size, read_size);
    exit(1);
  }
  fclose(file);
}

// Macros are preprocessor directives. They run before compiler sees code,
// no knowledge about types, variable, etc. It acts like text subtitute, in this
// case CUDA_CHECK is a wrapper function on variable `call` function.
// the purpose is that during runtime errors that emerge from a kernel launch is
// not directly visible, since kernels are asynchronous and the return
// immediately when launched. If error is not handled properly, error can emerge
// on different checkpoints of the program which makes debugging a nightmare.
#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t error = call;                                                  \
    if (error != cudaSuccess) {                                                \
      fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,         \
              cudaGetErrorString(error));                                      \
      cudaDeviceReset();                                                       \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

//---------------------------------------------------
// INITIALIZATION
//---------------------------------------------------
void initialize_weights(float *w, int in_dim, int out_dim) {
  float scale = sqrtf(6.0 / in_dim);
  for (int i = 0; i < in_dim * out_dim; ++i) {
    w[i] = ((float)rand() / RAND_MAX) * 2.0f * scale - scale;
  }
}

void initialize_biases(float *b, int dim) {
  for (int i = 0; i < dim; ++i) {
    b[i] = 0;
  }
}

void normalize_data(float *data, int size) {
  const float mean = 0.1307f;
  const float std = 0.3081f;
  for (int i = 0; i < size; i++) {
    data[i] = (data[i] - mean) / std;
  }
}

void initialize_neural_network(NeuralNetwork *nn, int in_dim, int hidden_dim,
                               int out_dim) {
  size_t w1_bytes = in_dim * hidden_dim * sizeof(float);
  float *h_w1 = (float *)malloc(w1_bytes);
  initialize_weights(h_w1, in_dim, hidden_dim);
  CUDA_CHECK(cudaMalloc(&nn->w1, w1_bytes));
  CUDA_CHECK(cudaMemcpy(nn->w1, h_w1, w1_bytes, cudaMemcpyHostToDevice));
  free(h_w1);

  size_t b1_bytes = hidden_dim * sizeof(float);
  float *h_b1 = (float *)malloc(b1_bytes);
  initialize_biases(h_b1, hidden_dim);
  CUDA_CHECK(cudaMalloc(&nn->b1, b1_bytes));
  CUDA_CHECK(cudaMemcpy(nn->b1, h_b1, b1_bytes, cudaMemcpyHostToDevice));
  free(h_b1);

  size_t w2_bytes = hidden_dim * out_dim * sizeof(float);
  float *h_w2 = (float *)malloc(w2_bytes);
  initialize_weights(h_w2, hidden_dim, out_dim);
  CUDA_CHECK(cudaMalloc(&nn->w2, w2_bytes));
  CUDA_CHECK(cudaMemcpy(nn->w2, h_w2, w2_bytes, cudaMemcpyHostToDevice));
  free(h_w2);

  size_t b2_bytes = out_dim * sizeof(float);
  float *h_b2 = (float *)malloc(b2_bytes);
  initialize_biases(h_b2, out_dim);
  CUDA_CHECK(cudaMalloc(&nn->b2, b2_bytes));
  CUDA_CHECK(cudaMemcpy(nn->b2, h_b2, b2_bytes, cudaMemcpyHostToDevice));
  free(h_b2);

  CUDA_CHECK(cudaMalloc(&nn->grad_w1, w1_bytes));
  CUDA_CHECK(cudaMalloc(&nn->grad_b1, b1_bytes));
  CUDA_CHECK(cudaMalloc(&nn->grad_w2, w2_bytes));
  CUDA_CHECK(cudaMalloc(&nn->grad_b2, b2_bytes));
}

//---------------------------------------------------
// FORWARD PASS
//---------------------------------------------------

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

__global__ void bias_forward_kernel(float *x, float *bias, int batch_size,
                                    int dim) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int b = idx / dim;
  int d = idx % dim;
  if (b < batch_size && d < dim) {
    x[b * dim + d] += bias[d];
  }
}

__global__ void relu_forward_kernel(float *x, int dim) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < dim) {
    x[idx] = fmaxf(0.0f, x[idx]);
  }
}

__global__ void softmax_kernel(float *x, int row_dim, int col_dim) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < row_dim && col < col_dim) {

    float row_max = x[row * col_dim + col];
    for (int c = 0; c < col_dim; ++c) {
      row_max = fmaxf(row_max, x[row * col_dim + c]);
    }

    float row_sum = 0.0f;
    for (int c = 0; c < col_dim; ++c) {
      row_sum += expf(x[row * col_dim + c] - row_max);
    }

    int idx = row * col_dim + col;
    x[idx] = fmaxf(expf(x[idx] - row_max) / row_sum, 1e-7f);
  }
}

void forward_timed(NeuralNetwork *nn, float *input, float *hidden,
                   float *output, int batch_size, int input_dim, int hidden_dim,
                   int output_dim, TimingStats *stats) {
  struct timespec start, end;

  dim3 block_size(32, 32);

  clock_gettime(CLOCK_MONOTONIC, &start);
  dim3 grid_dim_w1((hidden_dim + block_size.x - 1) / block_size.x,
                   (batch_size + block_size.y - 1) / block_size.y);
  matmul_a_b_kernel<<<grid_dim_w1, block_size>>>(
      input, nn->w1, hidden, batch_size, hidden_dim, input_dim);
  CUDA_CHECK(cudaDeviceSynchronize());
  clock_gettime(CLOCK_MONOTONIC, &end);
  stats->fwd_matmul1 += get_time_diff(start, end);

  clock_gettime(CLOCK_MONOTONIC, &start);
  bias_forward_kernel<<<(hidden_dim + 256 - 1) / 256, 256>>>(
      hidden, nn->b1, batch_size, hidden_dim);
  CUDA_CHECK(cudaDeviceSynchronize());
  clock_gettime(CLOCK_MONOTONIC, &end);
  stats->fwd_bias1 += get_time_diff(start, end);

  clock_gettime(CLOCK_MONOTONIC, &start);
  relu_forward_kernel<<<((batch_size * hidden_dim) + 256 - 1) / 256, 256>>>(
      hidden, batch_size * hidden_dim);
  CUDA_CHECK(cudaDeviceSynchronize());
  clock_gettime(CLOCK_MONOTONIC, &end);
  stats->fwd_relu += get_time_diff(start, end);

  clock_gettime(CLOCK_MONOTONIC, &start);
  dim3 grid_dim_w2((hidden_dim + block_size.x - 1) / block_size.x,
                   (batch_size + block_size.y - 1) / block_size.y);
  matmul_a_b_kernel<<<grid_dim_w2, block_size>>>(
      hidden, nn->w2, output, batch_size, output_dim, hidden_dim);
  CUDA_CHECK(cudaDeviceSynchronize());
  clock_gettime(CLOCK_MONOTONIC, &end);
  stats->fwd_matmul2 += get_time_diff(start, end);

  clock_gettime(CLOCK_MONOTONIC, &start);
  bias_forward_kernel<<<(output_dim + 256 - 1) / 256, 256>>>(
      output, nn->b2, batch_size, output_dim);
  CUDA_CHECK(cudaDeviceSynchronize());
  clock_gettime(CLOCK_MONOTONIC, &end);
  stats->fwd_bias2 += get_time_diff(start, end);

  clock_gettime(CLOCK_MONOTONIC, &start);
  dim3 grid_dim_softmax((hidden_dim + block_size.x - 1) / block_size.x,
                        (batch_size + block_size.y - 1) / block_size.y);
  softmax_kernel<<<grid_dim_softmax, block_size>>>(output, batch_size,
                                                   output_dim);
  CUDA_CHECK(cudaDeviceSynchronize());
  clock_gettime(CLOCK_MONOTONIC, &end);
  stats->fwd_softmax += get_time_diff(start, end);
}

//---------------------------------------------------
// LOSS
//---------------------------------------------------

float cross_entropy_loss(float *probs, int *y_test, int batch_size,
                         int num_classes) {
  float loss = 0.0f;
  for (int b = 0; b < batch_size; ++b) {
    loss += logf(fmaxf(probs[b * num_classes + y_test[b]], 1e-7f));
  }
  return -loss / (float)batch_size;
}

//---------------------------------------------------
// BACKWARD PASS
//---------------------------------------------------

__global__ void matmul_at_b_kernel(float *A, float *B, float *C, int m_A,
                                   int n_B, int k_shared) {

  // A: (k_shared, m) need to be transposed, such that
  // A^T: (m, k_shared) @ B: (k_shared, n) -> (m, n)
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < m_A && col < n_B) {
    float row_sum = 0.0f;
    for (int k = 0; k < k_shared; ++k) {
      row_sum += A[k * m_A + row] * B[k * n_B + col];
    }
    C[row * n_B + col] = row_sum;
  }
}

__global__ void matmul_a_bt_kernel(float *A, float *B, float *C, int m_A,
                                   int n_B, int k_shared) {
  // B: (n, k_shared) need to be transposed, such that
  // A: (m, k_shared) @ B^T: (k_shared, n) -> (m, n)
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < m_A && col < n_B) {
    float row_sum = 0.0f;
    for (int k = 0; k < k_shared; ++k) {
      row_sum += A[row * k_shared + k] * B[col * k_shared + k];
    }
    C[row * n_B + col] = row_sum;
  }
}

__global__ void relu_backward_kernel(float *grad, float *x, float *grad_out,
                                     int dim) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < dim) {
    grad_out[idx] = grad[idx] * (x[idx] > 0);
  }
}

__global__ void bias_backward_kernel(float *grad, float *grad_bias,
                                     int batch_size, int dim) {
  int d = blockIdx.x * blockDim.x + threadIdx.x;

  if (d < dim) {
    grad_bias[d] = 0.0f;
    for (int b = 0; b < batch_size; ++b) {
      grad_bias[d] += grad[b * dim + d];
    }
  }
}

__global__ void compute_output_gradients_kernel(float *probs, int *y_true,
                                                float *grad, int batch_size,
                                                int num_classes) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int b = idx / num_classes;
  int n = idx % num_classes;

  if (b < batch_size && n < num_classes) {
    int idx = b * num_classes + n;
    if (n == y_true[b]) {
      grad[idx] = (probs[idx] - 1.0f) / batch_size;
    } else {
      grad[idx] = probs[idx] / batch_size;
    }
  }
}

void backward_timed(NeuralNetwork *nn, float *input, float *hidden,
                    float *output, int *labels, int batch_size, int input_dim,
                    int hidden_dim, int output_dim, TimingStats *stats) {
  struct timespec start, end;

  dim3 block_size(32, 32);

  float *output_grad, *fc2_grad, *relu_grad;
  CUDA_CHECK(cudaMalloc(&output_grad, batch_size * output_dim * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&fc2_grad, batch_size * hidden_dim * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&relu_grad, batch_size * hidden_dim * sizeof(float)));

  clock_gettime(CLOCK_MONOTONIC, &start);
  compute_output_gradients_kernel<<<((batch_size * output_dim) + 255) / 256,
                                    256>>>(output, labels, output_grad,
                                           batch_size, output_dim);
  CUDA_CHECK(cudaDeviceSynchronize());
  clock_gettime(CLOCK_MONOTONIC, &end);
  stats->bwd_output_grad += get_time_diff(start, end);

  clock_gettime(CLOCK_MONOTONIC, &start);
  dim3 grid_dim_w2((output_dim + block_size.x - 1) / block_size.x,
                   (hidden_dim + block_size.y - 1) / block_size.y);
  matmul_at_b_kernel<<<grid_dim_w2, block_size>>>(
      hidden, output_grad, nn->grad_w2, hidden_dim, output_dim, batch_size);
  CUDA_CHECK(cudaDeviceSynchronize());
  clock_gettime(CLOCK_MONOTONIC, &end);
  stats->bwd_matmul2 += get_time_diff(start, end);

  clock_gettime(CLOCK_MONOTONIC, &start);
  bias_backward_kernel<<<(output_dim + 255) / 256, 256>>>(
      output_grad, nn->grad_b2, batch_size, output_dim);
  CUDA_CHECK(cudaDeviceSynchronize());
  clock_gettime(CLOCK_MONOTONIC, &end);
  stats->bwd_bias2 += get_time_diff(start, end);

  dim3 grid_dim_fc2((hidden_dim + block_size.x - 1) / block_size.x,
                    (batch_size + block_size.y - 1) / block_size.y);
  matmul_a_bt_kernel<<<grid_dim_fc2, block_size>>>(
      output_grad, nn->w2, fc2_grad, batch_size, hidden_dim, output_dim);

  clock_gettime(CLOCK_MONOTONIC, &start);
  relu_backward_kernel<<<((hidden_dim * batch_size) + 255) / 256, 256>>>(
      fc2_grad, hidden, relu_grad, hidden_dim * batch_size);
  CUDA_CHECK(cudaDeviceSynchronize());
  clock_gettime(CLOCK_MONOTONIC, &end);
  stats->bwd_relu += get_time_diff(start, end);

  clock_gettime(CLOCK_MONOTONIC, &start);
  dim3 grid_dim_w1((hidden_dim + block_size.x - 1) / block_size.x,
                   (input_dim + block_size.y - 1) / block_size.y);
  matmul_at_b_kernel<<<grid_dim_w1, block_size>>>(
      input, relu_grad, nn->grad_w1, input_dim, hidden_dim, batch_size);
  CUDA_CHECK(cudaDeviceSynchronize());
  clock_gettime(CLOCK_MONOTONIC, &end);
  stats->bwd_matmul1 += get_time_diff(start, end);

  clock_gettime(CLOCK_MONOTONIC, &start);
  bias_backward_kernel<<<(hidden_dim + 255) / 256, 256>>>(
      relu_grad, nn->grad_b1, batch_size, hidden_dim);
  CUDA_CHECK(cudaDeviceSynchronize());
  clock_gettime(CLOCK_MONOTONIC, &end);
  stats->bwd_bias1 += get_time_diff(start, end);

  CUDA_CHECK(cudaFree(output_grad));
  CUDA_CHECK(cudaFree(fc2_grad));
  CUDA_CHECK(cudaFree(relu_grad));
}

//---------------------------------------------------
// WEIGHTS UPDATE
//---------------------------------------------------
__global__ void update_weight_kernel(float *weights, float *grads, float dim,
                                     float lr) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < dim) {
    weights[idx] -= lr * grads[idx];
  }
}

void update_weights_timed(NeuralNetwork *nn, int input_dim, int hidden_dim,
                          int output_dim, float lr, TimingStats *stats) {
  struct timespec start, end;
  clock_gettime(CLOCK_MONOTONIC, &start);
  update_weight_kernel<<<((output_dim * hidden_dim) + 255) / 256, 256>>>(
      nn->w2, nn->grad_w2, output_dim * hidden_dim, lr);
  update_weight_kernel<<<((input_dim * hidden_dim) + 255) / 256, 256>>>(
      nn->w1, nn->grad_w1, input_dim * hidden_dim, lr);
  update_weight_kernel<<<(output_dim + 255) / 256, 256>>>(nn->b2, nn->grad_b2,
                                                          output_dim, lr);
  update_weight_kernel<<<(hidden_dim + 255) / 256, 256>>>(nn->b1, nn->grad_b1,
                                                          hidden_dim, lr);
  CUDA_CHECK(cudaDeviceSynchronize());
  clock_gettime(CLOCK_MONOTONIC, &end);
  stats->weight_updates += get_time_diff(start, end);
}

__global__ void zero_grads(float *grads, int dim) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < dim) {
    grads[idx] = 0;
  }
}

//---------------------------------------------------
// TRAINING
//---------------------------------------------------

int imin(int a, int b) { return a < b ? a : b; }

void train_timed(NeuralNetwork *nn, float *X_train, int *y_train, int epochs,
                 int batch_size, int input_dim, int hidden_dim, int num_classes,
                 int train_size, float lr) {
  struct timespec start, end, total_start, total_end;

  float *output, *hidden, *input;
  int *y_true;
  CUDA_CHECK(cudaMalloc(&hidden, batch_size * hidden_dim * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&output, batch_size * num_classes * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&input, batch_size * input_dim * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&y_true, batch_size * sizeof(int)));

  float *h_output = (float *)malloc(batch_size * num_classes * sizeof(float));

  TimingStats stats = {0};
  int num_batches = (train_size + batch_size - 1) / batch_size;

  clock_gettime(CLOCK_MONOTONIC, &total_start);
  for (int e = 0; e < epochs; ++e) {
    float epoch_loss = 0.0f;
    for (int n = 0; n < num_batches; ++n) {

      int cur_batch_size = imin(abs(n * batch_size - train_size), batch_size);

      clock_gettime(CLOCK_MONOTONIC, &start);
      float *h_input = &X_train[n * batch_size * input_dim];
      int *h_y_true = &y_train[n * batch_size];
      CUDA_CHECK(cudaMemcpy(input, h_input,
                            batch_size * input_dim * sizeof(float),
                            cudaMemcpyHostToDevice));
      CUDA_CHECK(cudaMemcpy(y_true, h_y_true, batch_size * sizeof(int),
                            cudaMemcpyHostToDevice));
      clock_gettime(CLOCK_MONOTONIC, &end);
      stats.data_loading += get_time_diff(start, end);

      forward_timed(nn, input, hidden, output, cur_batch_size, input_dim,
                    hidden_dim, num_classes, &stats);

      CUDA_CHECK(cudaMemcpy(h_output, output,
                            batch_size * num_classes * sizeof(float),
                            cudaMemcpyDeviceToHost));

      clock_gettime(CLOCK_MONOTONIC, &start);
      epoch_loss +=
          cross_entropy_loss(h_output, h_y_true, cur_batch_size, num_classes);
      clock_gettime(CLOCK_MONOTONIC, &end);
      stats.cross_entropy += get_time_diff(start, end);

      backward_timed(nn, input, hidden, output, y_true, cur_batch_size,
                     input_dim, hidden_dim, num_classes, &stats);

      update_weights_timed(nn, input_dim, hidden_dim, num_classes, lr, &stats);
      zero_grads<<<((input_dim * hidden_dim) + 255) / 256, 256>>>(
          nn->grad_w1, input_dim * hidden_dim);
      zero_grads<<<((num_classes * hidden_dim) + 255) / 256, 256>>>(
          nn->grad_w2, num_classes * hidden_dim);
      zero_grads<<<(hidden_dim + 255) / 256, 256>>>(nn->grad_b1, hidden_dim);
      zero_grads<<<(num_classes + 255) / 256, 256>>>(nn->grad_b2, num_classes);
    }
    printf("Epoch %d loss: %.4f\n", e, epoch_loss / num_batches);
  }
  clock_gettime(CLOCK_MONOTONIC, &total_end);
  stats.total_time = get_time_diff(total_start, total_end);

  // Print timing statistics
  printf("\n=== C CPU IMPLEMENTATION TIMING BREAKDOWN ===\n");
  printf("Total training time: %.1f seconds\n\n", stats.total_time);

  printf("Detailed Breakdown:\n");
  printf("  Data loading:     %6.3fs (%5.1f%%)\n", stats.data_loading,
         100.0 * stats.data_loading / stats.total_time);
  double forward_pass = stats.fwd_matmul1 + stats.fwd_bias1 + stats.fwd_relu +
                        stats.fwd_matmul2 + stats.fwd_bias2 + stats.fwd_softmax;
  printf("  Forward pass:     %6.3fs (%5.1f%%)\n", forward_pass,
         100.0 * forward_pass / stats.total_time);
  printf("    Matmul 1:       %6.3fs (%5.1f%%)\n", stats.fwd_matmul1,
         100.0 * stats.fwd_matmul1 / stats.total_time);
  printf("    Bias 1:         %6.3fs (%5.1f%%)\n", stats.fwd_bias1,
         100.0 * stats.fwd_bias1 / stats.total_time);
  printf("    ReLU:           %6.3fs (%5.1f%%)\n", stats.fwd_relu,
         100.0 * stats.fwd_relu / stats.total_time);
  printf("    Matmul 2:       %6.3fs (%5.1f%%)\n", stats.fwd_matmul2,
         100.0 * stats.fwd_matmul2 / stats.total_time);
  printf("    Bias 2:         %6.3fs (%5.1f%%)\n", stats.fwd_bias2,
         100.0 * stats.fwd_bias2 / stats.total_time);
  printf("    Softmax:        %6.3fs (%5.1f%%)\n", stats.fwd_softmax,
         100.0 * stats.fwd_softmax / stats.total_time);
  printf("  Loss computation: %6.3fs (%5.1f%%)\n", stats.cross_entropy,
         100.0 * stats.cross_entropy / stats.total_time);
  double backward_pass = stats.bwd_output_grad + stats.bwd_matmul2 +
                         stats.bwd_bias2 + stats.bwd_relu + stats.bwd_matmul1 +
                         stats.bwd_bias1;
  printf("  Backward pass:    %6.3fs (%5.1f%%)\n", backward_pass,
         100.0 * backward_pass / stats.total_time);
  printf("    Output gradient:%6.3fs (%5.1f%%)\n", stats.bwd_output_grad,
         100.0 * stats.bwd_output_grad / stats.total_time);
  printf("    Matmul 2:       %6.3fs (%5.1f%%)\n", stats.bwd_matmul2,
         100.0 * stats.bwd_matmul2 / stats.total_time);
  printf("    Bias 2:         %6.3fs (%5.1f%%)\n", stats.bwd_bias2,
         100.0 * stats.bwd_bias2 / stats.total_time);
  printf("    ReLU:           %6.3fs (%5.1f%%)\n", stats.bwd_relu,
         100.0 * stats.bwd_relu / stats.total_time);
  printf("    Matmul 1:       %6.3fs (%5.1f%%)\n", stats.bwd_matmul1,
         100.0 * stats.bwd_matmul1 / stats.total_time);
  printf("    Bias 1:         %6.3fs (%5.1f%%)\n", stats.bwd_bias1,
         100.0 * stats.bwd_bias1 / stats.total_time);
  printf("  Weight updates:   %6.3fs (%5.1f%%)\n", stats.weight_updates,
         100.0 * stats.weight_updates / stats.total_time);

  free(h_output);

  CUDA_CHECK(cudaFree(output));
  CUDA_CHECK(cudaFree(hidden));
  CUDA_CHECK(cudaFree(input));
  CUDA_CHECK(cudaFree(y_true));
}

int main() {
  srand(time(NULL));

  int train_size = 10000;
  int test_size = 10000;
  int in_dim = 784;
  int hidden_dim = 256;
  int num_classes = 10;
  int batch_size = 8;
  int epochs = 10;
  float lr = 0.01;

  NeuralNetwork nn;
  initialize_neural_network(&nn, in_dim, hidden_dim, num_classes);

  float *X_train = (float *)malloc(train_size * in_dim * sizeof(float));
  int *y_train = (int *)malloc(train_size * sizeof(int));
  float *X_test = (float *)malloc(test_size * in_dim * sizeof(float));
  int *y_test = (int *)malloc(test_size * sizeof(int));

  load_data("../data/X_train.bin", X_train, train_size * in_dim);
  normalize_data(X_train, train_size * in_dim);
  load_labels("../data/y_train.bin", y_train, train_size);
  load_data("../data/X_test.bin", X_test, test_size * in_dim);
  normalize_data(X_test, test_size * in_dim);
  load_labels("../data/y_test.bin", y_test, test_size);

  train_timed(&nn, X_train, y_train, epochs, batch_size, in_dim, hidden_dim,
              num_classes, train_size, lr);

  // Clean up device memory
  CUDA_CHECK(cudaFree(nn.w1));
  CUDA_CHECK(cudaFree(nn.w2));
  CUDA_CHECK(cudaFree(nn.b1));
  CUDA_CHECK(cudaFree(nn.b2));
  CUDA_CHECK(cudaFree(nn.grad_w1));
  CUDA_CHECK(cudaFree(nn.grad_w2));
  CUDA_CHECK(cudaFree(nn.grad_b1));
  CUDA_CHECK(cudaFree(nn.grad_b2));

  // Clean up host memory
  free(X_train);
  free(y_train);
  free(X_test);
  free(y_test);

  return 0;
}