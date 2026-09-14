#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef struct {
  double data_loading; // Time spent loading data
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

/**
 * Load binary labels from file
 * @param filename Path to binary file
 * @param labels Pointer to buffer to store labels
 * @param size Number of labels to read
 */
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
  nn->w1 = (float *)malloc(w1_bytes);
  initialize_weights(nn->w1, in_dim, hidden_dim);

  size_t b1_bytes = hidden_dim * sizeof(float);
  nn->b1 = (float *)malloc(b1_bytes);
  initialize_biases(nn->b1, hidden_dim);

  size_t w2_bytes = hidden_dim * out_dim * sizeof(float);
  nn->w2 = (float *)malloc(w2_bytes);
  initialize_weights(nn->w2, hidden_dim, out_dim);

  size_t b2_bytes = out_dim * sizeof(float);
  nn->b2 = (float *)malloc(b2_bytes);
  initialize_biases(nn->b2, out_dim);

  nn->grad_w1 = (float *)malloc(w1_bytes);
  nn->grad_b1 = (float *)malloc(b1_bytes);
  nn->grad_w2 = (float *)malloc(w2_bytes);
  nn->grad_b2 = (float *)malloc(b2_bytes);
}

//---------------------------------------------------
// FORWARD PASS
//---------------------------------------------------

void matmul_a_b(float *A, float *B, float *C, int m, int n, int k_shared) {
  // A: (m, k_shared) @ B: (k_shared, n) -> (m, n)
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      C[i * n + j] = 0.0f;
      for (int k = 0; k < k_shared; ++k) {
        C[i * n + j] += A[i * k_shared + k] * B[k * n + j];
      }
    }
  }
}

void bias_forward(float *x, float *bias, int batch_size, int dim) {
  // x: (batch_size, dim) + bias: (dim,)
  for (int b = 0; b < batch_size; ++b) {
    for (int d = 0; d < dim; ++d) {
      x[b * dim + d] += bias[d];
    }
  }
}

void relu_forward(float *x, int dim) {
  for (int d = 0; d < dim; ++d) {
    x[d] = fmax(0.0f, x[d]);
  }
}

void softmax(float *x, int row_dim, int col_dim) {

  for (int r = 0; r < row_dim; ++r) {

    float row_max = x[r * col_dim];
    for (int c = 0; c < col_dim; ++c) {
      int idx = r * col_dim + c;
      if (x[idx] > row_max) {
        row_max = x[idx];
      }
    }

    float row_sum = 0.0f;
    for (int c = 0; c < col_dim; ++c) {
      int idx = r * col_dim + c;
      x[idx] = expf(x[idx] - row_max);
      row_sum += x[idx];
    }

    for (int c = 0; c < col_dim; ++c) {
      int idx = r * col_dim + c;
      x[idx] = fmaxf(x[idx] / row_sum, 1e-7f);
    }
  }
}

void forward_timed(NeuralNetwork *nn, float *input, float *hidden,
                   float *output, int batch_size, int input_dim, int hidden_dim,
                   int output_dim, TimingStats *stats) {
  struct timespec start, end;

  clock_gettime(CLOCK_MONOTONIC, &start);
  matmul_a_b(input, nn->w1, hidden, batch_size, hidden_dim, input_dim);
  clock_gettime(CLOCK_MONOTONIC, &end);
  stats->fwd_matmul1 += get_time_diff(start, end);

  clock_gettime(CLOCK_MONOTONIC, &start);
  bias_forward(hidden, nn->b1, batch_size, hidden_dim);
  clock_gettime(CLOCK_MONOTONIC, &end);
  stats->fwd_bias1 += get_time_diff(start, end);

  clock_gettime(CLOCK_MONOTONIC, &start);
  relu_forward(hidden, batch_size * hidden_dim);
  clock_gettime(CLOCK_MONOTONIC, &end);
  stats->fwd_relu += get_time_diff(start, end);

  clock_gettime(CLOCK_MONOTONIC, &start);
  matmul_a_b(hidden, nn->w2, output, batch_size, output_dim, hidden_dim);
  clock_gettime(CLOCK_MONOTONIC, &end);
  stats->fwd_matmul2 += get_time_diff(start, end);

  clock_gettime(CLOCK_MONOTONIC, &start);
  bias_forward(output, nn->b2, batch_size, output_dim);
  clock_gettime(CLOCK_MONOTONIC, &end);
  stats->fwd_bias2 += get_time_diff(start, end);

  clock_gettime(CLOCK_MONOTONIC, &start);
  softmax(output, batch_size, output_dim);
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
void matmul_at_b(float *A, float *B, float *C, int m, int n, int k_shared) {
  // A: (k_shared, m) need to be transposed, such that
  // A^T: (m, k_shared) @ B: (k_shared, n) -> (m, n)
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      C[i * n + j] = 0.0f;
      for (int k = 0; k < k_shared; ++k) {
        C[i * n + j] += A[k * m + i] * B[k * n + j];
      }
    }
  }
}

void matmul_a_bt(float *A, float *B, float *C, int m, int n, int k_shared) {
  // B: (n, k_shared) need to be transposed, such that
  // A: (m, k_shared) @ B^T: (k_shared, n) -> (m, n)
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      C[i * n + j] = 0.0f;
      for (int k = 0; k < k_shared; ++k) {
        C[i * n + j] += A[i * k_shared + k] * B[j * k_shared + k];
      }
    }
  }
}

void relu_backward(float *grad, float *x, float *grad_out, int dim) {
  for (int i = 0; i < dim; ++i) {
    grad_out[i] = grad[i] * (x[i] > 0);
  }
}

void bias_backward(float *grad, float *grad_bias, int batch_size, int dim) {
  // grad: (batch_size, dim), grad_bias: (dim,)
  for (int d = 0; d < dim; ++d) {
    grad_bias[d] = 0.0f;
    for (int b = 0; b < batch_size; ++b) {
      grad_bias[d] += grad[b * dim + d];
    }
  }
}

void compute_output_gradients(float *probs, int *y_true, float *grad,
                              int batch_size, int num_classes) {
  for (int b = 0; b < batch_size; ++b) {
    for (int n = 0; n < num_classes; ++n) {
      int idx = b * num_classes + n;
      if (n == y_true[b]) {
        grad[idx] = (probs[idx] - 1.0f) / (float)batch_size;
      } else {
        grad[idx] = probs[idx] / (float)batch_size;
      }
    }
  }
}

void backward_timed(NeuralNetwork *nn, float *input, float *hidden,
                    float *output, int *labels, int batch_size, int input_dim,
                    int hidden_dim, int output_dim, TimingStats *stats) {
  struct timespec start, end;

  // calculate output gradients
  clock_gettime(CLOCK_MONOTONIC, &start);
  float *output_grad = (float *)malloc(batch_size * output_dim * sizeof(float));
  compute_output_gradients(output, labels, output_grad, batch_size, output_dim);
  clock_gettime(CLOCK_MONOTONIC, &end);
  stats->bwd_output_grad += get_time_diff(start, end);

  // calculate w2 gradients
  clock_gettime(CLOCK_MONOTONIC, &start);
  matmul_at_b(hidden, output_grad, nn->grad_w2, hidden_dim, output_dim,
              batch_size);
  clock_gettime(CLOCK_MONOTONIC, &end);
  stats->bwd_matmul2 += get_time_diff(start, end);

  // calculate b2 gradients
  clock_gettime(CLOCK_MONOTONIC, &start);
  bias_backward(output_grad, nn->grad_b2, batch_size, output_dim);
  clock_gettime(CLOCK_MONOTONIC, &end);
  stats->bwd_bias2 += get_time_diff(start, end);

  float *fc2_grad = (float *)malloc(batch_size * hidden_dim * sizeof(float));
  matmul_a_bt(output_grad, nn->w2, fc2_grad, batch_size, hidden_dim,
              output_dim);

  // calculate relu gradients
  clock_gettime(CLOCK_MONOTONIC, &start);
  float *relu_grad = (float *)malloc(hidden_dim * batch_size * sizeof(float));
  relu_backward(fc2_grad, hidden, relu_grad, hidden_dim * batch_size);
  clock_gettime(CLOCK_MONOTONIC, &end);
  stats->bwd_relu += get_time_diff(start, end);

  // calculate w1 gradients
  clock_gettime(CLOCK_MONOTONIC, &start);
  matmul_at_b(input, relu_grad, nn->grad_w1, input_dim, hidden_dim, batch_size);
  clock_gettime(CLOCK_MONOTONIC, &end);
  stats->bwd_matmul1 += get_time_diff(start, end);

  // calculate b1 gradients
  clock_gettime(CLOCK_MONOTONIC, &start);
  bias_backward(relu_grad, nn->grad_b1, batch_size, hidden_dim);
  clock_gettime(CLOCK_MONOTONIC, &end);
  stats->bwd_bias1 += get_time_diff(start, end);

  free(output_grad);
  free(fc2_grad);
  free(relu_grad);
}

//---------------------------------------------------
// WEIGHTS UPDATE
//---------------------------------------------------
void update_weights_timed(NeuralNetwork *nn, int input_dim, int hidden_dim,
                          int output_dim, float lr, TimingStats *stats) {
  struct timespec start, end;

  clock_gettime(CLOCK_MONOTONIC, &start);
  for (int i = 0; i < input_dim * hidden_dim; ++i) {
    nn->w1[i] -= lr * nn->grad_w1[i];
  }
  for (int i = 0; i < hidden_dim; ++i) {
    nn->b1[i] -= lr * nn->grad_b1[i];
  }
  for (int i = 0; i < output_dim * hidden_dim; ++i) {
    nn->w2[i] -= lr * nn->grad_w2[i];
  }
  for (int i = 0; i < output_dim; ++i) {
    nn->b2[i] -= lr * nn->grad_b2[i];
  }
  clock_gettime(CLOCK_MONOTONIC, &end);
  stats->weight_updates += get_time_diff(start, end);
}

void zero_grads(float *grads, int dim) {
  memset(grads, 0, dim * sizeof(float));
}

//---------------------------------------------------
// TRAINING
//---------------------------------------------------

int imin(int a, int b) { return a < b ? a : b; }

void train_timed(NeuralNetwork *nn, float *X_train, int *y_train, int epochs,
                 int batch_size, int input_dim, int hidden_dim, int num_classes,
                 int train_size, float lr) {
  struct timespec start, end, total_start, total_end;

  float *output = (float *)malloc(batch_size * num_classes * sizeof(float));
  float *hidden = (float *)malloc(batch_size * hidden_dim * sizeof(float));

  TimingStats stats = {0};
  int num_batches = (train_size + batch_size - 1) / batch_size;

  clock_gettime(CLOCK_MONOTONIC, &total_start);
  for (int e = 0; e < epochs; ++e) {
    float epoch_loss = 0.0f;
    for (int n = 0; n < num_batches; ++n) {

      clock_gettime(CLOCK_MONOTONIC, &start);
      float *input = &X_train[n * batch_size * input_dim];
      int *y_true = &y_train[n * batch_size];
      clock_gettime(CLOCK_MONOTONIC, &end);
      stats.data_loading += get_time_diff(start, end);

      int cur_batch_size = imin(abs(n * batch_size - train_size), batch_size);

      forward_timed(nn, input, hidden, output, cur_batch_size, input_dim,
                    hidden_dim, num_classes, &stats);
      clock_gettime(CLOCK_MONOTONIC, &start);
      epoch_loss +=
          cross_entropy_loss(output, y_true, cur_batch_size, num_classes);
      clock_gettime(CLOCK_MONOTONIC, &end);
      stats.cross_entropy += get_time_diff(start, end);

      backward_timed(nn, input, hidden, output, y_true, cur_batch_size,
                     input_dim, hidden_dim, num_classes, &stats);

      update_weights_timed(nn, input_dim, hidden_dim, num_classes, lr, &stats);

      zero_grads(nn->grad_w1, input_dim * hidden_dim);
      zero_grads(nn->grad_w2, num_classes * hidden_dim);
      zero_grads(nn->grad_b1, hidden_dim);
      zero_grads(nn->grad_b2, num_classes);
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

  free(hidden);
  free(output);
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

  free(nn.w1);
  free(nn.w2);
  free(nn.b1);
  free(nn.b2);
  free(nn.grad_w1);
  free(nn.grad_w2);
  free(nn.grad_b1);
  free(nn.grad_b2);
  free(X_train);
  free(y_train);
  free(X_test);
  free(y_test);

  return 0;
}