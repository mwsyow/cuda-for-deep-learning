#include <cstdio>
#include <cstdlib>

void maxpool2d_cpu(const float *in, float *out, int input_rows, int input_cols,
                   int pool_dim) {

  // int output_rows = input_rows - pool_dim + 1;
  // int output_cols = input_cols - pool_dim + 1;
  // for stride = pool_dim, according to the book
  int output_rows = input_rows / pool_dim;
  int output_cols = input_cols / pool_dim;

  for (int row{0}; row < output_rows; ++row) {
    for (int col{0}; col < output_cols; ++col) {
      float max{1e-20f};
      for (int k_row{0}; k_row < pool_dim; ++k_row) {
        for (int k_col{0}; k_col < pool_dim; ++k_col) {
          // int idx = (row + k_row) * input_cols + (col + k_col);
          // for stride = pool_dim, according to the book
          int idx =
              (row * pool_dim + k_row) * input_cols + (col * pool_dim + k_col);
          if (in[idx] > max) {
            max = in[idx];
          }
        }
      }
      out[row * output_cols + col] = max;
    }
  }
}

__global__ void maxpool2d_gpu(const float *in, float *out, int input_rows,
                              int input_cols, int pool_dim) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  int col = blockIdx.y * blockDim.y + threadIdx.y;

  // int output_rows = input_rows - pool_dim + 1;
  // int output_cols = input_cols - pool_dim + 1;
  // for stride = pool_dim, according to the book
  int output_rows = input_rows / pool_dim;
  int output_cols = input_cols / pool_dim;

  if (row < output_rows && col < output_cols) {
    float max{1e-20f};
    for (int k_row{0}; k_row < pool_dim; ++k_row) {
      for (int k_col{0}; k_col < pool_dim; ++k_col) {
        // int idx = (row + k_row) * input_cols + (col + k_col);
        // for stride = pool_dim, according to the book
        int idx =
            (row * pool_dim + k_row) * input_cols + (col * pool_dim + k_col);
        if (in[idx] > max) {
          max = in[idx];
        }
      }
    }
    out[row * output_cols + col] = max;
  }
}

int main() {
  //------------------------------------------------------
  // INITIALIZATION
  //------------------------------------------------------
  int input_rows{256};
  int input_cols{256};
  int pool_dim{2};
  // int output_rows = input_rows - pool_dim + 1;
  // int output_cols = input_cols - pool_dim + 1;
  // for stride = pool_dim, according to the book
  int output_rows = input_rows / pool_dim;
  int output_cols = input_cols / pool_dim;

  int n_in = input_rows * input_cols;
  int n_out = output_rows * output_cols;

  // size of vectors
  // float is 4 byte times the number of element 8 -> 32 bytes per vector
  size_t bytes_in = n_in * sizeof(float);
  size_t bytes_out = n_out * sizeof(float);

  // allocate memory on CPU
  // malloc returns void * hence needs to be cast to float *
  // h_ convention for host
  float *h_a = (float *)malloc(bytes_in);
  float *h_c_gpu = (float *)malloc(bytes_out);
  float *h_c_cpu = (float *)malloc(bytes_out);

  float max_a = n_in;
  for (int i = 0; i < n_in; ++i) {
    h_a[i] = (float)i / max_a; // h_a = [0, 1, 2, 3, 4, 5, 6, 7]
  }

  // allocate memory on GPU
  // cudaMalloc accepts pointer to void pointer
  // d_ convention for device
  float *d_a{};
  float *d_c{};
  cudaMalloc((void **)&d_a, bytes_in);
  cudaMalloc((void **)&d_c, bytes_out);

  // copy memory from CPU to GPU
  cudaMemcpy(d_a, h_a, bytes_in, cudaMemcpyHostToDevice);

  //------------------------------------------------------
  // DIFFERENT KERNEL FUNCTIONS
  //------------------------------------------------------
  // Transpose
  dim3 threadsPerBlock(16, 16);
  dim3 blocksPerGrid((output_rows + threadsPerBlock.x - 1) / threadsPerBlock.x,
                     (output_cols + threadsPerBlock.y - 1) / threadsPerBlock.y);
  maxpool2d_gpu<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_c, input_rows,
                                                    input_cols, pool_dim);
  maxpool2d_cpu(h_a, h_c_cpu, input_rows, input_cols, pool_dim);
  //------------------------------------------------------
  // KERNEL TEST + CLEANUP
  //------------------------------------------------------

  // copy memory from GPU to CPU
  cudaMemcpy(h_c_gpu, d_c, bytes_out, cudaMemcpyDeviceToHost);

  // check whether computation is correct
  int success{1};
  float tolerance{1e-5f};
  for (int i{0}; i < n_out; i++) {
    float diff = fabsf(h_c_gpu[i] - h_c_cpu[i]);
    if (diff > tolerance) {
      printf("Error at index %d: Got %f, expected %f\n", i, h_c_gpu[i],
             h_c_cpu[i]);
      success = 0;
      break;
    };
  };

  if (success) {
    printf("All elements are correct.");
  };

  // free memory on CPU
  free(h_a);
  free(h_c_gpu);
  free(h_c_cpu);

  // free memory on GPU
  cudaFree(d_a);
  cudaFree(d_c);
}