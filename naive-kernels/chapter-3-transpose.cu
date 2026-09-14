// transpose a matrix
// transpose operation is turning row-major to column-major in memory or vice
// versa
#include <cstdio>
#include <cstdlib>

void transpose_cpu(float *in, float *out, int num_rows, int num_columns) {
  for (int c{0}; c < num_columns; ++c) {
    for (int r{0}; r < num_rows; ++r) {
      out[c * num_rows + r] = in[r * num_columns + c];
    }
  }
}
__global__ void transpose_gpu(float *in, float *out, int num_rows,
                              int num_columns) {
  int x_dim = blockIdx.x * blockDim.x + threadIdx.x;
  int y_dim = blockIdx.y * blockDim.y + threadIdx.y;

  if (x_dim < num_rows && y_dim < num_columns) {
    out[y_dim * num_rows + x_dim] = in[x_dim * num_columns + y_dim];
  }
}

int main() {
  //------------------------------------------------------
  // INITIALIZATION
  //------------------------------------------------------
  // initialize for 2D case + batch for Transpose
  int num_columns{1000};
  int num_rows{500};
  int batch_size{10};
  int n = num_columns * num_rows * batch_size;

  // size of vectors
  // float is 4 byte times the number of element 8 -> 32 bytes per vector
  size_t bytes = n * sizeof(float);

  // allocate memory on CPU
  // malloc returns void * hence needs to be cast to float *
  // h_ convention for host
  float *h_a = (float *)malloc(bytes);
  float *h_b = (float *)malloc(bytes);
  float *h_c = (float *)malloc(bytes);
  float *h_c_true = (float *)malloc(bytes);

  for (int i = 0; i < n; ++i) {
    h_a[i] = (float)i;       // h_a = [0, 1, 2, 3, 4, 5, 6, 7]
    h_b[i] = (float)(i * 2); // h_a = [0, 2, 4, 6, 8, 10, 12, 14]
  }

  // allocate memory on GPU
  // cudaMalloc accepts pointer to void pointer
  // d_ convention for device
  float *d_a{};
  float *d_b{};
  float *d_c{};
  cudaMalloc((void **)&d_a, bytes);
  cudaMalloc((void **)&d_b, bytes);
  cudaMalloc((void **)&d_c, bytes);

  // copy memory from CPU to GPU
  cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

  //------------------------------------------------------
  // DIFFERENT KERNEL FUNCTIONS
  //------------------------------------------------------
  // Transpose
  dim3 threadsPerBlock(16, 16);
  dim3 blocksPerGrid((num_rows + threadsPerBlock.x - 1) / threadsPerBlock.x,
                     (num_columns + threadsPerBlock.y - 1) / threadsPerBlock.y);
  for (int b{0}; b < batch_size; ++b) {
    float *d_a_pb = d_a + (b * num_rows * num_columns);
    float *d_c_pb = d_c + (b * num_rows * num_columns);
    float *h_a_pb = h_a + (b * num_rows * num_columns);
    float *h_c_true_pb = h_c_true + (b * num_rows * num_columns);
    // GPU version
    transpose_gpu<<<blocksPerGrid, threadsPerBlock>>>(d_a_pb, d_c_pb, num_rows,
                                                      num_columns);
    // CPU version
    transpose_cpu(h_a_pb, h_c_true_pb, num_rows, num_columns);
  }
  //------------------------------------------------------
  // KERNEL TEST + CLEANUP
  //------------------------------------------------------

  // copy memory from GPU to CPU
  cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);

  // check whether computation is correct
  int success{1};
  float tolerance{1e-6f};
  for (int i{0}; i < n; i++) {
    float diff = fabsf(h_c[i] - h_c_true[i]);
    if (diff > tolerance) {
      printf("Error at index %d: Got %f, expected %f\n", i, h_c[i],
             h_c_true[i]);
      success = 0;
      break;
    };
  };

  if (success) {
    printf("All elements are correct.");
  };

  // free memory on CPU
  free(h_a);
  free(h_b);
  free(h_c);
  free(h_c_true);

  // free memory on GPU
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);
}