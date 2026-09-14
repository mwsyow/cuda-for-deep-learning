#include <cstdio>
#include <cstdlib>

void gemm_cpu(const float *A, const float *B, float *C, int M_rows, int N_cols,
              int K_shared_dim) {

  for (int m{0}; m < M_rows; ++m) {
    for (int n{0}; n < N_cols; ++n) {
      float sum{0};
      for (int k{0}; k < K_shared_dim; ++k) {
        sum = sum + (A[(m * K_shared_dim) + k] * B[(k * N_cols) + n]);
      }
      C[(m * N_cols) + n] = sum;
    }
  }
}

__global__ void gemm_gpu(const float *A, const float *B, float *C, int M_rows,
                         int N_cols, int K_shared_dim) {
  int m = blockIdx.x * blockDim.x + threadIdx.x;
  int n = blockIdx.y * blockDim.y + threadIdx.y;

  if (m < M_rows && n < N_cols) {
    float sum{0};
    for (int k{0}; k < K_shared_dim; ++k) {
      sum = sum + A[m * K_shared_dim + k] * B[k * N_cols + n];
    }
    C[m * N_cols + n] = sum;
  }
}

int main() {
  //------------------------------------------------------
  // INITIALIZATION
  //------------------------------------------------------
  int M_rows{1024};
  int N_cols{512};
  int K_shared_dim{256};
  int n_A = M_rows * K_shared_dim;
  int n_B = K_shared_dim * N_cols;
  int n_C = M_rows * N_cols;
  // initialize for 2D case for GEMM

  // size of vectors
  // float is 4 byte times the number of element 8 -> 32 bytes per vector
  size_t bytes_A = n_A * sizeof(float);
  size_t bytes_B = n_B * sizeof(float);
  size_t bytes_C = n_C * sizeof(float);

  // allocate memory on CPU
  // malloc returns void * hence needs to be cast to float *
  // h_ convention for host
  float *h_a = (float *)malloc(bytes_A);
  float *h_b = (float *)malloc(bytes_B);
  float *h_c_gpu = (float *)malloc(bytes_C);
  float *h_c_cpu = (float *)malloc(bytes_C);

  for (int i = 0; i < n_A; ++i) {
    h_a[i] = (float)i; // h_a = [0, 1, 2, 3, 4, 5, 6, 7]
  }
  for (int i = 0; i < n_B; ++i) {
    h_b[i] = (float)(i * 2); // h_a = [0, 2, 4, 6, 8, 10, 12, 14]
  }

  // allocate memory on GPU
  // cudaMalloc accepts pointer to void pointer
  // d_ convention for device
  float *d_a{};
  float *d_b{};
  float *d_c{};
  cudaMalloc((void **)&d_a, bytes_A);
  cudaMalloc((void **)&d_b, bytes_B);
  cudaMalloc((void **)&d_c, bytes_C);

  // copy memory from CPU to GPU
  cudaMemcpy(d_a, h_a, bytes_A, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, h_b, bytes_B, cudaMemcpyHostToDevice);

  //------------------------------------------------------
  // DIFFERENT KERNEL FUNCTIONS
  //------------------------------------------------------
  // Transpose
  dim3 threadsPerBlock(16, 16);
  dim3 blocksPerGrid((M_rows + threadsPerBlock.x - 1) / threadsPerBlock.x,
                     (N_cols + threadsPerBlock.y - 1) / threadsPerBlock.y);
  gemm_gpu<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, M_rows, N_cols,
                                               K_shared_dim);
  gemm_cpu(h_a, h_b, h_c_cpu, M_rows, N_cols, K_shared_dim);
  //------------------------------------------------------
  // KERNEL TEST + CLEANUP
  //------------------------------------------------------

  // copy memory from GPU to CPU
  cudaMemcpy(h_c_gpu, d_c, bytes_C, cudaMemcpyDeviceToHost);

  // check whether computation is correct
  int success{1};
  float tolerance{1e-6f};
  for (int i{0}; i < n_C; i++) {
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
  free(h_b);
  free(h_c_gpu);
  free(h_c_cpu);

  // free memory on GPU
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);
}