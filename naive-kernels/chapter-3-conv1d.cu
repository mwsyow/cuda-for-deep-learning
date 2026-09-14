#include <cstdio>
#include <cstdlib>

void conv1d_cpu(const float *in, float *out, const float *kernel,
                int input_size, int kernel_size) {

  int output_size = input_size - kernel_size + 1;
  for (int o{0}; o < output_size; ++o) {
    float sum{0};
    for (int k{0}; k < kernel_size; ++k) {
      sum = sum + in[o + k] * kernel[k];
    }
    out[o] = sum;
  }
}

__global__ void conv1d_gpu(const float *in, float *out, const float *kernel,
                           int input_size, int kernel_size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int output_size = input_size - kernel_size + 1;
  if (idx < output_size) {
    float sum{0};
    for (int k{0}; k < kernel_size; ++k) {
      sum = sum + in[idx + k] * kernel[k];
    }
    out[idx] = sum;
  }
}

int main() {
  //------------------------------------------------------
  // INITIALIZATION
  //------------------------------------------------------
  int input_size{10000000};
  int kernel_size{32};
  int output_size = input_size - kernel_size + 1;
  int n_in = input_size;
  int n_k = kernel_size;
  int n_out = output_size;

  // size of vectors
  // float is 4 byte times the number of element 8 -> 32 bytes per vector
  size_t bytes_in = n_in * sizeof(float);
  size_t bytes_k = n_k * sizeof(float);
  size_t bytes_out = n_out * sizeof(float);

  // allocate memory on CPU
  // malloc returns void * hence needs to be cast to float *
  // h_ convention for host
  float *h_a = (float *)malloc(bytes_in);
  float *h_b = (float *)malloc(bytes_k);
  float *h_c_gpu = (float *)malloc(bytes_out);
  float *h_c_cpu = (float *)malloc(bytes_out);

  for (int i = 0; i < n_in; ++i) {
    h_a[i] = (float)i; // h_a = [0, 1, 2, 3, 4, 5, 6, 7]
  }
  for (int i = 0; i < n_k; ++i) {
    h_b[i] = (float)(i * 2); // h_a = [0, 2, 4, 6, 8, 10, 12, 14]
  }

  // allocate memory on GPU
  // cudaMalloc accepts pointer to void pointer
  // d_ convention for device
  float *d_a{};
  float *d_b{};
  float *d_c{};
  cudaMalloc((void **)&d_a, bytes_in);
  cudaMalloc((void **)&d_b, bytes_k);
  cudaMalloc((void **)&d_c, bytes_out);

  // copy memory from CPU to GPU
  cudaMemcpy(d_a, h_a, bytes_in, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, h_b, bytes_k, cudaMemcpyHostToDevice);

  //------------------------------------------------------
  // DIFFERENT KERNEL FUNCTIONS
  //------------------------------------------------------
  // Transpose
  dim3 threadsPerBlock(256);
  dim3 blocksPerGrid((input_size + threadsPerBlock.x - 1) / threadsPerBlock.x);
  conv1d_gpu<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_c, d_b, input_size,
                                                 kernel_size);
  conv1d_cpu(h_a, h_c_cpu, h_b, input_size, kernel_size);
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
  free(h_b);
  free(h_c_gpu);
  free(h_c_cpu);

  // free memory on GPU
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);
}