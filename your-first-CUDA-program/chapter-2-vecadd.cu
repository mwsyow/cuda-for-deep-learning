#include <__clang_cuda_builtin_vars.h>
#include <cstddef>
#include <cstdio>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>

// kernel function indicated by __global__ identifier
// kernel function has void return type
// a, b and c pointing memory on GPU, so memory is already allocated by this
// time this function gets called
// #2 is the current thread id running the function
// #3 each element of vector a and b addition is run parallely, one
// thread per elements pair
__global__ void vecAdd(float *a, float *b, float *c) { // #1
  int i = threadIdx.x;                                 // #2
  c[i] = a[i] + b[i];                                  // #3
}

// vecAdd for 1D case, similar to above
// however this version is safer such that if we booked more threads than needed
// then the surplus threads won't corrupt the calculations of needed threads
__global__ void vecAdd(float *a, float *b, float *c, int n) {
  // this calculate the global index for the thread in 1D case
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < n) {
    c[i] = a[i] + b[i];
  }
}

// simple calculation of 3D tensor addition in CPU
void vecAdd3D_cpu(const float *a, const float *b, float *c, int width,
                  int height, int depth) {
  for (int w{0}; w < width; ++w) {
    for (int h{0}; h < height; ++h) {
      for (int d{0}; d < depth; ++d) {
        int index = w * (depth * height) + h * (depth) + d;
        c[index] = a[index] + b[index];
      }
    }
  }
}

// calculation of 3D tensor addition in GPU
// index calculation is similar to its CPU counterpart
// called row-major order where you turn nD indexing to 1D indexing
__global__ void vecAdd3D_gpu(const float *a, const float *b, float *c,
                             int width, int height, int depth) {
  int x_dim = blockIdx.x * blockDim.x + threadIdx.x;
  int y_dim = blockIdx.y * blockDim.y + threadIdx.y;
  int z_dim = blockIdx.z * blockDim.z + threadIdx.z;

  if (x_dim < width && y_dim < height && z_dim < depth) {
    int index = x_dim * (height * depth) + y_dim * (depth) + z_dim;
    c[index] = a[index] + b[index];
  }
}

// calculation for real use-case example
// General Rules: works for nD dimensions as shown in this example
// 1. compute global index w.r.t. each dimension
// 2. do the boudary check
// 3. Linearize the nD coordinates into a row-major 1D memory index
// 4. perform your computation
__global__ void normalizeImageBatch(const float *input, float *output,
                                    float mean, float std, int batch_size,
                                    int width, int height, int depth) {
  int x_dim = threadIdx.x * blockDim.x + threadIdx.x;
  int y_dim = threadIdx.y * blockDim.y + threadIdx.y;
  int z_dim = threadIdx.z * blockDim.z + threadIdx.z;

  if (x_dim < width && y_dim < height && z_dim < depth) {
    for (int b{0}; b < batch_size; ++b) {
      int index = b * (width * depth * height) + x_dim * (depth * height) +
                  y_dim * (depth) + z_dim;
      output[index] = (input[index] + mean) / std;
    }
  }
}

int main() {
  // initialize number of elements
  //   int n = 1000000;
  // initialize for 3D case
  int width{256};
  int height{256};
  int depth{32};
  int n = width * height * depth;

  // size of vectors
  // float is 4 byte times the number of element 8 -> 32 bytes per vector
  size_t bytes = n * sizeof(float);

  // allocate memory on CPU
  // malloc returns void * hence needs to be cast to float *
  // h_ convention for host
  float *h_a = (float *)malloc(bytes);
  float *h_b = (float *)malloc(bytes);
  float *h_c = (float *)malloc(bytes);

  for (int i = 0; i < n; i++) {
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
  // DIFFERENT VERSIONS OF KERNEL LAUNCH
  //------------------------------------------------------
  // launch the kernel
  // 8 means launch 8 threads
  //   vecAdd<<<1, 8>>>(d_a, d_b, d_c);

  // this uses the safer version
  //   vecAdd<<<1, 8>>>(d_a, d_b, d_c, n);

  // threads per block is only capped at 1024
  // calculate number of blocks as number of elements exceed number of threads
  // per block
  //   int threadsPerBlock = 256;
  //   int blocksPerGrid = ceil((double)n / threadsPerBlock);
  //   vecAdd<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, n);

  // 3D case
  dim3 threadsPerBlock(8, 8, 8);
  dim3 blocksPerGrid(ceil((double)width / threadsPerBlock.x),
                     ceil((double)height / threadsPerBlock.y),
                     ceil((double)depth / threadsPerBlock.z));
  vecAdd3D_gpu<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, width, height,
                                                   depth);
  //------------------------------------------------------

  // copy memory from GPU to CPU
  cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);

  // check whether computation is correct
  int success{1};
  for (int i{0}; i < n; i++) {
    if (h_c[i] != h_a[i] + h_b[i]) {
      printf("Error at index %d: Got %f, expected %f\n", i, h_c[i],
             h_a[i] + h_b[i]);
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

  // free memory on GPU
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);
}
