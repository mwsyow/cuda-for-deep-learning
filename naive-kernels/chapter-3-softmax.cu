// transpose a matrix
// transpose operation is turning row-major to column-major in memory or vice
// versa
#include <cmath>
#include <cstdio>
#include <cstdlib>

void softmax_cpu(const float *in, float *out, int num_rows, int num_columns) {
  for (int row{0}; row < num_rows; ++row) {
    // Calculation is done in 4 steps

    // STEP 1: find max for numerical stability
    // such that exponent function doesn't blow up
    float max = in[row * num_columns];
    for (int col{0}; col < num_columns; ++col) {
      float cur = in[row * num_columns + col];
      if (cur > max) {
        max = cur;
      }
    }

    // STEP 2 & 3: compute exp(in - max) & sum exponential
    float sum{0};
    for (int col{0}; col < num_columns; ++col) {
      int idx = row * num_columns + col;
      out[idx] = expf(in[idx] - max);
      sum = sum + out[idx];
    }

    // STEP 4: Normalize
    for (int col{0}; col < num_columns; ++col) {
      out[row * num_columns + col] = out[row * num_columns + col] / sum;
    }
  }
}
__global__ void softmax_gpu(const float *in, float *out, int num_rows,
                            int num_columns) {
  int row_dim = blockIdx.x * blockDim.x + threadIdx.x;
  int col_dim = blockIdx.y * blockDim.y + threadIdx.y;

  if (row_dim < num_rows && col_dim < num_columns) {
    // STEP 1: find max for numerical stability
    // redundant work for each thread working on the same row for computing max
    float max = -1e-20f; // if use in[...] then you're accessing memory which
                         // makes it even more slow
    for (int col{0}; col < num_columns; ++col) {
      float cur = in[row_dim * num_columns + col];
      if (cur > max) {
        max = cur;
      }
    }

    // STEP 2: compute sum exponential
    // redundant work for each thread working on the same row for computing sum
    // of exponential
    float sum{0};
    for (int col{0}; col < num_columns; ++col) {
      sum = sum + expf(in[row_dim * num_columns + col] - max);
    }

    // STEP 3: normalize
    int idx = row_dim * num_columns + col_dim;
    out[idx] = expf(in[idx] - max) / sum;
  }
}

int main() {
  //------------------------------------------------------
  // INITIALIZATION
  //------------------------------------------------------
  // initialize for 2D case
  int num_columns{10};
  int num_rows{10};
  int n = num_columns * num_rows;

  // size of vectors
  // float is 4 byte times the number of element 8 -> 32 bytes per vector
  size_t bytes = n * sizeof(float);

  // allocate memory on CPU
  // malloc returns void * hence needs to be cast to float *
  // h_ convention for host
  float *h_a = (float *)malloc(bytes);
  float *h_c_gpu = (float *)malloc(bytes);
  float *h_c_cpu = (float *)malloc(bytes);

  for (int i = 0; i < n; ++i) {
    h_a[i] = (float)i; // h_a = [0, 1, 2, 3, 4, 5, 6, 7]
  }

  // allocate memory on GPU
  // cudaMalloc accepts pointer to void pointer
  // d_ convention for device
  float *d_a{};
  float *d_c{};
  cudaMalloc((void **)&d_a, bytes);
  cudaMalloc((void **)&d_c, bytes);

  // copy memory from CPU to GPU
  cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);

  //------------------------------------------------------
  // DIFFERENT KERNEL FUNCTIONS
  //------------------------------------------------------
  // Transpose
  dim3 threadsPerBlock(16, 16);
  dim3 blocksPerGrid((num_rows + threadsPerBlock.x - 1) / threadsPerBlock.x,
                     (num_columns + threadsPerBlock.y - 1) / threadsPerBlock.y);
  // GPU version
  softmax_gpu<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_c, num_rows,
                                                  num_columns);
  // CPU version
  softmax_cpu(h_a, h_c_cpu, num_rows, num_columns);
  //------------------------------------------------------
  // KERNEL TEST + CLEANUP
  //------------------------------------------------------

  // copy memory from GPU to CPU
  cudaMemcpy(h_c_gpu, d_c, bytes, cudaMemcpyDeviceToHost);

  // check whether computation is correct
  int success{1};
  float tolerance{1e-6f};
  for (int i{0}; i < n; i++) {
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