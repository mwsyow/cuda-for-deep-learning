#include "common.cuh"
#include "forward-pass-cpu.cuh"
#include "forward-pass-gpu.cuh"
#include "transformer.cuh"
#include <cassert>
#include <cuda_device_runtime_api.h>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <iomanip>
#include <iostream>
#include <mpi.h>
#include <nccl.h>
#include <vector>

void is_equal(float *H, half *D, uint size) {
  double max_abs_err = 0.0;
  double max_ref = 0.0;
  for (uint i = 0; i < size; i++) {
    float ref = H[i];
    double diff = std::fabs(ref - __half2float(D[i]));
    max_abs_err = std::fmax(max_abs_err, diff);
    max_ref = std::fmax(max_ref, std::fabs(ref));
  }
  const auto previous_flags = std::cout.flags();
  const auto previous_precision = std::cout.precision();
  std::cout << std::right << std::scientific << std::setprecision(4)
            << std::setw(14) << max_ref << std::setw(14) << max_abs_err;
  if (max_ref > 0.0)
    std::cout << std::fixed << std::setprecision(3) << std::setw(10)
              << 100.0 * max_abs_err / max_ref;
  else
    std::cout << std::setw(10) << "n/a";
  std::cout << "  " << (max_abs_err > 0.01 * max_ref ? "FAIL" : "PASS") << '\n';
  std::cout.flags(previous_flags);
  std::cout.precision(previous_precision);
}

void is_transformer_equal(TransformerBlock<float> ref_transformer,
                          TransformerBlock<half> res_transformer, float *ref,
                          half *res, uint seq_len) {
  auto title = [](const char *name) {
    std::cout << "  " << std::left << std::setw(30) << name;
  };
  std::cout << "\n=== Correctness: GPU FP16 vs CPU FP32 ===\n"
            << "Criterion: max absolute error <= 1% of max |reference|\n\n";
  title("Stage");
  std::cout << std::right << std::setw(14) << "Max |ref|" << std::setw(14)
            << "Max abs error" << std::setw(10) << "Error (%)"
            << "  Result\n"
            << "  "
               "---------------------------------------------------------------"
               "-----------\n"
            << "Attention\n";
  title("Query projection (XQ)");
  is_equal(ref_transformer.salayer->X_q, res_transformer.salayer->X_q,
           seq_len * ref_transformer.sa_hidden_dim);
  title("Key projection (XK)");
  is_equal(ref_transformer.salayer->X_k, res_transformer.salayer->X_k,
           seq_len * ref_transformer.sa_hidden_dim);
  title("Value projection (XV)");
  is_equal(ref_transformer.salayer->X_v, res_transformer.salayer->X_v,
           seq_len * ref_transformer.sa_hidden_dim);
  title("Scaled attention logits");
  is_equal(ref_transformer.salayer->attention_scores,
           res_transformer.salayer->attention_scores, seq_len * seq_len);
  title("Softmax probabilities");
  is_equal(ref_transformer.salayer->attention_weights,
           res_transformer.salayer->attention_weights, seq_len * seq_len);
  title("Attention context (AV)");
  is_equal(ref_transformer.salayer->Z, res_transformer.salayer->Z,
           seq_len * ref_transformer.sa_hidden_dim);
  title("Attention output projection");
  is_equal(ref_transformer.salayer->Res, res_transformer.salayer->Res,
           seq_len * ref_transformer.sa_hidden_dim);

  std::cout << "\nFeed-forward\n";
  title("Hidden expansion (W1)");
  is_equal(ref_transformer.fflayer->X_w1, res_transformer.fflayer->X_w1,
           seq_len * ref_transformer.ff_hidden_dim);
  title("ReLU activation");
  is_equal(ref_transformer.fflayer->X_relu, res_transformer.fflayer->X_relu,
           seq_len * ref_transformer.ff_hidden_dim);

  std::cout << "\nTransformer block\n";
  title("Final output");
  is_equal(ref, res, seq_len * ref_transformer.in_dim);
}

void print_time_statistics(const TimeStats &s) {
  const auto previous_flags = std::cout.flags();
  const auto previous_precision = std::cout.precision();
  auto print = [&](const char *name, float ms) {
    std::cout << "  " << std::left << std::setw(30) << name << std::right
              << std::setw(12) << ms << " ms\n";
  };

  std::cout << "\n=== GPU timing breakdown ===\n"
            << "Single diagnostic run; synchronization after each stage.\n\n"
            << std::fixed << std::setprecision(3) << "Attention\n";
  print("Query projection (XQ)", s.X_q);
  print("Key projection (XK)", s.X_k);
  print("Value projection (XV)", s.X_v);
  print("Scaled attention logits", s.attention_scores);
  print("Softmax probabilities", s.attention_weights);
  print("Attention context (AV)", s.Z);
  print("Attention output projection", s.attention_result);
  print("Attention subtotal", s.salayer);

  std::cout << "\nFeed-forward\n";
  print("Hidden expansion (W1)", s.X_w1);
  print("ReLU activation", s.X_relu);
  print("Output projection (W2)", s.X_w2);
  print("Feed-forward subtotal", s.fflayer);
  std::cout << "  ---------------------------------------------\n";
  print("Full block total", s.total);
  std::cout.flags(previous_flags);
  std::cout.precision(previous_precision);
}

int main(int argc, char *argv[]) {
  MPI_Init(&argc, &argv);

  int rank, world_size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  int device_count;
  CUDA_CHECK(cudaGetDeviceCount(&device_count));
  if (world_size > device_count) {
    std::cerr << "World size " << world_size
              << "cannot exceed number of device " << device_count << std::endl;
    MPI_Finalize();
    return 1;
  }

  ncclUniqueId ncclCommId;
  if (rank == 0)
    NCCL_CHECK(ncclGetUniqueId(&ncclCommId));
  MPI_Bcast(&ncclCommId, sizeof(ncclCommId), MPI_BYTE, 0, MPI_COMM_WORLD);

  ncclComm_t ncclComm;
  NCCL_CHECK(ncclCommInitRank(&ncclComm, world_size, ncclCommId, rank));

  TransformerBlock<float> ref_transformer;
  TransformerBlock<half> res_transformer;
  TransformerBlock<half> device_transformer;

  ulong xseed = 7;
  ulong wseed = 42;
  uint in_dim = 128;
  uint hidden_dim = 128;
  uint seq_len = 64;

  std::vector<float> h_X(seq_len * in_dim);
  std::vector<half> h_X_half(seq_len * in_dim);
  std::vector<float> h_Ref(seq_len * in_dim);
  std::vector<half> h_Res(seq_len * in_dim);

  std::cout << "Initializing and allocating weights and caches on CPU"
            << std::endl;
  init_transformer_block(ref_transformer, in_dim, hidden_dim, seq_len, wseed);
  std::cout << "Transferring weights and allocating GPU caches" << std::endl;
  to_device(ref_transformer, device_transformer, seq_len);

  std::cout << "Creating input and context" << std::endl;
  std::mt19937 gen(xseed);
  std::uniform_real_distribution<float> distribution(-1.0f, 1.0f);

  for (auto &v : h_X)
    v = distribution(gen);

  for (int i = 0; i < h_X.size(); i++) {
    h_X_half[i] = __float2half(h_X[i]);
  }

  cublasHandle_t cublasHandle;
  cudaStream_t stream;
  CUBLAS_CHECK(cublasCreate(&cublasHandle));
  CUDA_CHECK(cudaStreamCreate(&stream));
  CUBLAS_CHECK(cublasSetStream(cublasHandle, stream));

  DeviceContext context{stream, cublasHandle};
  TimeStats stats{};

  half *d_X, *d_Res;
  CUDA_CHECK(cudaMalloc(&d_X, seq_len * in_dim * sizeof(half)));
  CUDA_CHECK(cudaMalloc(&d_Res, seq_len * in_dim * sizeof(half)));

  CUDA_CHECK(cudaMemcpy(d_X, h_X_half.data(), h_X_half.size() * sizeof(half),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_Res, h_Res.data(), h_Res.size() * sizeof(half),
                        cudaMemcpyHostToDevice));

  std::cout << "Running transformer on CPU for reference" << std::endl;
  run_transformer_block_cpu(ref_transformer, h_X.data(), h_Ref.data(), seq_len);

  std::cout << "Running transformer on GPU" << std::endl;
  run_transformer_block_gpu(context, device_transformer, d_X, d_Res, seq_len,
                            stats);

  std::cout
      << "Copying weights and caches from GPU to CPU for correctness check"
      << std::endl;
  CUDA_CHECK(cudaMemcpy(h_Res.data(), d_Res, h_Res.size() * sizeof(half),
                        cudaMemcpyDeviceToHost));
  to_host(device_transformer, res_transformer, seq_len);
  is_transformer_equal(ref_transformer, res_transformer, h_Ref.data(),
                       h_Res.data(), seq_len);
  print_time_statistics(stats);

  std::cout << "\nReleasing model and CUDA resources..." << std::endl;
  free_device_transformer_block(device_transformer);
  free_host_transformer_block(ref_transformer);
  free_host_transformer_block(res_transformer);

  CUBLAS_CHECK(cublasDestroy(cublasHandle));
  CUDA_CHECK(cudaStreamDestroy(stream));
  CUDA_CHECK(cudaFree(d_X));
  CUDA_CHECK(cudaFree(d_Res));

  NCCL_CHECK(ncclCommDestroy(ncclComm));
  MPI_Finalize();

  std::cout << "Run finished" << std::endl;
}
