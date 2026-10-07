#include "common.cuh"
#include "correctness_check.cuh"
#include "forward-pass-cpu.cuh"
#include "forward-pass-gpu.cuh"
#include "transformer.cuh"
#include <cuda_device_runtime_api.h>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <iostream>
#include <mpi.h>
#include <nccl.h>
#include <vector>

int main(int argc, char *argv[]) {
  //-------------------------------------------------------------------------------
  // MPI and NCCL startup
  //-------------------------------------------------------------------------------
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

  // Assuming this program is run on a single node
  CUDA_CHECK(cudaSetDevice(rank));

  //-------------------------------------------------------------------------------
  // Initialization
  //-------------------------------------------------------------------------------
  TimeStats single_stats{};
  TimeStats multi_stats{};

  TransformerBlock<float> ref_transformer;
  TransformerBlock<half> res_transformer;
  TransformerBlock<half> device_transformer;
  TransformerBlock<half> shard_transformer;

  ulong xseed = 7;
  ulong wseed = 42;
  uint in_dim = 128;
  uint hidden_dim = 128;
  uint seq_len = 64;

  std::vector<float> h_X(seq_len * in_dim);
  std::vector<half> h_X_half(seq_len * in_dim);
  std::vector<float> h_Ref(seq_len * in_dim);
  std::vector<half> h_Res(seq_len * in_dim);

  cublasHandle_t cublasHandle;
  cudaStream_t stream;
  CUBLAS_CHECK(cublasCreate(&cublasHandle));
  CUDA_CHECK(cudaStreamCreate(&stream));
  CUBLAS_CHECK(cublasSetStream(cublasHandle, stream));
  DeviceContext context{stream, cublasHandle};

  if (rank == 0) {
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
  }
  scatter_transformer(device_transformer, shard_transformer, world_size, rank,
                      seq_len, ncclComm, stream);

  half *d_X, *d_Res;
  CUDA_CHECK(cudaMalloc(&d_X, seq_len * in_dim * sizeof(half)));
  CUDA_CHECK(cudaMalloc(&d_Res, seq_len * in_dim * sizeof(half)));

  if (rank == 0) {
    CUDA_CHECK(cudaMemcpy(d_X, h_X_half.data(), h_X_half.size() * sizeof(half),
                          cudaMemcpyHostToDevice));
  }
  NCCL_CHECK(
      ncclBroadcast(d_X, d_X, seq_len * in_dim, ncclHalf, 0, ncclComm, stream));

  //-------------------------------------------------------------------------------
  // Run transformer on single GPU
  //-------------------------------------------------------------------------------
  if (rank == 0) {
    std::cout << "Running transformer on CPU for reference" << std::endl;
    run_transformer_block_cpu(ref_transformer, h_X.data(), h_Ref.data(),
                              seq_len);
    std::cout << "Running transformer on single GPU" << std::endl;
    run_transformer_block_single_gpu(context, device_transformer, d_X, d_Res,
                                     seq_len, single_stats);
    std::cout
        << "Copying weights and caches from GPU to CPU for correctness check"
        << std::endl;
    CUDA_CHECK(cudaMemcpy(h_Res.data(), d_Res, h_Res.size() * sizeof(half),
                          cudaMemcpyDeviceToHost));
    to_host(device_transformer, res_transformer, seq_len);
    is_ff_layer_equal(ref_transformer, res_transformer, h_Ref.data(),
                      h_Res.data(), seq_len);
    print_time_statistics(single_stats);
  }

  //-------------------------------------------------------------------------------
  // Run transformer on multi GPUs
  //-------------------------------------------------------------------------------
  std::cout << "Running transformer on" << world_size << " GPUs" << std::endl;
  run_transformer_block_multi_gpus(context, shard_transformer, d_X, d_Res,
                                   seq_len, multi_stats, ncclComm);

  std::cout << "Reducing time stats to root" << std::endl;
  reduce_time_statistics(multi_stats, rank);

  if (rank == 0) {
    std::cout
        << "Copying weights and caches from GPU to CPU for correctness check"
        << std::endl;
    CUDA_CHECK(cudaMemcpy(h_Res.data(), d_Res, h_Res.size() * sizeof(half),
                          cudaMemcpyDeviceToHost));
    // TODO:
    //  1. needs collective communication on each intermediate result on each
    //  attention or ff layer
    //  2. need collective communication on timing stats
    to_host(device_transformer, res_transformer, seq_len);
    is_ff_layer_equal(ref_transformer, res_transformer, h_Ref.data(),
                      h_Res.data(), seq_len);
    print_time_statistics(multi_stats);
  }

  //-------------------------------------------------------------------------------
  // Teardown
  //-------------------------------------------------------------------------------
  std::cout << "\nReleasing model and CUDA resources..." << std::endl;
  free_device_transformer_block(device_transformer);
  free_device_transformer_block(shard_transformer);
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
