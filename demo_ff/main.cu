#include "common.cuh"
#include "correctness_check.cuh"
#include "feed_forward.cuh"
#include "forward_pass_cpu.cuh"
#include "forward_pass_gpu.cuh"
#include <algorithm>
#include <cuda_device_runtime_api.h>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <exception>
#include <iostream>
#include <limits>
#include <mpi.h>
#include <nccl.h>
#include <stdexcept>
#include <string>
#include <vector>

struct CliOptions {
  ulong xseed = 7;
  ulong wseed = 42;
  uint in_dim = 128;
  uint hidden_dim = 4 * 128;
  uint seq_len = 64;
  uint warmup = 1;
  uint num_runs = 1;
  bool multi_gpu = false;
  bool help = false;
};

void print_usage(const char *program) {
  std::cout << "Usage: " << program << " [options]\n"
            << "  --xseed N          Input seed (default: 7)\n"
            << "  --wseed N          Weight seed (default: 42)\n"
            << "  --in_dim N         Input width (default: 128)\n"
            << "  --hidden_dim N     Base hidden width\n"
               "(default: 4 * 128)\n"
            << "  --seq_len N        Sequence length (default: 64)\n"
            << "  --warmup N         Warmup iterations (default: 1)\n"
            << "  --num_runs N       Number of run iterations (default: 1)\n"
            << "  --multi_gpu BOOL   true/false or 1/0 (default: false)\n"
            << "  --help             Show this help\n"
            << "Options accept --name value or --name=value; hyphens and "
               "underscores are accepted.\n";
}

CliOptions parse_cli_arguments(int argc, char *argv[]) {
  CliOptions options;
  for (int i = 1; i < argc; ++i) {
    std::string name = argv[i], value;
    const auto equals = name.find('=');
    if (equals != std::string::npos) {
      value = name.substr(equals + 1);
      name.resize(equals);
    }
    if (name == "--help" || name == "-h") {
      options.help = true;
      continue;
    }
    if (name.compare(0, 2, "--") != 0)
      throw std::invalid_argument("Expected a named option: " + name);
    std::replace(name.begin() + 2, name.end(), '-', '_');
    if (name != "--xseed" && name != "--wseed" && name != "--in_dim" &&
        name != "--hidden_dim" && name != "--seq_len" &&
        name != "--multi_gpu" && name != "--warmup" && name != "--num_runs")
      throw std::invalid_argument("Unknown option: " + name);
    if (equals == std::string::npos) {
      if (i + 1 == argc || std::string(argv[i + 1]).compare(0, 2, "--") == 0)
        throw std::invalid_argument("Missing value for " + name);
      value = argv[++i];
    }
    if (name == "--multi_gpu") {
      if (value == "true" || value == "1")
        options.multi_gpu = true;
      else if (value == "false" || value == "0")
        options.multi_gpu = false;
      else
        throw std::invalid_argument(name + " expects true, false, 1, or 0");
      continue;
    }
    if (value.empty() ||
        value.find_first_not_of("0123456789") != std::string::npos)
      throw std::invalid_argument(name + " expects an unsigned integer");
    const auto number = std::stoul(value);
    const bool seed = name == "--xseed" || name == "--wseed";
    const bool warmup = name == "--warmup";
    if (!seed && number > std::numeric_limits<uint>::max())
      throw std::out_of_range(name + " exceeds uint range");
    if (!seed && !warmup && number == 0)
      throw std::invalid_argument(name + " must be positive");
    if (name == "--xseed")
      options.xseed = static_cast<ulong>(number);
    else if (name == "--wseed")
      options.wseed = static_cast<ulong>(number);
    else if (name == "--in_dim")
      options.in_dim = static_cast<uint>(number);
    else if (name == "--hidden_dim")
      options.hidden_dim = static_cast<uint>(number);
    else if (name == "--num_runs")
      options.num_runs = static_cast<uint>(number);
    else if (warmup)
      options.warmup = static_cast<uint>(number);
    else
      options.seq_len = static_cast<uint>(number);
  }
  return options;
}

int main(int argc, char *argv[]) {
  //-------------------------------------------------------------------------------
  // MPI and NCCL startup
  //-------------------------------------------------------------------------------
  MPI_Init(&argc, &argv);

  int rank, world_size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  CliOptions options;
  try {
    options = parse_cli_arguments(argc, argv);
  } catch (const std::exception &e) {
    if (rank == 0)
      std::cerr << e.what() << '\n';
    MPI_Finalize();
    return 1;
  }

  ncclUniqueId ncclCommId;
  ncclComm_t ncclComm;
  if (options.multi_gpu) {
    int device_count;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    if (world_size > device_count) {
      std::cerr << "World size " << world_size
                << "cannot exceed number of device " << device_count
                << std::endl;
      MPI_Finalize();
      return 1;
    }
    if (rank == 0)
      NCCL_CHECK(ncclGetUniqueId(&ncclCommId));
    MPI_Bcast(&ncclCommId, sizeof(ncclCommId), MPI_BYTE, 0, MPI_COMM_WORLD);

    NCCL_CHECK(ncclCommInitRank(&ncclComm, world_size, ncclCommId, rank));
  }

  // Assuming this program is run on a single node
  CUDA_CHECK(cudaSetDevice(rank));

  //-------------------------------------------------------------------------------
  // Initialization
  //-------------------------------------------------------------------------------
  TimeStats stats{};
  TimeStats avg_stats{};

  FFLayer<float> ref_layer;
  FFLayer<half> res_layer;
  FFLayer<half> device_layer;
  FFLayer<half> shard_layer;

  std::vector<float> h_X(options.seq_len * options.in_dim);
  std::vector<half> h_X_half(options.seq_len * options.in_dim);
  std::vector<float> h_Ref(options.seq_len * options.in_dim);
  std::vector<half> h_Res(options.seq_len * options.in_dim);

  cublasHandle_t cublasHandle;
  cudaStream_t stream;
  CUBLAS_CHECK(cublasCreate(&cublasHandle));
  CUDA_CHECK(cudaStreamCreate(&stream));
  CUBLAS_CHECK(cublasSetStream(cublasHandle, stream));
  DeviceContext context{stream, cublasHandle};

  if (rank == 0) {
    std::cout << "Initializing and allocating weights and caches on CPU"
              << std::endl;
    init_ff_layer(ref_layer, options.in_dim, options.hidden_dim,
                  options.seq_len, options.wseed);
    std::cout << "Transferring weights and allocating GPU caches" << std::endl;
    to_device(ref_layer, device_layer, options.seq_len);
    std::cout << "Creating input and context" << std::endl;
    std::mt19937 gen(options.xseed);
    std::uniform_real_distribution<float> distribution(-1.0f, 1.0f);

    for (auto &v : h_X)
      v = distribution(gen);

    for (int i = 0; i < h_X.size(); i++) {
      h_X_half[i] = __float2half(h_X[i]);
    }
  }

  if (options.multi_gpu) {
    scatter_ff_layer(device_layer, shard_layer, world_size, rank,
                     options.seq_len, ncclComm, stream);
  }

  half *d_X, *d_Res;
  CUDA_CHECK(cudaMalloc(&d_X, options.seq_len * options.in_dim * sizeof(half)));
  CUDA_CHECK(
      cudaMalloc(&d_Res, options.seq_len * options.in_dim * sizeof(half)));

  if (rank == 0) {
    CUDA_CHECK(cudaMemcpy(d_X, h_X_half.data(), h_X_half.size() * sizeof(half),
                          cudaMemcpyHostToDevice));
  }

  if (options.multi_gpu) {
    NCCL_CHECK(ncclBroadcast(d_X, d_X, options.seq_len * options.in_dim,
                             ncclHalf, 0, ncclComm, stream));
  }
  //-------------------------------------------------------------------------------
  // Correctness check and warmup
  //-------------------------------------------------------------------------------
  if (rank == 0) {
    std::cout << "Running Feed Forward Layer on CPU for reference" << std::endl;
    run_ff_layer_cpu(ref_layer, h_X.data(), h_Ref.data(), options.seq_len);
  }
  if (!options.multi_gpu && rank == 0) {
    //-------------------------------------------------------------------------------
    // Warming up on single GPU
    //-------------------------------------------------------------------------------
    std::cout << "Warming up on single GPU" << std::endl;
    for (uint i = 0; i < options.warmup; i++) {
      run_ff_layer_single_gpu(context, device_layer, d_X, d_Res,
                              options.seq_len, stats);
    }
  } else {
    //-------------------------------------------------------------------------------
    // Warming up on multi GPUs
    //-------------------------------------------------------------------------------
    std::cout << "Warming up on " << world_size << " GPUs" << std::endl;
    for (uint i = 0; i < options.warmup; i++) {
      run_ff_layer_multi_gpus(context, shard_layer, d_X, d_Res, options.seq_len,
                              stats, ncclComm);
    }
    std::cout << "Unsharding intermediate results for correctness check"
              << std::endl;
    unsharding_intermediate_results(shard_layer, device_layer, options.seq_len,
                                    rank, ncclComm, stream);
  }
  if (rank == 0) {
    std::cout
        << "Copying weights and caches from GPU to CPU for correctness check"
        << std::endl;
    CUDA_CHECK(cudaMemcpy(h_Res.data(), d_Res, h_Res.size() * sizeof(half),
                          cudaMemcpyDeviceToHost));
    to_host(device_layer, res_layer, options.seq_len);
    is_ff_layer_equal(ref_layer, res_layer, h_Ref.data(), h_Res.data(),
                      options.seq_len);
  }
  //-------------------------------------------------------------------------------
  // Running Demo
  //-------------------------------------------------------------------------------
  for (int i = 0; i < options.num_runs; i++) {
    if (!options.multi_gpu) {
      run_ff_layer_single_gpu(context, device_layer, d_X, d_Res,
                              options.seq_len, stats);
    } else {
      run_ff_layer_multi_gpus(context, shard_layer, d_X, d_Res, options.seq_len,
                              stats, ncclComm);
      reduce_time_statistics(stats, rank);
    }

    avg_stats.X_w1 += stats.X_w1;
    avg_stats.X_relu += stats.X_relu;
    avg_stats.X_w2 += stats.X_w2;
    avg_stats.all_reduce_fflayer += stats.all_reduce_fflayer;
    avg_stats.total += stats.total;
  }
  print_time_statistics(stats);
  //-------------------------------------------------------------------------------
  // Teardown
  //-------------------------------------------------------------------------------
  std::cout << "\nReleasing layers and CUDA resources..." << std::endl;
  free_device_ff_layer(device_layer);
  free_host_ff_layer(ref_layer);
  free_host_ff_layer(res_layer);

  CUBLAS_CHECK(cublasDestroy(cublasHandle));
  CUDA_CHECK(cudaStreamDestroy(stream));
  CUDA_CHECK(cudaFree(d_X));
  CUDA_CHECK(cudaFree(d_Res));

  if (options.multi_gpu) {
    free_device_ff_layer(shard_layer);
    NCCL_CHECK(ncclCommDestroy(ncclComm));
  }
  MPI_Finalize();

  std::cout << "Run finished" << std::endl;
}
