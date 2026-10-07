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
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <locale>
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
  bool save_stats = false;
  std::string stats_path = "stats.csv";
  bool help = false;
};

void print_usage(const char *program) {
  std::cout
      << "Usage: " << program << " [options]\n"
      << "  --xseed N          Input seed (default: 7)\n"
      << "  --wseed N          Weight seed (default: 42)\n"
      << "  --in_dim N         Input width (default: 128)\n"
      << "  --hidden_dim N     Hidden width (default: 512)\n"
      << "  --seq_len N        Sequence length (default: 64)\n"
      << "  --warmup N         Warmup iterations (default: 1)\n"
      << "  --num_runs N       Number of run iterations (default: 1)\n"
      << "  --multi_gpu BOOL   true/false or 1/0 (default: false)\n"
      << "  --save_stats BOOL  Save average timings as CSV (default: false)\n"
      << "  --stats_path PATH  CSV output file, overwritten (default: "
         "stats.csv)\n"
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
        name != "--multi_gpu" && name != "--warmup" && name != "--num_runs" &&
        name != "--save_stats" && name != "--stats_path")
      throw std::invalid_argument("Unknown option: " + name);
    if (equals == std::string::npos) {
      if (i + 1 == argc || std::string(argv[i + 1]).compare(0, 2, "--") == 0)
        throw std::invalid_argument("Missing value for " + name);
      value = argv[++i];
    }
    if (name == "--stats_path") {
      if (value.empty())
        throw std::invalid_argument("--stats_path must not be empty");
      options.stats_path = value;
      continue;
    }
    if (name == "--multi_gpu" || name == "--save_stats") {
      bool &flag =
          name == "--multi_gpu" ? options.multi_gpu : options.save_stats;
      if (value == "true" || value == "1")
        flag = true;
      else if (value == "false" || value == "0")
        flag = false;
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

// Write one summary row; stats contains averages over measured runs, excluding
// warmup.
void save_time_statistics_csv(const TimeStats &stats, const TflopsStats &tflops,
                              const CliOptions &options, int world_size) {
  std::ofstream file(options.stats_path);
  if (!file)
    throw std::runtime_error("Cannot open stats file: " + options.stats_path);
  file.imbue(std::locale::classic());
  file << "xseed,wseed,in_dim,hidden_dim,seq_len,warmup,num_runs,multi_gpu,"
          "world_size,"
          "X_w1_ms,X_relu_ms,X_w2_ms,fflayer_ms,all_reduce_fflayer_ms,total_"
          "ms,X_w1_tflops,X_relu_tflops,X_w2_tflops,fflayer_tflops,total_"
          "tflops\n";
  file << std::setprecision(std::numeric_limits<float>::max_digits10)
       << options.xseed << ',' << options.wseed << ',' << options.in_dim << ','
       << options.hidden_dim << ',' << options.seq_len << ',' << options.warmup
       << ',' << options.num_runs << ','
       << (options.multi_gpu ? "true" : "false") << ',' << world_size << ','
       << stats.X_w1 << ',' << stats.X_relu << ',' << stats.X_w2 << ','
       << stats.fflayer << ',' << stats.all_reduce_fflayer << ',' << stats.total
       << ',' << tflops.X_w1 << ',' << tflops.X_relu << ',' << tflops.X_w2
       << ',' << tflops.fflayer << ',' << tflops.total << '\n';
  file.close();
  if (!file)
    throw std::runtime_error("Failed to write stats file: " +
                             options.stats_path);
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

  if (options.help) {
    if (rank == 0)
      print_usage(argv[0]);
    MPI_Finalize();
    return 0;
  }

  // Assuming this program is run on a single node
  CUDA_CHECK(cudaSetDevice(rank));

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

  //-------------------------------------------------------------------------------
  // Initialization
  //-------------------------------------------------------------------------------
  TimeStats stats{};
  TimeStats avg_stats{};

  FFLayer<float> ref_layer;
  FFLayer<half> res_layer;
  FFLayer<half> shard_layer;
  FFLayer<half> device_layer{};
  device_layer.hidden_dim = options.hidden_dim;
  device_layer.in_dim = options.in_dim;

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

    std::cout << "Creating input" << std::endl;
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
    std::cout << "Running feed-forward CPU reference" << std::endl;
    run_ff_layer_cpu(ref_layer, h_X.data(), h_Ref.data(), options.seq_len);
  }
  if (!options.multi_gpu && rank == 0) {
    //-------------------------------------------------------------------------------
    // Warming up on single GPU
    //-------------------------------------------------------------------------------
    std::cout << "Single-GPU warmup: " << options.warmup << " iterations"
              << std::endl;
    for (uint i = 0; i < options.warmup; i++) {
      if (rank == 0)
        std::cout << "  Warmup " << i + 1 << "/" << options.warmup << std::endl;
      run_ff_layer_single_gpu(context, device_layer, d_X, d_Res,
                              options.seq_len, stats);
    }
  } else {
    //-------------------------------------------------------------------------------
    // Warming up on multi GPUs
    //-------------------------------------------------------------------------------
    if (rank == 0)
      std::cout << "Multi-GPU warmup: " << options.warmup
                << " iterations across " << world_size << " ranks" << std::endl;
    for (uint i = 0; i < options.warmup; i++) {
      if (rank == 0)
        std::cout << "  Warmup " << i + 1 << "/" << options.warmup << std::endl;
      run_ff_layer_multi_gpus(context, shard_layer, d_X, d_Res, options.seq_len,
                              stats, ncclComm);
    }
    if (rank == 0)
      std::cout << "Gathering intermediate results for correctness check"
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
  if (rank == 0)
    std::cout << "Measured feed-forward runs: " << options.num_runs
              << std::endl;
  for (uint i = 0; i < options.num_runs; i++) {
    if (rank == 0)
      std::cout << "  Run " << i + 1 << "/" << options.num_runs << std::endl;
    if (!options.multi_gpu) {
      run_ff_layer_single_gpu(context, device_layer, d_X, d_Res,
                              options.seq_len, stats);
    } else {
      // Align ranks so a rank that starts early does not wait inside the timed
      // region for a late peer (rank 0 does extra host work between runs).
      MPI_Barrier(MPI_COMM_WORLD);
      run_ff_layer_multi_gpus(context, shard_layer, d_X, d_Res, options.seq_len,
                              stats, ncclComm);
      reduce_time_statistics(stats, rank);
    }

    avg_stats.X_w1 += stats.X_w1;
    avg_stats.X_relu += stats.X_relu;
    avg_stats.X_w2 += stats.X_w2;
    avg_stats.fflayer += stats.fflayer;
    avg_stats.all_reduce_fflayer += stats.all_reduce_fflayer;
    avg_stats.total += stats.total;
  }
  int exit_status = 0;
  if (rank == 0) {
    avg_stats.X_w1 /= options.num_runs;
    avg_stats.X_relu /= options.num_runs;
    avg_stats.X_w2 /= options.num_runs;
    avg_stats.fflayer /= options.num_runs;
    avg_stats.all_reduce_fflayer /= options.num_runs;
    avg_stats.total /= options.num_runs;
    std::cout << "Average over " << options.num_runs << " measured runs\n";
    const TflopsStats tflops = compute_tflops(
        avg_stats, options.in_dim, options.hidden_dim, options.seq_len);
    print_time_statistics(avg_stats, tflops);
    if (options.save_stats) {
      try {
        save_time_statistics_csv(avg_stats, tflops, options, world_size);
        std::cout << "Saved timing statistics to " << options.stats_path
                  << '\n';
      } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        exit_status = 1;
      }
    }
  }
  MPI_Bcast(&exit_status, 1, MPI_INT, 0, MPI_COMM_WORLD);
  //-------------------------------------------------------------------------------
  // Teardown
  //-------------------------------------------------------------------------------
  if (rank == 0) {
    std::cout << "\nReleasing feed-forward layers and CUDA resources..."
              << std::endl;
    free_device_ff_layer(device_layer);
    free_host_ff_layer(ref_layer);
    free_host_ff_layer(res_layer);
  }

  CUBLAS_CHECK(cublasDestroy(cublasHandle));
  CUDA_CHECK(cudaStreamDestroy(stream));
  CUDA_CHECK(cudaFree(d_X));
  CUDA_CHECK(cudaFree(d_Res));

  if (options.multi_gpu) {
    free_device_ff_layer(shard_layer);
    NCCL_CHECK(ncclCommDestroy(ncclComm));
  }
  MPI_Finalize();

  if (rank == 0)
    std::cout << "Feed-forward demo finished" << std::endl;
  return exit_status;
}
