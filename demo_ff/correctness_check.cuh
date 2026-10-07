#include "feed_forward.cuh"
#include "mpi.h"
#include <cassert>
#include <cuda_fp16.h>
#include <iomanip>
#include <ios>
#include <iostream>

#pragma once

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

void is_ff_layer_equal(FFLayer<float> ref_layer, FFLayer<half> res_layer,
                       float *ref, half *res, uint seq_len) {
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
            << "\nFeed-forward\n";
  title("Hidden expansion (W1)");
  is_equal(ref_layer.X_w1, res_layer.X_w1, seq_len * ref_layer.hidden_dim);
  title("ReLU activation");
  is_equal(ref_layer.X_relu, res_layer.X_relu, seq_len * ref_layer.hidden_dim);


  title("Final output (W2)");
  is_equal(ref, res, seq_len * ref_layer.in_dim);
}

void reduce_time_statistics(TimeStats &stats, const int rank) {

  std::vector<float> send_buff(sizeof(TimeStats) / sizeof(float));
  std::vector<float> recv_buff(sizeof(TimeStats) / sizeof(float));

  send_buff[0] = stats.X_w1;
  send_buff[1] = stats.X_relu;
  send_buff[2] = stats.X_w2;
  send_buff[3] = stats.fflayer;
  send_buff[4] = stats.all_reduce_fflayer;
  send_buff[5] = stats.total;

  MPI_Reduce(send_buff.data(), recv_buff.data(),
             static_cast<int>(send_buff.size()), MPI_FLOAT, MPI_MAX, 0,
             MPI_COMM_WORLD);

  if (rank == 0) {
    stats.X_w1 = recv_buff[0];
    stats.X_relu = recv_buff[1];
    stats.X_w2 = recv_buff[2];
    stats.fflayer = recv_buff[3];
    stats.all_reduce_fflayer = recv_buff[4];
    stats.total = recv_buff[5];
  }
}

void print_time_statistics(const TimeStats &stats) {
  const auto previous_flags = std::cout.flags();
  const auto previous_precision = std::cout.precision();
  auto print = [&](const char *name, float ms) {
    std::cout << "  " << std::left << std::setw(30) << name << std::right
              << std::setw(12) << ms << " ms\n";
  };

  std::cout << "\n=== Average GPU timing breakdown ===\n"
            << "Measured runs only; warmup excluded. Synchronization after each stage.\n\n"
            << std::fixed << std::setprecision(3);
  std::cout << "\nFeed-forward\n";
  print("Hidden expansion (W1)", stats.X_w1);
  print("ReLU activation", stats.X_relu);
  print("Output projection (W2)", stats.X_w2);
  print("Feed-forward subtotal", stats.fflayer);
  print("Feed-forward all reduce", stats.all_reduce_fflayer);
  std::cout << "  ---------------------------------------------\n";
  print("Feed-forward total", stats.total);
  std::cout.flags(previous_flags);
  std::cout.precision(previous_precision);
}
