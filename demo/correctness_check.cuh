#include "mpi.h"
#include "transformer.cuh"
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

void is_ff_layer_equal(TransformerBlock<float> ref_transformer,
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

void reduce_time_statistics(TimeStats &stats, const int rank) {

  std::vector<float> stats_buff(sizeof(TimeStats) / sizeof(float));

  stats_buff[0] = stats.X_q;
  stats_buff[1] = stats.X_k;
  stats_buff[2] = stats.X_v;
  stats_buff[3] = stats.attention_scores;
  stats_buff[4] = stats.attention_weights;
  stats_buff[5] = stats.Z;
  stats_buff[6] = stats.attention_result;
  stats_buff[7] = stats.X_w1;
  stats_buff[8] = stats.X_relu;
  stats_buff[9] = stats.X_w2;
  stats_buff[10] = stats.salayer;
  stats_buff[11] = stats.fflayer;
  stats_buff[12] = stats.all_reduce_salayer;
  stats_buff[13] = stats.all_reduce_fflayer;
  stats_buff[14] = stats.total;

  MPI_Reduce(stats_buff.data(), stats_buff.data(), sizeof(TimeStats), MPI_BYTE,
             MPI_MAX, 0, MPI_COMM_WORLD);

  if (rank == 0) {
    stats.X_q = stats_buff[0];
    stats.X_k = stats_buff[1];
    stats.X_v = stats_buff[2];
    stats.attention_scores = stats_buff[3];
    stats.attention_weights = stats_buff[4];
    stats.Z = stats_buff[5];
    stats.attention_result = stats_buff[6];
    stats.X_w1 = stats_buff[7];
    stats.X_relu = stats_buff[8];
    stats.X_w2 = stats_buff[9];
    stats.salayer = stats_buff[10];
    stats.fflayer = stats_buff[11];
    stats.all_reduce_salayer = stats_buff[12];
    stats.all_reduce_fflayer = stats_buff[13];
    stats.total = stats_buff[14];
  }
}

void print_time_statistics(const TimeStats &stats) {
  const auto previous_flags = std::cout.flags();
  const auto previous_precision = std::cout.precision();
  auto print = [&](const char *name, float ms) {
    std::cout << "  " << std::left << std::setw(30) << name << std::right
              << std::setw(12) << ms << " ms\n";
  };

  std::cout << "\n=== GPU timing breakdown ===\n"
            << "Single diagnostic run; synchronization after each stage.\n\n"
            << std::fixed << std::setprecision(3) << "Attention\n";
  print("Query projection (XQ)", stats.X_q);
  print("Key projection (XK)", stats.X_k);
  print("Value projection (XV)", stats.X_v);
  print("Scaled attention logits", stats.attention_scores);
  print("Softmax probabilities", stats.attention_weights);
  print("Attention context (AV)", stats.Z);
  print("Attention output projection", stats.attention_result);
  print("Attention subtotal", stats.salayer);

  std::cout << "\nFeed-forward\n";
  print("Hidden expansion (W1)", stats.X_w1);
  print("ReLU activation", stats.X_relu);
  print("Output projection (W2)", stats.X_w2);
  print("Feed-forward subtotal", stats.fflayer);
  std::cout << "  ---------------------------------------------\n";
  print("Full block total", stats.total);
  std::cout.flags(previous_flags);
  std::cout.precision(previous_precision);
}
