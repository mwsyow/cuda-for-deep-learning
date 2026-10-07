# Size sweep

On the SSH server, from the repository root (after copying these scripts there):

```bash
make -C demo_ff
python3 demo_ff/sweep.py --gpus 2 --output demo_ff/results/run1
```

Then on your local machine, from the repository root, download and plot:

```bash
mkdir -p demo_ff/results/run1
rsync -av -e 'ssh -p 16981' root@84.50.156.116:/workspace/cuda-for-deep-learning/demo_ff/results/run1/ demo_ff/results/run1/
python3 -m pip install matplotlib
python3 demo_ff/plot_stats.py demo_ff/results/run1/stats.csv
```

The sweep runs sizes 128, 256, 512, 1024, 2048, and 4096 sequentially,
with hidden_dim = 4 * in_dim and seq_len = 64. Each size gets a single-GPU
run with one MPI process and a multi-GPU run with the requested process count.
Defaults are 3 warmups and 10 measured iterations. Root execution adds
Open MPI's `--allow-run-as-root` automatically. All requested GPUs must be visible.

Each configuration retains its own log and CSV; stats.csv combines completed
measurements. A failed process stops the sweep with its log path. Existing
output directories are allowed: matching logs and per-configuration CSVs are
overwritten, and the first successful result replaces the combined stats.csv
so reruns do not append duplicate measurements. Unrelated old files remain.
Use `--dry-run` to inspect commands without running them.

The 4096 cap is deliberately conservative for 16 GB RAM and 32 GB VRAM.
At that size, the current implementation's CPU float weights and oversized
ReLU cache take about 768 MiB, and the copied half weights add 256 MiB,
plus activations and temporary buffers. GPU weight buffers are also well
below 32 GB, even with full weights, shards, and transpose buffers present.
These are array estimates, not a guarantee of available memory; MPI, CUDA,
NCCL and other processes need additional space. The naive CPU correctness
reference can make larger configurations slow even when GPU execution is fast.
The script limits seq_len to at most the smallest input size because the
current CPU ReLU cache is allocated using in_dim instead of seq_len.

For a shorter sweep:

```bash
python3 demo_ff/sweep.py --sizes 128 256 512 1024 --gpus 2
```

The plot uses total_ms from the executable (GPU timing, not process wall time).
Choose `--metric all_reduce_fflayer_ms` or `--metric fflayer_ms` to inspect
communication or feed-forward timing separately. Fixed seq_len makes this a
width sweep; it does not guarantee that multiple GPUs will outperform one.
