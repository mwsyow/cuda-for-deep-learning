#!/usr/bin/env python3
"""Plot a sweep's single- vs multi-GPU timing and throughput (requires matplotlib).

One figure, 2x2 subplots sharing the in_dim axis; columns are multi GPU
including / excluding the NCCL all-reduce, rows are time / throughput:
  1. time incl. NCCL (multi total_ms)
  2. time excl. NCCL (multi fflayer_ms)
  3. GFLOP/s incl. NCCL (multi total_gflops)
  4. GFLOP/s excl. NCCL (multi fflayer_gflops)
Single GPU has no all-reduce, so it uses total_ms / total_gflops in every panel.
"""
import argparse
import csv
from pathlib import Path

SINGLE_COLOR = '#2a78d6'
MULTI_COLOR = '#eb6834'


def series(rows, multi, metric):
    """Group rows of one mode into {label: sorted [(in_dim, value)]}."""
    seq_lens = {row['seq_len'] for row in rows}
    groups = {}
    for row in rows:
        if (row['multi_gpu'].lower() in ('true', '1')) != multi:
            continue
        n = int(row['world_size'])
        label = f'multi, {n} GPU{"s" if n > 1 else ""}' if multi else 'single, 1 GPU'
        if len(seq_lens) > 1:
            label += f', seq_len={row["seq_len"]}'
        groups.setdefault(label, []).append((int(row['in_dim']), float(row[metric])))
    return {label: sorted(points) for label, points in groups.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('csv', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.ticker

    with args.csv.open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        parser.error('CSV contains no measurements')
    if 'total_gflops' not in rows[0]:
        parser.error('CSV has no GFLOP/s columns; rerun the sweep with the current binary')

    panels = [
        ('total_ms', 'total_ms', 'Execution time (ms)',
         'Time, multi GPU incl. NCCL all-reduce'),
        ('total_ms', 'fflayer_ms', 'Execution time (ms)',
         'Time, multi GPU excl. NCCL all-reduce'),
        ('total_gflops', 'total_gflops', 'Throughput (GFLOP/s)',
         'Throughput, multi GPU incl. NCCL all-reduce'),
        ('total_gflops', 'fflayer_gflops', 'Throughput (GFLOP/s)',
         'Throughput, multi GPU excl. NCCL all-reduce'),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    ticks = sorted({int(row['in_dim']) for row in rows})
    for ax, (single_metric, multi_metric, ylabel, title) in zip(axes.flat, panels):
        for multi, metric, color, marker in [
                (False, single_metric, SINGLE_COLOR, 'o'),
                (True, multi_metric, MULTI_COLOR, 's')]:
            for i, (label, points) in enumerate(series(rows, multi, metric).items()):
                ax.plot(*zip(*points), color=color, marker=marker, markersize=6,
                        linewidth=2, linestyle=['-', '--', ':', '-.'][i % 4],
                        label=label)
        ax.set(xlabel='Input dimension (hidden_dim = 4 × in_dim)', ylabel=ylabel,
               title=title)
        ax.set_xscale('log', base=2)
        ax.set_yscale('log')
        ax.yaxis.set_major_locator(matplotlib.ticker.LogLocator(subs=(1, 2, 5)))
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f'{v:g}'))
        ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_xticks(ticks, labels=[str(n) for n in ticks])
        ax.grid(True, which='major', alpha=0.3)
        ax.spines[['top', 'right']].set_visible(False)
        ax.legend(frameon=False)
    fig.suptitle('Feed-forward size sweep: single vs multi GPU')
    fig.tight_layout()
    output = args.output or args.csv.with_name('sweep.png')
    fig.savefig(output, dpi=180)
    print(output)


if __name__ == '__main__':
    main()
