#!/usr/bin/env python3
"""Plot a sweep's measured average GPU execution time (requires matplotlib)."""
import argparse
import csv
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('csv', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--metric', default='total_ms', choices=[
        'total_ms', 'fflayer_ms', 'all_reduce_fflayer_ms',
        'X_w1_ms', 'X_relu_ms', 'X_w2_ms'])
    args = parser.parse_args()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    with args.csv.open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        parser.error('CSV contains no measurements')
    groups = {}
    for row in rows:
        mode = (f'{row["world_size"]} GPUs' if row['multi_gpu'].lower() in ('true', '1')
                else '1 GPU')
        label = f'{mode}, seq_len={row["seq_len"]}'
        groups.setdefault(label, []).append((int(row['in_dim']), float(row[args.metric])))
    fig, ax = plt.subplots(figsize=(8, 5))
    for label, points in groups.items():
        points.sort()
        ax.plot(*zip(*points), marker='o', label=label)
    ax.set(xlabel='Input dimension (hidden_dim = 4 × in_dim)',
           ylabel=f'Mean execution time (ms): {args.metric}',
           title='Feed-forward size sweep')
    ax.set_xscale('log', base=2)
    ticks = sorted({int(row['in_dim']) for row in rows})
    ax.set_xticks(ticks, labels=[str(n) for n in ticks])
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    output = args.output or args.csv.with_name(f'{args.metric}.png')
    fig.savefig(output, dpi=180)
    print(output)


if __name__ == '__main__':
    main()
