#!/usr/bin/env python3
"""Run paired single/multi-GPU benchmarks, retaining CSVs and logs."""

import argparse
import csv
import os
import shlex
import subprocess
from datetime import datetime
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sizes", nargs="+", type=int, default=[128, 256, 512, 1024, 2048, 4096]
    )
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--num-runs", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent
        / "results"
        / datetime.now().strftime("%Y%m%d-%H%M%S"),
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    binary = Path(__file__).resolve().parent / "main"
    output = args.output.resolve()
    if not args.dry_run:
        if not os.access(binary, os.X_OK):
            parser.error("build the executable first: make -C demo_ff")
        output.mkdir(parents=True, exist_ok=True)
    aggregate = output / "stats.csv"
    first_result = True
    for size in sorted(set(args.sizes)):
        for multi, ranks in [(False, 1), (True, args.gpus)]:
            name = f"in{size}_" + (f"multi{ranks}" if multi else "single")
            stats = output / f"{name}.csv"
            command = ["mpirun"]
            if os.geteuid() == 0:
                command.append("--allow-run-as-root")
            command += [
                "-np",
                str(ranks),
                str(binary),
                "--multi_gpu",
                str(int(multi)),
                "--in_dim",
                str(size),
                "--hidden_dim",
                str(4 * size),
                "--seq_len",
                str(args.seq_len),
                "--num_runs",
                str(args.num_runs),
                "--warmup",
                str(args.warmup),
                "--save_stats",
                "1",
                "--stats_path",
                str(stats),
            ]
            print(shlex.join(command), flush=True)
            if args.dry_run:
                continue
            log = output / f"{name}.log"
            with log.open("w") as stream:
                result = subprocess.run(
                    command, stdout=stream, stderr=subprocess.STDOUT
                )
            if result.returncode:
                raise SystemExit(f"Benchmark failed ({result.returncode}); see {log}")
            with stats.open(newline="") as stream:
                reader = csv.DictReader(stream)
                fields = reader.fieldnames
                rows = list(reader)
            if (
                len(rows) != 1
                or int(rows[0]["in_dim"]) != size
                or int(rows[0]["world_size"]) != ranks
            ):
                raise SystemExit(f"Unexpected benchmark statistics: {stats}")
            with aggregate.open("w" if first_result else "a", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                if first_result:
                    writer.writeheader()
                writer.writerows(rows)
            first_result = False
            print(f"Saved {name}: {rows[0]['total_ms']} ms", flush=True)
    if not args.dry_run:
        print(f"Combined results: {aggregate}")


if __name__ == "__main__":
    main()
