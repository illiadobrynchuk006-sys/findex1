"""Median wall/CPU/RSS benchmarks and speedup chart for lab 05."""
from __future__ import annotations

import argparse
import csv
import os
import statistics
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def benchmark(corpus: Path, executor: str, workers: int, repeats: int):
    command = [
        sys.executable, "-m", "findex.bench_worker", str(corpus),
        "--executor", executor, "--workers", str(workers),
    ]
    records = []
    for _ in range(repeats):
        completed = subprocess.run(command, check=True, text=True, capture_output=True)
        wall, cpu, rss = map(float, completed.stdout.strip().split(","))
        records.append((wall, cpu, rss))
    return tuple(statistics.median(row[k] for row in records) for k in range(3))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("corpus", type=Path)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--outdir", type=Path, default=Path("benchmarks/lab05"))
    args = parser.parse_args()
    if args.repeats < 3:
        parser.error("Use at least 3 repetitions")
    counts = sorted({1, 2, 4, 8, os.cpu_count() or 1})
    rows = []
    for executor in ("serial", "threads", "processes"):
        for workers in counts:
            print(f"Measuring {executor}/{workers} ...", file=sys.stderr)
            wall, cpu, rss = benchmark(args.corpus, executor, workers, args.repeats)
            rows.append(dict(executor=executor, workers=workers, wall_s=wall,
                             cpu_s=cpu, peak_rss_kib=rss))
    baseline = next(row["wall_s"] for row in rows
                    if row["executor"] == "serial" and row["workers"] == 1)
    for row in rows:
        row["speedup"] = baseline / row["wall_s"] if row["wall_s"] else 0.0
    args.outdir.mkdir(parents=True, exist_ok=True)
    with (args.outdir / "measurements.csv").open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for mode in ("serial", "threads", "processes"):
        matching = [r for r in rows if r["executor"] == mode]
        plt.plot([r["workers"] for r in matching],
                 [r["speedup"] for r in matching], marker="o", label=mode)
    plt.plot(counts, counts, linestyle="--", label="ideal linear")
    plt.xlabel("Workers")
    plt.ylabel("Speedup vs serial/1")
    plt.title("findex indexing speedup")
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.outdir / "speedup.png", dpi=150)
    print(f"CSV: {args.outdir / 'measurements.csv'}")
    print(f"PNG: {args.outdir / 'speedup.png'}")


if __name__ == "__main__":
    main()
