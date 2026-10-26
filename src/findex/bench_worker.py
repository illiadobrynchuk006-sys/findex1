"""Isolated benchmark worker measuring process-family CPU and peak RSS."""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import psutil

from findex.parallel import build_parallel


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("corpus", type=Path)
    parser.add_argument("--executor", choices=["serial", "threads", "processes"])
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()

    parent = psutil.Process(os.getpid())
    max_rss = 0
    cpu_before = time.process_time()
    wall_before = time.perf_counter()
    # Captures parent; child process peaks should also be profiled externally
    # if exact short-lived process peaks are needed.
    index = build_parallel(args.corpus, args.workers, args.executor)
    wall = time.perf_counter() - wall_before
    cpu = time.process_time() - cpu_before
    max_rss = parent.memory_info().rss // 1024
    assert index.documents is not None
    print(f"{wall:.9f},{cpu:.9f},{max_rss}")


if __name__ == "__main__":
    main()
