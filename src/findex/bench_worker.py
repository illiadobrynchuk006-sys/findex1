"""Isolated worker: wall time, CPU of parent and children, and RSS high water mark."""
from __future__ import annotations

import argparse
import resource
import time
from pathlib import Path

from findex.parallel import build_parallel


def cpu_time() -> float:
    parent = resource.getrusage(resource.RUSAGE_SELF)
    children = resource.getrusage(resource.RUSAGE_CHILDREN)
    return parent.ru_utime + parent.ru_stime + children.ru_utime + children.ru_stime


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("corpus", type=Path)
    parser.add_argument("--executor", choices=["serial", "threads", "processes"])
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    before_cpu = cpu_time()
    before_wall = time.perf_counter()
    index = build_parallel(args.corpus, args.workers, args.executor)
    wall = time.perf_counter() - before_wall
    cpu = cpu_time() - before_cpu
    parent_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    child_rss = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    # Linux maxrss is KiB. These are separate process high-water marks;
    # max(parent, child) is not combined simultaneous memory use.
    rss = max(parent_rss, child_rss)
    assert index.documents is not None
    print(f"{wall:.9f},{cpu:.9f},{rss}")


if __name__ == "__main__":
    main()
