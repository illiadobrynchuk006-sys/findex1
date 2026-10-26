"""Measure concurrent HTTP latency (p50/p95/p99) against findex API.

Example:
 python scripts/benchmark_lab07.py http://127.0.0.1:8000 --requests 100 --concurrency 20
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import statistics
import time

import httpx


def percentile(values: list[float], pct: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * pct / 100
    low = math.floor(position)
    high = math.ceil(position)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


async def measure(base_url: str, path: str, query: str, requests: int, concurrency: int):
    durations: list[float] = []
    errors = 0
    gate = asyncio.Semaphore(concurrency)
    limits = httpx.Limits(max_connections=concurrency)
    async with httpx.AsyncClient(timeout=30, limits=limits) as client:
        async def single():
            nonlocal errors
            async with gate:
                start = time.perf_counter()
                try:
                    response = await client.get(
                        base_url.rstrip("/") + path, params={"q": query}
                    )
                    response.raise_for_status()
                except httpx.HTTPError:
                    errors += 1
                finally:
                    durations.append((time.perf_counter() - start) * 1000)

        start = time.perf_counter()
        await asyncio.gather(*(single() for _ in range(requests)))
        wall = time.perf_counter() - start
    return {
        "url": base_url, "route": path, "concurrency": concurrency,
        "requests": requests, "errors": errors, "wall_s": round(wall, 4),
        "rps": round(requests / wall, 2),
        "p50_ms": round(percentile(durations, 50), 3),
        "p95_ms": round(percentile(durations, 95), 3),
        "p99_ms": round(percentile(durations, 99), 3),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("url")
    parser.add_argument("--query", default="python")
    parser.add_argument("--requests", type=int, default=100)
    parser.add_argument("--concurrency", type=int, default=20)
    parser.add_argument("--route", choices=["/search", "/search-async"], default="/search")
    parser.add_argument("--out", default="benchmarks/lab07/results.jsonl")
    args = parser.parse_args()
    if args.requests < 1 or args.concurrency < 1:
        parser.error("Request count and concurrency must be positive")
    row = asyncio.run(measure(
        args.url, args.route, args.query, args.requests, args.concurrency
    ))
    from pathlib import Path
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps(row, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
