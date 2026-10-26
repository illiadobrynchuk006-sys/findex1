"""Compare Python and NumPy scoring; profile hot paths and save measurements."""
from __future__ import annotations

import argparse
import cProfile
import json
import pstats
import statistics
import time
from pathlib import Path

from findex.index import load_binary
from findex.model import Index
from findex.numpy_rank import NumpyScorer
from findex.rank import Scorer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("index", type=Path)
    parser.add_argument("--query", default="python AND пошук AND документ")
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--output", type=Path, default=Path("benchmarks/lab08/results.json"))
    args = parser.parse_args()
    index = Index(load_binary(args.index))
    python_scorer = Scorer(index)
    numpy_scorer = NumpyScorer(index)
    rows = []
    for method in ("bm25", "tfidf"):
        for label, scorer in (("python", python_scorer), ("numpy", numpy_scorer)):
            timings = []
            for _ in range(args.repeats):
                start = time.perf_counter()
                # Fresh Python scorer to avoid cached top-k results.
                instance = Scorer(index) if label == "python" else scorer
                instance.search(args.query, method=method, top_k=10)
                timings.append((time.perf_counter() - start) * 1000)
            rows.append({
                "engine": label, "method": method,
                "median_ms": round(statistics.median(timings), 5),
                "repeats": args.repeats,
            })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(json.dumps(rows, indent=2))
    profile = cProfile.Profile()
    profile.enable()
    Scorer(index).search(args.query, method="bm25")
    profile.disable()
    profile.dump_stats(str(args.output.parent / "search.prof"))
    with (args.output.parent / "profile.txt").open("w") as stream:
        pstats.Stats(profile, stream=stream).sort_stats("cumulative").print_stats(25)


if __name__ == "__main__":
    main()
