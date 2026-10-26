"""Benchmark index, serialization, Boolean search and posting memory."""

import argparse
import gc
import time
import tracemalloc
from dataclasses import dataclass
from pathlib import Path

from findex.index import (
    build_index,
    build_packed_postings,
    load_binary,
    load_pickle,
    save_binary,
    save_pickle,
)
from findex.search import boolean_search


@dataclass
class RegularPosting:
    doc_id: str
    count: int


@dataclass(slots=True)
class SlotsPosting:
    doc_id: str
    count: int


def measure(fn):
    gc.collect()
    tracemalloc.start()
    t = time.perf_counter()
    try:
        result = fn()
        elapsed = time.perf_counter() - t
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return result, elapsed, peak / 1048576


def main(root):
    idx, seconds, mem = measure(lambda: build_index(root))
    print(
        f"Build: {seconds:.6f}s, {mem:.3f} MiB, {len(idx.documents)} docs, {len(idx.postings)} terms"
    )
    for name, fn in [
        (
            "dataclass",
            lambda: [RegularPosting(*p) for posts in idx.postings.values() for p in posts],
        ),
        ("slots", lambda: [SlotsPosting(*p) for posts in idx.postings.values() for p in posts]),
        ("array(I)", lambda: build_packed_postings(idx)),
    ]:
        obj, t, peak = measure(fn)
        print(f"Posting {name}: {t:.6f}s, {peak:.3f} MiB")
        del obj
    for name, save, load in [
        ("binary", save_binary, load_binary),
        ("pickle", save_pickle, load_pickle),
    ]:
        output = root.parent / f"lab02-{name}-benchmark.bin"
        _, st, sm = measure(lambda: save(idx, output))
        restored, lt, lm = measure(lambda: load(output))
        assert restored == idx
        print(
            f"{name}: {output.stat().st_size} bytes, save {st:.6f}s/{sm:.3f} MiB, load {lt:.6f}s/{lm:.3f} MiB"
        )
        output.unlink()
    terms = sorted(idx.postings, key=lambda x: (-len(idx.postings[x]), x))
    for label, ts in [("common", terms[:2]), ("rare", terms[-2:])]:
        if len(ts) < 2:
            continue
        query = " AND ".join(ts)
        for engine in ["merge", "set"]:
            matched, t, mem = measure(lambda: boolean_search(idx, query, engine))
            print(
                f"{label}/{engine} {query!r}: matches={len(matched)} time={t:.6f}s memory={mem:.3f}MiB"
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data", type=Path)
    main(parser.parse_args().data)
