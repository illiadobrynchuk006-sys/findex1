"""Boolean search over an inverted index."""

from __future__ import annotations

import argparse
import time
import tracemalloc
from pathlib import Path

from findex.index import load_binary, load_pickle
from findex.tokenize import tokenize


def merge_intersection(a, b):
    i = j = 0
    out = []
    while i < len(a) and j < len(b):
        if a[i] == b[j]:
            out.append(a[i])
            i += 1
            j += 1
        elif a[i] < b[j]:
            i += 1
        else:
            j += 1
    return out


def merge_union(a, b):
    i = j = 0
    out = []
    while i < len(a) or j < len(b):
        if j == len(b) or (i < len(a) and a[i] < b[j]):
            x = a[i]
            i += 1
        elif i == len(a) or b[j] < a[i]:
            x = b[j]
            j += 1
        else:
            x = a[i]
            i += 1
            j += 1
        out.append(x)
    return out


def merge_difference(a, b):
    i = j = 0
    out = []
    while i < len(a):
        if j == len(b) or a[i] < b[j]:
            out.append(a[i])
            i += 1
        elif a[i] == b[j]:
            i += 1
            j += 1
        else:
            j += 1
    return out


def boolean_search(index, query, engine="merge"):
    if engine not in ("merge", "set"):
        raise ValueError("Invalid engine")
    parts = query.split()
    if not parts:
        raise ValueError("Empty query")
    result = None
    operator = "AND"
    expecting_term = True
    for part in parts:
        if part.upper() in {"AND", "OR", "NOT"}:
            if not expecting_term:
                operator = part.upper()
                expecting_term = True
                continue
            if result is None and part.upper() == "NOT":
                operator = "NOT"
                continue
            raise ValueError("Unexpected operator")
        tokens = list(tokenize(part))
        if len(tokens) != 1:
            raise ValueError("Expected one term")
        ids = [doc for doc, _ in index.postings.get(tokens[0], ())]
        if result is None:
            result = merge_difference(sorted(index.documents), ids) if operator == "NOT" else ids
        elif engine == "merge":
            result = {"AND": merge_intersection, "OR": merge_union, "NOT": merge_difference}[
                operator
            ](result, ids)
        else:
            a, b = set(result), set(ids)
            result = sorted({"AND": a & b, "OR": a | b, "NOT": a - b}[operator])
        operator = "AND"
        expecting_term = False
    if expecting_term:
        raise ValueError("Query ends with operator")
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("index", type=Path)
    p.add_argument("query")
    p.add_argument("--engine", choices=("merge", "set"), default="merge")
    p.add_argument("--format", choices=("binary", "pickle"), default="binary")
    a = p.parse_args()
    tracemalloc.start()
    t = time.perf_counter()
    idx = (load_binary if a.format == "binary" else load_pickle)(a.index)
    matches = boolean_search(idx, a.query, a.engine)
    elapsed = time.perf_counter() - t
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    print(f"Matches: {len(matches)}")
    for doc in matches[:50]:
        meta = idx.documents[doc]
        print(f"{doc}\tlen={meta.length}\t{meta.path}")
    print(f"Time: {elapsed:.4f} s; peak: {peak / 1048576:.3f} MiB")


if __name__ == "__main__":
    main()
