"""Parallel indexing with module-level workers and deterministic merging."""
from __future__ import annotations

import multiprocessing as mp
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path
from typing import Literal

from findex.index import DocMeta, InvertedIndex
from findex.tokenize import tokenize

ExecutorKind = Literal["serial", "threads", "processes"]
Partial = tuple[dict[str, list[tuple[str, int]]], dict[str, DocMeta]]


def build_partial(paths: list[tuple[str, str]]) -> Partial:
    """Read files in the worker; accept (unique doc_id, path) rather than text."""
    postings: dict[str, list[tuple[str, int]]] = defaultdict(list)
    documents: dict[str, DocMeta] = {}
    for doc_id, filename in paths:
        with Path(filename).open("r", encoding="utf-8", errors="replace") as stream:
            counts = Counter(tokenize(stream.read()))
        if doc_id in documents:
            raise ValueError(f"Duplicate document id: {doc_id}")
        documents[doc_id] = DocMeta(sum(counts.values()), filename)
        for term, tf in counts.items():
            postings[term].append((doc_id, tf))
    return dict(postings), documents


def merge(parts: list[Partial]) -> InvertedIndex:
    """Merge chunks in any order into a stable sorted index."""
    postings: dict[str, list[tuple[str, int]]] = defaultdict(list)
    documents: dict[str, DocMeta] = {}
    for partial_posts, partial_docs in parts:
        overlap = documents.keys() & partial_docs.keys()
        if overlap:
            raise ValueError(f"Duplicate document IDs: {sorted(overlap)!r}")
        documents.update(partial_docs)
        for term, items in partial_posts.items():
            postings[term].extend(items)
    for term, items in postings.items():
        items.sort(key=lambda pair: pair[0])
        if any(a[0] == b[0] for a, b in zip(items, items[1:])):
            raise ValueError(f"Duplicate posting for {term}")
    return InvertedIndex(dict(sorted(postings.items())), dict(sorted(documents.items())))


def build_parallel(
    root: Path,
    workers: int = 1,
    executor: ExecutorKind = "serial",
) -> InvertedIndex:
    if workers < 1:
        raise ValueError("workers must be >= 1")
    if executor not in ("serial", "threads", "processes"):
        raise ValueError("Unknown executor")
    root = Path(root)
    if not root.is_dir():
        raise NotADirectoryError(root)
    files = [(p.relative_to(root).as_posix(), str(p.resolve()))
             for p in sorted(root.rglob("*.txt")) if p.is_file()]
    if not files:
        return merge([])
    count = min(workers, len(files))
    chunks = [files[i::count] for i in range(count)]
    if executor == "serial":
        parts = [build_partial(chunk) for chunk in chunks]
    elif executor == "threads":
        with ThreadPoolExecutor(max_workers=count) as pool:
            parts = list(pool.map(build_partial, chunks))
    else:
        with ProcessPoolExecutor(
            max_workers=count, mp_context=mp.get_context("spawn")
        ) as pool:
            parts = list(pool.map(build_partial, chunks))
    return merge(parts)
