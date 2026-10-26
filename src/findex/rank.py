"""TF-IDF/BM25 scoring, timed searches, cached queries and snippets."""

from __future__ import annotations

import argparse
import heapq
import logging
import math
import time
from dataclasses import dataclass
from functools import wraps
from pathlib import Path

from findex.model import Index
from findex.query import And, Not, Or, Phrase, Term, evaluate, parse

logger = logging.getLogger(__name__)


@dataclass(order=True, frozen=True, slots=True)
class SearchResult:
    score: float
    doc_id: str
    path: str


def timed(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        try:
            return fn(*args, **kwargs)
        finally:
            logger.info("%s took %.6fs", fn.__name__, time.perf_counter() - start)

    return wrapper


def query_terms(node):
    if isinstance(node, Term):
        return [node.value]
    if isinstance(node, Phrase):
        return list(node.words)
    if isinstance(node, Not):
        return []
    if isinstance(node, (And, Or)):
        return query_terms(node.left) + query_terms(node.right)
    return []


class Scorer:
    def __init__(self, index: Index, *, k1: float = 1.5, b: float = 0.75):
        self.index = index
        self.k1 = k1
        self.b = b
        self._freq = {}
        self._cache = {}
        self.cache_hits = 0

    def _frequencies(self, term):
        if term not in self._freq:
            self._freq[term] = dict(self.index.raw.postings.get(term, ()))
        return self._freq[term]

    def tfidf(self, term: str, doc_id: str) -> float:
        tf = self._frequencies(term).get(doc_id, 0)
        if not tf:
            return 0.0
        df = self.index.df(term)
        return (1.0 + math.log(tf)) * (math.log((1 + self.index.num_docs) / (1 + df)) + 1)

    def bm25(self, term: str, doc_id: str) -> float:
        tf = self._frequencies(term).get(doc_id, 0)
        if not tf:
            return 0.0
        n = self.index.num_docs
        df = self.index.df(term)
        idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
        length = self.index.documents[doc_id].length
        avg = self.index.avg_doc_length or 1
        denom = tf + self.k1 * (1 - self.b + self.b * length / avg)
        return idf * tf * (self.k1 + 1) / denom

    @timed
    def search(self, query: str, *, method="bm25", top_k=10):
        if method not in {"tfidf", "bm25"}:
            raise ValueError("method must be tfidf or bm25")
        if top_k < 0:
            raise ValueError("top_k must be nonnegative")
        key = (query, method, top_k)
        if key in self._cache:
            self.cache_hits += 1
            logger.info("cache hit: %r", query)
            return list(self._cache[key])
        tree = parse(query)
        candidates = evaluate(tree, self.index)
        terms = set(query_terms(tree))
        score_fn = self.bm25 if method == "bm25" else self.tfidf
        ranked = heapq.nlargest(
            top_k,
            (
                SearchResult(
                    sum(score_fn(term, doc) for term in terms), doc, self.index.documents[doc].path
                )
                for doc in candidates
            ),
        )
        ranked.sort(key=lambda r: (-r.score, r.doc_id))
        self._cache[key] = tuple(ranked)
        return list(ranked)


def snippet(path: str, terms: list[str], width=160) -> str:
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    folded = text.casefold()
    locations = [folded.find(term.casefold()) for term in terms]
    found = [p for p in locations if p >= 0]
    pos = min(found) if found else 0
    start = max(0, pos - width // 2)
    end = min(len(text), start + width)
    return (
        ("…" if start else "")
        + text[start:end].replace("\n", " ")
        + ("…" if end < len(text) else "")
    )


def main():
    parser = argparse.ArgumentParser(description="Ranked retrieval using lab-03")
    parser.add_argument("data", type=Path)
    parser.add_argument("query")
    parser.add_argument("--method", choices=("tfidf", "bm25"), default="bm25")
    parser.add_argument("--top", type=int, default=5)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    start = time.perf_counter()
    idx = Index.from_path(args.data)
    logger.info("build took %.4fs", time.perf_counter() - start)
    scorer = Scorer(idx)
    tree = parse(args.query)
    for result in scorer.search(args.query, method=args.method, top_k=args.top):
        print(f"{result.score:.4f}\t{result.doc_id}\t{snippet(result.path, query_terms(tree))}")


if __name__ == "__main__":
    main()
