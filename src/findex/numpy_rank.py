"""Vectorized BM25 and TF-IDF using compact NumPy posting arrays."""
from __future__ import annotations

import math

import numpy as np

from findex.model import Index
from findex.query import evaluate, parse
from findex.rank import SearchResult, query_terms


class NumpyScorer:
    """Drop-in ranked search; preserve the lab-03 Boolean candidate set."""

    def __init__(self, index: Index, k1: float = 1.5, b: float = 0.75):
        self.index = index
        self.k1 = k1
        self.b = b
        self.ids = sorted(index.documents)
        self.ordinals = {doc: i for i, doc in enumerate(self.ids)}
        self.lengths = np.asarray(
            [index.documents[doc].length for doc in self.ids], dtype=np.int32
        )
        self.postings: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for term, pairs in index.raw.postings.items():
            self.postings[term] = (
                np.asarray([self.ordinals[doc] for doc, _ in pairs], dtype=np.int32),
                np.asarray([freq for _, freq in pairs], dtype=np.int32),
            )

    def score(self, term: str, method: str) -> tuple[np.ndarray, np.ndarray]:
        docs, counts = self.postings.get(
            term, (np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int32))
        )
        if not len(docs):
            return docs, np.empty(0, dtype=np.float64)
        tf = counts.astype(np.float64)
        n, df = len(self.ids), len(docs)
        if method == "tfidf":
            values = (1 + np.log(tf)) * (math.log((1 + n) / (1 + df)) + 1)
        elif method == "bm25":
            idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
            avg = float(self.lengths.mean()) if n else 1.0
            avg = avg or 1.0
            denom = tf + self.k1 * (
                1 - self.b + self.b * self.lengths[docs] / avg
            )
            values = idf * tf * (self.k1 + 1) / denom
        else:
            raise ValueError("method must be bm25 or tfidf")
        return docs, values

    def search(self, query: str, *, method: str = "bm25", top_k: int = 10):
        if method not in ("bm25", "tfidf") or top_k < 0:
            raise ValueError("Invalid scorer or top_k")
        tree = parse(query)
        candidates = evaluate(tree, self.index)
        if not candidates or top_k == 0:
            return []
        scores = np.zeros(len(self.ids), dtype=np.float64)
        for term in set(query_terms(tree)):
            docs, values = self.score(term, method)
            np.add.at(scores, docs, values)
        candidate_ids = np.asarray(
            sorted(self.ordinals[doc] for doc in candidates), dtype=np.int32
        )
        candidate_scores = scores[candidate_ids]
        k = min(top_k, len(candidate_ids))
        # Include any tie at the cut; stable tie-break uses the document ID.
        if k < len(candidate_ids):
            threshold = np.partition(candidate_scores, -k)[-k]
            keep = candidate_ids[candidate_scores >= threshold]
        else:
            keep = candidate_ids
        ranked = sorted(keep.tolist(), key=lambda i: (-scores[i], self.ids[i]))[:k]
        return [
            SearchResult(float(scores[i]), self.ids[i], self.index.documents[self.ids[i]].path)
            for i in ranked
        ]
