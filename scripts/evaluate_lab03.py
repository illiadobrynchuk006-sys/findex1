"""Interactive relevance judgments: compute precision@5 for 10 queries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from findex.model import Index
from findex.rank import Scorer


def precision_at_k(predicted, relevant, k=5):
    return len(set(predicted[:k]) & set(relevant)) / k


def main():
    p = argparse.ArgumentParser()
    p.add_argument("corpus", type=Path)
    p.add_argument("judgments", type=Path, help="JSON: [{query:..., relevant:[doc_ids...]}, ...]")
    args = p.parse_args()
    items = json.loads(args.judgments.read_text(encoding="utf-8"))
    if len(items) != 10:
        raise ValueError("Provide exactly 10 judged queries")
    scorer = Scorer(Index.from_path(args.corpus))
    for item in items:
        query, relevant = item["query"], item["relevant"]
        tf = [hit.doc_id for hit in scorer.search(query, method="tfidf", top_k=5)]
        bm = [hit.doc_id for hit in scorer.search(query, method="bm25", top_k=5)]
        print(
            f"{query!r}: TF-IDF P@5={precision_at_k(tf, relevant):.2f}; BM25 P@5={precision_at_k(bm, relevant):.2f}"
        )


if __name__ == "__main__":
    main()
