"""Count document and token statistics in one pass.

Run: python -m findex.stats data/ [--limit 100] [--mode lazy|eager]
"""

import argparse
import logging
import time
import tracemalloc
from collections import Counter
from dataclasses import dataclass
from itertools import islice
from pathlib import Path

from findex.corpus import iter_documents
from findex.tokenize import tokenize


@dataclass
class CorpusStats:
    documents: int
    tokens: int
    vocabulary: int
    most_common: list[tuple[str, int]]


def analyze(root: Path, limit: int | None = None, mode: str = "lazy") -> CorpusStats:
    """Return stats in one pass; eager mode intentionally stores lists."""
    if limit is not None and limit < 0:
        raise ValueError("--limit має бути невід'ємним")
    docs = iter_documents(root)
    if limit is not None:
        docs = islice(docs, limit)

    counts: Counter[str] = Counter()
    document_count = 0
    token_count = 0

    if mode == "lazy":
        for doc in docs:
            document_count += 1
            # update consumes the generator without a full token list
            for token in tokenize(doc.text):
                counts[token] += 1
                token_count += 1
    elif mode == "eager":
        # A purposely wasteful baseline: all documents and token lists in RAM.
        document_list = list(docs)
        token_lists = [list(tokenize(doc.text)) for doc in document_list]
        document_count = len(document_list)
        for tokens in token_lists:
            counts.update(tokens)
            token_count += len(tokens)
    else:
        raise ValueError("mode має бути 'lazy' або 'eager'")

    return CorpusStats(document_count, token_count, len(counts), counts.most_common(50))


def measure(
    root: Path, limit: int | None = None, mode: str = "lazy"
) -> tuple[CorpusStats, float, int]:
    """Measure wall time and peak traced Python memory for a single analysis."""
    tracemalloc.start()
    start = time.perf_counter()
    try:
        stats = analyze(root, limit=limit, mode=mode)
        elapsed = time.perf_counter() - start
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return stats, elapsed, peak


def main() -> None:
    parser = argparse.ArgumentParser(description="Статистика текстового корпусу")
    parser.add_argument("root", type=Path, help="Каталог із .txt документами")
    parser.add_argument(
        "--limit", type=int, default=None, help="Опрацювати тільки перші N документів"
    )
    parser.add_argument("--mode", choices=("lazy", "eager"), default="lazy")
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
    try:
        stats, elapsed, peak = measure(args.root, args.limit, args.mode)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(f"Режим: {args.mode}")
    print(f"Документів: {stats.documents}")
    print(f"Токенів: {stats.tokens}")
    print(f"Розмір словника: {stats.vocabulary}")
    print("Топ-50 термінів:")
    for idx, (token, count) in enumerate(stats.most_common, start=1):
        print(f"{idx:2}. {token:24} {count}")
    print(f"Час: {elapsed:.4f} с")
    print(f"Пікова пам'ять (tracemalloc): {peak / (1024 * 1024):.3f} MiB")


if __name__ == "__main__":
    main()
