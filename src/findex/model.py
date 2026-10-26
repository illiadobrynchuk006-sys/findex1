"""Pythonic index wrapper: Mapping protocol, dunders and context manager."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from functools import cached_property
from pathlib import Path

from findex.index import InvertedIndex, build_index


class Index(Mapping[str, list[tuple[str, int]]]):
    def __init__(self, raw: InvertedIndex):
        self.raw = raw

    def __getitem__(self, term: str) -> list[tuple[str, int]]:
        return self.raw.postings[term]

    def __iter__(self) -> Iterator[str]:
        return iter(self.raw.postings)

    def __len__(self) -> int:
        return len(self.raw.postings)

    def __repr__(self) -> str:
        return f"Index(terms={len(self)}, documents={self.num_docs})"

    @property
    def num_docs(self) -> int:
        return len(self.raw.documents)

    @cached_property
    def avg_doc_length(self) -> float:
        lengths = [doc.length for doc in self.raw.documents.values()]
        return sum(lengths) / len(lengths) if lengths else 0.0

    def df(self, term: str) -> int:
        return len(self.raw.postings.get(term, ()))

    @property
    def documents(self):
        return self.raw.documents

    @classmethod
    def from_path(cls, path: Path) -> "Index":
        return cls(build_index(Path(path)))


@contextmanager
def open_index(path: Path):
    """Build an index and expose it in a managed context."""
    index = Index.from_path(path)
    try:
        yield index
    finally:
        index.raw.postings.clear()
        index.raw.documents.clear()
        index.__dict__.pop("avg_doc_length", None)
