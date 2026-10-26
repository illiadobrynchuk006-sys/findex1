"""NumPy scorer must match the Python reference on ranking and score."""
from pathlib import Path

import pytest

from findex.index import build_index
from findex.model import Index
from findex.numpy_rank import NumpyScorer
from findex.rank import Scorer


@pytest.fixture
def index(tmp_path: Path) -> Index:
    for name, text in {
        "a.txt": "python python пошук дані",
        "b.txt": "python пошук",
        "c.txt": "пошук дані дані",
        "d.txt": "семантичний код",
    }.items():
        (tmp_path / name).write_text(text, encoding="utf-8")
    return Index(build_index(tmp_path))


@pytest.mark.parametrize("method", ["tfidf", "bm25"])
@pytest.mark.parametrize("query", [
    "python", "пошук", "python AND пошук", "python OR дані",
    "python AND NOT дані", "невідоме",
])
def test_same_ranking(index: Index, method: str, query: str):
    expected = Scorer(index).search(query, method=method, top_k=10)
    actual = NumpyScorer(index).search(query, method=method, top_k=10)
    assert [r.doc_id for r in actual] == [r.doc_id for r in expected]
    for left, right in zip(actual, expected):
        assert left.score == pytest.approx(right.score)


def test_topk_and_inputs(index: Index):
    scorer = NumpyScorer(index)
    assert len(scorer.search("пошук", top_k=1)) == 1
    assert scorer.search("пошук", top_k=0) == []
    with pytest.raises(ValueError):
        scorer.search("пошук", method="bad")
