"""Parallel indexing invariants on a small fixed UTF-8 corpus."""
from pathlib import Path

import pytest

from findex.parallel import build_parallel, build_partial, merge


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    for i, content in enumerate(["python кіт", "python python", "кіт пошук", "пошук"]):
        (tmp_path / f"{i:03}.txt").write_text(content, encoding="utf-8")
    return tmp_path


@pytest.mark.parametrize("executor", ["serial", "threads", "processes"])
@pytest.mark.parametrize("workers", [1, 2])
def test_same_index(corpus: Path, executor: str, workers: int) -> None:
    expected = build_parallel(corpus, workers=1, executor="serial")
    actual = build_parallel(corpus, workers=workers, executor=executor)
    assert actual == expected
    for postings in actual.postings.values():
        assert [doc for doc, _ in postings] == sorted(doc for doc, _ in postings)


def test_serial_merge_matches_direct(corpus: Path) -> None:
    paths = [(p.name, str(p.resolve())) for p in sorted(corpus.glob("*.txt"))]
    direct = build_partial(paths)
    assert merge([direct]) == build_parallel(corpus, 1, "serial")
    left = build_partial(paths[:2])
    right = build_partial(paths[2:])
    assert merge([right, left]) == merge([left, right])


def test_bad_arguments(corpus: Path) -> None:
    with pytest.raises(ValueError):
        build_parallel(corpus, workers=0)
    with pytest.raises(ValueError):
        build_parallel(corpus, executor="invalid")  # type: ignore[arg-type]


def test_worker_failure_is_propagated(tmp_path: Path) -> None:
    with pytest.raises(OSError):
        build_partial([("missing", str(tmp_path / "absent.txt"))])


def test_duplicate_document_id(corpus: Path) -> None:
    name = str(corpus / "000.txt")
    with pytest.raises(ValueError):
        merge([build_partial([("same", name)]), build_partial([("same", name)])])
