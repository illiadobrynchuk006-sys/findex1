import inspect
from pathlib import Path

from findex.corpus import iter_documents
from findex.stats import analyze
from findex.tokenize import tokenize


def test_generators():
    assert inspect.isgeneratorfunction(iter_documents)
    assert inspect.isgeneratorfunction(tokenize)


def test_documents_and_stats(tmp_path: Path):
    (tmp_path / "sub").mkdir()
    (tmp_path / "a.txt").write_text("Кіт кіт собака", encoding="utf-8")
    (tmp_path / "sub" / "b.txt").write_text("Собака 123", encoding="utf-8")
    docs = iter_documents(tmp_path)
    assert next(docs).doc_id == "a.txt"
    assert next(docs).doc_id == "sub/b.txt"
    for mode in ("lazy", "eager"):
        stats = analyze(tmp_path, mode=mode)
        assert (stats.documents, stats.tokens, stats.vocabulary) == (2, 5, 3)
        assert stats.most_common[:2] == [("кіт", 2), ("собака", 2)]
        limited = analyze(tmp_path, limit=1, mode=mode)
        assert (limited.documents, limited.tokens) == (1, 3)


def test_invalid_utf8_does_not_crash(tmp_path: Path):
    (tmp_path / "bad.txt").write_bytes(b"hello \xff world")
    docs = list(iter_documents(tmp_path))
    assert len(docs) == 1
    assert "\ufffd" in docs[0].text


def test_empty_corpus(tmp_path: Path):
    assert analyze(tmp_path).documents == 0
    assert analyze(tmp_path).tokens == 0
