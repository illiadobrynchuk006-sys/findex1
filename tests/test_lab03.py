"""Lab 03 integration tests."""
import pytest
from findex.model import Index, open_index
from findex.query import parse, evaluate, And, Or, Not, Phrase
from findex.rank import Scorer, timed, snippet

@pytest.fixture
def corpus(tmp_path):
    (tmp_path / "a.txt").write_text("python python швидкий пошук", encoding="utf-8")
    (tmp_path / "b.txt").write_text("python повільний пошук", encoding="utf-8")
    (tmp_path / "c.txt").write_text("індекс даних", encoding="utf-8")
    return tmp_path

def test_mapping(corpus):
    idx = Index.from_path(corpus)
    assert len(idx) > 0
    assert "python" in idx
    assert idx["python"] == [("a.txt", 2), ("b.txt", 1)]
    assert idx.num_docs == 3
    assert idx.df("python") == 2
    assert idx.avg_doc_length > 0
    assert "Index(" in repr(idx)
    assert list(iter(idx))

def test_open_index(corpus):
    with open_index(corpus) as idx:
        assert idx.num_docs == 3
    assert idx.num_docs == 0

@pytest.mark.parametrize("query,expected", [
    ("python AND пошук", {"a.txt", "b.txt"}),
    ("python AND NOT повільний", {"a.txt"}),
    ("(python OR індекс) AND NOT повільний", {"a.txt", "c.txt"}),
    ('"швидкий пошук"', {"a.txt"}),
    ('"пошук швидкий"', set()),
])
def test_evaluate(corpus, query, expected):
    assert evaluate(parse(query), Index.from_path(corpus)) == expected

@pytest.mark.parametrize("query", ["", "python AND", "(python", "python OR )", '""'])
def test_bad_queries(query):
    with pytest.raises(ValueError):
        parse(query)

def test_ranking(corpus):
    scorer = Scorer(Index.from_path(corpus))
    for method in ("tfidf", "bm25"):
        hits = scorer.search("python", method=method)
        assert [r.doc_id for r in hits] == ["a.txt", "b.txt"]
        assert hits[0].score >= hits[1].score
        assert scorer.search("python", method=method) == hits
    assert scorer.cache_hits == 2

def test_scoring_checks(corpus):
    scorer = Scorer(Index.from_path(corpus))
    assert scorer.bm25("невідомий", "a.txt") == 0
    assert scorer.tfidf("невідомий", "a.txt") == 0
    with pytest.raises(ValueError):
        scorer.search("python", method="bad")

def test_snippet(corpus):
    assert "python" in snippet(str(corpus / "a.txt"), ["python"])

def test_timed_preserves_metadata():
    @timed
    def hello():
        return 42
    assert hello() == 42
    assert hello.__name__ == "hello"
