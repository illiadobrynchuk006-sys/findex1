from pathlib import Path
import pytest
from findex.index import build_index, build_packed_postings, save_binary, load_binary, save_pickle, load_pickle
from findex.search import boolean_search, merge_intersection, merge_union, merge_difference

@pytest.fixture
def corpus(tmp_path):
    (tmp_path / "a.txt").write_text("Кіт кіт пес", encoding="utf-8")
    (tmp_path / "b.txt").write_text("кіт риба", encoding="utf-8")
    (tmp_path / "c.txt").write_text("пес риба", encoding="utf-8")
    return tmp_path

def test_build(corpus):
    idx = build_index(corpus)
    assert idx.postings["кіт"] == [("a.txt", 2), ("b.txt", 1)]
    assert idx.documents["a.txt"].length == 3

@pytest.mark.parametrize("mode", ["binary", "pickle"])
def test_roundtrip(corpus, tmp_path, mode):
    idx = build_index(corpus)
    p = tmp_path / "index.bin"
    save = save_binary if mode == "binary" else save_pickle
    load = load_binary if mode == "binary" else load_pickle
    save(idx, p)
    assert load(p) == idx

@pytest.mark.parametrize("query,expected", [
    ("кіт AND пес", ["a.txt"]),
    ("кіт OR пес", ["a.txt", "b.txt", "c.txt"]),
    ("кіт NOT пес", ["b.txt"]),
    ("NOT кіт", ["c.txt"]),
    ("кіт пес", ["a.txt"]),
    ("невідомий", []),
    ("КІТ AND риба", ["b.txt"]),
])
def test_search(corpus, query, expected):
    idx = build_index(corpus)
    assert boolean_search(idx, query, "merge") == expected
    assert boolean_search(idx, query, "set") == expected

def test_merge():
    assert merge_intersection(["a", "c"], ["b", "c"]) == ["c"]
    assert merge_union(["a", "c"], ["b", "c"]) == ["a", "b", "c"]
    assert merge_difference(["a", "c"], ["b", "c"]) == ["a"]

def test_array(corpus):
    assert len(build_packed_postings(build_index(corpus))["кіт"]) == 4

def test_bad_binary(tmp_path):
    p = tmp_path / "broken"
    p.write_bytes(b"bad")
    with pytest.raises(ValueError):
        load_binary(p)
