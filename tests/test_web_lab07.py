"""FastAPI API tests using an in-memory index override."""
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from findex.index import build_index
from findex.model import Index
from findex.rank import Scorer
from findex.web.app import app, get_index, get_scorer


@pytest.fixture
def client(tmp_path: Path):
    (tmp_path / "a.txt").write_text("python пошук python", encoding="utf-8")
    (tmp_path / "b.txt").write_text("java дані", encoding="utf-8")
    index = Index(build_index(tmp_path))
    scorer = Scorer(index)
    app.dependency_overrides[get_index] = lambda: index
    app.dependency_overrides[get_scorer] = lambda: scorer
    try:
        with TestClient(app) as test_client:
            yield test_client
    finally:
        app.dependency_overrides.clear()


def test_success(client):
    r = client.get("/search", params={"q": "python"})
    assert r.status_code == 200
    assert r.json()["results"][0]["doc_id"] == "a.txt"


def test_invalid_search(client):
    assert client.get("/search", params={"q": ""}).status_code == 422
    assert client.get("/search", params={"q": "python", "page": 0}).status_code == 422
    assert client.get("/search", params={"q": "python AND"}).status_code == 422


def test_document_missing(client):
    assert client.get("/docs/absent.txt").status_code == 404
    assert "python" in client.get("/docs/a.txt").text


def test_stats(client):
    assert client.get("/stats").json()["documents"] == 2


def test_html(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "findex" in r.text


def test_health_unloaded(monkeypatch, tmp_path):
    monkeypatch.setenv("INDEX_PATH", str(tmp_path / "missing-index.bin"))
    with TestClient(app) as client:
        assert client.get("/health").status_code == 503
