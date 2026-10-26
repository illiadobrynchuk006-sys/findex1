"""CLI integration tests."""

import json

from typer.testing import CliRunner

from findex.cli import app

runner = CliRunner()


def test_help():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "index" in result.stdout
    assert "search" in result.stdout


def test_missing_index(tmp_path):
    result = runner.invoke(app, ["search", str(tmp_path / "missing.bin"), "python"])
    assert result.exit_code != 0
    assert "Error:" in result.output


def test_roundtrip(tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "one.txt").write_text("python програмування python", encoding="utf-8")
    (docs / "two.txt").write_text("java програмування", encoding="utf-8")
    destination = tmp_path / "index.bin"
    result = runner.invoke(app, ["index", str(docs), "--out", str(destination)])
    assert result.exit_code == 0, result.output
    assert destination.exists()
    result = runner.invoke(app, ["search", str(destination), "python", "--json"])
    assert result.exit_code == 0, result.output
    values = json.loads(result.stdout)
    assert [item["doc_id"] for item in values] == ["one.txt"]
    assert all("score" in item for item in values)


def test_bad_query(tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "one.txt").write_text("python", encoding="utf-8")
    output = tmp_path / "idx.bin"
    assert runner.invoke(app, ["index", str(docs), "--out", str(output)]).exit_code == 0
    result = runner.invoke(app, ["search", str(output), "python AND"])
    assert result.exit_code != 0
