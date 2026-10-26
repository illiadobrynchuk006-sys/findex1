"""Installed findex command-line interface."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Literal

import typer
from rich.console import Console
from rich.progress import Progress
from rich.table import Table

from findex.index import load_binary, save_binary
from findex.parallel import build_parallel
from findex.model import Index
from findex.rank import Scorer, snippet

app = typer.Typer(no_args_is_help=True, help="findex: corpus indexing and ranked search")
console = Console(stderr=True)
log = logging.getLogger("findex")


def _log(verbose: int) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose > 1 else logging.INFO if verbose else logging.WARNING
    )


@app.callback()
def callback(
    verbose: int = typer.Option(0, "-v", count=True, help="Increase diagnostic verbosity"),
) -> None:
    _log(verbose)


@app.command("index")
def index_command(
    data: Path = typer.Argument(..., help="Directory of .txt documents"),
    out: Path = typer.Option(Path("index.bin"), "--out", help="Output index"),
    workers: int = typer.Option(1, "--workers", min=1),
    executor: Literal["serial", "threads", "processes"] = typer.Option(
        "serial", "--executor"
    ),
) -> None:
    try:
        if not data.is_dir():
            raise ValueError(f"Corpus directory not found: {data}")
        with Progress(console=console) as progress:
            task = progress.add_task("Building index", total=None)
            index_data = build_parallel(data, workers=workers, executor=executor)
            progress.update(task, description="Saving index")
            out.parent.mkdir(parents=True, exist_ok=True)
            save_binary(index_data, out)
        console.print(
            f"Indexed {len(index_data.documents)} documents, {len(index_data.postings)} terms → {out}"
        )
    except (ValueError, OSError) as exc:
        log.error("%s", exc)
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc


@app.command("search")
def search_command(
    index_file: Path = typer.Argument(..., help="Saved index"),
    query: str = typer.Argument(..., help="Boolean expression"),
    scorer: Literal["bm25", "tfidf"] = typer.Option("bm25", "--scorer"),
    top: int = typer.Option(5, "--top", min=0),
    as_json: bool = typer.Option(False, "--json", help="Machine-readable stdout"),
) -> None:
    try:
        if not index_file.is_file():
            raise FileNotFoundError(index_file)
        index = Index(load_binary(index_file))
        hits = Scorer(index).search(query, method=scorer, top_k=top)
        rows = [{"doc_id": h.doc_id, "score": round(h.score, 6), "path": h.path} for h in hits]
        if as_json:
            typer.echo(json.dumps(rows, ensure_ascii=False))
        else:
            table = Table(title="Search results")
            table.add_column("Document")
            table.add_column("Score", justify="right")
            table.add_column("Snippet")
            terms = [w for w in query.split() if w.upper() not in ("AND", "OR", "NOT")]
            for h in hits:
                try:
                    preview = snippet(h.path, terms)
                except OSError:
                    preview = "[source not available]"
                table.add_row(h.doc_id, f"{h.score:.4f}", preview)
            console.print(table)
    except (ValueError, OSError) as exc:
        log.error("%s", exc)
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc


@app.command("stats")
def stats_command(data: Path = typer.Argument(...)) -> None:
    from findex.stats import analyze

    try:
        if not data.is_dir():
            raise FileNotFoundError(data)
        result = analyze(data)
        console.print(
            f"Documents: {result.documents}; tokens: {result.tokens}; vocabulary: {result.vocabulary}"
        )
    except (ValueError, OSError) as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc



@app.command("crawl")
def crawl_command(
    url: str = typer.Argument(..., help="Permitted starting URL"),
    out: Path = typer.Option(Path("crawl.jsonl"), "--out"),
    max_pages: int = typer.Option(200, "--max-pages", min=1),
    concurrency: int = typer.Option(5, "--concurrency", min=1),
) -> None:
    """Crawl a site, writing each fetched page to JSONL."""
    import asyncio

    from findex.crawler import crawl_to_jsonl

    try:
        total = asyncio.run(
            crawl_to_jsonl(
                url, out, max_pages=max_pages, concurrency=concurrency
            )
        )
        console.print(f"Saved {total} pages to {out}")
    except (ValueError, OSError) as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc


@app.command("serve")
def serve_command(
    port: int = typer.Option(8000, "--port", min=1, max=65535),
    workers: int = typer.Option(1, "--workers", min=1),
) -> None:
    """Start the findex FastAPI application."""
    import uvicorn

    uvicorn.run("findex.web.app:app", host="0.0.0.0", port=port, workers=workers)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
