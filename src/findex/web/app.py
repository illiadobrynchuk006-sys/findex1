"""FastAPI search service over an index loaded at startup."""
from __future__ import annotations

import logging
import os
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Literal

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from starlette.concurrency import run_in_threadpool

from findex.index import load_binary
from findex.model import Index
from findex.rank import Scorer, snippet
from findex.numpy_rank import NumpyScorer
from findex.query import evaluate, parse

logger = logging.getLogger(__name__)
templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))


class Settings(BaseSettings):
    index_path: Path = Path("index.bin")
    model_config = SettingsConfigDict(env_prefix="", extra="ignore")


class SearchItem(BaseModel):
    doc_id: str
    score: float
    snippet: str


class SearchResponse(BaseModel):
    q: str
    page: int
    total: int
    results: list[SearchItem]


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = Settings()
    app.state.index = None
    app.state.scorer = None
    if settings.index_path.is_file():
        try:
            index = Index(load_binary(settings.index_path))
            app.state.index = index
            app.state.scorer = Scorer(index)
        except (OSError, ValueError, KeyError) as exc:
            logger.error("Index loading failed: %s", exc)
    yield
    app.state.scorer = None
    app.state.index = None


app = FastAPI(title="findex", version="1.0.0", lifespan=lifespan)


def get_index(request: Request) -> Index:
    index = getattr(request.app.state, "index", None)
    if index is None:
        raise HTTPException(status_code=503, detail="Index is not loaded")
    return index


def get_scorer(request: Request, index: Annotated[Index, Depends(get_index)]) -> Scorer:
    scorer = getattr(request.app.state, "scorer", None)
    if scorer is None:
        scorer = Scorer(index)
    return scorer


@app.exception_handler(Exception)
async def unhandled_error(request: Request, exc: Exception):
    from fastapi.responses import JSONResponse

    error_id = str(uuid.uuid4())
    logger.exception("Unhandled error request_id=%s path=%s", error_id, request.url.path)
    return JSONResponse(status_code=500, content={"detail": "Internal error", "request_id": error_id})


@app.get("/health")
def health(request: Request):
    if getattr(request.app.state, "index", None) is None:
        raise HTTPException(status_code=503, detail="Index is not loaded")
    return {"status": "ok"}


@app.get("/stats")
def stats(index: Annotated[Index, Depends(get_index)]):
    return {"documents": index.num_docs, "terms": len(index), "avg_doc_length": index.avg_doc_length}


@app.get("/docs/{doc_id:path}")
def get_document(doc_id: str, index: Annotated[Index, Depends(get_index)]):
    from fastapi.responses import PlainTextResponse

    meta = index.documents.get(doc_id)
    if meta is None:
        raise HTTPException(status_code=404, detail="Document not found")
    path = Path(meta.path)
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Source document not found")
    return PlainTextResponse(path.read_text(encoding="utf-8", errors="replace"))


@app.get("/search", response_model=SearchResponse)
def search(
    scorer_instance: Annotated[Scorer, Depends(get_scorer)],
    q: Annotated[str, Query(min_length=1)],
    scorer: Literal["bm25", "tfidf"] = "bm25",
    page: Annotated[int, Query(ge=1)] = 1,
    engine: Literal["python", "numpy"] = "python",
):
    try:
        # Search engine currently exposes top-k rather than all hits.
        total = len(evaluate(parse(q), scorer_instance.index))
        ranker = NumpyScorer(scorer_instance.index) if engine == "numpy" else scorer_instance
        hits = ranker.search(q, method=scorer, top_k=page * 10)
        results = [
            SearchItem(doc_id=h.doc_id, score=h.score, snippet=snippet(h.path, q.split()))
            for h in hits[(page - 1) * 10 : page * 10]
        ]
        return SearchResponse(q=q, page=page, total=total, results=results)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc



@app.get("/search-async", response_model=SearchResponse)
async def search_async(
    scorer_instance: Annotated[Scorer, Depends(get_scorer)],
    q: Annotated[str, Query(min_length=1)],
    scorer: Literal["bm25", "tfidf"] = "bm25",
    page: Annotated[int, Query(ge=1)] = 1,
    engine: Literal["python", "numpy"] = "python",
):
    """Async route: offload CPU/blocking document I/O to a thread pool."""
    return await run_in_threadpool(search, scorer_instance, q, scorer, page, engine)


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(request, "home.html", {})
