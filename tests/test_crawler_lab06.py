"""Offline tests for async HTTP crawling and retry behavior."""
from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from findex.crawler import canonical, crawl, crawl_to_jsonl, fetch


def transport() -> httpx.MockTransport:
    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/robots.txt":
            return httpx.Response(200, text="User-agent: *\nDisallow: /private\n")
        if request.url.path == "/":
            return httpx.Response(
                200,
                headers={"content-type": "text/html"},
                text='<h1>Hello</h1><a href="/a">A</a><a href="/private">No</a>',
            )
        if request.url.path == "/a":
            return httpx.Response(
                200,
                headers={"content-type": "text/html"},
                text='<p>Second page</p><a href="/">Home</a>',
            )
        return httpx.Response(404)
    return httpx.MockTransport(handler)


def test_canonical() -> None:
    assert canonical("HTTPS://EXAMPLE.ORG/a#part") == "https://example.org/a"
    with pytest.raises(ValueError):
        canonical("file:///etc/passwd")


@pytest.mark.asyncio
async def test_crawl_robots_links_and_dedupe() -> None:
    async with httpx.AsyncClient(transport=transport()) as client:
        pages = [page async for page in crawl(
            "https://example.org/", client=client, max_pages=4, concurrency=3, delay=0
        )]
    assert {page.url for page in pages} == {
        "https://example.org/", "https://example.org/a"
    }


@pytest.mark.asyncio
async def test_jsonl(tmp_path) -> None:
    destination = tmp_path / "crawl.jsonl"
    async with httpx.AsyncClient(transport=transport()) as client:
        count = await crawl_to_jsonl(
            "https://example.org/", destination,
            client=client, max_pages=2, delay=0,
        )
    assert count == 2
    lines = [json.loads(line) for line in destination.read_text().splitlines()]
    assert len(lines) == 2
    assert all("url" in line and "text" in line for line in lines)


@pytest.mark.asyncio
async def test_transient_retry(monkeypatch) -> None:
    attempts = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return httpx.Response(503, headers={"Retry-After": "0"})
        return httpx.Response(200, text="ok", headers={"content-type": "text/html"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        page = await fetch(client, "https://example.org/")
    assert attempts == 2
    assert page is not None and "ok" in page.text
