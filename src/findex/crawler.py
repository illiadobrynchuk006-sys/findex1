"""Polite asyncio web crawler with streaming JSONL output."""
from __future__ import annotations

import asyncio
import json
import logging
import random
import time
from collections import defaultdict
from collections.abc import AsyncIterator
from dataclasses import asdict, dataclass
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urldefrag, urljoin, urlsplit, urlunsplit
from urllib.robotparser import RobotFileParser

import httpx

logger = logging.getLogger(__name__)


class PageParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.links: list[str] = []
        self.words: list[str] = []
        self.hidden = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in ("script", "style", "noscript"):
            self.hidden += 1
        if tag == "a":
            for name, value in attrs:
                if name == "href" and value:
                    self.links.append(value)

    def handle_endtag(self, tag: str) -> None:
        if tag in ("script", "style", "noscript"):
            self.hidden = max(0, self.hidden - 1)

    def handle_data(self, data: str) -> None:
        if not self.hidden:
            self.words.append(data)


def canonical(url: str) -> str:
    url, _ = urldefrag(url)
    parts = urlsplit(url)
    if parts.scheme.lower() not in ("http", "https") or not parts.hostname:
        raise ValueError("Only absolute HTTP(S) URLs are supported")
    hostname = parts.hostname.lower()
    port = parts.port
    netloc = hostname if port in (None, 80 if parts.scheme == "http" else 443) else f"{hostname}:{port}"
    return urlunsplit((parts.scheme.lower(), netloc, parts.path or "/", parts.query, ""))


@dataclass(frozen=True)
class Page:
    url: str
    status: int
    text: str
    elapsed: float
    links: tuple[str, ...] = ()

    def as_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


class RobotsCache:
    def __init__(self, client: httpx.AsyncClient, agent: str):
        self.client = client
        self.agent = agent
        self.entries: dict[str, RobotFileParser] = {}
        self.locks: defaultdict[str, asyncio.Lock] = defaultdict(asyncio.Lock)

    async def allowed(self, url: str) -> bool:
        parts = urlsplit(url)
        origin = f"{parts.scheme}://{parts.netloc}"
        async with self.locks[origin]:
            if origin not in self.entries:
                parser = RobotFileParser()
                parser.set_url(origin + "/robots.txt")
                try:
                    response = await self.client.get(origin + "/robots.txt")
                    if response.status_code == 404:
                        parser.parse([])
                    elif response.status_code == 200:
                        parser.parse(response.text.splitlines())
                    else:
                        logger.warning("robots status %s; deny %s", response.status_code, origin)
                        return False
                except httpx.HTTPError:
                    logger.warning("robots unavailable; deny %s", origin)
                    return False
                self.entries[origin] = parser
            return self.entries[origin].can_fetch(self.agent, url)


async def fetch(
    client: httpx.AsyncClient, url: str, *, retries: int = 3
) -> Page | None:
    for attempt in range(retries + 1):
        start = time.perf_counter()
        try:
            response = await client.get(url)
            elapsed = time.perf_counter() - start
            logger.info("fetch %s %d %.3fs", url, response.status_code, elapsed)
            if response.status_code == 429 or response.status_code >= 500:
                if attempt < retries:
                    after = response.headers.get("Retry-After", "")
                    try:
                        delay = max(0, min(float(after), 30))
                    except ValueError:
                        delay = min(8, 0.3 * (2**attempt) + random.uniform(0, 0.1))
                    await asyncio.sleep(delay)
                    continue
            if response.is_error:
                return None
            if "text/html" not in response.headers.get("content-type", "").lower():
                return None
            parser = PageParser()
            parser.feed(response.text)
            return Page(url, response.status_code, " ".join(parser.words), elapsed, tuple(parser.links))
        except (httpx.TimeoutException, httpx.TransportError) as exc:
            logger.warning("fetch failure %s: %s", url, exc)
            if attempt == retries:
                return None
            await asyncio.sleep(min(8, 0.3 * (2**attempt) + random.uniform(0, 0.1)))
    return None


async def crawl(
    start_url: str,
    *,
    max_pages: int = 200,
    concurrency: int = 5,
    delay: float = 0.1,
    client: httpx.AsyncClient | None = None,
) -> AsyncIterator[Page]:
    if max_pages < 1 or concurrency < 1 or delay < 0:
        raise ValueError("Invalid crawler limits")
    origin = canonical(start_url)
    host = urlsplit(origin).hostname
    user_agent = "findex-course-crawler/0.6 (+https://github.com/illiadobrynchuk006-sys/findex1)"
    own_client = client is None
    if own_client:
        client = httpx.AsyncClient(
            timeout=httpx.Timeout(10), follow_redirects=False,
            headers={"User-Agent": user_agent}, max_redirects=5,
        )
    assert client is not None
    try:
        robots = RobotsCache(client, user_agent)
        queue: asyncio.Queue[str] = asyncio.Queue()
        queue.put_nowait(origin)
        seen = {origin}
        requested = 0
        lock = asyncio.Lock()
        semaphore = asyncio.Semaphore(concurrency)
        last_request: defaultdict[str, float] = defaultdict(float)
        results: asyncio.Queue[Page] = asyncio.Queue()

        async def worker() -> None:
            nonlocal requested
            while True:
                url = await queue.get()
                try:
                    if not await robots.allowed(url):
                        logger.info("robots deny %s", url)
                        continue
                    async with lock:
                        if requested >= max_pages:
                            continue
                        requested += 1
                    async with semaphore:
                        async with lock:
                            waiting = max(0.0, delay - (time.monotonic() - last_request[urlsplit(url).netloc]))
                            if waiting:
                                await asyncio.sleep(waiting)
                            last_request[urlsplit(url).netloc] = time.monotonic()
                        page = await fetch(client, url)
                    if page is None:
                        continue
                    await results.put(page)
                    for href in page.links:
                        try:
                            target = canonical(urljoin(url, href))
                        except (ValueError, UnicodeError):
                            continue
                        if urlsplit(target).hostname != host:
                            continue
                        async with lock:
                            if target in seen or len(seen) >= max_pages:
                                continue
                            seen.add(target)
                            queue.put_nowait(target)
                finally:
                    queue.task_done()

        async def supervise() -> None:
            async with asyncio.TaskGroup() as group:
                tasks = [group.create_task(worker()) for _ in range(concurrency)]
                await queue.join()
                for task in tasks:
                    task.cancel()

        supervisor = asyncio.create_task(supervise())
        try:
            while True:
                if supervisor.done() and results.empty():
                    await supervisor
                    break
                try:
                    page = await asyncio.wait_for(results.get(), timeout=0.1)
                except asyncio.TimeoutError:
                    continue
                yield page
        finally:
            if not supervisor.done():
                supervisor.cancel()
            try:
                await supervisor
            except asyncio.CancelledError:
                pass
    finally:
        if own_client:
            await client.aclose()


async def crawl_to_jsonl(start_url: str, destination: Path, **kwargs: object) -> int:
    destination.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with destination.open("w", encoding="utf-8") as stream:
        async for page in crawl(start_url, **kwargs):
            stream.write(page.as_json() + "\n")
            count += 1
    return count
