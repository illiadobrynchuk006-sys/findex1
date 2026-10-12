"""Lazy document reader for a directory tree of UTF-8 .txt files."""

import logging
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Document:
    doc_id: str
    path: Path
    text: str


def iter_documents(root: Path) -> Iterator[Document]:
    """Yield one document at a time, never materializing the whole corpus.

    Invalid UTF-8 bytes are replaced with U+FFFD and reported as a warning.
    The document text itself is necessarily held in memory until the next yield.
    """
    root = Path(root)
    if not root.is_dir():
        raise NotADirectoryError(f"Папка корпусу не існує: {root}")
    for path in sorted(root.rglob("*.txt")):
        if not path.is_file():
            continue
        try:
            with path.open("r", encoding="utf-8", errors="replace") as file:
                text = file.read()
        except OSError as exc:
            logger.warning("Пропущено %s: %s", path, exc)
            continue
        if "\ufffd" in text:
            logger.warning("Пошкоджене UTF-8 у %s: байти замінено символом U+FFFD", path)
        yield Document(doc_id=path.relative_to(root).as_posix(), path=path, text=text)
