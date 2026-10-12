"""Tokenizer: NFC, casefold, Unicode words; apostrophes, hyphens and digits."""

import re
import unicodedata
from collections.abc import Iterator

# Apostrophes inside a word are accepted (ASCII ' and typographic ’).
# A hyphen separates words. Digits are retained, including standalone numbers.
# Underscores are treated as separators; \w is Unicode-aware in Python.
WORD_RE = re.compile(r"[^\W_]+(?:['’][^\W_]+)*", re.UNICODE)


def tokenize(text: str) -> Iterator[str]:
    """Yield normalized tokens lazily using re.finditer()."""
    normalized = unicodedata.normalize("NFC", text).casefold()
    for match in WORD_RE.finditer(normalized):
        yield match.group()
