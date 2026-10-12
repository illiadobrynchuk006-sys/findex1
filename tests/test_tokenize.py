from findex.tokenize import tokenize


def test_casefold_mixed_case():
    assert list(tokenize("PyThOn PYTHON python")) == ["python"] * 3


def test_cyrillic():
    assert list(tokenize("Українська МОВА")) == ["українська", "мова"]


def test_combining_accent_nfc():
    assert list(tokenize("cafe\u0301 café")) == ["café", "café"]


def test_punctuation():
    assert list(tokenize("Hello, world! (OK).")) == ["hello", "world", "ok"]


def test_empty_string():
    assert list(tokenize("")) == []


def test_apostrophes():
    assert list(tokenize("don't п'ять п’ять")) == ["don't", "п'ять", "п’ять"]


def test_hyphen_separates():
    assert list(tokenize("веб-пошук well-known")) == ["веб", "пошук", "well", "known"]


def test_digits_preserved():
    assert list(tokenize("Python3 2026 12.5")) == ["python3", "2026", "12", "5"]


def test_underscore_separator():
    assert list(tokenize("hello_world")) == ["hello", "world"]


def test_unicode_casefold():
    assert list(tokenize("Straße STRASSE")) == ["strasse", "strasse"]
