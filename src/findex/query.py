"""Recursive-descent query parser: AND, OR, NOT, parentheses, quoted phrases."""

from __future__ import annotations

import re
from dataclasses import dataclass

from findex.tokenize import tokenize


@dataclass(frozen=True, slots=True)
class Term:
    value: str


@dataclass(frozen=True, slots=True)
class Phrase:
    words: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class And:
    left: object
    right: object


@dataclass(frozen=True, slots=True)
class Or:
    left: object
    right: object


@dataclass(frozen=True, slots=True)
class Not:
    child: object


LEX = re.compile(r'\s*("[^"]*"|\(|\)|[^\s()"]+)')


def parse(text: str):
    tokens = LEX.findall(text)
    if not tokens:
        raise ValueError("Empty query")
    position = 0

    def peek():
        return tokens[position] if position < len(tokens) else None

    def consume():
        nonlocal position
        result = peek()
        position += 1
        return result

    def atom():
        token = peek()
        if token is None:
            raise ValueError("Unexpected end of query")
        if token.upper() == "NOT":
            consume()
            return Not(atom())
        if token == "(":
            consume()
            result = parse_or()
            if peek() != ")":
                raise ValueError("Missing closing parenthesis")
            consume()
            return result
        if token in (")", "AND", "OR"):
            raise ValueError(f"Unexpected token {token}")
        consume()
        if token.startswith('"'):
            words = tuple(tokenize(token[1:-1]))
            if not words:
                raise ValueError("Empty phrase")
            return Phrase(words)
        words = tuple(tokenize(token))
        if len(words) != 1:
            raise ValueError(f"Invalid term {token!r}")
        return Term(words[0])

    def parse_and():
        result = atom()
        while peek() is not None and peek().upper() == "AND":
            consume()
            result = And(result, atom())
        return result

    def parse_or():
        result = parse_and()
        while peek() is not None and peek().upper() == "OR":
            consume()
            result = Or(result, parse_and())
        return result

    tree = parse_or()
    if peek() is not None:
        raise ValueError(f"Unexpected token {peek()!r}")
    return tree


def evaluate(node, index):
    """Evaluate AST against document ID sets; phrase checks original documents."""
    from pathlib import Path

    from findex.tokenize import tokenize

    if isinstance(node, Term):
        return {doc for doc, _ in index.raw.postings.get(node.value, ())}
    if isinstance(node, Phrase):
        candidate = None
        for word in node.words:
            hits = {doc for doc, _ in index.raw.postings.get(word, ())}
            candidate = hits if candidate is None else candidate & hits
        answer = set()
        for doc_id in candidate or ():
            text = Path(index.documents[doc_id].path).read_text(encoding="utf-8", errors="replace")
            tokens = list(tokenize(text))
            n = len(node.words)
            if any(tuple(tokens[i : i + n]) == node.words for i in range(len(tokens) - n + 1)):
                answer.add(doc_id)
        return answer
    if isinstance(node, Not):
        return set(index.documents) - evaluate(node.child, index)
    if isinstance(node, And):
        return evaluate(node.left, index) & evaluate(node.right, index)
    if isinstance(node, Or):
        return evaluate(node.left, index) | evaluate(node.right, index)
    raise TypeError(f"Unsupported query node {type(node)}")
