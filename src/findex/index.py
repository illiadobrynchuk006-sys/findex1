"""Inverted index building and persistence."""
from __future__ import annotations
import argparse
import pickle
import struct
import time
import tracemalloc
from array import array
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from findex.corpus import iter_documents
from findex.tokenize import tokenize

@dataclass(frozen=True, slots=True)
class DocMeta:
    length: int
    path: str

@dataclass(slots=True)
class InvertedIndex:
    postings: dict[str, list[tuple[str, int]]]
    documents: dict[str, DocMeta]

def build_index(root: Path) -> InvertedIndex:
    postings = defaultdict(list)
    documents = {}
    for doc in iter_documents(root):
        counts = Counter(tokenize(doc.text))
        if doc.doc_id in documents:
            raise ValueError("Duplicate document ID")
        documents[doc.doc_id] = DocMeta(sum(counts.values()), str(doc.path))
        for term, count in counts.items():
            postings[term].append((doc.doc_id, count))
    return InvertedIndex({term: sorted(items) for term, items in postings.items()}, documents)

def save_pickle(index, path):
    with Path(path).open("wb") as f:
        pickle.dump(index, f, protocol=pickle.HIGHEST_PROTOCOL)

def load_pickle(path):
    # Never use pickle with untrusted files.
    with Path(path).open("rb") as f:
        return pickle.load(f)

def save_binary(index, path):
    """Binary format: length-prefixed JSON UTF-8 payload."""
    import json
    data = {"documents": {k: [v.length, v.path] for k, v in index.documents.items()},
            "postings": index.postings}
    payload = json.dumps(data, ensure_ascii=False).encode("utf-8")
    with Path(path).open("wb") as f:
        f.write(b"FIDX02\0")
        f.write(struct.pack("<I", len(payload)))
        f.write(payload)

def load_binary(path):
    import json
    with Path(path).open("rb") as f:
        if f.read(7) != b"FIDX02\0":
            raise ValueError("Invalid index header")
        size_bytes = f.read(4)
        if len(size_bytes) != 4:
            raise ValueError("Truncated index")
        size = struct.unpack("<I", size_bytes)[0]
        if size > 100_000_000:
            raise ValueError("Index too large")
        payload = f.read(size)
        if len(payload) != size or f.read(1):
            raise ValueError("Corrupt index")
    data = json.loads(payload)
    return InvertedIndex({k: [tuple(p) for p in v] for k, v in data["postings"].items()},
                         {k: DocMeta(*v) for k, v in data["documents"].items()})

def build_packed_postings(index):
    ordinals = {key: i for i, key in enumerate(sorted(index.documents))}
    return {term: array("I", (item for doc, tf in posts for item in (ordinals[doc], tf)))
            for term, posts in index.postings.items()}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data", type=Path)
    parser.add_argument("--out", type=Path, default=Path("index.bin"))
    parser.add_argument("--format", choices=["binary", "pickle"], default="binary")
    args = parser.parse_args()
    tracemalloc.start()
    start = time.perf_counter()
    idx = build_index(args.data)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    (save_binary if args.format == "binary" else save_pickle)(idx, args.out)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    print(f"Documents: {len(idx.documents)}; terms: {len(idx.postings)}")
    print(f"Saved {args.out} ({args.out.stat().st_size} bytes)")
    print(f"Time: {time.perf_counter()-start:.4f} s; peak: {peak/1048576:.3f} MiB")

if __name__ == "__main__":
    main()
