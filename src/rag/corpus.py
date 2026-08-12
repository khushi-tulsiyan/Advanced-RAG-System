"""Persistence for the chunk corpus.

Chunks are stored as JSONL in index order. Row *i* of the vector index always
corresponds to line *i* of this file, which is the invariant the previous
implementation lacked: it built BM25 by iterating ``os.listdir`` while the
dense index came from a separately written JSON file, so the two orderings
could disagree and retrieval returned unrelated passages.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List

from .types import Chunk


def save_chunks(chunks: Iterable[Chunk], path: Path) -> int:
    """Writes chunks to ``path`` as JSONL, preserving order. Returns the count."""
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for chunk in chunks:
            handle.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n")
            count += 1
    return count


def load_chunks(path: Path) -> List[Chunk]:
    """Reads chunks previously written by :func:`save_chunks`."""
    if not path.exists():
        raise FileNotFoundError(f"Chunk corpus not found at {path}. Run scripts/build_index.py first.")

    chunks: List[Chunk] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                chunks.append(Chunk.from_dict(json.loads(line)))
            except (json.JSONDecodeError, KeyError) as exc:
                raise ValueError(f"Malformed chunk on line {line_number} of {path}: {exc}") from exc
    return chunks
