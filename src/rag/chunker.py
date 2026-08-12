"""Document loading and chunking.

Implements a self-contained recursive character splitter so the project does
not need to pull in the whole ``langchain`` dependency tree for one utility.
Splits are attempted on progressively finer separators (paragraph, line,
sentence, word, character) so a chunk boundary lands on a natural break
whenever one exists inside the size budget.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from pathlib import Path
from typing import Iterable, List, Sequence

from .config import Settings, settings as default_settings
from .types import Chunk

logger = logging.getLogger(__name__)

DEFAULT_SEPARATORS: Sequence[str] = ("\n\n", "\n", ". ", " ", "")
SUPPORTED_SUFFIXES = (".txt", ".md", ".markdown")


class TextChunker:
    """Splits text into overlapping chunks suitable for embedding."""

    def __init__(
        self,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
        separators: Sequence[str] = DEFAULT_SEPARATORS,
        settings: Settings = default_settings,
    ) -> None:
        self.chunk_size = chunk_size if chunk_size is not None else settings.chunk_size
        self.chunk_overlap = chunk_overlap if chunk_overlap is not None else settings.chunk_overlap
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if not 0 <= self.chunk_overlap < self.chunk_size:
            raise ValueError("chunk_overlap must be non-negative and smaller than chunk_size")
        self.separators = tuple(separators)

    # -- splitting -------------------------------------------------------
    def chunk_text(self, text: str) -> List[str]:
        """Splits ``text`` into overlapping chunks of at most ``chunk_size`` characters."""
        text = _normalise_whitespace(text)
        if not text:
            return []
        pieces = self._split_recursive(text, self.separators)
        return self._merge_with_overlap(pieces)

    def _split_recursive(self, text: str, separators: Sequence[str]) -> List[str]:
        """Breaks ``text`` into fragments that each fit within ``chunk_size``."""
        if len(text) <= self.chunk_size:
            return [text]
        if not separators:
            return _hard_wrap(text, self.chunk_size)

        separator, rest = separators[0], separators[1:]
        if separator == "":
            return _hard_wrap(text, self.chunk_size)

        parts = [part for part in text.split(separator) if part]
        if len(parts) == 1:
            # Separator absent; try the next finer one.
            return self._split_recursive(text, rest)

        # Re-attach punctuation-bearing separators so sentence ends survive the split.
        suffix = separator.rstrip() if separator.strip() else ""

        fragments: List[str] = []
        for part in parts:
            piece = part + suffix
            if len(piece) > self.chunk_size:
                fragments.extend(self._split_recursive(piece, rest))
            else:
                fragments.append(piece)
        return fragments

    def _merge_with_overlap(self, pieces: Iterable[str]) -> List[str]:
        """Greedily packs fragments up to ``chunk_size``, carrying an overlap tail forward."""
        chunks: List[str] = []
        current = ""
        for piece in pieces:
            candidate = f"{current} {piece}".strip() if current else piece.strip()
            if len(candidate) <= self.chunk_size:
                current = candidate
                continue
            if current:
                chunks.append(current)
                tail = current[-self.chunk_overlap :] if self.chunk_overlap else ""
                current = f"{tail} {piece}".strip() if tail else piece.strip()
                # A single oversized fragment can still exceed the budget.
                while len(current) > self.chunk_size:
                    chunks.append(current[: self.chunk_size])
                    current = current[self.chunk_size - self.chunk_overlap :]
            else:
                chunks.extend(_hard_wrap(piece.strip(), self.chunk_size))
                current = ""
        if current:
            chunks.append(current)
        return [chunk for chunk in (c.strip() for c in chunks) if chunk]

    # -- document level --------------------------------------------------
    def chunk_document(self, text: str, source: str) -> List[Chunk]:
        """Chunks a single document, attaching provenance metadata to each piece."""
        return [
            Chunk(
                id=_chunk_id(source, position, piece),
                text=piece,
                source=source,
                position=position,
                metadata={"char_length": len(piece)},
            )
            for position, piece in enumerate(self.chunk_text(text))
        ]

    def chunk_directory(self, input_dir: Path | str) -> List[Chunk]:
        """Loads and chunks every supported document under ``input_dir``."""
        input_path = Path(input_dir)
        if not input_path.exists():
            raise FileNotFoundError(f"Input directory does not exist: {input_path}")

        chunks: List[Chunk] = []
        files = sorted(p for p in input_path.rglob("*") if p.suffix.lower() in SUPPORTED_SUFFIXES)
        for path in files:
            text = path.read_text(encoding="utf-8", errors="replace")
            source = str(path.relative_to(input_path))
            doc_chunks = self.chunk_document(text, source)
            chunks.extend(doc_chunks)
            logger.info("Chunked %s into %d chunks", source, len(doc_chunks))

        if not files:
            logger.warning("No %s files found under %s", "/".join(SUPPORTED_SUFFIXES), input_path)
        return chunks

    def process_documents(self, input_dir: Path | str, output_dir: Path | str) -> List[Chunk]:
        """Chunks ``input_dir`` and writes one JSON file per source document."""
        chunks = self.chunk_directory(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        by_source: dict[str, List[Chunk]] = {}
        for chunk in chunks:
            by_source.setdefault(chunk.source, []).append(chunk)

        for source, source_chunks in by_source.items():
            stem = Path(source).with_suffix("").as_posix().replace("/", "__")
            target = output_path / f"{stem}_chunks.json"
            target.write_text(
                json.dumps([c.to_dict() for c in source_chunks], indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            logger.info("Wrote %d chunks to %s", len(source_chunks), target)
        return chunks


def _normalise_whitespace(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _hard_wrap(text: str, size: int) -> List[str]:
    return [text[i : i + size] for i in range(0, len(text), size)] if text else []


def _chunk_id(source: str, position: int, text: str) -> str:
    digest = hashlib.sha1(f"{source}:{position}:{text}".encode("utf-8")).hexdigest()
    return digest[:16]
