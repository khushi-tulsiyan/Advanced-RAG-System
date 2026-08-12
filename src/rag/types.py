"""Shared data structures.

Chunks carry provenance (source file, position) so that answers can be cited.
The original implementation passed bare strings around, which made it
impossible to tell a caller *where* a retrieved passage came from.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class Chunk:
    """A single retrievable passage."""

    id: str
    text: str
    source: str
    position: int
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> Chunk:
        return cls(
            id=payload["id"],
            text=payload["text"],
            source=payload.get("source", "unknown"),
            position=int(payload.get("position", 0)),
            metadata=payload.get("metadata", {}) or {},
        )


@dataclass(frozen=True)
class ScoredChunk:
    """A chunk together with the score that surfaced it."""

    chunk: Chunk
    score: float
    #: Per-retriever diagnostics, e.g. ``{"dense_rank": 3, "bm25_rank": 11}``.
    components: dict[str, float] = field(default_factory=dict)

    @property
    def text(self) -> str:
        return self.chunk.text

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.chunk.id,
            "text": self.chunk.text,
            "source": self.chunk.source,
            "position": self.chunk.position,
            "score": self.score,
            "metadata": self.chunk.metadata,
            "components": self.components,
        }
