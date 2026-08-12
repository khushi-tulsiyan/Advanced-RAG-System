"""Cross-encoder reranking.

The original code fed cross-encoder pairs to a ``text-classification`` pipeline
as ``f"{query} [SEP] {doc}"`` strings. That produces a single-segment encoding,
so the model never sees the query/document pair it was trained on. This module
uses ``sentence_transformers.CrossEncoder``, which builds the correct paired
encoding, and degrades to a no-op when the model is unavailable.
"""

from __future__ import annotations

import logging
from typing import List, Protocol, Sequence, runtime_checkable

from .config import settings as default_settings
from .types import ScoredChunk

logger = logging.getLogger(__name__)


@runtime_checkable
class Reranker(Protocol):
    def rerank(self, query: str, candidates: Sequence[ScoredChunk], top_k: int | None = None) -> List[ScoredChunk]: ...


class NoOpReranker:
    """Passes retrieval order through unchanged."""

    model_name = "noop"

    def rerank(self, query: str, candidates: Sequence[ScoredChunk], top_k: int | None = None) -> List[ScoredChunk]:
        results = list(candidates)
        return results[:top_k] if top_k else results


class CrossEncoderReranker:
    """Rescores retrieved chunks with a query/document cross-encoder."""

    def __init__(self, model_name: str | None = None, batch_size: int = 32) -> None:
        from sentence_transformers import CrossEncoder  # imported lazily

        self.model_name = model_name or default_settings.reranker_model
        self.batch_size = batch_size
        self._model = CrossEncoder(self.model_name)

    def rerank(self, query: str, candidates: Sequence[ScoredChunk], top_k: int | None = None) -> List[ScoredChunk]:
        if not candidates:
            return []

        pairs = [(query, candidate.chunk.text) for candidate in candidates]
        scores = self._model.predict(pairs, batch_size=self.batch_size)

        rescored = [
            ScoredChunk(
                chunk=candidate.chunk,
                score=float(score),
                # Keep the retrieval score so callers can inspect what the reranker changed.
                components={**candidate.components, "retrieval_score": candidate.score},
            )
            for candidate, score in zip(candidates, scores)
        ]
        rescored.sort(key=lambda item: item.score, reverse=True)
        return rescored[:top_k] if top_k else rescored


def load_reranker(model_name: str | None = None, *, allow_fallback: bool = True) -> Reranker:
    """Returns a cross-encoder reranker, or a no-op one if it cannot be loaded."""
    name = model_name or default_settings.reranker_model
    try:
        return CrossEncoderReranker(name)
    except Exception as exc:  # pragma: no cover - depends on optional install
        if not allow_fallback:
            raise
        logger.warning(
            "Could not load cross-encoder %r (%s). Reranking is disabled; "
            "results will use fused retrieval order only.",
            name,
            exc,
        )
        return NoOpReranker()
