"""End-to-end retrieval pipeline: hybrid retrieval followed by reranking."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence

from .config import Settings, settings as default_settings
from .reranker import NoOpReranker, Reranker, load_reranker
from .retriever import HybridRetriever
from .types import Chunk, ScoredChunk

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RetrievalResult:
    """Everything a caller needs to render and debug an answer."""

    query: str
    results: List[ScoredChunk]
    elapsed_ms: float
    reranked: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "results": [result.to_dict() for result in self.results],
            "elapsed_ms": round(self.elapsed_ms, 2),
            "reranked": self.reranked,
        }

    @property
    def context(self) -> str:
        """Retrieved passages joined into a citation-friendly prompt context."""
        return "\n\n".join(
            f"[{i}] ({result.chunk.source}#{result.chunk.position}) {result.chunk.text}"
            for i, result in enumerate(self.results, start=1)
        )


class RAGPipeline:
    """Composes a :class:`HybridRetriever` with an optional reranker.

    The retriever over-fetches ``candidate_k`` chunks so the cross-encoder has a
    meaningful pool to reorder; reranking the final ``top_k`` alone cannot
    improve recall.
    """

    def __init__(
        self,
        retriever: HybridRetriever,
        reranker: Reranker | None = None,
        settings: Settings = default_settings,
    ) -> None:
        self.retriever = retriever
        self.settings = settings
        self.reranker = reranker if reranker is not None else NoOpReranker()

    @classmethod
    def load(cls, settings: Settings = default_settings, reranker: Reranker | None = None) -> "RAGPipeline":
        """Builds a pipeline from persisted index artefacts."""
        retriever = HybridRetriever.load(settings)
        if reranker is None:
            reranker = load_reranker(settings.reranker_model) if settings.use_reranker else NoOpReranker()
        return cls(retriever, reranker, settings)

    @classmethod
    def from_documents(
        cls,
        chunks: Sequence[Chunk],
        settings: Settings = default_settings,
        reranker: Reranker | None = None,
        **retriever_kwargs: Any,
    ) -> "RAGPipeline":
        """Builds an in-memory pipeline directly from chunks."""
        retriever = HybridRetriever.from_documents(chunks, settings=settings, **retriever_kwargs)
        return cls(retriever, reranker if reranker is not None else NoOpReranker(), settings)

    @classmethod
    def from_directory(cls, input_dir: Path | str, settings: Settings = default_settings) -> "RAGPipeline":
        from .retriever import build_retriever_from_directory

        retriever = build_retriever_from_directory(input_dir, settings=settings)
        reranker = load_reranker(settings.reranker_model) if settings.use_reranker else NoOpReranker()
        return cls(retriever, reranker, settings)

    # -- query -----------------------------------------------------------
    def query(self, query: str, top_k: int | None = None, candidate_k: int | None = None) -> RetrievalResult:
        """Retrieves, reranks and returns the ``top_k`` best passages."""
        if not query or not query.strip():
            raise ValueError("query must be a non-empty string")

        top_k = top_k or self.settings.top_k
        candidate_k = candidate_k or max(self.settings.candidate_k, top_k)

        started = time.perf_counter()
        candidates = self.retriever.retrieve(query, top_k=candidate_k, candidate_k=candidate_k)
        results = self.reranker.rerank(query, candidates, top_k=top_k)
        elapsed_ms = (time.perf_counter() - started) * 1000

        logger.info("Query %r returned %d results in %.1fms", query, len(results), elapsed_ms)
        return RetrievalResult(
            query=query,
            results=results,
            elapsed_ms=elapsed_ms,
            reranked=not isinstance(self.reranker, NoOpReranker),
        )

    # Backwards-compatible shim for the original string-list API.
    def retrieve_answer(self, query: str, top_k: int = 5) -> List[str]:
        return [result.text for result in self.query(query, top_k=top_k).results]

    @property
    def stats(self) -> Dict[str, Any]:
        return {
            "num_chunks": len(self.retriever),
            "vector_backend": getattr(self.retriever.vector_index, "backend", "unknown"),
            "embedding_model": getattr(self.retriever.embedder, "model_name", "unknown"),
            "reranker_model": getattr(self.reranker, "model_name", "unknown"),
        }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    pipeline = RAGPipeline.load()
    outcome = pipeline.query("What is deep learning?")
    for rank, item in enumerate(outcome.results, start=1):
        print(f"{rank}. [{item.score:.4f}] ({item.chunk.source}) {item.text[:200]}")
