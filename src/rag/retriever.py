"""Hybrid sparse + dense retrieval.

BM25 and the dense index are fused with Reciprocal Rank Fusion (RRF). RRF
combines *ranks* rather than raw scores, which avoids the central flaw of the
original implementation: BM25 scores are unbounded and corpus-dependent while
cosine similarities live in ``[-1, 1]``, so adding or comparing them directly
is meaningless.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
from rank_bm25 import BM25Okapi

from .config import Settings, settings as default_settings
from .corpus import load_chunks, save_chunks
from .embeddings import Embedder, embed_texts, load_embedder
from .types import Chunk, ScoredChunk
from .vector_store import (
    VectorIndex,
    build_vector_index,
    load_index,
    read_manifest,
    save_index,
    write_manifest,
)

logger = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"[a-z0-9]+")

# A short list is enough to stop the most common BM25 noise without a
# heavyweight NLP dependency.
STOPWORDS = frozenset(
    """a an and are as at be by for from has have how in is it its of on or that the
    this to was were what when where which who why will with""".split()
)


def tokenize(text: str) -> List[str]:
    """Lowercases, strips punctuation and drops stopwords for BM25 matching."""
    return [token for token in _TOKEN_RE.findall(text.lower()) if token not in STOPWORDS]


class HybridRetriever:
    """Retrieves chunks using BM25 and dense vectors, fused with RRF."""

    def __init__(
        self,
        chunks: Sequence[Chunk],
        vector_index: VectorIndex,
        embedder: Embedder,
        settings: Settings = default_settings,
    ) -> None:
        if len(chunks) != vector_index.size:
            raise ValueError(
                f"Corpus/index mismatch: {len(chunks)} chunks but {vector_index.size} vectors. "
                "Rebuild the index with scripts/build_index.py."
            )
        self.chunks = list(chunks)
        self.vector_index = vector_index
        self.embedder = embedder
        self.settings = settings
        tokenized = [tokenize(chunk.text) for chunk in self.chunks]
        # BM25Okapi divides by the corpus average length, so it needs a non-empty corpus.
        self.bm25 = BM25Okapi(tokenized) if tokenized else None

    # -- construction ----------------------------------------------------
    @classmethod
    def from_documents(
        cls,
        chunks: Sequence[Chunk],
        embedder: Embedder | None = None,
        settings: Settings = default_settings,
        prefer_faiss: bool = True,
    ) -> "HybridRetriever":
        """Embeds ``chunks`` and builds an in-memory retriever."""
        embedder = embedder or load_embedder(settings.embedding_model)
        vectors = embed_texts(embedder, [c.text for c in chunks], settings.embedding_batch_size)
        index = build_vector_index(vectors, prefer_faiss=prefer_faiss)
        return cls(chunks, index, embedder, settings)

    @classmethod
    def load(
        cls,
        settings: Settings = default_settings,
        embedder: Embedder | None = None,
    ) -> "HybridRetriever":
        """Loads a retriever from the artefacts written by ``scripts/build_index.py``."""
        chunks = load_chunks(settings.chunks_path)
        index = load_index(settings.faiss_index_path, settings.embeddings_path)
        embedder = embedder or load_embedder(settings.embedding_model)

        manifest = read_manifest(settings.index_manifest_path)
        built_with = manifest.get("embedding_model")
        current = getattr(embedder, "model_name", None)
        if built_with and current and built_with != current:
            # Mixing models silently returns nonsense; the original code had a
            # 384-dim index and a 768-dim query model wired together.
            raise ValueError(
                f"Index was built with embedding model {built_with!r} but {current!r} is loaded. "
                "Rebuild the index or set RAG_EMBEDDING_MODEL to match."
            )
        return cls(chunks, index, embedder, settings)

    def save(self, settings: Settings | None = None) -> None:
        """Persists the corpus, vector index and a manifest describing the build."""
        settings = settings or self.settings
        settings.ensure_dirs()
        save_chunks(self.chunks, settings.chunks_path)
        save_index(self.vector_index, settings.faiss_index_path, settings.embeddings_path)
        write_manifest(
            settings.index_manifest_path,
            embedding_model=getattr(self.embedder, "model_name", "unknown"),
            dimension=self.embedder.dimension,
            num_chunks=len(self.chunks),
            backend=getattr(self.vector_index, "backend", "unknown"),
        )

    # -- search ----------------------------------------------------------
    def sparse_search(self, query: str, top_k: int) -> List[tuple[int, float]]:
        """BM25 ranking. Returns ``(chunk_index, score)`` best-first."""
        tokens = tokenize(query)
        if self.bm25 is None or not tokens:
            return []
        scores = np.asarray(self.bm25.get_scores(tokens), dtype=np.float32)
        k = min(top_k, scores.shape[0])
        top = np.argpartition(-scores, kth=k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        # A zero score means no query term matched; keeping those adds only noise.
        return [(int(i), float(scores[i])) for i in top if scores[i] > 0]

    def dense_search(self, query: str, top_k: int) -> List[tuple[int, float]]:
        """Cosine ranking over the vector index. Returns ``(chunk_index, score)``."""
        if self.vector_index.size == 0:
            return []
        query_vector = embed_texts(self.embedder, [query])
        scores, indices = self.vector_index.search(query_vector, top_k)
        # FAISS pads short result sets with -1; those must not index the corpus.
        return [
            (int(idx), float(score))
            for idx, score in zip(indices[0], scores[0])
            if 0 <= int(idx) < len(self.chunks)
        ]

    def retrieve(self, query: str, top_k: int | None = None, candidate_k: int | None = None) -> List[ScoredChunk]:
        """Runs both retrievers and fuses their rankings with RRF."""
        if not query or not query.strip():
            raise ValueError("query must be a non-empty string")

        top_k = top_k or self.settings.top_k
        candidate_k = candidate_k or max(self.settings.candidate_k, top_k)

        sparse = self.sparse_search(query, candidate_k)
        dense = self.dense_search(query, candidate_k)

        fused: Dict[int, float] = {}
        components: Dict[int, Dict[str, float]] = {}
        rrf_k = self.settings.rrf_k

        for weight, results, label in (
            (self.settings.sparse_weight, sparse, "bm25"),
            (self.settings.dense_weight, dense, "dense"),
        ):
            for rank, (chunk_index, raw_score) in enumerate(results, start=1):
                fused[chunk_index] = fused.get(chunk_index, 0.0) + weight / (rrf_k + rank)
                entry = components.setdefault(chunk_index, {})
                entry[f"{label}_rank"] = float(rank)
                entry[f"{label}_score"] = float(raw_score)

        ordered = sorted(fused.items(), key=lambda item: item[1], reverse=True)[:top_k]
        return [
            ScoredChunk(chunk=self.chunks[idx], score=score, components=components.get(idx, {}))
            for idx, score in ordered
        ]

    # Kept for backwards compatibility with the original API.
    def hybrid_search(self, query: str, top_k: int = 5) -> List[ScoredChunk]:
        return self.retrieve(query, top_k=top_k)

    def __len__(self) -> int:
        return len(self.chunks)


def build_retriever_from_directory(
    input_dir: Path | str,
    settings: Settings = default_settings,
    embedder: Embedder | None = None,
) -> HybridRetriever:
    """Convenience helper: chunk a directory and build a retriever over it."""
    from .chunker import TextChunker

    chunks = TextChunker(settings=settings).chunk_directory(input_dir)
    if not chunks:
        raise ValueError(f"No documents found under {input_dir}")
    return HybridRetriever.from_documents(chunks, embedder=embedder, settings=settings)
