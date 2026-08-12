"""Central configuration.

Every path in the project is derived from :data:`REPO_ROOT` so that scripts,
tests and the API behave identically regardless of the working directory they
are launched from. Values can be overridden with ``RAG_*`` environment
variables.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# src/rag/config.py -> src/rag -> src -> repo root
REPO_ROOT = Path(__file__).resolve().parents[2]


def _env_path(name: str, default: Path) -> Path:
    raw = os.getenv(name)
    return Path(raw).expanduser().resolve() if raw else default


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    return int(raw) if raw else default


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    return float(raw) if raw else default


def _env_str(name: str, default: str) -> str:
    return os.getenv(name) or default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    """Resolved settings for the whole system."""

    # --- storage layout -------------------------------------------------
    data_dir: Path = field(default_factory=lambda: _env_path("RAG_DATA_DIR", REPO_ROOT / "data"))

    # --- chunking -------------------------------------------------------
    chunk_size: int = field(default_factory=lambda: _env_int("RAG_CHUNK_SIZE", 512))
    chunk_overlap: int = field(default_factory=lambda: _env_int("RAG_CHUNK_OVERLAP", 64))

    # --- models ---------------------------------------------------------
    embedding_model: str = field(
        default_factory=lambda: _env_str("RAG_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    )
    reranker_model: str = field(
        default_factory=lambda: _env_str("RAG_RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    )
    embedding_batch_size: int = field(default_factory=lambda: _env_int("RAG_EMBEDDING_BATCH_SIZE", 32))

    # --- retrieval ------------------------------------------------------
    top_k: int = field(default_factory=lambda: _env_int("RAG_TOP_K", 5))
    candidate_k: int = field(default_factory=lambda: _env_int("RAG_CANDIDATE_K", 30))
    rrf_k: int = field(default_factory=lambda: _env_int("RAG_RRF_K", 60))
    dense_weight: float = field(default_factory=lambda: _env_float("RAG_DENSE_WEIGHT", 1.0))
    sparse_weight: float = field(default_factory=lambda: _env_float("RAG_SPARSE_WEIGHT", 1.0))
    use_reranker: bool = field(default_factory=lambda: _env_bool("RAG_USE_RERANKER", True))

    # --- derived paths --------------------------------------------------
    @property
    def raw_docs_dir(self) -> Path:
        return self.data_dir / "raw_docs"

    @property
    def processed_chunks_dir(self) -> Path:
        return self.data_dir / "processed_chunks"

    @property
    def vector_store_dir(self) -> Path:
        return self.data_dir / "vector_store"

    @property
    def chunks_path(self) -> Path:
        """Single source of truth for chunk text + metadata, aligned with the index."""
        return self.vector_store_dir / "chunks.jsonl"

    @property
    def faiss_index_path(self) -> Path:
        return self.vector_store_dir / "faiss.index"

    @property
    def embeddings_path(self) -> Path:
        """Raw embedding matrix, used by the numpy fallback and by index rebuilds."""
        return self.vector_store_dir / "embeddings.npy"

    @property
    def index_manifest_path(self) -> Path:
        return self.vector_store_dir / "manifest.json"

    def ensure_dirs(self) -> None:
        for path in (self.raw_docs_dir, self.processed_chunks_dir, self.vector_store_dir):
            path.mkdir(parents=True, exist_ok=True)


settings = Settings()
