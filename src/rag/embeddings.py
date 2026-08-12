"""Embedding backends.

``sentence-transformers`` (and therefore torch) is an optional dependency. When
it is unavailable the system falls back to :class:`HashingEmbedder`, a
deterministic, dependency-free encoder. The fallback is far weaker than a real
model but keeps the pipeline runnable and the test suite fast and offline.
"""

from __future__ import annotations

import hashlib
import logging
import re
from typing import List, Protocol, Sequence, runtime_checkable

import numpy as np

from .config import settings as default_settings

logger = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"[a-z0-9]+")


@runtime_checkable
class Embedder(Protocol):
    """Minimal interface the retriever needs from an embedding model."""

    @property
    def dimension(self) -> int: ...

    def encode(self, texts: Sequence[str], batch_size: int = 32) -> np.ndarray: ...


def normalize(matrix: np.ndarray) -> np.ndarray:
    """L2-normalises rows so inner product equals cosine similarity."""
    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (matrix / norms).astype(np.float32)


class SentenceTransformerEmbedder:
    """Wraps a ``sentence-transformers`` model and always returns unit vectors."""

    def __init__(self, model_name: str | None = None) -> None:
        from sentence_transformers import SentenceTransformer  # imported lazily

        self.model_name = model_name or default_settings.embedding_model
        self._model = SentenceTransformer(self.model_name)
        self._dimension = int(self._model.get_sentence_embedding_dimension())

    @property
    def dimension(self) -> int:
        return self._dimension

    def encode(self, texts: Sequence[str], batch_size: int = 32) -> np.ndarray:
        if not texts:
            return np.zeros((0, self._dimension), dtype=np.float32)
        vectors = self._model.encode(
            list(texts),
            batch_size=batch_size,
            convert_to_numpy=True,
            show_progress_bar=len(texts) > 512,
        )
        return normalize(vectors)


class HashingEmbedder:
    """Deterministic hashed bag-of-words encoder used when no ML stack is present."""

    def __init__(self, dimension: int = 384, model_name: str = "hashing-bow") -> None:
        if dimension <= 0:
            raise ValueError("dimension must be positive")
        self._dimension = dimension
        self.model_name = model_name

    @property
    def dimension(self) -> int:
        return self._dimension

    def encode(self, texts: Sequence[str], batch_size: int = 32) -> np.ndarray:
        matrix = np.zeros((len(texts), self._dimension), dtype=np.float32)
        for row, text in enumerate(texts):
            for token in _TOKEN_RE.findall(text.lower()):
                digest = hashlib.md5(token.encode("utf-8")).digest()
                bucket = int.from_bytes(digest[:4], "little") % self._dimension
                # Signed hashing keeps unrelated collisions from always reinforcing.
                sign = 1.0 if digest[4] & 1 else -1.0
                matrix[row, bucket] += sign
        return normalize(matrix)


def load_embedder(model_name: str | None = None, *, allow_fallback: bool = True) -> Embedder:
    """Returns the best available embedder, degrading to hashing if needed."""
    name = model_name or default_settings.embedding_model
    try:
        return SentenceTransformerEmbedder(name)
    except Exception as exc:  # pragma: no cover - depends on optional install
        if not allow_fallback:
            raise
        logger.warning(
            "Could not load sentence-transformers model %r (%s). "
            "Falling back to HashingEmbedder; retrieval quality will be reduced. "
            "Install with: pip install 'sentence-transformers'",
            name,
            exc,
        )
        return HashingEmbedder()


def embed_texts(embedder: Embedder, texts: List[str], batch_size: int = 32) -> np.ndarray:
    """Encodes ``texts``, guaranteeing a correctly shaped float32 matrix."""
    vectors = embedder.encode(texts, batch_size=batch_size)
    vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.shape[0] != len(texts):
        raise ValueError(f"Embedder returned {vectors.shape[0]} vectors for {len(texts)} texts")
    return vectors
