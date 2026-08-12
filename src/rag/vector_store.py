"""Dense vector index with a FAISS backend and a pure-numpy fallback.

Both backends store L2-normalised vectors and use inner product, so scores are
cosine similarities in ``[-1, 1]`` (the original code used ``IndexFlatL2`` on
unnormalised vectors, which produces unbounded distances that cannot be fused
with BM25 scores in any meaningful way).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np

from .embeddings import normalize

logger = logging.getLogger(__name__)

try:  # pragma: no cover - import guard
    import faiss  # type: ignore

    FAISS_AVAILABLE = True
except ImportError:  # pragma: no cover
    faiss = None  # type: ignore
    FAISS_AVAILABLE = False


@runtime_checkable
class VectorIndex(Protocol):
    """Search interface shared by the FAISS and numpy backends."""

    @property
    def size(self) -> int: ...

    def search(self, query_vectors: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]: ...


class NumpyVectorIndex:
    """Exact cosine search over an in-memory matrix. No external dependencies."""

    backend = "numpy"

    def __init__(self, vectors: np.ndarray) -> None:
        self.vectors = normalize(np.asarray(vectors, dtype=np.float32))

    @property
    def size(self) -> int:
        return int(self.vectors.shape[0])

    @property
    def dimension(self) -> int:
        return int(self.vectors.shape[1]) if self.vectors.size else 0

    def search(self, query_vectors: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:
        queries = normalize(np.asarray(query_vectors, dtype=np.float32))
        if self.size == 0:
            empty = np.zeros((queries.shape[0], 0), dtype=np.float32)
            return empty, empty.astype(np.int64)

        k = min(top_k, self.size)
        similarities = queries @ self.vectors.T
        # argpartition finds the top-k cheaply, then we sort just that slice.
        partitioned = np.argpartition(-similarities, kth=k - 1, axis=1)[:, :k]
        rows = np.arange(queries.shape[0])[:, None]
        ordering = np.argsort(-similarities[rows, partitioned], axis=1)
        indices = partitioned[rows, ordering]
        scores = similarities[rows, indices]
        return scores.astype(np.float32), indices.astype(np.int64)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, self.vectors)

    @classmethod
    def load(cls, path: Path) -> NumpyVectorIndex:
        return cls(np.load(path))


class FaissVectorIndex:
    """FAISS ``IndexFlatIP`` over normalised vectors, i.e. exact cosine search."""

    backend = "faiss"

    def __init__(self, index: faiss.Index) -> None:  # type: ignore[name-defined]
        self._index = index

    @classmethod
    def build(cls, vectors: np.ndarray) -> FaissVectorIndex:
        if not FAISS_AVAILABLE:
            raise RuntimeError("faiss is not installed; use NumpyVectorIndex instead")
        vectors = normalize(np.asarray(vectors, dtype=np.float32))
        index = faiss.IndexFlatIP(vectors.shape[1])
        if vectors.shape[0]:
            index.add(vectors)
        return cls(index)

    @property
    def size(self) -> int:
        return int(self._index.ntotal)

    @property
    def dimension(self) -> int:
        return int(self._index.d)

    def search(self, query_vectors: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:
        queries = normalize(np.asarray(query_vectors, dtype=np.float32))
        if self.size == 0:
            empty = np.zeros((queries.shape[0], 0), dtype=np.float32)
            return empty, empty.astype(np.int64)
        scores, indices = self._index.search(queries, min(top_k, self.size))
        return scores.astype(np.float32), indices.astype(np.int64)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(path))

    @classmethod
    def load(cls, path: Path) -> FaissVectorIndex:
        if not FAISS_AVAILABLE:
            raise RuntimeError("faiss is not installed")
        return cls(faiss.read_index(str(path)))


def build_vector_index(vectors: np.ndarray, prefer_faiss: bool = True) -> VectorIndex:
    """Builds the best available index backend for ``vectors``."""
    if prefer_faiss and FAISS_AVAILABLE:
        return FaissVectorIndex.build(vectors)
    if prefer_faiss:
        logger.warning("faiss not installed; using the numpy vector index (exact but slower at scale)")
    return NumpyVectorIndex(vectors)


def save_index(index: VectorIndex, faiss_path: Path, numpy_path: Path) -> None:
    """Persists whichever backend is in use to its matching location."""
    if isinstance(index, FaissVectorIndex):
        index.save(faiss_path)
    elif isinstance(index, NumpyVectorIndex):
        index.save(numpy_path)
    else:  # pragma: no cover - defensive
        raise TypeError(f"Cannot persist index of type {type(index)!r}")


def load_index(faiss_path: Path, numpy_path: Path) -> VectorIndex:
    """Loads a persisted index, preferring FAISS when both artefacts exist."""
    if FAISS_AVAILABLE and faiss_path.exists():
        return FaissVectorIndex.load(faiss_path)
    if numpy_path.exists():
        return NumpyVectorIndex.load(numpy_path)
    if faiss_path.exists() and not FAISS_AVAILABLE:
        raise RuntimeError(
            f"Found a FAISS index at {faiss_path} but faiss is not installed. "
            "Install faiss-cpu or rebuild the index with scripts/build_index.py."
        )
    raise FileNotFoundError(f"No vector index found at {faiss_path} or {numpy_path}. Run scripts/build_index.py first.")


def write_manifest(path: Path, **fields: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(fields, indent=2, sort_keys=True), encoding="utf-8")


def read_manifest(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:  # pragma: no cover - corrupt file
        logger.warning("Ignoring unreadable manifest at %s", path)
        return {}
