from __future__ import annotations

import numpy as np
import pytest

from rag.embeddings import HashingEmbedder, normalize
from rag.vector_store import (
    FAISS_AVAILABLE,
    FaissVectorIndex,
    NumpyVectorIndex,
    build_vector_index,
    load_index,
    save_index,
)

BACKENDS = [NumpyVectorIndex] + ([FaissVectorIndex] if FAISS_AVAILABLE else [])


def make_index(cls, vectors):
    return cls.build(vectors) if cls is FaissVectorIndex else cls(vectors)


@pytest.fixture
def vectors():
    rng = np.random.default_rng(0)
    return normalize(rng.normal(size=(50, 16)).astype(np.float32))


def test_normalize_produces_unit_vectors():
    matrix = normalize(np.array([[3.0, 4.0], [0.0, 0.0]], dtype=np.float32))
    assert np.isclose(np.linalg.norm(matrix[0]), 1.0)
    # A zero vector must not produce NaNs.
    assert not np.isnan(matrix).any()


@pytest.mark.parametrize("cls", BACKENDS)
def test_index_finds_the_exact_vector_first(cls, vectors):
    index = make_index(cls, vectors)
    scores, indices = index.search(vectors[7:8], top_k=5)

    assert index.size == 50
    assert indices[0][0] == 7
    assert scores[0][0] == pytest.approx(1.0, abs=1e-4)


@pytest.mark.parametrize("cls", BACKENDS)
def test_results_are_sorted_by_descending_similarity(cls, vectors):
    index = make_index(cls, vectors)
    scores, _ = index.search(vectors[:3], top_k=10)
    for row in scores:
        assert list(row) == sorted(row, reverse=True)


@pytest.mark.parametrize("cls", BACKENDS)
def test_top_k_larger_than_corpus_is_clamped(cls):
    index = make_index(cls, normalize(np.eye(3, 8, dtype=np.float32)))
    _, indices = index.search(np.eye(1, 8, dtype=np.float32), top_k=100)
    assert indices.shape[1] == 3
    assert all(0 <= int(i) < 3 for i in indices[0])


def test_numpy_and_faiss_agree(vectors):
    if not FAISS_AVAILABLE:
        pytest.skip("faiss is not installed")
    query = vectors[3:4]
    _, numpy_ids = NumpyVectorIndex(vectors).search(query, top_k=5)
    _, faiss_ids = FaissVectorIndex.build(vectors).search(query, top_k=5)
    assert list(numpy_ids[0]) == list(faiss_ids[0])


def test_index_round_trips_through_disk(vectors, tmp_path):
    index = build_vector_index(vectors)
    faiss_path, numpy_path = tmp_path / "faiss.index", tmp_path / "embeddings.npy"

    save_index(index, faiss_path, numpy_path)
    reloaded = load_index(faiss_path, numpy_path)

    assert reloaded.size == index.size
    _, before = index.search(vectors[1:2], top_k=5)
    _, after = reloaded.search(vectors[1:2], top_k=5)
    assert list(before[0]) == list(after[0])


def test_loading_a_missing_index_explains_the_fix(tmp_path):
    with pytest.raises(FileNotFoundError, match="build_index"):
        load_index(tmp_path / "nope.index", tmp_path / "nope.npy")


def test_hashing_embedder_is_deterministic_and_normalised():
    embedder = HashingEmbedder(dimension=64)
    first = embedder.encode(["deep learning networks"])
    second = embedder.encode(["deep learning networks"])

    assert first.shape == (1, 64)
    assert np.allclose(first, second)
    assert np.isclose(np.linalg.norm(first[0]), 1.0)


def test_hashing_embedder_separates_unrelated_text():
    embedder = HashingEmbedder(dimension=256)
    vectors = embedder.encode(
        ["neural networks and deep learning", "neural networks and deep learning models", "cooking pasta recipes"]
    )
    similar = float(vectors[0] @ vectors[1])
    unrelated = float(vectors[0] @ vectors[2])
    assert similar > unrelated


def test_hashing_embedder_handles_empty_input():
    assert HashingEmbedder(dimension=32).encode([]).shape == (0, 32)
