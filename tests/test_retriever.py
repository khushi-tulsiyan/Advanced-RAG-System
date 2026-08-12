from __future__ import annotations

import numpy as np
import pytest

from rag.corpus import load_chunks, save_chunks
from rag.embeddings import HashingEmbedder
from rag.retriever import HybridRetriever, tokenize
from rag.types import Chunk


def test_tokenize_lowercases_and_strips_punctuation_and_stopwords():
    tokens = tokenize("What IS Deep-Learning, really?")
    assert "deep" in tokens and "learning" in tokens
    assert "is" not in tokens and "what" not in tokens
    assert all(token.isalnum() for token in tokens)


def test_retrieval_finds_the_relevant_document(retriever):
    results = retriever.retrieve("What is self-attention in transformers?", top_k=3)
    assert results
    assert results[0].chunk.source == "transformers.txt"


def test_results_are_ordered_by_descending_score(retriever):
    results = retriever.retrieve("deep learning neural networks", top_k=5)
    scores = [result.score for result in results]
    assert scores == sorted(scores, reverse=True)


def test_retrieval_respects_top_k(retriever):
    assert len(retriever.retrieve("neural networks", top_k=2)) <= 2


def test_results_are_deduplicated(retriever):
    results = retriever.retrieve("BM25 FAISS hybrid retrieval", top_k=8)
    ids = [result.chunk.id for result in results]
    assert len(ids) == len(set(ids))


def test_fusion_reports_which_retriever_contributed(retriever):
    results = retriever.retrieve("BM25 sparse lexical ranking", top_k=5)
    components = {key for result in results for key in result.components}
    assert components & {"bm25_rank", "dense_rank"}


def test_hybrid_beats_a_single_retriever_on_a_lexical_query(retriever):
    """An exact rare term should rank via BM25 even if the dense model is weak."""
    query = "FAISS"
    sparse_ids = {retriever.chunks[i].id for i, _ in retriever.sparse_search(query, 5)}
    hybrid_ids = {result.chunk.id for result in retriever.retrieve(query, top_k=5)}
    assert sparse_ids & hybrid_ids


def test_sparse_search_drops_zero_score_matches(retriever):
    results = retriever.sparse_search("zzzznonexistenttoken", top_k=5)
    assert results == []


def test_dense_search_never_returns_out_of_range_indices(retriever):
    results = retriever.dense_search("anything at all", top_k=100)
    assert all(0 <= index < len(retriever.chunks) for index, _ in results)


def test_empty_query_is_rejected(retriever):
    with pytest.raises(ValueError):
        retriever.retrieve("   ")


def test_corpus_index_mismatch_is_caught(chunks, embedder, settings):
    """The original code could silently pair a corpus with a differently ordered index."""
    vectors = embedder.encode([c.text for c in chunks])
    from rag.vector_store import NumpyVectorIndex

    with pytest.raises(ValueError, match="mismatch"):
        HybridRetriever(chunks[:-1], NumpyVectorIndex(vectors), embedder, settings)


def test_retriever_round_trips_through_disk(retriever, settings, embedder):
    retriever.save(settings)
    reloaded = HybridRetriever.load(settings, embedder=embedder)

    assert len(reloaded) == len(retriever)
    query = "hybrid retrieval combines BM25 and dense vectors"
    assert [r.chunk.id for r in reloaded.retrieve(query, top_k=3)] == [
        r.chunk.id for r in retriever.retrieve(query, top_k=3)
    ]


def test_loading_with_a_different_embedding_model_is_refused(retriever, settings):
    retriever.save(settings)
    mismatched = HashingEmbedder(dimension=128, model_name="some-other-model")

    with pytest.raises(ValueError, match="embedding model"):
        HybridRetriever.load(settings, embedder=mismatched)


def test_saved_corpus_preserves_index_order(retriever, settings, tmp_path):
    path = tmp_path / "chunks.jsonl"
    save_chunks(retriever.chunks, path)
    assert [c.id for c in load_chunks(path)] == [c.id for c in retriever.chunks]


def test_corpus_survives_unicode_and_newlines(tmp_path):
    path = tmp_path / "chunks.jsonl"
    original = [Chunk(id="x1", text="Grüße\nand 日本語 text", source="a.txt", position=0)]
    save_chunks(original, path)
    assert load_chunks(path)[0].text == original[0].text


def test_empty_corpus_does_not_crash(settings, embedder):
    from rag.vector_store import NumpyVectorIndex

    retriever = HybridRetriever([], NumpyVectorIndex(np.zeros((0, 128), dtype=np.float32)), embedder, settings)
    assert retriever.retrieve("anything", top_k=5) == []
