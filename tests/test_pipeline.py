from __future__ import annotations

import pytest

from rag.pipeline import RAGPipeline
from rag.reranker import NoOpReranker
from rag.types import ScoredChunk


class ReverseReranker:
    """Test double that scores by inverse rank, so it always reverses the order."""

    model_name = "reverse"

    def rerank(self, query, candidates, top_k=None):
        rescored = [
            ScoredChunk(chunk=candidate.chunk, score=float(i), components=candidate.components)
            for i, candidate in enumerate(candidates)
        ]
        rescored.sort(key=lambda item: item.score, reverse=True)
        return rescored[:top_k] if top_k else rescored


def test_query_returns_requested_number_of_results(pipeline):
    outcome = pipeline.query("What is deep learning?", top_k=2)
    assert len(outcome.results) == 2
    assert outcome.query == "What is deep learning?"
    assert outcome.elapsed_ms >= 0


def test_query_surfaces_the_right_document(pipeline):
    outcome = pipeline.query("self-attention mechanism in transformers", top_k=3)
    assert outcome.results[0].chunk.source == "transformers.txt"


def test_empty_query_is_rejected(pipeline):
    with pytest.raises(ValueError):
        pipeline.query("")


def test_reranker_reorders_and_truncates_results(retriever, settings):
    baseline = RAGPipeline(retriever, NoOpReranker(), settings).query("neural networks", top_k=3)
    reranked = RAGPipeline(retriever, ReverseReranker(), settings).query("neural networks", top_k=3)

    assert len(reranked.results) == 3
    assert reranked.reranked is True
    assert baseline.reranked is False
    assert [r.chunk.id for r in reranked.results] != [r.chunk.id for r in baseline.results]


def test_reranker_sees_more_candidates_than_it_returns(retriever, settings):
    """Reranking only the final top_k cannot improve recall, so we over-fetch."""
    seen = {}

    class RecordingReranker:
        model_name = "recording"

        def rerank(self, query, candidates, top_k=None):
            seen["count"] = len(candidates)
            return list(candidates)[:top_k]

    pipeline = RAGPipeline(retriever, RecordingReranker(), settings)
    pipeline.query("retrieval", top_k=1, candidate_k=5)
    assert seen["count"] > 1


def test_reranker_preserves_the_retrieval_score_for_debugging(retriever, settings):
    outcome = RAGPipeline(retriever, ReverseReranker(), settings).query("BM25", top_k=2)
    assert all("bm25_rank" in r.components or "dense_rank" in r.components for r in outcome.results)


def test_context_property_includes_citations(pipeline):
    outcome = pipeline.query("hybrid retrieval", top_k=2)
    context = outcome.context
    assert "[1]" in context
    assert outcome.results[0].chunk.source in context


def test_to_dict_is_json_serialisable(pipeline):
    import json

    payload = pipeline.query("transformers", top_k=2).to_dict()
    json.dumps(payload)
    assert {"query", "results", "elapsed_ms", "reranked"} == set(payload)
    assert {"id", "text", "source", "score"} <= set(payload["results"][0])


def test_legacy_retrieve_answer_still_returns_strings(pipeline):
    results = pipeline.retrieve_answer("deep learning", top_k=2)
    assert len(results) == 2
    assert all(isinstance(text, str) for text in results)


def test_stats_describe_the_loaded_components(pipeline):
    stats = pipeline.stats
    assert stats["num_chunks"] > 0
    assert stats["vector_backend"] in {"faiss", "numpy"}


def test_pipeline_loads_from_persisted_artifacts(retriever, settings, embedder):
    retriever.save(settings)
    from rag.retriever import HybridRetriever

    reloaded = RAGPipeline(HybridRetriever.load(settings, embedder=embedder), NoOpReranker(), settings)
    assert reloaded.query("transformers", top_k=2).results
