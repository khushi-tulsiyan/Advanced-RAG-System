"""Shared fixtures.

Everything here runs offline: the hashing embedder means no model downloads and
no torch, so the suite is fast enough to run on every commit.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from rag.chunker import TextChunker
from rag.config import settings as base_settings
from rag.embeddings import HashingEmbedder
from rag.pipeline import RAGPipeline
from rag.retriever import HybridRetriever

DOCUMENTS = {
    "deep_learning.txt": (
        "Deep learning is a subfield of machine learning that uses neural networks "
        "with many layers. Deep neural networks learn hierarchical representations "
        "directly from raw data, which removes the need for manual feature engineering."
    ),
    "transformers.txt": (
        "Transformers are a neural network architecture built around the self-attention "
        "mechanism. Self-attention lets every token attend to every other token in the "
        "sequence, which is why transformers handle long-range dependencies well in NLP."
    ),
    "retrieval.txt": (
        "BM25 is a sparse lexical ranking function used by search engines to estimate "
        "document relevance from term frequency. FAISS is a library for efficient "
        "similarity search over dense vectors. Hybrid retrieval combines both signals."
    ),
}


@pytest.fixture
def settings(tmp_path):
    """Settings pointing at a temporary data directory."""
    configured = replace(base_settings, data_dir=tmp_path / "data", chunk_size=200, chunk_overlap=40)
    configured.ensure_dirs()
    return configured


@pytest.fixture
def raw_docs_dir(settings):
    for name, text in DOCUMENTS.items():
        (settings.raw_docs_dir / name).write_text(text, encoding="utf-8")
    return settings.raw_docs_dir


@pytest.fixture
def embedder():
    return HashingEmbedder(dimension=128)


@pytest.fixture
def chunks(settings, raw_docs_dir):
    return TextChunker(settings=settings).chunk_directory(raw_docs_dir)


@pytest.fixture
def retriever(chunks, embedder, settings):
    return HybridRetriever.from_documents(chunks, embedder=embedder, settings=settings)


@pytest.fixture
def pipeline(retriever, settings):
    return RAGPipeline(retriever, settings=settings)
