"""Advanced RAG System: hybrid retrieval with cross-encoder reranking."""

from .chunker import TextChunker
from .config import Settings, settings
from .evaluator import EvalExample, GenerationEvaluator, RetrievalEvaluator
from .pipeline import RAGPipeline, RetrievalResult
from .reranker import CrossEncoderReranker, NoOpReranker, load_reranker
from .retriever import HybridRetriever
from .types import Chunk, ScoredChunk

__version__ = "2.0.0"

__all__ = [
    "Chunk",
    "CrossEncoderReranker",
    "EvalExample",
    "GenerationEvaluator",
    "HybridRetriever",
    "NoOpReranker",
    "RAGPipeline",
    "RetrievalEvaluator",
    "RetrievalResult",
    "ScoredChunk",
    "Settings",
    "TextChunker",
    "load_reranker",
    "settings",
    "__version__",
]
