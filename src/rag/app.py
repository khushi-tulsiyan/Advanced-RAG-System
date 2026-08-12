"""FastAPI service exposing the RAG pipeline.

The pipeline is loaded once during startup via the lifespan hook rather than at
import time, so an import failure or a missing index surfaces as a 503 on
``/query`` instead of preventing the process from starting at all.

Run with: ``uvicorn rag.app:app --reload`` (from ``src/``) or ``make serve``.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

from fastapi import Depends, FastAPI, HTTPException, status
from pydantic import BaseModel, Field

from .config import settings
from .pipeline import RAGPipeline

logger = logging.getLogger(__name__)

# Populated on startup; ``None`` means the index could not be loaded.
_state: Dict[str, Any] = {"pipeline": None, "error": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    try:
        _state["pipeline"] = RAGPipeline.load(settings)
        logger.info("RAG pipeline ready: %s", _state["pipeline"].stats)
    except Exception as exc:  # noqa: BLE001 - report, don't crash the server
        _state["error"] = str(exc)
        logger.error("Failed to load RAG pipeline: %s", exc)
    yield
    _state.clear()


app = FastAPI(
    title="Advanced RAG System",
    version="2.0",
    description="Hybrid (BM25 + dense) retrieval with cross-encoder reranking.",
    lifespan=lifespan,
)


def get_pipeline() -> RAGPipeline:
    """Dependency that yields the loaded pipeline or fails with 503."""
    pipeline = _state.get("pipeline")
    if pipeline is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Retrieval index unavailable: {_state.get('error') or 'not loaded'}. "
            "Run scripts/build_index.py to build it.",
        )
    return pipeline


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000, description="Natural-language question")
    top_k: int = Field(default=5, ge=1, le=50, description="Number of passages to return")
    candidate_k: Optional[int] = Field(
        default=None, ge=1, le=200, description="Candidate pool size before reranking"
    )


class ResultItem(BaseModel):
    id: str
    text: str
    source: str
    position: int
    score: float
    metadata: Dict[str, Any] = {}
    components: Dict[str, float] = {}


class QueryResponse(BaseModel):
    query: str
    results: List[ResultItem]
    elapsed_ms: float
    reranked: bool


@app.get("/", tags=["meta"])
def home() -> Dict[str, str]:
    return {
        "message": "Advanced RAG System",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health", tags=["meta"])
def health() -> Dict[str, Any]:
    """Reports readiness plus index statistics for monitoring."""
    pipeline = _state.get("pipeline")
    if pipeline is None:
        return {"status": "degraded", "error": _state.get("error") or "pipeline not loaded"}
    return {"status": "ok", **pipeline.stats}


@app.post("/query", response_model=QueryResponse, tags=["retrieval"])
def retrieve(request: QueryRequest, pipeline: RAGPipeline = Depends(get_pipeline)) -> QueryResponse:
    """Retrieves the passages most relevant to ``query``."""
    try:
        outcome = pipeline.query(request.query, top_k=request.top_k, candidate_k=request.candidate_k)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("Query failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Retrieval failed"
        ) from exc
    return QueryResponse(**outcome.to_dict())


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("rag.app:app", host="127.0.0.1", port=8000, reload=True)
