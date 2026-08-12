"""API tests.

These run against the app in-process with ``TestClient``. The original file was
a script that POSTed to ``127.0.0.1:8000`` and printed the responses: it needed
a manually started server, asserted nothing, and failed collection under pytest.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from rag import app as app_module
from rag.app import app, get_pipeline


@pytest.fixture
def client(pipeline, monkeypatch):
    """Client wired to the in-memory test pipeline.

    The lifespan hook is stubbed out: left alone it calls ``RAGPipeline.load``
    against the *real* settings, which downloads models and reads whatever
    index happens to exist on the machine running the tests.
    """
    monkeypatch.setattr(app_module.RAGPipeline, "load", classmethod(lambda cls, *a, **kw: pipeline))
    app.dependency_overrides[get_pipeline] = lambda: pipeline
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()
    app_module._state.clear()


@pytest.fixture
def degraded_client():
    """Client where the index failed to load."""
    app.dependency_overrides.clear()
    app_module._state["pipeline"] = None
    app_module._state["error"] = "index missing"
    yield TestClient(app)
    app_module._state.clear()


def test_home_lists_entrypoints(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "docs" in response.json()


def test_health_reports_index_stats(client):
    payload = client.get("/health").json()
    assert payload["status"] == "ok"
    assert payload["num_chunks"] > 0


def test_query_returns_scored_passages(client):
    response = client.post("/query", json={"query": "What is deep learning?", "top_k": 3})
    assert response.status_code == 200

    payload = response.json()
    assert payload["query"] == "What is deep learning?"
    assert 0 < len(payload["results"]) <= 3
    assert {"id", "text", "source", "score"} <= set(payload["results"][0])
    assert payload["elapsed_ms"] >= 0


def test_query_results_are_ordered_by_score(client):
    results = client.post("/query", json={"query": "neural networks", "top_k": 5}).json()["results"]
    scores = [result["score"] for result in results]
    assert scores == sorted(scores, reverse=True)


def test_query_finds_the_relevant_source(client):
    results = client.post("/query", json={"query": "self-attention in transformers", "top_k": 3}).json()["results"]
    assert results[0]["source"] == "transformers.txt"


@pytest.mark.parametrize(
    "payload",
    [
        {"query": "", "top_k": 3},
        {"query": "valid", "top_k": 0},
        {"query": "valid", "top_k": 1000},
        {"top_k": 3},
        {"query": "valid", "top_k": "many"},
    ],
)
def test_invalid_requests_are_rejected_with_422(client, payload):
    assert client.post("/query", json=payload).status_code == 422


def test_query_defaults_top_k_when_omitted(client):
    results = client.post("/query", json={"query": "retrieval"}).json()["results"]
    assert 0 < len(results) <= 5


def test_query_returns_503_when_the_index_is_unavailable(degraded_client):
    response = degraded_client.post("/query", json={"query": "anything", "top_k": 3})
    assert response.status_code == 503
    assert "build_index" in response.json()["detail"]


def test_health_reports_degraded_when_the_index_is_unavailable(degraded_client):
    payload = degraded_client.get("/health").json()
    assert payload["status"] == "degraded"
    assert payload["error"] == "index missing"
