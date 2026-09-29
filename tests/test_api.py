"""FastAPI endpoint tests. The real startup lifespan (which loads the vector store
and builds the agent) is replaced with a no-op so these never touch the network or
need a real .env; dependencies are swapped via FastAPI's dependency_overrides."""
from __future__ import annotations

import contextlib

import pytest
from fastapi.testclient import TestClient

import backend.main as backend_main
from tests.fakes import FakeChroma


@contextlib.asynccontextmanager
async def _noop_lifespan(app):
    yield


def _fake_ask_with_sources(question, agent, thread_id="default"):
    if question == "boom":
        raise RuntimeError("simulated failure")
    return {
        "answer": f"Answer to: {question}",
        "sources": [{"source": "a.pdf", "page": "1"}],
        "attempts": 0,
    }


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(backend_main.app.router, "lifespan_context", _noop_lifespan)
    monkeypatch.setattr(backend_main, "ask_with_sources", _fake_ask_with_sources)
    backend_main.app.dependency_overrides[backend_main.get_agent] = lambda: object()
    backend_main.app.dependency_overrides[backend_main.get_db] = lambda: FakeChroma(
        metadatas=[{"source": "a.pdf"}, {"source": "b.pdf"}]
    )
    with TestClient(backend_main.app) as c:
        yield c
    backend_main.app.dependency_overrides.clear()


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok", "vectors": 2}


def test_papers(client):
    resp = client.get("/papers")
    assert resp.status_code == 200
    assert resp.json() == {"papers": {"a.pdf": 1, "b.pdf": 1}}


def test_query_success(client):
    resp = client.post("/query", json={"question": "hello", "thread_id": "t1"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["answer"] == "Answer to: hello"
    assert body["sources"] == [{"source": "a.pdf", "page": "1"}]
    assert body["attempts"] == 0


def test_query_defaults_thread_id(client):
    resp = client.post("/query", json={"question": "hello"})
    assert resp.status_code == 200


def test_query_rejects_empty_question(client):
    resp = client.post("/query", json={"question": ""})
    assert resp.status_code == 422


def test_query_rejects_missing_question(client):
    resp = client.post("/query", json={})
    assert resp.status_code == 422


def test_query_failure_returns_500_not_a_crash(client):
    resp = client.post("/query", json={"question": "boom"})
    assert resp.status_code == 500


def test_health_returns_503_when_not_initialized(monkeypatch):
    monkeypatch.setattr(backend_main.app.router, "lifespan_context", _noop_lifespan)
    backend_main.app.dependency_overrides.clear()
    with TestClient(backend_main.app) as c:
        resp = c.get("/health")
    assert resp.status_code == 503
