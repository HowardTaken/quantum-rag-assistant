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
    # Rate limiting is covered by its own dedicated tests below; disable it here so the
    # rest of the suite hammering /query and /papers doesn't trip it and fail on order.
    backend_main.limiter.enabled = False
    with TestClient(backend_main.app) as c:
        yield c
    backend_main.app.dependency_overrides.clear()
    backend_main.limiter.enabled = True
    backend_main.limiter.reset()


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


# --- auth -------------------------------------------------------------------------
# The `client` fixture runs with no API_KEY set (conftest's autouse fixture only sets
# GOOGLE_API_KEY), so auth is disabled by default there. These tests set API_KEY
# explicitly to exercise the enforced path.


def test_query_unauthenticated_when_no_api_key_configured(client):
    # Belt-and-suspenders check that the default (API_KEY unset) truly skips the check,
    # since every other test in this file implicitly relies on that.
    resp = client.post("/query", json={"question": "hi"})
    assert resp.status_code == 200


def test_query_rejects_missing_key_when_configured(client, monkeypatch):
    monkeypatch.setenv("API_KEY", "secret123")
    backend_main.get_settings.cache_clear()
    resp = client.post("/query", json={"question": "hi"})
    assert resp.status_code == 401


def test_query_rejects_wrong_key_when_configured(client, monkeypatch):
    monkeypatch.setenv("API_KEY", "secret123")
    backend_main.get_settings.cache_clear()
    resp = client.post("/query", json={"question": "hi"}, headers={"X-API-Key": "wrong"})
    assert resp.status_code == 401


def test_query_accepts_correct_key_when_configured(client, monkeypatch):
    monkeypatch.setenv("API_KEY", "secret123")
    backend_main.get_settings.cache_clear()
    resp = client.post(
        "/query", json={"question": "hi"}, headers={"X-API-Key": "secret123"}
    )
    assert resp.status_code == 200


def test_papers_also_requires_the_key_when_configured(client, monkeypatch):
    monkeypatch.setenv("API_KEY", "secret123")
    backend_main.get_settings.cache_clear()
    assert client.get("/papers").status_code == 401
    assert client.get("/papers", headers={"X-API-Key": "secret123"}).status_code == 200


def test_health_never_requires_a_key(client, monkeypatch):
    # /health has no require_api_key dependency at all -- it should stay reachable for
    # uptime monitoring regardless of whether API_KEY is configured.
    monkeypatch.setenv("API_KEY", "secret123")
    backend_main.get_settings.cache_clear()
    assert client.get("/health").status_code == 200


# --- rate limiting ------------------------------------------------------------------


def test_query_rate_limit_returns_429_after_exceeding_the_limit(client, monkeypatch):
    monkeypatch.setenv("RATE_LIMIT_QUERY", "2/minute")
    backend_main.get_settings.cache_clear()
    backend_main.limiter.enabled = True
    backend_main.limiter.reset()
    try:
        for _ in range(2):
            assert client.post("/query", json={"question": "hi"}).status_code == 200
        resp = client.post("/query", json={"question": "hi"})
        assert resp.status_code == 429
    finally:
        backend_main.limiter.enabled = False
        backend_main.limiter.reset()


def test_rate_limit_is_tracked_per_endpoint_not_globally(client, monkeypatch):
    monkeypatch.setenv("RATE_LIMIT_QUERY", "1/minute")
    monkeypatch.setenv("RATE_LIMIT_DEFAULT", "5/minute")
    backend_main.get_settings.cache_clear()
    backend_main.limiter.enabled = True
    backend_main.limiter.reset()
    try:
        assert client.post("/query", json={"question": "hi"}).status_code == 200
        assert client.post("/query", json={"question": "hi"}).status_code == 429
        # /papers has its own, much higher limit and shouldn't be affected by /query's.
        assert client.get("/papers").status_code == 200
    finally:
        backend_main.limiter.enabled = False
        backend_main.limiter.reset()
