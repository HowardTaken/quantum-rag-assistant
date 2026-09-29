"""FastAPI service exposing the RAG agent to external clients (e.g. a frontend).

Run with: uvicorn backend.main:app --reload
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Request, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from slowapi.util import get_remote_address

from agent import ask_with_sources, build_agent
from backend.schemas import HealthResponse, PapersResponse, QueryRequest, QueryResponse, Source
from checkpointer import build_sqlite_checkpointer
from config import get_settings
from tools import load_vector_store, source_counts

logger = logging.getLogger(__name__)

limiter = Limiter(key_func=get_remote_address)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    logging.basicConfig(level=settings.log_level)
    try:
        db = load_vector_store()
        checkpointer = build_sqlite_checkpointer(str(settings.checkpoint_db_path))
        app.state.db = db
        app.state.agent = build_agent(db=db, checkpointer=checkpointer)
        logger.info("Agent ready with %d vectors", db._collection.count())
        if not settings.api_key:
            logger.warning("API_KEY not set -- /query and /papers are unauthenticated")
    except Exception:
        logger.exception("Failed to initialize agent on startup")
        raise
    yield


app = FastAPI(title="Quantum RAG Agent API", version="1.0.0", lifespan=lifespan)

app.state.limiter = limiter


@app.exception_handler(RateLimitExceeded)
def rate_limit_exceeded_handler(request: Request, exc: RateLimitExceeded) -> JSONResponse:
    # Match the rest of the API's error shape ({"detail": ...}) instead of slowapi's
    # default {"error": ...}, so frontend error handling doesn't need a special case.
    return JSONResponse(status_code=429, content={"detail": f"Rate limit exceeded: {exc.detail}"})


app.add_middleware(SlowAPIMiddleware)

app.add_middleware(
    CORSMiddleware,
    allow_origins=get_settings().cors_origins_list,
    allow_methods=["*"],
    allow_headers=["*"],
)

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def require_api_key(provided: str | None = Security(_api_key_header)) -> None:
    """Gate the expensive endpoints behind a shared key.

    This is a soft gate, not real authentication: the key ships inside the public
    frontend's JS bundle (there's no server-side proxy to keep it secret behind), so
    anyone who inspects the bundle can read it. Its purpose is to block naive scrapers
    and randomly-discovered-URL traffic, not a determined attacker -- the rate limit
    below is the layer that actually bounds worst-case cost regardless of who has the key.
    If API_KEY isn't configured (e.g. local dev), the check is skipped entirely.
    """
    settings = get_settings()
    if settings.api_key is None:
        return
    if provided != settings.api_key:
        raise HTTPException(status_code=401, detail="Missing or invalid API key")


def get_agent(request: Request):
    agent = getattr(request.app.state, "agent", None)
    if agent is None:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    return agent


def get_db(request: Request):
    db = getattr(request.app.state, "db", None)
    if db is None:
        raise HTTPException(status_code=503, detail="Vector store not initialized")
    return db


@app.get("/health", response_model=HealthResponse)
def health(db=Depends(get_db)) -> HealthResponse:
    return HealthResponse(status="ok", vectors=db._collection.count())


@app.get("/papers", response_model=PapersResponse, dependencies=[Depends(require_api_key)])
@limiter.limit(lambda: get_settings().rate_limit_default)
def papers(request: Request, db=Depends(get_db)) -> PapersResponse:
    return PapersResponse(papers=source_counts(db))


@app.post("/query", response_model=QueryResponse, dependencies=[Depends(require_api_key)])
@limiter.limit(lambda: get_settings().rate_limit_query)
def query(request: Request, req: QueryRequest, agent=Depends(get_agent)) -> QueryResponse:
    try:
        result = ask_with_sources(req.question, agent, thread_id=req.thread_id)
    except Exception as exc:
        logger.exception("Query failed for question=%r", req.question)
        raise HTTPException(status_code=500, detail="Failed to process query") from exc
    return QueryResponse(
        answer=result["answer"],
        sources=[Source(**s) for s in result["sources"]],
        attempts=result["attempts"],
    )
