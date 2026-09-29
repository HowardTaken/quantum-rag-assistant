"""FastAPI service exposing the RAG agent to external clients (e.g. a frontend).

Run with: uvicorn backend.main:app --reload
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

from agent import ask_with_sources, build_agent
from backend.schemas import HealthResponse, PapersResponse, QueryRequest, QueryResponse, Source
from config import get_settings
from tools import load_vector_store, source_counts

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    logging.basicConfig(level=settings.log_level)
    try:
        db = load_vector_store()
        app.state.db = db
        app.state.agent = build_agent(db=db)
        logger.info("Agent ready with %d vectors", db._collection.count())
    except Exception:
        logger.exception("Failed to initialize agent on startup")
        raise
    yield


app = FastAPI(title="Quantum RAG Agent API", version="1.0.0", lifespan=lifespan)

# Permissive for local dev / a same-origin-less SPA during development; tighten
# allow_origins to the deployed frontend's URL before shipping this publicly.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


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


@app.get("/papers", response_model=PapersResponse)
def papers(db=Depends(get_db)) -> PapersResponse:
    return PapersResponse(papers=source_counts(db))


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest, agent=Depends(get_agent)) -> QueryResponse:
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
