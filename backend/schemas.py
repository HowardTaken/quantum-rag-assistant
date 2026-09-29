"""Pydantic request/response models for the API."""
from __future__ import annotations

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    thread_id: str = Field(
        default="default",
        max_length=128,
        description="Conversation thread ID; reuse it to keep multi-turn context.",
    )


class Source(BaseModel):
    source: str
    page: str


class QueryResponse(BaseModel):
    answer: str
    sources: list[Source]
    attempts: int = Field(description="Number of self-critique re-retrieval retries used")


class PapersResponse(BaseModel):
    papers: dict[str, int] = Field(description="Source filename -> ingested chunk count")


class HealthResponse(BaseModel):
    status: str
    vectors: int
