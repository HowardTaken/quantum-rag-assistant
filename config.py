"""Centralized configuration, loaded from environment variables / .env."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Optional at the config-loading level so Settings() always constructs -- code that
    # merely reads unrelated fields (e.g. CORS origins, at import time) shouldn't have to
    # provide a Google API key it doesn't need. Anything that actually calls the Google API
    # must go through require_google_api_key() below, which is where "fail fast with a
    # clear message" actually happens.
    google_api_key: str | None = Field(default=None, description="Google Generative AI API key")

    papers_dir: Path = Path("papers")
    chroma_dir: Path = Path("chroma_db")
    collection_name: str = "rag_docs"

    embedding_model: str = "models/gemini-embedding-001"
    llm_model: str = "gemini-2.5-flash"

    chunk_size: int = 1000
    chunk_overlap: int = 200
    inter_chunk_delay_seconds: float = 0.65  # throttles ingestion to stay under API rate limits

    retrieval_k: int = 5
    max_agent_iterations: int = 2  # self-critique retry budget before the agent gives up

    log_level: str = "INFO"

    # Comma-separated allowed origins for the API's CORS policy. Defaults cover local
    # frontend dev servers plus the GitHub Pages deployment; override via the
    # CORS_ORIGINS env var if the frontend ever moves.
    cors_origins: str = (
        "http://localhost:5173,http://127.0.0.1:5173,https://howardtaken.github.io"
    )

    # Shared secret checked against the X-API-Key header on protected endpoints. None
    # (the default, e.g. local dev with no .env override) disables the check entirely --
    # see require_api_key() in backend/main.py for why this is a soft gate, not real auth.
    api_key: str | None = Field(default=None, description="Shared API key for the backend")

    # Per-IP limit on the expensive /query endpoint. Calibrated against reality, not a
    # round number: Gemini's free tier caps gemini-2.5-flash at 20 requests/DAY, total,
    # across every user of this deployment combined -- found by exhausting it during
    # manual testing while building this feature. A per-minute limit generous enough to
    # feel normal for one user is still nearly meaningless against that shared daily cap,
    # so the real ceiling on a free-tier deployment is Google's quota, not this number.
    # This limit's job is narrower: stop one IP from single-handedly burning the whole
    # day's budget before anyone else gets a turn.
    rate_limit_query: str = "5/hour"
    rate_limit_default: str = "60/minute"  # per-IP limit on everything else (cheap, no LLM calls)

    checkpoint_db_path: Path = Path("checkpoints.sqlite")

    @property
    def cors_origins_list(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]

    def require_google_api_key(self) -> str:
        if not self.google_api_key:
            raise RuntimeError(
                "Missing required environment variable: GOOGLE_API_KEY. "
                "Copy .env.example to .env and fill in your Google API key."
            )
        return self.google_api_key


@lru_cache
def get_settings() -> Settings:
    return Settings()
