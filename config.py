"""Centralized configuration, loaded from environment variables / .env."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, ValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    google_api_key: str = Field(..., description="Google Generative AI API key")

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

    @property
    def cors_origins_list(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]


def _missing_env_error(exc: ValidationError) -> Exception:
    missing = [str(e["loc"][0]) for e in exc.errors() if e["type"] == "missing"]
    if not missing:
        return exc
    return RuntimeError(
        f"Missing required environment variable(s): {', '.join(missing)}. "
        "Copy .env.example to .env and fill in your Google API key."
    )


@lru_cache
def get_settings() -> Settings:
    try:
        return Settings()
    except ValidationError as exc:
        raise _missing_env_error(exc) from exc
