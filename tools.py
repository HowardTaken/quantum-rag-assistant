"""Vector store access and the tools exposed to the agent."""
from __future__ import annotations

import logging

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.tools import BaseTool, tool
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from config import get_settings

logger = logging.getLogger(__name__)


def load_vector_store() -> Chroma:
    settings = get_settings()
    embeddings = GoogleGenerativeAIEmbeddings(
        model=settings.embedding_model, google_api_key=settings.require_google_api_key()
    )
    return Chroma(
        collection_name=settings.collection_name,
        embedding_function=embeddings,
        persist_directory=str(settings.chroma_dir),
    )


def format_docs(docs: list[Document]) -> str:
    if not docs:
        return "No relevant passages found."
    return "\n\n---\n\n".join(
        f"[Source: {d.metadata.get('source', 'unknown')}, page {d.metadata.get('page', '?')}]\n"
        f"{d.page_content}"
        for d in docs
    )


def source_counts(db: Chroma) -> dict[str, int]:
    raw = db.get(include=["metadatas"])
    counts: dict[str, int] = {}
    for meta in raw.get("metadatas") or []:
        name = meta.get("source", "unknown")
        counts[name] = counts.get(name, 0) + 1
    return counts


def make_tools(db: Chroma) -> list[BaseTool]:
    """Build the tools bound to a specific vector store instance."""
    settings = get_settings()

    @tool
    def search_papers(query: str) -> str:
        """Search the ingested physics papers for passages relevant to the query.

        Returns the top matching excerpts, each tagged with its source filename and
        page number. Use this whenever a question needs specific facts, figures, or
        claims from the papers -- never answer factual questions from memory alone.
        """
        docs = db.similarity_search(query, k=settings.retrieval_k)
        logger.info("search_papers query=%r hits=%d", query, len(docs))
        return format_docs(docs)

    @tool
    def list_available_papers() -> str:
        """List the source documents currently ingested, with their chunk counts.

        Use this when the user asks what papers/documents are available, instead of
        searching paper content.
        """
        counts = source_counts(db)
        if not counts:
            return "No papers are currently ingested."
        return "\n".join(f"- {name} ({n} chunks)" for name, n in sorted(counts.items()))

    return [search_papers, list_available_papers]
