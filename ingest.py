import logging
import time
from pathlib import Path
from typing import Iterable, Protocol

import chromadb
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import get_settings

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


class UploadedFile(Protocol):
    """Matches Streamlit's UploadedFile interface -- narrow enough that tests don't need
    Streamlit installed to construct a fake one."""

    name: str

    def getvalue(self) -> bytes: ...


def save_uploaded_pdfs(papers_dir: Path, files: Iterable[UploadedFile]) -> list[Path]:
    """Write uploaded PDFs into papers_dir, ready for load_and_split()/embed_and_store().

    Not exposed over the public API on purpose (see backend/main.py's docstring on scope) --
    this is for the local Streamlit app, run by the paper library's owner, not the public.
    """
    papers_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    for f in files:
        if not f.name.lower().endswith(".pdf"):
            raise ValueError(f"Not a PDF: {f.name}")
        dest = papers_dir / Path(f.name).name  # basename only, no directory traversal
        dest.write_bytes(f.getvalue())
        saved.append(dest)
    return saved


def load_and_split():
    settings = get_settings()
    if not settings.papers_dir.is_dir():
        raise RuntimeError(f"Papers directory not found: {settings.papers_dir}/")

    loader = PyPDFDirectoryLoader(str(settings.papers_dir))
    documents = loader.load()
    if not documents:
        raise RuntimeError(f"No PDFs found in {settings.papers_dir}/")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    chunks = splitter.split_documents(documents)
    for chunk in chunks:
        # PyPDFDirectoryLoader stores the full (OS-specific) path; normalize to the
        # bare filename so citations shown to users don't leak "papers\..." paths.
        source = chunk.metadata.get("source")
        if source:
            chunk.metadata["source"] = Path(source).name
    logger.info("Loaded %d page(s) from %s/", len(documents), settings.papers_dir)
    logger.info("Total chunks created: %d", len(chunks))
    return chunks

def embed_with_retry(embed_fn, text, retries=6):
    for attempt in range(retries):
        try:
            return embed_fn([text])[0]
        except Exception as e:
            msg = str(e)
            if any(x in msg for x in ("429", "RESOURCE_EXHAUSTED", "403", "PERMISSION_DENIED")):
                wait = 70 * (attempt + 1)
                logger.warning("API limit hit (attempt %d), waiting %ds...", attempt + 1, wait)
                time.sleep(wait)
            else:
                raise
    raise RuntimeError("Embedding failed after all retries")

def embed_and_store(chunks):
    settings = get_settings()
    embedding_model = GoogleGenerativeAIEmbeddings(
        model=settings.embedding_model, google_api_key=settings.require_google_api_key()
    )
    client = chromadb.PersistentClient(path=str(settings.chroma_dir))
    collection = client.get_or_create_collection(settings.collection_name)

    # Resume: skip chunks whose IDs are already in the collection
    existing_ids = set(collection.get(include=[])["ids"])
    pending = [(i, c) for i, c in enumerate(chunks) if f"chunk_{i}" not in existing_ids]

    if not pending:
        logger.info("All %d chunks already stored. Nothing to do.", len(chunks))
        return

    logger.info("%d chunks already stored. Embedding %d remaining...", len(existing_ids), len(pending))

    for pos, (i, chunk) in enumerate(pending):
        vector = embed_with_retry(embedding_model.embed_documents, chunk.page_content)
        collection.add(
            ids=[f"chunk_{i}"],
            embeddings=[vector],
            documents=[chunk.page_content],
            metadatas=[chunk.metadata],
        )
        logger.info("[%d/%d] chunk_%d stored", pos + 1, len(pending), i)
        if pos < len(pending) - 1:
            time.sleep(settings.inter_chunk_delay_seconds)

    logger.info("Done. %d vectors stored in %s/", collection.count(), settings.chroma_dir)

if __name__ == "__main__":
    chunks = load_and_split()
    embed_and_store(chunks)
