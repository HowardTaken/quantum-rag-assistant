import os
import time
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import chromadb
import subprocess

load_dotenv()

PAPERS_DIR = "papers"
CHROMA_DIR = "chroma_db"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
EMBEDDING_MODEL = "models/gemini-embedding-001"
INTER_CHUNK_DELAY = 0.65  # seconds between chunks; keeps us under 100 req/min

def load_and_split():
    loader = PyPDFDirectoryLoader(PAPERS_DIR)
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(documents)
    print(f"Loaded {len(documents)} page(s) from {PAPERS_DIR}/")
    print(f"Total chunks created: {len(chunks)}")
    return chunks

def embed_with_retry(embed_fn, text, retries=6):
    for attempt in range(retries):
        try:
            return embed_fn([text])[0]
        except Exception as e:
            msg = str(e)
            if any(x in msg for x in ("429", "RESOURCE_EXHAUSTED", "403", "PERMISSION_DENIED")):
                wait = 70 * (attempt + 1)
                print(f"\n  API limit hit (attempt {attempt + 1}), waiting {wait}s...")
                time.sleep(wait)
            else:
                raise
    raise RuntimeError("Embedding failed after all retries")

def embed_and_store(chunks):
    embedding_model = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = client.get_or_create_collection("rag_docs")

    # Resume: skip chunks whose IDs are already in the collection
    existing_ids = set(collection.get(include=[])["ids"])
    pending = [(i, c) for i, c in enumerate(chunks) if f"chunk_{i}" not in existing_ids]

    if not pending:
        print(f"All {len(chunks)} chunks already stored. Nothing to do.")
        return

    print(f"{len(existing_ids)} chunks already stored. Embedding {len(pending)} remaining...")

    for pos, (i, chunk) in enumerate(pending):
        vector = embed_with_retry(embedding_model.embed_documents, chunk.page_content)
        collection.add(
            ids=[f"chunk_{i}"],
            embeddings=[vector],
            documents=[chunk.page_content],
            metadatas=[chunk.metadata],
        )
        print(f"  [{pos + 1}/{len(pending)}] chunk_{i} stored", end="\r")
        if pos < len(pending) - 1:
            time.sleep(INTER_CHUNK_DELAY)

    print(f"\nDone. {collection.count()} vectors stored in {CHROMA_DIR}/")

def auto_git_push():
    print("\n📦 Starting automated GitHub synchronization...")
    try:
        # 1. Stage all changes (including the updated chroma_db folder)
        subprocess.run(["git", "add", "."], check=True)
        
        # 2. Create the commit message
        commit_message = "Auto-update: Embedded new research papers and updated vector database"
        subprocess.run(["git", "commit", "-m", commit_message], check=True)
        
        # 3. Push to your main branch
        subprocess.run(["git", "push", "origin", "main"], check=True)
        print("🚀 Successfully pushed updates to GitHub! Your cloud app is updating now.")
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Git automation failed: {e}")
    except Exception as e:
        print(f"❌ An unexpected error occurred during Git push: {e}")

# SINGLE ENTRY POINT
if __name__ == "__main__":
    # 1. Load data
    chunks = load_and_split()
    
    # 2. Process data and save to Chroma
    embed_and_store(chunks)
    print("Database ingestion complete!")
    
    # 3. Synchronize changes to the cloud
    auto_git_push()
