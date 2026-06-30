import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

CHROMA_DIR = "chroma_db"
EMBEDDING_MODEL = "models/gemini-embedding-001"
LLM_MODEL = "gemini-2.5-flash"
NUM_RESULTS = 5

SYSTEM_PROMPT = (
    "You are an elite ECE quantum physics research assistant. "
    "Use the provided context to answer the question. "
    "If the answer is not in the context, say you do not know. "
    "Always cite the source document name in your answer."
)

PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "Context:\n{context}\n\nQuestion: {question}"),
])

def format_docs(docs):
    return "\n\n---\n\n".join(
        f"[Source: {d.metadata.get('source', 'unknown')}, page {d.metadata.get('page', '?')}]\n{d.page_content}"
        for d in docs
    )

def load_vector_store():
    embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)
    db = Chroma(
        collection_name="rag_docs",
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR,
    )
    return db

def build_chain(db):
    retriever = db.as_retriever(search_kwargs={"k": NUM_RESULTS})
    llm = ChatGoogleGenerativeAI(model=LLM_MODEL, temperature=0)

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | PROMPT
        | llm
        | StrOutputParser()
    )
    return chain

def ask(question: str, chain) -> str:
    return chain.invoke(question)

if __name__ == "__main__":
    print("Loading vector store...")
    db = load_vector_store()
    print(f"Loaded {db._collection.count()} vectors from {CHROMA_DIR}/\n")

    chain = build_chain(db)

    while True:
        question = input("Ask a question (or 'quit'): ").strip()
        if question.lower() in ("quit", "exit", "q"):
            break
        if not question:
            continue
        print("\nThinking...\n")
        answer = ask(question, chain)
        print(f"Answer:\n{answer}\n")
        print("-" * 60)
