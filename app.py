import uuid

import streamlit as st

from agent import ask, build_agent
from config import get_settings
from ingest import embed_and_store, load_and_split, save_uploaded_pdfs
from tools import load_vector_store, source_counts

st.set_page_config(
    page_title="Quantum Literature RAG Assistant",
    page_icon="⚛",
    layout="centered",
)

st.title("⚛ Quantum Literature RAG Assistant")
st.caption(
    "Ask questions about your semiconductor physics papers. "
    "Powered by Gemini + ChromaDB + a LangGraph agent (search tools, self-critique retry)."
)

# Load the vector store and agent once per process and cache them (the LLM/tools/graph are
# shared across sessions; per-visitor conversation state lives in the checkpointer, keyed by
# thread_id below). Split into two cached functions so the sidebar's "currently ingested"
# list can reuse the same connection instead of opening a second one.
@st.cache_resource(show_spinner="Loading vector store...")
def get_db():
    try:
        return load_vector_store()
    except RuntimeError as exc:
        st.error(f"Startup failed: {exc}")
        st.stop()

@st.cache_resource(show_spinner="Loading agent...")
def get_agent():
    return build_agent(db=get_db())

agent = get_agent()

with st.sidebar:
    st.header("Add papers")
    st.caption(
        "Runs locally and embeds via the Gemini API. The free tier caps embedding requests "
        "at a small number per day, shared across everything using this key -- each chunk "
        "costs one request, so a handful of large papers can exhaust it fast."
    )
    uploaded_files = st.file_uploader(
        "Upload PDF(s)", type="pdf", accept_multiple_files=True, key="paper_uploads"
    )
    if uploaded_files and st.button("Ingest uploaded papers"):
        settings = get_settings()
        try:
            with st.spinner("Saving files..."):
                save_uploaded_pdfs(settings.papers_dir, uploaded_files)
            with st.spinner("Chunking and embedding (can take a while)..."):
                chunks = load_and_split()
                embed_and_store(chunks)
        except Exception as exc:
            st.error(f"Ingestion failed: {exc}")
        else:
            st.cache_resource.clear()
            st.success(f"Ingested {len(uploaded_files)} file(s). Reloading...")
            st.rerun()

    st.divider()
    st.caption("Currently ingested:")
    for name, count in sorted(source_counts(get_db()).items()):
        st.caption(f"- {name} ({count} chunks)")

# Each browser session gets its own conversation thread so multi-turn context doesn't leak
# between visitors.
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

# Initialise chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Render existing conversation
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Handle new input
if prompt := st.chat_input("Ask a question about your papers..."):
    # Show and store the user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate and stream the assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = ask(prompt, agent, thread_id=st.session_state.thread_id)
            except Exception as exc:
                response = f"Sorry, something went wrong answering that: {exc}"
        st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})
