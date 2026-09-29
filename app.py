import uuid

import streamlit as st

from agent import ask, build_agent
from tools import load_vector_store

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

# Load the agent once per process and cache it (the LLM/tools/graph are shared across sessions;
# per-visitor conversation state lives in the checkpointer, keyed by thread_id below).
@st.cache_resource(show_spinner="Loading vector store...")
def get_agent():
    try:
        db = load_vector_store()
        return build_agent(db=db)
    except RuntimeError as exc:
        st.error(f"Startup failed: {exc}")
        st.stop()

agent = get_agent()

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
