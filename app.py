import streamlit as st
from rag_agent import load_vector_store, build_chain, ask

st.set_page_config(
    page_title="Quantum Literature RAG Assistant",
    page_icon="⚛",
    layout="centered",
)

st.title("⚛ Quantum Literature RAG Assistant")
st.caption("Ask questions about your semiconductor physics papers. Powered by Gemini + ChromaDB.")

# Load chain once per session and cache it
@st.cache_resource(show_spinner="Loading vector store...")
def get_chain():
    db = load_vector_store()
    return build_chain(db)

chain = get_chain()

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
            response = ask(prompt, chain)
        st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})
