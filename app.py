import streamlit as st
import tempfile
import os
import time
import shutil  # Added for copying the database
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rag_agent import load_vector_store, build_chain, ask

# 1. MUST BE FIRST STREAMLIT COMMAND
st.set_page_config(
    page_title="Quantum Literature RAG Assistant",
    page_icon="⚛",
    layout="centered",
)

st.title("⚛ Quantum Literature RAG Assistant")
st.caption("Ask questions about your semiconductor physics papers. Powered by Gemini + ChromaDB.")

# 2. CACHE CHAIN AND VECTOR STORE
@st.cache_resource(show_spinner="Loading vector store...")
def get_backend():
    # --- STREAMLIT CLOUD READ-ONLY WORKAROUND ---
    writable_db_path = "/tmp/chroma_db_writable"
    
    # If the temporary copy doesn't exist yet in this session, create it
    if not os.path.exists(writable_db_path):
        if os.path.exists("chroma_db"):
            shutil.copytree("chroma_db", writable_db_path)
        else:
            # Fallback if chroma_db doesn't exist at all yet
            os.makedirs(writable_db_path)
    
    # Pass the writable path to the agent
    db = load_vector_store(db_path=writable_db_path)  
    chain = build_chain(db)
    return db, chain

vector_store, chain = get_backend()

# 3. SIDEBAR AND LITERATURE MANAGEMENT
st.sidebar.header("🔬 Literature Management")
uploaded_file = st.sidebar.file_uploader("Upload a temporary paper (Session-only)", type=["pdf"])

if uploaded_file is not None:
    if f"processed_{uploaded_file.name}" not in st.session_state:
        with st.sidebar.status("Processing temporary PDF...", expanded=True) as status:
            try:
                # Create a temporary file on the server
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_path = tmp_file.name

                st.write("Reading document structure...")
                loader = PyPDFLoader(tmp_path)
                pages = loader.load()
                
                # Define text splitter explicitly so it doesn't break
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                chunks = text_splitter.split_documents(pages)
                
                st.write(f"Injecting {len(chunks)} chunks into active session context...")
                st.write(f"Trickle-feeding {len(chunks)} chunks to avoid API limits...")
                progress_bar = st.progress(0)

                # --- RATE LIMIT FIX: PROPERLY INDENTED LOOP ---
                for i, chunk in enumerate(chunks):
                    vector_store.add_documents([chunk])
                    
                    # Update progress bar
                    progress_bar.progress((i + 1) / len(chunks))
                    
                    # Wait 0.65 seconds to stay under the 100 requests/minute limit
                    time.sleep(0.65)
                # ----------------------------------------------
                
                os.remove(tmp_path)
                
                st.session_state[f"processed_{uploaded_file.name}"] = True
                status.update(label="✅ Upload Complete! Paper added to current brain.", state="complete")
                st.rerun()
                
            except Exception as e:
                status.update(label=f"❌ Ingestion Failed: {e}", state="error")

# 4. CHAT HISTORY AND INTERACTION
if "messages" not in st.session_state:
    st.session_state.messages = []

# Render existing conversation
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Handle new input
if prompt := st.chat_input("Ask a question about your papers..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = ask(prompt, chain)
        st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})
