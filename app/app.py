import streamlit as st
import sys
import os

# Add project root to sys.path so we can import from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.download_dataset import download_data
from src.ingest_pdfs import load_pdfs
from src.chunk_text import chunk_documents
from src.build_faiss_index import build_index
from src.retrieve import load_faiss_index
from src.rag_pipeline import RAGPipeline

st.set_page_config(page_title="DOCU-QUERY-RAG", layout="wide")

st.title("📄 DOCU-QUERY-RAG")
st.markdown("### Student Handbook & Academic Rules Assistant")
st.markdown("---")

# Sidebar for System Operations
with st.sidebar:
    st.header("🔧 System Management")
    
    if st.button("🚀 Re-build Knowledge Base"):
        with st.status("Building Knowledge Base...", expanded=True) as status:
            try:
                st.write("📥 Downloading dataset...")
                data_path = download_data()
                
                st.write("📑 Loading PDFs...")
                raw_docs = load_pdfs(data_path)
                st.write(f"✅ Loaded {len(raw_docs)} pages.")
                
                if not raw_docs:
                    st.error("No documents found! Check the data path.")
                    st.stop()
                
                st.write("✂️ Chunking text...")
                chunks = chunk_documents(raw_docs)
                st.write(f"✅ Created {len(chunks)} chunks.")
                
                st.write("🧠 Building FAISS index...")
                build_index(chunks)
                
                status.update(label="System Ready!", state="complete", expanded=False)
                st.success("Knowledge base rebuilt successfully!")
                
                # Clear cache to force reload of index
                st.cache_resource.clear()
                
            except Exception as e:
                st.error(f"Build failed: {e}")
                status.update(label="Build Failed", state="error")

# Check for API Key
if not os.getenv("OPENAI_API_KEY"):
    st.sidebar.warning("⚠️ OPENAI_API_KEY not found in environment.")

# Main Chat Interface
if "messages" not in st.session_state:
    st.session_state.messages = []

# Load the RAG Pipeline
@st.cache_resource(show_spinner=False)
def get_rag_pipeline():
    vector_store = load_faiss_index()
    if vector_store:
        try:
            return RAGPipeline(vector_store)
        except Exception as e:
            st.error(f"Failed to initialize RAG pipeline: {e}")
            return None
    return None

rag_pipeline = get_rag_pipeline()

if rag_pipeline is None:
    st.info("👋 Welcome! Please click **'Re-build Knowledge Base'** in the sidebar to start.")
else:
    # Display Chat History
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # User Input
    if prompt := st.chat_input("Ask a question about the university rules..."):
        # 1. Add User Message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # 2. Generate Answer
        with st.chat_message("assistant"):
            with st.spinner("Analyzing documents..."):
                answer, sources = rag_pipeline.answer_question(prompt)
                
                st.markdown(answer)
                
                # Show Sources in an expander
                if sources:
                    with st.expander("📚 Referenced Sources"):
                        for i, doc in enumerate(sources):
                            source_name = doc.metadata.get("source", "Unknown")
                            page_num = doc.metadata.get("page", "Unknown")
                            st.markdown(f"**{i+1}. {source_name} (Page {page_num})**")
                            st.caption(doc.page_content[:300] + "...")

        # 3. Add Assistant Message
        st.session_state.messages.append({"role": "assistant", "content": answer})
