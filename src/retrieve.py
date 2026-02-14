import os
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# Must match the path in build_faiss_index.py
INDEX_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../data/faiss_index")

def load_faiss_index():
    """
    Loads the FAISS index from disk.
    """
    if not os.path.exists(INDEX_DIR):
        print(f"Index directory not found at {INDEX_DIR}")
        return None
    
    print(f"Loading FAISS index from {INDEX_DIR}...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    try:
        vector_store = FAISS.load_local(INDEX_DIR, embeddings, allow_dangerous_deserialization=True)
        print("Index loaded successfully.")
        return vector_store
    except Exception as e:
        print(f"Error loading index: {e}")
        return None

def retrieve_context(vector_store, query, k=5):
    """
    Retrieves the top-k most relevant documents for the query.
    """
    if not vector_store:
        return []
    
    print(f"Retrieving top {k} chunks for query: '{query}'")
    docs = vector_store.similarity_search(query, k=k)
    return docs
