import os
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# Define where the index will be saved
INDEX_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../data/faiss_index")

def build_index(chunks):
    """
    Creates embeddings for the chunks and builds a FAISS index.
    Saves the index to disk.
    """
    if not chunks:
        print("No chunks to index.")
        return None

    print("Initializing SentenceTransformers embeddings (all-MiniLM-L6-v2)...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    print(f"Building FAISS index for {len(chunks)} chunks...")
    vector_store = FAISS.from_documents(chunks, embeddings)

    print(f"Saving index to {INDEX_DIR}...")
    if not os.path.exists(INDEX_DIR):
        os.makedirs(INDEX_DIR)
        
    vector_store.save_local(INDEX_DIR)
    print("Index built and saved successfully.")
    return vector_store
