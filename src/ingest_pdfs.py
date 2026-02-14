import os
import glob
from langchain_community.document_loaders import PyPDFLoader

def load_pdfs(source_dir):
    """
    Scans the source directory for PDF files and loads them.
    Returns a list of LangChain Document objects.
    """
    if not os.path.isdir(source_dir):
        raise ValueError(f"Source directory '{source_dir}' does not exist.")

    pdf_files = glob.glob(os.path.join(source_dir, "**/*.pdf"), recursive=True)
    documents = []

    print(f"Found {len(pdf_files)} PDF files in '{source_dir}'.")

    for file_path in pdf_files:
        try:
            print(f"Loading: {os.path.basename(file_path)}")
            loader = PyPDFLoader(file_path)
            docs = loader.load()
            
            # Ensure metadata is robust
            for doc in docs:
                doc.metadata["source"] = os.path.basename(file_path)
                # Ensure page is an integer
                doc.metadata["page"] = int(doc.metadata.get("page", 0)) + 1
                
            documents.extend(docs)
        except Exception as e:
            print(f"Failed to load {file_path}: {e}")

    print(f"Successfully loaded {len(documents)} document pages.")
    return documents
