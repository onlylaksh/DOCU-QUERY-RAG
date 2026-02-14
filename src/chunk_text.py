from langchain.text_splitter import RecursiveCharacterTextSplitter

def chunk_documents(documents, chunk_size=1000, chunk_overlap=200):
    """
    Splits the documents into overlapping chunks.
    """
    if not documents:
        print("No documents provided to chunk.")
        return []

    print(f"Splitting {len(documents)} documents with chunk_size={chunk_size}, overlap={chunk_overlap}...")
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    
    chunks = splitter.split_documents(documents)
    print(f"Generated {len(chunks)} text chunks.")
    return chunks
