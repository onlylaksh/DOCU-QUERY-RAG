# DOCU-QUERY-RAG

## Project Goal
Build a fully working Retrieval-Augmented Generation (RAG) system that allows users to ask natural language questions and get answers from a large collection of PDF documents.

## Dataset
This project uses the "Dataset of PDF Files" from Kaggle, downloaded programmatically using `kagglehub`.

## Architecture
1.  **Data Ingestion**: Download dataset via `kagglehub` and parse PDFs.
2.  **Indexing**: Create embeddings using `SentenceTransformers` and store in `FAISS`.
3.  **Retrieval**: Search for relevant chunks using vector similarity.
4.  **Generation**: Generate answers using an LLM (OpenAI GPT).
5.  **UI**: Interactive web interface using `Streamlit`.

## Setup
1.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
2.  Set up environment variables (create `.env`):
    ```
    OPENAI_API_KEY=your_api_key_here
    ```
3.  Run the application:
    ```bash
    streamlit run app.py
    ```
