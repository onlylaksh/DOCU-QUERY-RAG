import os
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from dotenv import load_dotenv

load_dotenv()

class RAGPipeline:
    def __init__(self, vector_store):
        self.vector_store = vector_store
        
        # Initialize LLM
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set.")
            
        self.llm = ChatOpenAI(
            model_name="gpt-3.5-turbo",
            temperature=0,
            openai_api_key=api_key
        )
        
        # Create Retrieval Chain
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vector_store.as_retriever(search_kwargs={"k": 5}),
            return_source_documents=True
        )

    def answer_question(self, question):
        """
        Processes the question and returns the answer + sources.
        """
        if not self.vector_store:
            return "Knowledge base not available.", []
            
        print(f"Processing question: {question}")
        try:
            response = self.qa_chain.invoke({"query": question})
            return response["result"], response["source_documents"]
        except Exception as e:
            return f"Error regenerating answer: {e}", []
