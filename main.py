from fastapi import FastAPI, Depends, HTTPException, status, Request
# from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials  # Removed: No longer needed
from pydantic import BaseModel, HttpUrl
from typing import List
import uvicorn
import os  # Make sure this is imported if using os.environ

from config import API_BEARER_TOKEN  # This is unused now, but safe to keep for future
from core.document_processor import load_document_from_url, split_documents
from core.vector_store import get_retriever
from core.llm_handler import create_qa_chain, get_answers

# --- API Setup ---
app = FastAPI(
    title="LLM-Powered Intelligent Query–Retrieval System",
    description="Processes documents to answer questions using a RAG pipeline.",
    version="1.0.0"
)

# --- Authentication (DISABLED) ---
# Commented out the following lines as token verification is no longer needed.

# auth_scheme = HTTPBearer()

# def verify_token(credentials: HTTPAuthorizationCredentials = Depends(auth_scheme)):
#     """Dependency to verify the bearer token."""
#     if not credentials or credentials.scheme != "Bearer" or credentials.credentials != API_BEARER_TOKEN:
#         raise HTTPException(
#             status_code=status.HTTP_401_UNAUTHORIZED,
#             detail="Invalid or missing authentication token",
#             headers={"WWW-Authenticate": "Bearer"},
#         )
#     return credentials

# --- Pydantic Models for API I/O ---
class RunRequest(BaseModel):
    documents: HttpUrl
    questions: List[str]

class RunResponse(BaseModel):
    answers: List[str]

# --- API Endpoints ---
@app.post("/api/v1/hackrx/run", response_model=RunResponse)
# Removed: dependencies=[Depends(verify_token)] to disable Bearer token authentication
async def run_submission(request: RunRequest):
    """
    This endpoint orchestrates the entire query-retrieval process:
    1.  Downloads and parses the document from the provided URL.
    2.  Splits the document into chunks.
    3.  Creates a vector store and retriever using Pinecone.
    4.  Initializes the RAG chain with an LLM.
    5.  Processes each question to generate an answer based on the document.
    """
    try:
        # 1. Input Document Processing
        print(f"Loading document from URL: {request.documents}")
        documents = load_document_from_url(str(request.documents))
        if not documents:
            raise HTTPException(status_code=400, detail="Could not load or parse the document from the URL.")

        # 2. Chunking
        chunks = split_documents(documents)

        # 3. Embedding and Retrieval Setup
        print("Initializing retriever...")
        retriever = get_retriever(chunks)

        # 4. LLM Chain Setup
        print("Creating QA chain...")
        rag_chain = create_qa_chain(retriever)

        # 5. Logic Evaluation (Answering Questions)
        print("Generating answers...")
        results = await get_answers(request.questions, rag_chain)

        # 6. JSON Output
        final_answers = [res["answer"] for res in results]
        
        return RunResponse(answers=final_answers)

    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def read_root():
    return {"message": "Welcome to the HackRx Query-Retrieval System API. Head to /docs for the API documentation."}

# --- Main execution ---
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
