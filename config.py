import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# API Keys and Environment
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") # <-- Changed
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")
PINECONE_INDEX_NAME = "hackrx-retrieval-system-gemini" # Use a new index for different embedding dimensions

# Authentication Token for the API
API_BEARER_TOKEN = "14136f363323c2af817656c9a0fb6542f8b7c643682417a1e4e55c9ad87b81f2"

# LLM and Embedding Configuration (Switched to Google)
LLM_MODEL = "gemini-1.5-flash"
EMBEDDING_MODEL = "models/text-embedding-004"
EMBEDDING_DIMENSION = 768 # <-- Important: Google's embedding dimension

# Document Processing Configuration
CHUNK_SIZE = 1500
CHUNK_OVERLAP = 200

# Vector Store Configuration
TOP_K_RESULTS = 10