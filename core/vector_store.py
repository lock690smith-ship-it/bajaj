from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from typing import List
from langchain.docstore.document import Document

from config import (
    GOOGLE_API_KEY, # Changed
    PINECONE_API_KEY,
    PINECONE_ENVIRONMENT,
    PINECONE_INDEX_NAME,
    EMBEDDING_MODEL,
    EMBEDDING_DIMENSION, # Added
    TOP_K_RESULTS
)

pc = Pinecone(api_key=PINECONE_API_KEY)

def get_or_create_vector_store(chunks: List[Document], force_recreate: bool = False) -> PineconeVectorStore:
    """
    Initializes the Pinecone vector store using Google's embedding model.
    """
    # Use GoogleGenerativeAIEmbeddings
    embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL, google_api_key=GOOGLE_API_KEY)

    if force_recreate and PINECONE_INDEX_NAME in pc.list_indexes().names():
        print(f"Deleting existing index '{PINECONE_INDEX_NAME}'...")
        pc.delete_index(PINECONE_INDEX_NAME)

    if PINECONE_INDEX_NAME not in pc.list_indexes().names():
        print(f"Creating new index '{PINECONE_INDEX_NAME}' for Google embeddings...")
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=EMBEDDING_DIMENSION,  # Use the correct dimension for Google
            metric='cosine',
            spec=ServerlessSpec(
                cloud='aws',
                region='us-east-1'
            )
        )
        print("Index created. Populating with new document chunks...")
        vector_store = PineconeVectorStore.from_documents(chunks, embeddings, index_name=PINECONE_INDEX_NAME)
    else:
        print("Index already exists. Populating with new document chunks...")
        # If index exists, ensure you are adding to the correct one
        vector_store = PineconeVectorStore.from_documents(chunks, embeddings, index_name=PINECONE_INDEX_NAME)
    
    print("Vector store is ready.")
    return vector_store

def get_retriever(chunks: List[Document]):
    """
    Creates a retriever from the vector store to find relevant document chunks.
    """
    vector_store = get_or_create_vector_store(chunks)
    return vector_store.as_retriever(search_kwargs={"k": TOP_K_RESULTS})