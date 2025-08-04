import requests
import tempfile
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, UnstructuredEmailLoader
from langchain.docstore.document import Document
from typing import List

from config import CHUNK_SIZE, CHUNK_OVERLAP

def load_document_from_url(url: str) -> List[Document]:
    """
    Downloads a document from a URL, determines its type, and loads its content
    into LangChain Document objects.
    """
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()

        # Use a temporary file to handle the document
        with tempfile.NamedTemporaryFile(delete=False, suffix=_get_suffix(url)) as tmp_file:
            tmp_file.write(response.content)
            tmp_file_path = tmp_file.name

        loader = _get_loader(tmp_file_path)
        documents = loader.load()
        
        # Clean up the temporary file
        os.remove(tmp_file_path)

        return documents

    except requests.exceptions.RequestException as e:
        print(f"Error downloading document from {url}: {e}")
        return []
    except Exception as e:
        print(f"Error processing document: {e}")
        return []

def _get_suffix(url: str) -> str:
    """Infers file extension from URL."""
    file_name = url.split('?')[0].split('/')[-1]
    if '.' in file_name:
        return f".{file_name.split('.')[-1]}"
    # Default suffixes for common document types if not in URL
    if 'pdf' in url.lower(): return '.pdf'
    if 'docx' in url.lower(): return '.docx'
    return '.tmp' # Default

def _get_loader(file_path: str):
    """Returns the appropriate LangChain document loader based on file extension."""
    if file_path.endswith(".pdf"):
        return PyPDFLoader(file_path)
    elif file_path.endswith(".docx"):
        return Docx2txtLoader(file_path)
    elif file_path.endswith(".eml"):
        return UnstructuredEmailLoader(file_path)
    else:
        # A fallback for other text-based formats if needed
        raise ValueError(f"Unsupported file type for: {file_path}")

def split_documents(documents: List[Document]) -> List[Document]:
    """
    Splits the loaded documents into smaller chunks for efficient embedding and retrieval.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        add_start_index=True,
    )
    chunked_documents = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} document(s) into {len(chunked_documents)} chunks.")
    return chunked_documents