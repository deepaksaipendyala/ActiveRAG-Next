# retrievers/vectorstore.py

import logging
from typing import List, Optional

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document

from utils.config import settings

# Setup logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Embedding Model Selection ---

def get_embedding_model() -> Embeddings:
    """
    Initializes the embedding model based on settings.
    Supports HuggingFace or OpenAI.
    """
    provider = settings.EMBEDDING_PROVIDER.lower()

    if provider == "huggingface":
        logger.info(f"Loading HuggingFace Embedding model: {settings.EMBEDDING_MODEL}")
        return HuggingFaceEmbeddings(
            model_name=settings.EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},   # Change to "cuda" if you have GPU
            encode_kwargs={"normalize_embeddings": True}
        )
    elif provider == "openai":
        logger.info(f"Loading OpenAI Embedding model: {settings.EMBEDDING_MODEL}")
        return OpenAIEmbeddings(
            model=settings.EMBEDDING_MODEL,
            api_key=settings.OPENAI_API_KEY
        )
    else:
        raise ValueError(f"Unsupported embedding provider: {provider}")

# --- Vectorstore Setup ---

_vectordb: Optional[Chroma] = None

def get_vectorstore() -> Chroma:
    """
    Returns a singleton instance of the Chroma vector store.
    """
    global _vectordb
    if _vectordb is not None:
        return _vectordb

    embeddings = get_embedding_model()
    vectordb = Chroma(
        persist_directory=settings.CHROMA_DB_PATH,
        embedding_function=embeddings
    )

    _vectordb = vectordb
    return vectordb

# --- Add Documents ---

def add_documents(texts: List[str]):
    """
    Adds a list of raw text passages to the vectorstore.
    Each passage becomes a Document.
    """
    if not texts:
        logger.warning("No documents provided to add to vectorstore.")
        return

    vectordb = get_vectorstore()
    documents = [Document(page_content=text) for text in texts]
    vectordb.add_documents(documents)
    logger.info(f"Successfully added {len(documents)} documents to vectorstore.")

# --- Query Documents ---

def query(text: str, k: int = 5) -> List[str]:
    """
    Queries the vectorstore for top-k similar documents.
    Returns list of page_content strings.
    """
    vectordb = get_vectorstore()
    docs = vectordb.similarity_search(text, k=k)
    return [doc.page_content for doc in docs]
