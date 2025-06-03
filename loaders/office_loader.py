# loaders/office_loader.py

import os
import logging
from typing import List

from langchain_core.documents import Document
from langchain_community.document_loaders import (
    UnstructuredWordDocumentLoader,
    UnstructuredExcelLoader,
    UnstructuredPowerPointLoader,
)

# Setup logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_office_documents(directory: str) -> List[Document]:
    """
    Loads Word, Excel, and PowerPoint documents from a directory into Document objects.

    Args:
        directory (str): Directory path containing .docx, .xlsx, .pptx files.

    Returns:
        List[Document]: Loaded Document objects.
    """
    if not os.path.exists(directory):
        logger.warning(f"Office docs directory not found: {directory}")
        return []

    documents = []
    logger.info(f"Loading Office documents from: {directory}")

    for filename in os.listdir(directory):
        full_path = os.path.join(directory, filename)

        try:
            if filename.lower().endswith(".docx"):
                loader = UnstructuredWordDocumentLoader(full_path)
            elif filename.lower().endswith(".xlsx"):
                loader = UnstructuredExcelLoader(full_path)
            elif filename.lower().endswith(".pptx"):
                loader = UnstructuredPowerPointLoader(full_path)
            else:
                continue  # Skip non-office files

            docs = loader.load()

            # Add source info if missing
            for doc in docs:
                if 'source' not in doc.metadata:
                    doc.metadata['source'] = full_path

            documents.extend(docs)
            logger.info(f"Loaded {len(docs)} documents from {filename}")

        except Exception as e:
            logger.error(f"Failed to load {filename}: {e}")

    logger.info(f"Total Office documents loaded: {len(documents)}")
    return documents

# --- Example usage (for quick testing) ---
if __name__ == "__main__":
    test_dir = "./data/office_docs"
    loaded_docs = load_office_documents(test_dir)
    print(f"Loaded {len(loaded_docs)} office documents.")
