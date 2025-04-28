# loaders/pdf_loader.py

import logging
from typing import List
from langchain_core.documents import Document
from langchain_community.document_loaders import PyMuPDFLoader

# Setup logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def load_pdf(file_path: str) -> List[Document]:
    """
    Load a PDF and return a list of LangChain Document objects.
    """
    logger.info(f"Loading PDF: {file_path}")

    try:
        loader = PyMuPDFLoader(file_path)
        documents = loader.load()

        for doc in documents:
            if 'source' not in doc.metadata:
                doc.metadata['source'] = file_path

        logger.info(f"Successfully loaded {len(documents)} pages from {file_path}")
        return documents

    except Exception as e:
        logger.error(f"Failed to load PDF {file_path}: {e}")
        return []


# --- Example usage ---
# if __name__ == "__main__":
#     documents = load_pdf("path/to/your/file.pdf")
#     print(f"Loaded {len(documents)} documents.")

#     documents = load_pdfs_from_directory("path/to/your/pdf/folder/")
#     print(f"Total documents loaded: {len(documents)}")

