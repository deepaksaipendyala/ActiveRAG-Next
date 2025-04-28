# loaders/web_loader.py

import logging
from typing import List
from langchain.schema import Document
import trafilatura

# Setup logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_webpage(url: str) -> List[Document]:
    try:
        downloaded = trafilatura.fetch_url(url)
        if downloaded is None:
            raise Exception(f"Failed to fetch content from {url}")

        extracted_text = trafilatura.extract(downloaded, include_comments=False, include_tables=True, include_images=False)
        if extracted_text is None:
            raise Exception(f"Failed to extract content from {url}")

        return [Document(page_content=extracted_text, metadata={"source": url})]

    except Exception as e:
        logger.error(f"Error loading web page {url}: {e}")
        return []

def load_webpages(urls: List[str]) -> List[Document]:
    all_documents = []
    logger.info(f"Starting batch load for {len(urls)} URL(s)...")
    for url in urls:
        docs = load_webpage(url)
        if docs:
            all_documents.extend(docs)
    logger.info(f"Completed loading {len(all_documents)} total web documents.")
    return all_documents

# --- Example usage ---
if __name__ == "__main__":
    test_urls = [
        "https://lilianweng.github.io/posts/2023-06-23-agent/",
        "https://arxiv.org/abs/2305.12345"
    ]
    docs = load_webpages(test_urls)
    for doc in docs:
        print(f"\n--- Source: {doc.metadata['source']} ---")
        print(f"Title: {doc.metadata.get('title', 'No Title')}")
        print(f"Content Preview: {doc.page_content[:500]}...\n")
