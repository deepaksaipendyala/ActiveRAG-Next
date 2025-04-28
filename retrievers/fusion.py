# retrievers/fusion.py

import logging
from typing import List
from langchain_core.documents import Document
from utils.llm import get_llm
from retrievers.vectorstore import get_vectorstore

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- 1. Generate Multi-Queries using LLM ---

def generate_queries(query: str, n: int = 3) -> List[str]:
    """
    Generate multiple rephrased versions of the query for better retrieval coverage.
    """
    logger.info(f"Generating {n} rephrased queries for: '{query}'")

    llm = get_llm()

    prompt = (
        f"Given the user question:\n\n'{query}'\n\n"
        f"Generate {n} diverse, relevant rephrasings of the query.\n"
        f"Return each query on a new line."
    )

    response = llm.invoke(prompt)
    queries = [q.strip("- ").strip() for q in response.content.strip().split("\n") if q.strip()]

    if not queries:
        queries = [query]  # fallback to original

    logger.info(f"Generated rephrased queries: {queries}")
    return queries

# --- 2. Reciprocal Rank Fusion (RRF) ---

def reciprocal_rank_fusion(results: List[List[Document]], k: int = 5) -> List[Document]:
    """
    Apply Reciprocal Rank Fusion (RRF) to merge multiple retrieval lists.
    """
    logger.info("Applying Reciprocal Rank Fusion...")

    scores = {}
    for docs in results:
        for rank, doc in enumerate(docs):
            key = doc.page_content
            scores[key] = scores.get(key, 0) + 1 / (rank + 1 + 60)  # 60 is smoothing constant

    # Rank by combined RRF scores
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    # Deduplicate and map back to Document objects
    unique_docs = {}
    for content, _ in ranked:
        if content not in unique_docs:
            unique_docs[content] = Document(page_content=content)

    logger.info(f"RRF produced {min(k, len(unique_docs))} top documents.")
    return list(unique_docs.values())[:k]

# --- 3. Full Fusion Retrieval Pipeline ---

def retrieve_with_fusion(
    query: str,
    vectordb,
    llm,
    num_queries: int = 3,  # <-- ADD THIS
    top_n: int = 5
) -> List[str]:
    """
    Full pipeline:
    1. Generate multiple queries (fusion queries)
    2. Retrieve documents for each
    3. Merge results using Reciprocal Rank Fusion (RRF)
    """
    logger.info(f"Starting RAG-Fusion for query: '{query}'")

    if vectordb is None:
        vectordb = get_vectorstore()

    # Generate multiple queries
    queries = generate_queries(query, n=num_queries)

    all_results = []
    for q in queries:
        docs = vectordb.similarity_search(q, k=top_n)
        all_results.append(docs)

    # Apply RRF
    fused_docs = reciprocal_rank_fusion(all_results, k=top_n)

    return [doc.page_content for doc in fused_docs]
