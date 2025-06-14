import sys
from pathlib import Path
from typing import List

from loaders.pdf_loader import load_pdf
from retrievers.vectorstore import get_vectorstore, add_documents
from retrievers.fusion import retrieve_with_fusion
from utils.llm import get_llm


def load_local_pdfs(directory: Path) -> List[str]:
    """Return page contents from all PDFs in the given directory."""
    texts: List[str] = []
    for pdf_file in directory.glob("*.pdf"):
        for doc in load_pdf(str(pdf_file)):
            texts.append(doc.page_content)
    return texts


def ingest_demo_documents() -> None:
    """Load PDF documents from the project resources into the vector store."""
    resource_dir = Path(__file__).resolve().parents[1] / "resources"
    texts = load_local_pdfs(resource_dir)
    if texts:
        add_documents(texts)


def run_rag(query: str) -> str:
    """Run a basic RAG pipeline using the existing retrieval utilities."""
    llm = get_llm()
    vectordb = get_vectorstore()

    ingest_demo_documents()

    passages = retrieve_with_fusion(query, vectordb, llm, num_queries=3, top_n=5)
    context = "\n\n".join(passages)

    prompt = (
        "Answer the question based only on the following context.\n\n" +
        context +
        f"\n\nQuestion: {query}"
    )
    response = llm.invoke(prompt)
    return response.content.strip()


if __name__ == "__main__":
    question = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "What is phosphorous?"
    print(run_rag(question))
