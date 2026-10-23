"""`search_governance_documents`: FAISS top-10 -> CrossEncoder rerank -> top-4.

Reuses the existing pipeline in src/ (load_vectorstore, retrieve_candidates,
DocumentReranker). Models load lazily on the first search and are cached for
the process. Tests swap the retriever with `set_retriever()` so no model is
ever downloaded.
"""

from __future__ import annotations

import re
import threading
from typing import Any, Callable

from google.adk.tools.tool_context import ToolContext

from .settings import RERANK_TOP_N, RETRIEVAL_K
from .turn_state import register_citations

Chunk = dict[str, Any]
Retriever = Callable[[str], list[Chunk]]

_retriever: Retriever | None = None
_lock = threading.Lock()


def _document_id(metadata: dict[str, Any]) -> str:
    if metadata.get("document_id"):
        return str(metadata["document_id"])
    chunk_id = str(metadata.get("chunk_id", ""))
    match = re.match(r"(.+)-p\d+-c\d+$", chunk_id)
    return match.group(1) if match else str(metadata.get("filename", "DOC"))


def document_to_chunk(document: Any) -> Chunk:
    """Convert a LangChain Document from src/ into a JSON-safe tool result."""
    metadata = dict(document.metadata)
    return {
        "chunk_id": str(metadata.get("chunk_id", "")),
        "document_id": _document_id(metadata),
        "title": str(metadata.get("title", metadata.get("filename", "Unknown"))),
        "source": str(metadata.get("filename", metadata.get("source", "Unknown"))),
        "page": int(metadata.get("page", 0)),
        "rerank_score": round(float(metadata.get("rerank_score", 0.0)), 4),
        "content": str(document.page_content),
    }


def _build_default_retriever() -> Retriever:
    # Imported lazily: these pull in FAISS, torch and sentence-transformers.
    from src.reranker import get_reranker
    from src.retrieval import retrieve_candidates
    from src.vectorstore import load_vectorstore

    vectorstore = load_vectorstore()
    reranker = get_reranker()

    def retrieve(query: str) -> list[Chunk]:
        candidates = retrieve_candidates(vectorstore, query, k=RETRIEVAL_K)
        documents = reranker.rerank(query, candidates, top_n=RERANK_TOP_N)
        return [document_to_chunk(document) for document in documents]

    return retrieve


def get_retriever() -> Retriever:
    global _retriever
    if _retriever is None:
        with _lock:
            if _retriever is None:
                _retriever = _build_default_retriever()
    return _retriever


def set_retriever(retriever: Retriever | None) -> None:
    """Override the retriever (tests, or a remote retrieval service)."""
    global _retriever
    _retriever = retriever


def search_governance_documents(query: str, tool_context: ToolContext) -> dict[str, Any]:
    """Search the indexed financial-AI governance documents (SHARC, SR 26-2 GenAI framework).

    Runs dense FAISS retrieval (top 10) followed by CrossEncoder reranking and
    returns the 4 most relevant chunks. Cite each chunk you use by its
    chunk_id in square brackets, for example [SHARC-p9-c1].

    Args:
        query: A focused natural-language search query.

    Returns:
        A dict with "status" and "results"; each result has chunk_id, document_id,
        title, source, page, rerank_score and content.
    """
    try:
        chunks = get_retriever()(query.strip())
    except FileNotFoundError as exc:
        return {"status": "error", "error_message": str(exc), "results": []}
    except Exception as exc:  # surface retrieval failures to the model, don't crash the run
        return {"status": "error", "error_message": f"Retrieval failed: {exc}", "results": []}

    register_citations(
        tool_context,
        [chunk["chunk_id"] for chunk in chunks],
        pages=[f"{chunk['document_id']}:{chunk['page']}" for chunk in chunks],
    )
    if not chunks:
        return {"status": "no_results", "results": []}
    return {"status": "success", "results": chunks}
