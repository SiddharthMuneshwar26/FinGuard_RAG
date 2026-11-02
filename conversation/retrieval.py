"""Retriever used by the conversation graph: FAISS top-10 -> CrossEncoder rerank -> top-4.

Reuses the existing pipeline in src/. Models load lazily on first use and stay
in memory for the life of the process (loaded once, bounded memory). Tests pass
a fake retriever instead, so no model is ever downloaded in CI.
"""

from __future__ import annotations

import re
import threading
from typing import Any, Callable

Chunk = dict[str, Any]
Retriever = Callable[[str], list[Chunk]]


def _document_id(metadata: dict[str, Any]) -> str:
    if metadata.get("document_id"):
        return str(metadata["document_id"])
    match = re.match(r"(.+)-p\d+-c\d+$", str(metadata.get("chunk_id", "")))
    return match.group(1) if match else str(metadata.get("filename", "DOC"))


def document_to_chunk(document: Any) -> Chunk:
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


class FaissRetriever:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._loaded: tuple[Any, Any] | None = None

    def _load(self) -> tuple[Any, Any]:
        if self._loaded is None:
            with self._lock:
                if self._loaded is None:
                    from src.reranker import get_reranker
                    from src.vectorstore import load_vectorstore

                    self._loaded = (load_vectorstore(), get_reranker())
        return self._loaded

    def __call__(self, query: str) -> list[Chunk]:
        from src.config import RERANK_TOP_N, RETRIEVAL_K
        from src.retrieval import retrieve_candidates

        vectorstore, reranker = self._load()
        candidates = retrieve_candidates(vectorstore, query, k=RETRIEVAL_K)
        documents = reranker.rerank(query, candidates, top_n=RERANK_TOP_N)
        return [document_to_chunk(document) for document in documents]
