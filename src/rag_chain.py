"""End-to-end retrieval, reranking, generation, and citation validation."""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Any

from langchain_core.documents import Document

from .config import ABSTENTION_MESSAGE
from .llm import OllamaClient, get_llm
from .prompts import build_messages
from .reranker import DocumentReranker
from .retrieval import retrieve_candidates
from .vectorstore import load_vectorstore


class FinGuardRAG:
    """Complete FinGuard retrieval-augmented generation pipeline."""

    def __init__(self) -> None:
        self.vectorstore = load_vectorstore()
        self.reranker = DocumentReranker()
        self.llm: OllamaClient = get_llm()

    def answer(self, question: str) -> dict[str, Any]:
        """Generate a grounded answer and return only sources actually cited."""
        question = question.strip()
        if not question:
            raise ValueError("Question cannot be empty")

        candidates = retrieve_candidates(self.vectorstore, question)
        documents = self.reranker.rerank(question, candidates)
        if not documents:
            return {"answer": ABSTENTION_MESSAGE, "sources": [], "documents": []}

        answer = self.llm.invoke(build_messages(question, documents))
        return {
            "answer": answer,
            "sources": self._create_source_list(answer, documents),
            "documents": documents,
        }

    @staticmethod
    def _create_source_list(answer: str, documents: list[Document]) -> list[dict[str, Any]]:
        """Map valid inline citation IDs back to page-level source metadata."""
        cited_ids = re.findall(r"\[([A-Za-z0-9_-]+-p\d+-c\d+)\]", answer)
        by_id = {str(document.metadata.get("chunk_id")): document for document in documents}
        sources: list[dict[str, Any]] = []
        seen: set[str] = set()
        for chunk_id in cited_ids:
            if chunk_id in seen or chunk_id not in by_id:
                continue
            seen.add(chunk_id)
            metadata = by_id[chunk_id].metadata
            filename = str(metadata.get("filename", metadata.get("source", "Unknown")))
            sources.append(
                {
                    "title": str(metadata.get("title", filename)),
                    "filename": filename,
                    "file": filename,
                    "page": int(metadata.get("page", 0)),
                    "chunk_id": chunk_id,
                    "rerank_score": float(metadata.get("rerank_score", 0.0)),
                }
            )
        return sources


@lru_cache(maxsize=1)
def get_rag() -> FinGuardRAG:
    """Return one cached pipeline per process so models load only once."""
    return FinGuardRAG()


def answer_question(question: str) -> dict[str, Any]:
    """Compatibility entry point used by the CLI and chatbot facade."""
    return get_rag().answer(question)
