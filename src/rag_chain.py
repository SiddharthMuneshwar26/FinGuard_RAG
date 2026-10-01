"""End-to-end retrieval, reranking, generation, and citation validation."""

from __future__ import annotations

import re
import time
from functools import lru_cache
from typing import Any

from langchain_core.documents import Document

from .config import ABSTENTION_MESSAGE, RERANK_TOP_N
from .llm import OllamaClient, get_llm
from .prompts import build_messages
from .reranker import DocumentReranker, get_reranker
from .retrieval import retrieve_candidates
from .vectorstore import load_vectorstore


class FinGuardRAG:
    """Complete FinGuard retrieval-augmented generation pipeline."""

    def __init__(self) -> None:
        self.vectorstore = load_vectorstore()
        self.reranker: DocumentReranker = get_reranker()
        self.llm: OllamaClient = get_llm()

    def answer(
        self,
        question: str,
        top_k: int = RERANK_TOP_N,
        generate: bool = True,
    ) -> dict[str, Any]:
        """Run retrieval/reranking and optionally generate a grounded answer."""
        question = question.strip()
        if not question:
            raise ValueError("Question cannot be empty")
        if top_k <= 0:
            raise ValueError("top_k must be positive")

        total_start = time.perf_counter()

        retrieval_start = time.perf_counter()
        candidates = retrieve_candidates(self.vectorstore, question)
        retrieval_ms = (time.perf_counter() - retrieval_start) * 1000

        rerank_start = time.perf_counter()
        documents = self.reranker.rerank(
            question,
            candidates,
            top_n=top_k,
        )
        rerank_ms = (time.perf_counter() - rerank_start) * 1000

        if not documents:
            total_ms = (time.perf_counter() - total_start) * 1000
            return {
                "answer": ABSTENTION_MESSAGE if generate else None,
                "sources": [],
                "documents": [],
                "latency_ms": {
                    "retrieval": round(retrieval_ms, 2),
                    "rerank": round(rerank_ms, 2),
                    "generation": 0.0,
                    "total": round(total_ms, 2),
                },
            }

        if not generate:
            total_ms = (time.perf_counter() - total_start) * 1000
            return {
                "answer": None,
                "sources": self._create_source_list_from_documents(documents),
                "documents": documents,
                "latency_ms": {
                    "retrieval": round(retrieval_ms, 2),
                    "rerank": round(rerank_ms, 2),
                    "generation": 0.0,
                    "total": round(total_ms, 2),
                },
            }

        generation_start = time.perf_counter()
        answer = self.llm.invoke(build_messages(question, documents))
        generation_ms = (time.perf_counter() - generation_start) * 1000
        total_ms = (time.perf_counter() - total_start) * 1000

        return {
            "answer": answer,
            "sources": self._create_source_list(answer, documents),
            "documents": documents,
            "latency_ms": {
                "retrieval": round(retrieval_ms, 2),
                "rerank": round(rerank_ms, 2),
                "generation": round(generation_ms, 2),
                "total": round(total_ms, 2),
            },
        }

    @staticmethod
    def _create_source_list(
        answer: str,
        documents: list[Document],
    ) -> list[dict[str, Any]]:
        """Map valid inline citation IDs back to page-level source metadata."""
        cited_ids = re.findall(
            r"\[([A-Za-z0-9_-]+-p\d+-c\d+)\]",
            answer,
        )

        by_id = {
            str(document.metadata.get("chunk_id")): document
            for document in documents
        }

        sources: list[dict[str, Any]] = []
        seen: set[str] = set()

        for chunk_id in cited_ids:
            if chunk_id in seen or chunk_id not in by_id:
                continue

            seen.add(chunk_id)
            metadata = by_id[chunk_id].metadata
            filename = str(
                metadata.get(
                    "filename",
                    metadata.get("source", "Unknown"),
                )
            )

            sources.append(
                {
                    "title": str(metadata.get("title", filename)),
                    "filename": filename,
                    "file": filename,
                    "page": int(metadata.get("page", 0)),
                    "chunk_id": chunk_id,
                    "rerank_score": float(
                        metadata.get("rerank_score", 0.0)
                    ),
                }
            )

        return sources

    @staticmethod
    def _create_source_list_from_documents(
        documents: list[Document],
    ) -> list[dict[str, Any]]:
        """Create source metadata for retrieval-only responses."""
        sources: list[dict[str, Any]] = []

        for document in documents:
            metadata = document.metadata
            filename = str(
                metadata.get(
                    "filename",
                    metadata.get("source", "Unknown"),
                )
            )

            sources.append(
                {
                    "title": str(metadata.get("title", filename)),
                    "filename": filename,
                    "file": filename,
                    "page": int(metadata.get("page", 0)),
                    "chunk_id": str(
                        metadata.get("chunk_id", "")
                    ),
                    "rerank_score": float(
                        metadata.get("rerank_score", 0.0)
                    ),
                }
            )

        return sources


@lru_cache(maxsize=1)
def get_rag() -> FinGuardRAG:
    """Return one cached pipeline per process so models load only once."""
    return FinGuardRAG()


def answer_question(
    question: str,
    top_k: int = RERANK_TOP_N,
    generate: bool = True,
) -> dict[str, Any]:
    """Compatibility entry point used by the CLI and chatbot facade."""
    return get_rag().answer(
        question,
        top_k=top_k,
        generate=generate,
    )