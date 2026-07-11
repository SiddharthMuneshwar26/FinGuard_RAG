"""Backward-compatible facade for the modular RAG chain."""

from __future__ import annotations

from typing import Any

from .rag_chain import answer_question


def answer(question: str) -> dict[str, Any]:
    """Return an answer and cited sources using the new backend."""
    result = answer_question(question)
    return {"answer": result["answer"], "sources": result["sources"], "chunks": []}
