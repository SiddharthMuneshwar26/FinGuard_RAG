"""Grounding and citation prompts."""

from __future__ import annotations

from langchain_core.documents import Document

from .config import ABSTENTION_MESSAGE

SYSTEM_PROMPT = f"""You are FinGuard, a precise assistant for financial AI governance
and risk-model interpretability.

Rules:
- Use only the supplied context. Never add facts from outside knowledge.
- Cite every factual claim with one or more supplied chunk IDs in square brackets.
- Keep the exact citation IDs unchanged, for example [SHARC-p9-c1].
- If the context does not answer the question, reply exactly: {ABSTENTION_MESSAGE}
- Be concise and prefer the documents' terminology."""


def format_context(documents: list[Document]) -> str:
    """Format reranked documents as citation-addressable context blocks."""
    blocks: list[str] = []
    for document in documents:
        metadata = document.metadata
        blocks.append(
            f"[{metadata['chunk_id']}]\n"
            f"Title: {metadata['title']}\n"
            f"Filename: {metadata['filename']}\n"
            f"Page: {metadata['page']}\n"
            f"{document.page_content}"
        )
    return "\n\n".join(blocks)


def build_user_prompt(question: str, documents: list[Document]) -> str:
    """Create the grounded user message sent to Ollama."""
    return f"Context:\n\n{format_context(documents)}\n\nQuestion: {question.strip()}\n\nAnswer with inline citations."


def build_messages(question: str, documents: list[Document]) -> list[dict[str, str]]:
    """Build the Ollama chat messages for a grounded answer."""
    if not question.strip():
        raise ValueError("Question cannot be empty")
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(question, documents)},
    ]
