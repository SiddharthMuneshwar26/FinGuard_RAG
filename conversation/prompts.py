"""Structured system instructions and message assembly for the conversation graph."""

from __future__ import annotations

from typing import Any

from src.config import ABSTENTION_MESSAGE

SYSTEM_INSTRUCTIONS = f"""You are FinGuard, an assistant for financial AI governance and
risk-model interpretability, serving customers of a regulated financial institution.

## Scope
- Answer only from the CONTEXT block in the latest user message.
- Earlier conversation turns are for resolving references ("it", "that framework");
  they are not evidence.

## Citations
- Cite every factual sentence with the chunk id(s) it relies on, in square
  brackets, copied exactly, e.g. [SHARC-p9-c1].
- Never invent a chunk id. Uncited sentences may be removed.

## Abstention
- If the context does not answer the question, reply exactly:
  {ABSTENTION_MESSAGE}

## Style
- Concise: at most 6 sentences. Use the documents' terminology.
- Never ask for or repeat personal data. Placeholders such as [EMAIL] or [PAN]
  mean data was masked; do not try to reconstruct it."""


def format_context(chunks: list[dict[str, Any]]) -> str:
    return "\n\n".join(
        f"[{chunk['chunk_id']}] {chunk['title']} (page {chunk['page']})\n{chunk['content']}" for chunk in chunks
    )


def build_messages(
    question: str,
    chunks: list[dict[str, Any]],
    history: list[dict[str, str]],
    history_turns: int,
) -> list[dict[str, str]]:
    """System instructions + the last `history_turns` exchanges + grounded question."""
    messages = [{"role": "system", "content": SYSTEM_INSTRUCTIONS}]
    if history_turns > 0:
        for item in history[-2 * history_turns :]:
            messages.append({"role": item["role"], "content": item["content"]})
    messages.append(
        {
            "role": "user",
            "content": f"CONTEXT:\n\n{format_context(chunks)}\n\nQUESTION: {question}\n\nAnswer with inline citations.",
        }
    )
    return messages


def retrieval_only_answer(chunks: list[dict[str, Any]], reason: str) -> str:
    """Degraded answer used when generation is disabled or every model failed."""
    lines = [reason, "", "Most relevant passages:"]
    for chunk in chunks:
        snippet = " ".join(chunk["content"].split())[:220]
        lines.append(f"- [{chunk['chunk_id']}] {chunk['title']}, p. {chunk['page']}: {snippet}…")
    return "\n".join(lines)
