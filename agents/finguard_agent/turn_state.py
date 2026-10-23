"""Per-turn bookkeeping shared by the tools and the guardrail callbacks.

The record is tagged with the ADK invocation id: a new user turn is a new
invocation, so counters reset automatically without needing an extra callback.
Each write assigns a fresh dict so ADK records it in the state delta.
"""

from __future__ import annotations

from typing import Any

from .settings import STATE_TURN


def _empty(invocation_id: str) -> dict[str, Any]:
    return {"invocation_id": invocation_id, "searches": 0, "citations": [], "pages": []}


def get_turn(context: Any) -> dict[str, Any]:
    """Return this invocation's record (a copy), creating it if needed."""
    record = context.state.get(STATE_TURN)
    invocation_id = str(getattr(context, "invocation_id", "") or "")
    if not isinstance(record, dict) or record.get("invocation_id") != invocation_id:
        return _empty(invocation_id)
    return {
        "invocation_id": invocation_id,
        "searches": int(record.get("searches", 0)),
        "citations": list(record.get("citations", [])),
        "pages": list(record.get("pages", [])),
    }


def save_turn(context: Any, record: dict[str, Any]) -> None:
    context.state[STATE_TURN] = dict(record)


def increment_searches(context: Any) -> int:
    record = get_turn(context)
    record["searches"] += 1
    save_turn(context, record)
    return record["searches"]


def register_citations(context: Any, citation_ids: list[str], pages: list[str] | None = None) -> None:
    """Record citation ids (and 'DOC:page' keys) that a tool actually returned."""
    record = get_turn(context)
    for citation in citation_ids:
        if citation and citation not in record["citations"]:
            record["citations"].append(citation)
    for page in pages or []:
        if page and page not in record["pages"]:
            record["pages"].append(page)
    save_turn(context, record)
