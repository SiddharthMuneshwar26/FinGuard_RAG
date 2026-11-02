"""LangGraph conversation graph for multi-step, stateful FinGuard sessions.

    guard_input ──blocked──────────────────────────────┐
        │ ok                                           │
    contextualize → retrieve → generate ──degraded/────┤
                                   │      abstained    │
                                   ▼                   ▼
                                 ground ─────────► finalize → END

State is checkpointed per session (thread_id = session_id): PostgreSQL in
production (PostgresSaver), in memory for tests. `history` and `turns` carry
across turns; every other field is reset at the start of each turn by the
service, so a blocked turn can never leak the previous turn's chunks or answer.

Blocked turns are audited but NOT added to `history`, so an injection attempt
is never replayed into later prompts as conversation context.
"""

from __future__ import annotations

import operator
import re
import time
from dataclasses import dataclass
from typing import Annotated, Any, TypedDict

from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph

from src.config import ABSTENTION_MESSAGE
from src.grounding import strip_ungrounded_citations
from src.safety import detect_pii, detect_prompt_injection, mask_pii

from .audit import AuditEvent, AuditSink
from .cache import Cache, retrieval_key
from .llm import ChatModel, LLMUnavailable, strip_think
from .prompts import build_messages, retrieval_only_answer
from .retrieval import Retriever

INJECTION_MESSAGE = (
    "I can't process that request. FinGuard only answers questions about the indexed governance documents."
)
PII_BLOCKED_MESSAGE = (
    "Your message appears to contain personal data ({kinds}). Please remove it and ask again. "
    "FinGuard does not process personal information on this channel."
)
TOO_LONG_MESSAGE = "Your message is too long ({length} characters; limit {limit}). Please ask a shorter question."
LLM_DOWN_REASON = "The answer service is temporarily unavailable, so here is the retrieved evidence instead."
NO_GENERATION_REASON = "Generation is disabled for this channel; returning retrieved evidence."

_FOLLOW_UP_RE = re.compile(
    r"\b(it|its|this|that|these|those|they|them|their|the same|above|previous|what about|how about|and)\b",
    re.IGNORECASE,
)


class ConversationState(TypedDict, total=False):
    # Persisted across turns.
    history: Annotated[list[dict[str, str]], operator.add]
    turns: int
    # Per turn (reset by the service before each invoke).
    request_id: str
    session_id: str
    channel: str
    caps: dict[str, Any]
    question: str
    search_query: str
    status: str  # ok | blocked | degraded | abstained
    block_reason: str
    chunks: list[dict[str, Any]]
    answer: str
    citations: list[str]
    cache_hit: bool
    model: str


@dataclass
class GraphDeps:
    retriever: Retriever
    model: ChatModel
    cache: Cache
    audit: AuditSink
    retrieval_cache_ttl_s: int = 3600


def is_follow_up(question: str) -> bool:
    return len(question.split()) <= 6 or bool(_FOLLOW_UP_RE.search(question))


def build_graph(deps: GraphDeps, checkpointer: Any) -> Any:
    def audit(state: ConversationState, event_type: str, **detail: Any) -> None:
        deps.audit.write(
            AuditEvent(
                request_id=state["request_id"],
                session_id=state["session_id"],
                channel=state["channel"],
                event_type=event_type,
                detail=detail,
            )
        )

    # ---------------------------------------------------------------- nodes
    def guard_input(state: ConversationState) -> dict[str, Any]:
        caps = state["caps"]
        question = state["question"].strip()

        def block(reason: str, message: str, **detail: Any) -> dict[str, Any]:
            audit(state, "guardrail_blocked", reason=reason, **detail)
            return {"status": "blocked", "block_reason": reason, "answer": message, "question": mask_pii(question)}

        if len(question) > caps["max_input_chars"]:
            return block(
                "too_long",
                TOO_LONG_MESSAGE.format(length=len(question), limit=caps["max_input_chars"]),
                length=len(question),
            )
        if detect_prompt_injection(question):
            return block("prompt_injection", INJECTION_MESSAGE)

        kinds = detect_pii(question)
        if kinds:
            if caps["pii_policy"] == "block":
                return block("pii", PII_BLOCKED_MESSAGE.format(kinds=", ".join(kinds)), pii_kinds=kinds)
            question = mask_pii(question)
            audit(state, "pii_masked", pii_kinds=kinds)

        audit(state, "guardrail_passed")
        return {"question": question, "status": "ok"}

    def contextualize(state: ConversationState) -> dict[str, Any]:
        """Resolve follow-ups ("what about its limitations?") against the last user turn."""
        question = state["question"]
        previous = [item["content"] for item in state.get("history", []) if item["role"] == "user"]
        if previous and is_follow_up(question):
            return {"search_query": f"{previous[-1]} {question}"}
        return {"search_query": question}

    def retrieve(state: ConversationState) -> dict[str, Any]:
        query = state["search_query"]
        use_cache = state["caps"]["retrieval_cache"]
        started = time.perf_counter()
        chunks = deps.cache.get_json(retrieval_key(query)) if use_cache else None
        cache_hit = chunks is not None
        if not cache_hit:
            chunks = deps.retriever(query)
            if use_cache:
                deps.cache.set_json(retrieval_key(query), chunks, deps.retrieval_cache_ttl_s)
        audit(
            state,
            "retrieval",
            query=query,
            cache_hit=cache_hit,
            chunk_ids=[chunk["chunk_id"] for chunk in chunks],
            latency_ms=round((time.perf_counter() - started) * 1000, 2),
        )
        return {"chunks": chunks, "cache_hit": cache_hit}

    def generate(state: ConversationState) -> dict[str, Any]:
        chunks = state.get("chunks", [])
        caps = state["caps"]
        if not chunks:
            return {"status": "abstained", "answer": ABSTENTION_MESSAGE}
        if not caps["generation"]:
            return {"status": "degraded", "answer": retrieval_only_answer(chunks, NO_GENERATION_REASON)}

        messages = build_messages(state["question"], chunks, state.get("history", []), caps["history_turns"])
        write = get_stream_writer()
        started = time.perf_counter()
        parts: list[str] = []
        try:
            for delta in deps.model.stream(messages):
                parts.append(delta)
                write({"type": "delta", "text": delta})
        except LLMUnavailable as exc:
            audit(state, "llm_call", model=deps.model.name, ok=False, error=str(exc)[:300])
            return {"status": "degraded", "answer": retrieval_only_answer(chunks, LLM_DOWN_REASON)}

        audit(
            state,
            "llm_call",
            model=deps.model.name,
            ok=True,
            latency_ms=round((time.perf_counter() - started) * 1000, 2),
            history_turns_sent=min(caps["history_turns"], len(state.get("history", [])) // 2),
        )
        return {"answer": strip_think("".join(parts)), "model": deps.model.name}

    def ground(state: ConversationState) -> dict[str, Any]:
        answer = state["answer"]
        if answer.strip() == ABSTENTION_MESSAGE:
            return {"status": "abstained"}
        chunk_ids = {chunk["chunk_id"] for chunk in state["chunks"]}
        pages = {f"{chunk['document_id']}:{chunk['page']}" for chunk in state["chunks"]}
        cleaned, valid = strip_ungrounded_citations(answer, chunk_ids, pages)
        cleaned = re.sub(r"[ \t]*" + re.escape(ABSTENTION_MESSAGE), "", cleaned).strip()
        if valid == 0 or not cleaned:
            audit(state, "grounding", outcome="abstained", removed_all=True)
            return {"status": "abstained", "answer": ABSTENTION_MESSAGE, "citations": []}
        cited = [c for c in dict.fromkeys(re.findall(r"[A-Za-z0-9_]+-p\d+-c\d+", cleaned)) if c in chunk_ids]
        if cleaned != answer.strip():
            audit(state, "grounding", outcome="citations_stripped")
        return {"answer": cleaned, "citations": cited}

    def finalize(state: ConversationState) -> dict[str, Any]:
        if state["status"] == "blocked":
            return {}  # never replay a blocked message as context
        return {
            "history": [
                {"role": "user", "content": state["question"]},
                {"role": "assistant", "content": state["answer"]},
            ],
            "turns": state.get("turns", 0) + 1,
        }

    # ---------------------------------------------------------------- edges
    def after_guard(state: ConversationState) -> str:
        return "finalize" if state["status"] == "blocked" else "contextualize"

    def after_generate(state: ConversationState) -> str:
        return "finalize" if state.get("status") in {"degraded", "abstained"} else "ground"

    graph = StateGraph(ConversationState)
    graph.add_node("guard_input", guard_input)
    graph.add_node("contextualize", contextualize)
    graph.add_node("retrieve", retrieve)
    graph.add_node("generate", generate)
    graph.add_node("ground", ground)
    graph.add_node("finalize", finalize)
    graph.add_edge(START, "guard_input")
    graph.add_conditional_edges("guard_input", after_guard, ["contextualize", "finalize"])
    graph.add_edge("contextualize", "retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_conditional_edges("generate", after_generate, ["ground", "finalize"])
    graph.add_edge("ground", "finalize")
    graph.add_edge("finalize", END)
    return graph.compile(checkpointer=checkpointer)


def turn_input(
    *, request_id: str, session_id: str, channel: str, caps: dict[str, Any], question: str
) -> dict[str, Any]:
    """Fresh per-turn fields. `history` and `turns` are left to the checkpoint."""
    return {
        "request_id": request_id,
        "session_id": session_id,
        "channel": channel,
        "caps": caps,
        "question": question,
        "search_query": "",
        "status": "ok",
        "block_reason": "",
        "chunks": [],
        "answer": "",
        "citations": [],
        "cache_hit": False,
        "model": "",
    }
