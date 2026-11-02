"""ConversationService: wires the graph to its checkpointer, cache, audit log and metrics.

Storage is chosen from settings:
  FINGUARD_DATABASE_URL set  -> PostgreSQL checkpointer (LangGraph PostgresSaver)
                                + PostgreSQL append-only audit_log
  FINGUARD_REDIS_URL set     -> Redis session metadata + retrieval cache
Unset -> in-process equivalents (tests / single-process demo only).

Compliance rule: if the audit log cannot be written, the turn fails ("no audit,
no answer") and the API returns 503.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterator

from prometheus_client import CollectorRegistry, Counter, Histogram

from .audit import AuditEvent, AuditSink, InMemoryAuditSink, PostgresAuditSink
from .cache import Cache, build_cache, session_key
from .capabilities import Capabilities, CapabilityRegistry
from .graph import GraphDeps, build_graph, turn_input
from .llm import ChatModel, LiteLLMChatModel
from .retrieval import FaissRetriever, Retriever
from .settings import Settings

logger = logging.getLogger("finguard.conversation")


class AuditUnavailable(RuntimeError):
    pass


class CapabilityDenied(PermissionError):
    pass


@dataclass
class TurnResult:
    request_id: str
    session_id: str
    channel: str
    status: str
    answer: str
    citations: list[str]
    chunks: list[dict[str, Any]]
    turn: int
    cache_hit: bool
    latency_ms: float
    block_reason: str = ""
    model: str = ""


@dataclass
class Metrics:
    registry: CollectorRegistry = field(default_factory=CollectorRegistry)

    def __post_init__(self) -> None:
        r = self.registry
        self.turns = Counter("finguard_turns_total", "Conversation turns by outcome", ["channel", "status"], registry=r)
        self.latency = Histogram(
            "finguard_turn_latency_seconds",
            "End-to-end turn latency",
            ["channel"],
            buckets=(0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 30, 60),
            registry=r,
        )
        self.blocks = Counter("finguard_guardrail_blocks_total", "Blocked turns", ["reason"], registry=r)
        self.cache = Counter("finguard_retrieval_cache_total", "Retrieval cache lookups", ["result"], registry=r)
        self.audit_failures = Counter("finguard_audit_write_failures_total", "Failed audit writes", registry=r)


class ConversationService:
    def __init__(
        self,
        settings: Settings | None = None,
        *,
        retriever: Retriever | None = None,
        model: ChatModel | None = None,
        cache: Cache | None = None,
        audit: AuditSink | None = None,
        checkpointer: Any = None,
        capabilities: CapabilityRegistry | None = None,
    ) -> None:
        self.settings = settings or Settings()
        s = self.settings
        self._pool = None

        if (checkpointer is None or audit is None) and s.database_url:
            from langgraph.checkpoint.postgres import PostgresSaver
            from psycopg.rows import dict_row
            from psycopg_pool import ConnectionPool

            self._pool = ConnectionPool(
                s.database_url,
                min_size=1,
                max_size=10,
                kwargs={"autocommit": True, "prepare_threshold": 0, "row_factory": dict_row},
                open=True,
            )
            try:
                if checkpointer is None:
                    checkpointer = PostgresSaver(self._pool)
                    checkpointer.setup()
                if audit is None:
                    audit = PostgresAuditSink(self._pool)
            except Exception:
                self._pool.close()
                raise

        if checkpointer is None:
            from langgraph.checkpoint.memory import InMemorySaver

            checkpointer = InMemorySaver()

        self.audit = audit or InMemoryAuditSink()
        self.cache = cache or build_cache(s.redis_url)
        self.capabilities = capabilities or CapabilityRegistry.from_file(s.channels_file)
        self.model = model or LiteLLMChatModel(
            s.model, s.fallback_models, timeout_s=s.llm_timeout_s, retries=s.llm_retries, max_tokens=s.llm_max_tokens
        )
        self.metrics = Metrics()
        deps = GraphDeps(
            retriever=retriever or FaissRetriever(),
            model=self.model,
            cache=self.cache,
            audit=_CountingAudit(self.audit, self.metrics),
            retrieval_cache_ttl_s=s.retrieval_cache_ttl_s,
        )
        self.graph = build_graph(deps, checkpointer)

    # ------------------------------------------------------------------ turns
    def _prepare(self, session_id: str, question: str, channel: str | None, request_id: str, streaming: bool):
        caps = self.capabilities.resolve(channel)
        if streaming and not caps.streaming:
            raise CapabilityDenied(f"streaming is disabled for channel {caps.channel!r}")
        self._audit(
            AuditEvent(
                request_id=request_id,
                session_id=session_id,
                channel=caps.channel,
                event_type="request_received",
                detail={"question": question, "streaming": streaming, "capabilities": caps.as_dict()},
            )
        )
        payload = turn_input(
            request_id=request_id, session_id=session_id, channel=caps.channel, caps=caps.as_dict(), question=question
        )
        config = {"configurable": {"thread_id": session_id}}
        return caps, payload, config

    def run_turn(self, session_id: str, question: str, channel: str | None, request_id: str) -> TurnResult:
        started = time.perf_counter()
        caps, payload, config = self._prepare(session_id, question, channel, request_id, streaming=False)
        state = self._invoke(payload, config)
        return self._complete(state, caps, started)

    def stream_turn(
        self, session_id: str, question: str, channel: str | None, request_id: str
    ) -> Iterator[dict[str, Any]]:
        """Yield {"type": "delta"} events while the model writes, then one "final" event.

        Deltas are provisional: the final event carries the grounded answer (any
        fabricated citation stripped), which the client should render as the
        source of truth.
        """
        started = time.perf_counter()
        caps, payload, config = self._prepare(session_id, question, channel, request_id, streaming=True)
        yield {"type": "start", "request_id": request_id, "session_id": session_id, "channel": caps.channel}
        state: dict[str, Any] = {}
        try:
            for mode, chunk in self.graph.stream(payload, config, stream_mode=["custom", "values"]):
                if mode == "custom":
                    yield chunk
                else:
                    state = chunk
        except AuditUnavailable:
            yield {"type": "error", "detail": "audit log unavailable"}
            return
        result = self._complete(state, caps, started)
        yield {"type": "final", **_result_payload(result)}

    def _invoke(self, payload: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
        return self.graph.invoke(payload, config)

    def _complete(self, state: dict[str, Any], caps: Capabilities, started: float) -> TurnResult:
        latency = time.perf_counter() - started
        result = TurnResult(
            request_id=state["request_id"],
            session_id=state["session_id"],
            channel=caps.channel,
            status=state["status"],
            answer=state["answer"],
            citations=state.get("citations", []),
            chunks=state.get("chunks", []),
            turn=state.get("turns", 0),
            cache_hit=state.get("cache_hit", False),
            latency_ms=round(latency * 1000, 2),
            block_reason=state.get("block_reason", ""),
            model=state.get("model", ""),
        )
        self._audit(
            AuditEvent(
                request_id=result.request_id,
                session_id=result.session_id,
                channel=result.channel,
                event_type="response_sent",
                detail={
                    "status": result.status,
                    "citations": result.citations,
                    "answer": result.answer,
                    "latency_ms": result.latency_ms,
                },
            )
        )
        self.metrics.turns.labels(caps.channel, result.status).inc()
        self.metrics.latency.labels(caps.channel).observe(latency)
        if result.status == "blocked":
            self.metrics.blocks.labels(result.block_reason).inc()
        elif result.chunks or result.cache_hit:
            self.metrics.cache.labels("hit" if result.cache_hit else "miss").inc()
        self.cache.set_json(
            session_key(result.session_id),
            {
                "turns": result.turn,
                "channel": result.channel,
                "last_status": result.status,
                "last_seen": datetime.now(timezone.utc).isoformat(),
            },
            self.settings.session_ttl_s,
        )
        logger.info(
            "turn complete",
            extra={
                "request_id": result.request_id,
                "session_id": result.session_id,
                "channel": result.channel,
                "status": result.status,
                "latency_ms": result.latency_ms,
                "cache_hit": result.cache_hit,
            },
        )
        return result

    def _audit(self, event: AuditEvent) -> None:
        _CountingAudit(self.audit, self.metrics).write(event)

    # --------------------------------------------------------------- sessions
    def session(self, session_id: str) -> dict[str, Any] | None:
        snapshot = self.graph.get_state({"configurable": {"thread_id": session_id}})
        values = snapshot.values if snapshot else {}
        if not values:
            return None
        return {
            "session_id": session_id,
            "turns": values.get("turns", 0),
            "history": values.get("history", []),
            "meta": self.cache.get_json(session_key(session_id)),
        }

    def readiness(self) -> dict[str, bool]:
        checks = {"cache": self.cache.ping(), "audit": self.audit.ping()}
        if self._pool is not None:
            try:
                with self._pool.connection() as conn:
                    conn.execute("SELECT 1")
                checks["database"] = True
            except Exception:
                checks["database"] = False
        return checks

    def close(self) -> None:
        if self._pool is not None:
            self._pool.close()


class _CountingAudit:
    """Wraps a sink: counts failures and converts them into AuditUnavailable."""

    def __init__(self, sink: AuditSink, metrics: Metrics) -> None:
        self._sink = sink
        self._metrics = metrics

    def write(self, event: AuditEvent) -> None:
        try:
            self._sink.write(event)
        except Exception as exc:
            self._metrics.audit_failures.inc()
            raise AuditUnavailable(str(exc)) from exc

    def for_request(self, request_id: str) -> list[AuditEvent]:
        return self._sink.for_request(request_id)

    def ping(self) -> bool:
        return self._sink.ping()


def _result_payload(result: TurnResult) -> dict[str, Any]:
    return {
        "request_id": result.request_id,
        "session_id": result.session_id,
        "channel": result.channel,
        "status": result.status,
        "answer": result.answer,
        "citations": result.citations,
        "turn": result.turn,
        "cache_hit": result.cache_hit,
        "latency_ms": result.latency_ms,
        "block_reason": result.block_reason,
        "model": result.model,
        "sources": [
            {k: chunk[k] for k in ("chunk_id", "document_id", "title", "page", "rerank_score")}
            for chunk in result.chunks
        ],
    }
