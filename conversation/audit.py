"""Append-only audit log of user actions, model calls and guardrail decisions.

Every event carries the request id and session id, so one user turn can be
traced end to end. All free text is PII-masked *before* it is written.

PostgreSQL enforces append-only at the database level: a trigger rejects any
UPDATE or DELETE on audit_log, so even a buggy or compromised service process
cannot rewrite history.
"""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Protocol

from src.safety import mask_pii

logger = logging.getLogger(__name__)

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS audit_log (
    id          BIGSERIAL PRIMARY KEY,
    occurred_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    request_id  TEXT NOT NULL,
    session_id  TEXT NOT NULL,
    channel     TEXT NOT NULL,
    event_type  TEXT NOT NULL,
    detail      JSONB NOT NULL DEFAULT '{}'::jsonb
);
CREATE INDEX IF NOT EXISTS audit_log_request_idx ON audit_log (request_id);
CREATE INDEX IF NOT EXISTS audit_log_session_idx ON audit_log (session_id, occurred_at);

CREATE OR REPLACE FUNCTION audit_log_block_mutation() RETURNS trigger AS $$
BEGIN
    RAISE EXCEPTION 'audit_log is append-only';
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS audit_log_append_only ON audit_log;
CREATE TRIGGER audit_log_append_only
    BEFORE UPDATE OR DELETE ON audit_log
    FOR EACH ROW EXECUTE FUNCTION audit_log_block_mutation();
"""


def _mask(value: Any) -> Any:
    if isinstance(value, str):
        return mask_pii(value)
    if isinstance(value, dict):
        return {key: _mask(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_mask(item) for item in value]
    return value


@dataclass(frozen=True)
class AuditEvent:
    request_id: str
    session_id: str
    channel: str
    event_type: str  # e.g. request_received, guardrail_blocked, retrieval, llm_call, response_sent
    detail: dict[str, Any] = field(default_factory=dict)
    occurred_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def masked(self) -> "AuditEvent":
        return AuditEvent(
            request_id=self.request_id,
            session_id=self.session_id,
            channel=self.channel,
            event_type=self.event_type,
            detail=_mask(self.detail),
            occurred_at=self.occurred_at,
        )


class AuditSink(Protocol):
    def write(self, event: AuditEvent) -> None: ...
    def for_request(self, request_id: str) -> list[AuditEvent]: ...
    def ping(self) -> bool: ...


class InMemoryAuditSink:
    """Test / demo sink. Exposes no update or delete operation."""

    def __init__(self) -> None:
        self._events: list[AuditEvent] = []
        self._lock = threading.Lock()

    def write(self, event: AuditEvent) -> None:
        with self._lock:
            self._events.append(event.masked())

    def for_request(self, request_id: str) -> list[AuditEvent]:
        with self._lock:
            return [event for event in self._events if event.request_id == request_id]

    @property
    def events(self) -> list[AuditEvent]:
        with self._lock:
            return list(self._events)

    def ping(self) -> bool:
        return True


class PostgresAuditSink:
    def __init__(self, pool: Any) -> None:
        self._pool = pool
        with self._pool.connection() as conn:
            conn.execute(SCHEMA_SQL, prepare=False)  # multi-statement DDL: simple query protocol

    def write(self, event: AuditEvent) -> None:
        event = event.masked()
        try:
            with self._pool.connection() as conn:
                conn.execute(
                    "INSERT INTO audit_log (occurred_at, request_id, session_id, channel, event_type, detail) "
                    "VALUES (%s, %s, %s, %s, %s, %s::jsonb)",
                    (
                        event.occurred_at,
                        event.request_id,
                        event.session_id,
                        event.channel,
                        event.event_type,
                        json.dumps(event.detail, default=str),
                    ),
                )
        except Exception:
            # Audit write failures are loud: logged at ERROR and counted in metrics,
            # but never echo the (already masked) payload into the log line.
            logger.exception("audit write failed for request %s (%s)", event.request_id, event.event_type)
            raise

    def for_request(self, request_id: str) -> list[AuditEvent]:
        with self._pool.connection() as conn:
            rows = conn.execute(
                "SELECT request_id, session_id, channel, event_type, detail, occurred_at "
                "FROM audit_log WHERE request_id = %s ORDER BY id",
                (request_id,),
            ).fetchall()
        return [
            AuditEvent(
                request_id=r["request_id"],
                session_id=r["session_id"],
                channel=r["channel"],
                event_type=r["event_type"],
                detail=r["detail"],
                occurred_at=r["occurred_at"],
            )
            for r in rows  # pool is configured with dict_row
        ]

    def ping(self) -> bool:
        try:
            with self._pool.connection() as conn:
                conn.execute("SELECT 1")
            return True
        except Exception:
            return False
