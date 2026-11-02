"""Integration tests against real PostgreSQL and Redis.

Skipped unless both are configured:

    docker compose up -d postgres redis
    set FINGUARD_TEST_DATABASE_URL=postgresql://finguard:finguard@localhost:5432/finguard
    set FINGUARD_TEST_REDIS_URL=redis://localhost:6379/0
    python -m pytest tests/test_conversation_integration.py -v

CI runs them with Postgres and Redis service containers.
"""

from __future__ import annotations

import json
import os
import uuid

import pytest

pytest.importorskip("langgraph")

DB_URL = os.getenv("FINGUARD_TEST_DATABASE_URL", "")
REDIS_URL = os.getenv("FINGUARD_TEST_REDIS_URL", "")

pytestmark = pytest.mark.skipif(
    not (DB_URL and REDIS_URL), reason="set FINGUARD_TEST_DATABASE_URL and FINGUARD_TEST_REDIS_URL"
)


@pytest.fixture
def make_service():
    from conversation.service import ConversationService
    from conversation.settings import Settings
    from tests.conversation_fakes import FakeRetriever, ScriptedModel

    created = []

    def factory(model=None, retriever=None):
        service = ConversationService(
            Settings(database_url=DB_URL, redis_url=REDIS_URL),
            retriever=retriever or FakeRetriever(),
            model=model or ScriptedModel(),
        )
        created.append(service)
        return service

    yield factory
    for service in created:
        service.close()


def test_session_state_survives_a_service_restart(make_service):
    """The checkpoint lives in PostgreSQL, so a new process continues the conversation."""
    from tests.conversation_fakes import ScriptedModel

    session_id = "it-" + uuid.uuid4().hex[:12]
    first = make_service(ScriptedModel("SHAP attributes predictions [SHARC-p9-c1]."))
    assert first.run_turn(session_id, "What is SHAP?", "mobile_sdk", uuid.uuid4().hex).turn == 1
    first.close()

    model = ScriptedModel("SHARC applies it to capital models [SHARC-p10-c2].")
    restarted = make_service(model)
    result = restarted.run_turn(session_id, "Where is it applied?", "mobile_sdk", uuid.uuid4().hex)
    assert result.turn == 2
    assert [m["role"] for m in model.prompts[-1]] == ["system", "user", "assistant", "user"]
    session = restarted.session(session_id)
    assert session["turns"] == 2 and session["meta"]["turns"] == 2  # meta read back from Redis


def test_retrieval_cache_is_shared_through_redis(make_service):
    from tests.conversation_fakes import FakeRetriever

    retriever = FakeRetriever()
    question = f"What is SHAP? ({uuid.uuid4().hex[:6]})"
    a = make_service(retriever=retriever)
    b = make_service(retriever=retriever)  # a second "replica"
    assert a.run_turn("it-" + uuid.uuid4().hex[:12], question, "web", uuid.uuid4().hex).cache_hit is False
    assert b.run_turn("it-" + uuid.uuid4().hex[:12], question, "web", uuid.uuid4().hex).cache_hit is True
    assert len(retriever.queries) == 1


def test_audit_log_is_masked_and_append_only(make_service):
    import psycopg

    service = make_service()
    request_id = uuid.uuid4().hex
    service.run_turn("it-" + uuid.uuid4().hex[:12], "I'm a.b@x.com, what is SHAP?", "mobile_sdk", request_id)

    events = service.audit.for_request(request_id)
    assert [e.event_type for e in events] == [
        "request_received", "pii_masked", "guardrail_passed", "retrieval", "llm_call", "response_sent",
    ]
    assert "a.b@x.com" not in json.dumps([e.detail for e in events])

    with psycopg.connect(DB_URL, autocommit=True) as conn:
        for statement in (
            "UPDATE audit_log SET event_type = 'tampered' WHERE request_id = %s",
            "DELETE FROM audit_log WHERE request_id = %s",
        ):
            with pytest.raises(psycopg.errors.RaiseException, match="append-only"):
                conn.execute(statement, (request_id,))
        count = conn.execute("SELECT count(*) FROM audit_log WHERE request_id = %s", (request_id,)).fetchone()[0]
    assert count == 6


def test_readiness_reports_all_dependencies(make_service):
    assert make_service().readiness() == {"cache": True, "audit": True, "database": True}
