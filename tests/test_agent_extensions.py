"""Offline tests for the Phase 2 agent add-ons: SQL tool, Google Drive tool,
multimodal handling and the tracing flag. Google APIs are fully mocked."""

from __future__ import annotations

import json
import sqlite3

import pytest

pytest.importorskip("google.adk")

from google.adk.models.llm_request import LlmRequest  # noqa: E402
from google.genai import types  # noqa: E402

from agents.finguard_agent import retrieval_tool, sql_tool, workspace_tool  # noqa: E402
from agents.finguard_agent.agent import build_agent  # noqa: E402
from agents.finguard_agent.guardrails import tool_guardrail  # noqa: E402
from agents.finguard_agent.multimodal import TEXT_ONLY_NOTE, build_user_content, make_attachment_callback  # noqa: E402
from agents.finguard_agent.tracing import maybe_enable_cloud_tracing  # noqa: E402
from tests.agent_fakes import FakeContext, FakeRetriever, FakeTool, call, make_llm, run, text  # noqa: E402


# --------------------------------------------------------------------------- SQL
@pytest.fixture
def sample_db(tmp_path, monkeypatch):
    path = tmp_path / "internal.db"
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE freight_rates (lane TEXT, mode TEXT, rate_usd REAL)")
        connection.executemany(
            "INSERT INTO freight_rates VALUES (?, ?, ?)",
            [("SHA-RTM", "ocean", 1800.0), ("SHA-LAX", "ocean", 2100.0), ("FRA-ORD", "air", 4.5)],
        )
    monkeypatch.setenv("FINGUARD_SQL_DB", str(path))
    return path


def test_sql_validator_rejects_anything_but_select():
    for bad in [
        "DELETE FROM freight_rates",
        "DROP TABLE freight_rates",
        "SELECT 1; DROP TABLE freight_rates",
        "PRAGMA table_info(freight_rates)",
        "ATTACH DATABASE 'x.db' AS x",
        "WITH x AS (SELECT 1) DELETE FROM freight_rates",
        "INSERT INTO freight_rates VALUES ('a','b',1)",
    ]:
        assert sql_tool.validate_select(bad), bad
    assert sql_tool.validate_select("SELECT * FROM freight_rates WHERE mode = ? -- delete later") is None
    assert sql_tool.validate_select("SELECT 'drop table' AS label") is None  # keyword inside a string literal

    guard = tool_guardrail(FakeTool("query_internal_database"), {"sql": "UPDATE freight_rates SET rate_usd = 0"}, FakeContext())
    assert guard["status"] == "error"


def test_sql_tool_parameterised_read_only(sample_db):
    ctx = FakeContext()
    result = sql_tool.query_internal_database(
        "SELECT lane, rate_usd FROM freight_rates WHERE mode = ? ORDER BY rate_usd DESC", ctx, parameters=["ocean"]
    )
    assert result["status"] == "success"
    assert result["rows"] == [["SHA-LAX", 2100.0], ["SHA-RTM", 1800.0]]
    assert f"sql:{result['query_id']}" in ctx.state["finguard_turn"]["citations"]

    # Injection through a parameter stays a plain value.
    assert sql_tool.query_internal_database(
        "SELECT * FROM freight_rates WHERE mode = ?", ctx, parameters=["ocean' OR '1'='1"]
    )["rows"] == []

    # Even if the validator were bypassed, the connection itself refuses writes.
    connection = sql_tool._connect_read_only(sample_db)
    with pytest.raises(sqlite3.DatabaseError):
        connection.execute("DELETE FROM freight_rates")
    connection.close()

    schema = sql_tool.describe_internal_database(ctx)
    assert schema["tables"] == {"freight_rates": ["lane", "mode", "rate_usd"]}


# ------------------------------------------------------------------------ Drive
class _FakeRequest:
    def __init__(self, value):
        self.value = value

    def execute(self):
        return self.value


class _FakeFiles:
    def __init__(self):
        self.list_kwargs = None

    def list(self, **kwargs):
        self.list_kwargs = kwargs
        return _FakeRequest({"files": [
            {"id": "doc123", "name": "Model risk policy", "mimeType": "application/vnd.google-apps.document",
             "webViewLink": "https://docs.google.com/document/d/doc123", "modifiedTime": "2026-10-01T10:00:00Z"},
            {"id": "pdf456", "name": "board pack.pdf", "mimeType": "application/pdf"},
        ]})

    def export(self, fileId, mimeType):
        assert mimeType == "text/plain"
        return _FakeRequest(b"All tier-1 models need annual validation.")


class _FakeDrive:
    def __init__(self):
        self._files = _FakeFiles()

    def files(self):
        return self._files


@pytest.fixture
def fake_drive():
    drive = _FakeDrive()
    workspace_tool.set_service_factory(lambda: drive)
    yield drive
    workspace_tool.set_service_factory(None)


def test_drive_tool_read_only_search(fake_drive):
    ctx = FakeContext()
    result = workspace_tool.search_google_drive("Sid's model policy", ctx, max_results=10)

    assert workspace_tool.SCOPES == ["https://www.googleapis.com/auth/drive.readonly"]
    kwargs = fake_drive.files().list_kwargs
    assert kwargs["q"] == "fullText contains 'Sid\\'s model policy' and trashed = false"
    assert kwargs["pageSize"] == 5  # capped
    assert result["results"][0]["snippet"] == "All tier-1 models need annual validation."
    assert "snippet" not in result["results"][1]
    assert ctx.state["finguard_turn"]["citations"] == ["drive:doc123", "drive:pdf456"]


def test_drive_credentials_never_hard_coded(monkeypatch, tmp_path):
    for name in ("FINGUARD_GOOGLE_OAUTH_SECRET", "FINGUARD_GOOGLE_OAUTH_TOKEN_JSON", "FINGUARD_GOOGLE_OAUTH_TOKEN_FILE"):
        monkeypatch.delenv(name, raising=False)
    result = workspace_tool.search_google_drive("policy", FakeContext())
    assert result["status"] == "error" and "not configured" in result["error_message"]

    token = {"token": "t", "refresh_token": "r", "client_id": "c", "client_secret": "s",
             "scopes": ["https://www.googleapis.com/auth/drive"]}  # broader than allowed
    monkeypatch.setenv("FINGUARD_GOOGLE_OAUTH_TOKEN_JSON", json.dumps(token))
    with pytest.raises(RuntimeError, match="extra scopes"):
        workspace_tool.load_credentials()


def test_drive_searches_count_toward_the_search_cap(fake_drive):
    retriever = FakeRetriever()
    retrieval_tool.set_retriever(retriever)
    try:
        llm = make_llm(
            call("search_governance_documents", query="tiers"),
            call("search_google_drive", query="policy"),
            call("search_governance_documents", query="tiers again"),
            call("search_google_drive", query="policy again"),
            text("Tier-1 models need annual validation [drive:doc123]."),
        )
        agent = build_agent(model=llm, enable_drive=True, enable_sql=False)
        (turn,) = run(agent, "What does my policy say about tiers?")
    finally:
        retrieval_tool.set_retriever(None)

    assert [r["status"] for r in turn.tool_responses] == ["success", "success", "success", "error"]
    assert turn.final_text == "Tier-1 models need annual validation [drive:doc123]."


# ------------------------------------------------------------------- Multimodal
def _request_with(*parts):
    return LlmRequest(contents=[types.Content(role="user", parts=list(parts))])


def test_multimodal_text_only_fallback_and_gemini_passthrough(tmp_path):
    image = types.Part.from_bytes(data=b"\x89PNG fake", mime_type="image/png")

    request = _request_with(image, types.Part(text="Explain this chart"))
    make_attachment_callback(model_supports_media=False)(FakeContext(), request)
    assert request.contents[0].parts[0].text == TEXT_ONLY_NOTE.format(mime="image/png")

    request = _request_with(image, types.Part(text="Explain this chart"))
    make_attachment_callback(model_supports_media=True)(FakeContext(), request)
    assert request.contents[0].parts[0].inline_data.mime_type == "image/png"

    request = _request_with(types.Part.from_bytes(data=b"MZ", mime_type="application/x-msdownload"))
    make_attachment_callback(model_supports_media=True)(FakeContext(), request)
    assert "unsupported type" in request.contents[0].parts[0].text

    audio = tmp_path / "question.wav"
    audio.write_bytes(b"RIFF fake")
    content = build_user_content(audio_path=audio)
    assert content.parts[0].inline_data.mime_type in {"audio/wav", "audio/x-wav"}
    assert content.parts[1].text == "Answer the question asked in the audio."


def test_tracing_is_off_unless_flag_set():
    assert maybe_enable_cloud_tracing() is False
