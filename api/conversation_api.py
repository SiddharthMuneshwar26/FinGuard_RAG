"""FinGuard conversation API (v1): stateful, multi-turn, streaming.

    POST /v1/chat                 one turn, JSON response
    POST /v1/chat/stream          one turn, Server-Sent Events (start, delta*, final)
    GET  /v1/sessions/{id}        conversation history + session metadata
    GET  /health                  liveness
    GET  /ready                   readiness (database, cache, audit sink)
    GET  /metrics                 Prometheus metrics

Headers:
    X-Channel     web | mobile_sdk | internal  (capability profile, config/channels.yaml)
    X-Request-ID  optional; generated if absent and echoed on every response

Run:  uvicorn api.conversation_api:app --port 8081
"""

from __future__ import annotations

import json
import re
import uuid
from contextlib import asynccontextmanager
from typing import Any, Iterator, Literal

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel, Field

from conversation.logging_setup import configure_logging
from conversation.service import AuditUnavailable, CapabilityDenied, ConversationService

API_VERSION = "1.0"
_ID_RE = re.compile(r"^[A-Za-z0-9_\-]{8,64}$")


class ChatRequest(BaseModel):
    session_id: str | None = Field(
        default=None,
        pattern=r"^[A-Za-z0-9_\-]{8,64}$",
        description="Omit to start a new session; reuse the returned id for follow-up turns.",
    )
    message: str = Field(..., min_length=1, max_length=8000)


class Source(BaseModel):
    chunk_id: str
    document_id: str
    title: str
    page: int
    rerank_score: float


class ChatResponse(BaseModel):
    api_version: str = API_VERSION
    request_id: str
    session_id: str
    channel: str
    status: Literal["ok", "blocked", "degraded", "abstained"]
    answer: str
    citations: list[str]
    sources: list[Source]
    turn: int
    cache_hit: bool
    latency_ms: float
    block_reason: str = ""
    model: str = ""


class HistoryItem(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class SessionResponse(BaseModel):
    api_version: str = API_VERSION
    session_id: str
    turns: int
    history: list[HistoryItem]
    meta: dict[str, Any] | None


def create_app(service: ConversationService | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if service is None:
            configure_logging()
        app.state.service = service or ConversationService()
        yield
        app.state.service.close()

    app = FastAPI(title="FinGuard Conversation API", version=API_VERSION, lifespan=lifespan)

    @app.middleware("http")
    async def request_id_middleware(request: Request, call_next):
        incoming = request.headers.get("x-request-id", "")
        request.state.request_id = incoming if _ID_RE.match(incoming) else uuid.uuid4().hex
        response = await call_next(request)
        response.headers["X-Request-ID"] = request.state.request_id
        return response

    def svc(request: Request) -> ConversationService:
        return request.app.state.service

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/ready")
    def ready(request: Request) -> JSONResponse:
        checks = svc(request).readiness()
        ok = all(checks.values())
        return JSONResponse({"ready": ok, "checks": checks}, status_code=200 if ok else 503)

    @app.get("/metrics")
    def metrics(request: Request) -> Response:
        return Response(generate_latest(svc(request).metrics.registry), media_type=CONTENT_TYPE_LATEST)

    @app.post("/v1/chat", response_model=ChatResponse)
    def chat(body: ChatRequest, request: Request, x_channel: str | None = Header(default=None)) -> ChatResponse:
        session_id = body.session_id or uuid.uuid4().hex
        try:
            result = svc(request).run_turn(session_id, body.message, x_channel, request.state.request_id)
        except AuditUnavailable as exc:
            raise HTTPException(503, "Audit log unavailable; request not processed.") from exc
        return ChatResponse(
            request_id=result.request_id,
            session_id=result.session_id,
            channel=result.channel,
            status=result.status,
            answer=result.answer,
            citations=result.citations,
            sources=[Source(**{k: c[k] for k in Source.model_fields}) for c in result.chunks],
            turn=result.turn,
            cache_hit=result.cache_hit,
            latency_ms=result.latency_ms,
            block_reason=result.block_reason,
            model=result.model,
        )

    @app.post("/v1/chat/stream")
    def chat_stream(body: ChatRequest, request: Request, x_channel: str | None = Header(default=None)):
        session_id = body.session_id or uuid.uuid4().hex
        service_ = svc(request)
        try:
            events = service_.stream_turn(session_id, body.message, x_channel, request.state.request_id)
            first = next(events)  # surfaces capability/audit errors before headers are sent
        except CapabilityDenied as exc:
            raise HTTPException(403, str(exc)) from exc
        except AuditUnavailable as exc:
            raise HTTPException(503, "Audit log unavailable; request not processed.") from exc

        def sse() -> Iterator[str]:
            for event in [first, *events]:
                yield f"event: {event['type']}\ndata: {json.dumps(event)}\n\n"

        return StreamingResponse(
            sse(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.get("/v1/sessions/{session_id}", response_model=SessionResponse)
    def get_session(session_id: str, request: Request) -> SessionResponse:
        if not _ID_RE.match(session_id):
            raise HTTPException(422, "invalid session id")
        data = svc(request).session(session_id)
        if data is None:
            raise HTTPException(404, "session not found")
        return SessionResponse(**data)

    return app


app = create_app()
