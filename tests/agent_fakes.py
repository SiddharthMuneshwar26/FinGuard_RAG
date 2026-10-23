"""Shared fakes for the offline ADK agent tests (no API keys, no model downloads)."""

from __future__ import annotations

import asyncio
import itertools
from typing import Any, AsyncGenerator

from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.runners import InMemoryRunner
from google.genai import types

_ids = itertools.count(1)


def call(name: str, **args: Any) -> types.Part:
    """A scripted model turn that calls a tool."""
    return types.Part(function_call=types.FunctionCall(id=f"call-{next(_ids)}", name=name, args=args))


def text(value: str) -> types.Part:
    return types.Part(text=value)


class ScriptedLlm(BaseLlm):
    """Replays a fixed list of model turns and records every request it receives."""

    script: list[Any] = []
    requests: list[LlmRequest] = []

    @classmethod
    def supported_models(cls) -> list[str]:
        return [r"scripted-.*"]

    async def generate_content_async(
        self, llm_request: LlmRequest, stream: bool = False
    ) -> AsyncGenerator[LlmResponse, None]:
        self.requests.append(llm_request)
        if not self.script:
            raise AssertionError("ScriptedLlm ran out of scripted turns")
        turn = self.script.pop(0)
        parts = turn if isinstance(turn, list) else [turn]
        yield LlmResponse(content=types.Content(role="model", parts=parts))


def make_llm(*turns: Any) -> ScriptedLlm:
    return ScriptedLlm(model="scripted-llm", script=list(turns), requests=[])


class FakeRetriever:
    """Stands in for FAISS + CrossEncoder; returns canned chunks and counts calls."""

    def __init__(self, chunks: list[dict[str, Any]] | None = None) -> None:
        self.chunks = chunks if chunks is not None else [chunk("SHARC", 9, 1), chunk("GAICF", 12, 2)]
        self.queries: list[str] = []

    def __call__(self, query: str) -> list[dict[str, Any]]:
        self.queries.append(query)
        return [dict(c) for c in self.chunks]


def chunk(doc: str, page: int, number: int, content: str = "Evidence text.") -> dict[str, Any]:
    return {
        "chunk_id": f"{doc}-p{page}-c{number}",
        "document_id": doc,
        "title": f"{doc} paper",
        "source": f"{doc.lower()}.pdf",
        "page": page,
        "rerank_score": 1.0,
        "content": content,
    }


class Turn:
    """Result of one user turn: final text plus every tool response seen."""

    def __init__(self) -> None:
        self.final_text = ""
        self.tool_responses: list[dict[str, Any]] = []


async def _run_turns(agent: Any, messages: list[types.Content]) -> list[Turn]:
    runner = InMemoryRunner(agent=agent, app_name="finguard_test")
    session = await runner.session_service.create_session(app_name="finguard_test", user_id="tester")
    turns: list[Turn] = []
    for message in messages:
        turn = Turn()
        async for event in runner.run_async(user_id="tester", session_id=session.id, new_message=message):
            for response in event.get_function_responses():
                turn.tool_responses.append(dict(response.response or {}))
            if event.is_final_response() and event.content and event.content.parts:
                turn.final_text = "".join(p.text or "" for p in event.content.parts if not p.thought)
        turns.append(turn)
    return turns


def user(message: str) -> types.Content:
    return types.Content(role="user", parts=[types.Part(text=message)])


def run(agent: Any, *messages: str | types.Content) -> list[Turn]:
    contents = [user(m) if isinstance(m, str) else m for m in messages]
    return asyncio.run(_run_turns(agent, contents))


class FakeContext:
    """Minimal stand-in for ToolContext / CallbackContext in direct callback tests."""

    def __init__(self, invocation_id: str = "inv-1") -> None:
        self.state: dict[str, Any] = {}
        self.invocation_id = invocation_id


class FakeTool:
    def __init__(self, name: str) -> None:
        self.name = name
