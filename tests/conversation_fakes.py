"""Offline fakes for the conversation-service tests: no models, no network."""

from __future__ import annotations

from typing import Any, Iterator

from conversation.llm import LLMUnavailable

CHUNKS = {
    "shap": [
        {
            "chunk_id": "SHARC-p9-c1",
            "document_id": "SHARC",
            "title": "SHARC: SHAP-Based Interpretability for Regulatory Capital",
            "source": "SHARC.pdf",
            "page": 9,
            "rerank_score": 7.1,
            "content": "SHAP attributes each prediction to its input features using Shapley values.",
        },
        {
            "chunk_id": "SHARC-p10-c2",
            "document_id": "SHARC",
            "title": "SHARC: SHAP-Based Interpretability for Regulatory Capital",
            "source": "SHARC.pdf",
            "page": 10,
            "rerank_score": 5.3,
            "content": "SHARC applies SHAP to ICAAP and CCAR capital models.",
        },
    ],
    "genai": [
        {
            "chunk_id": "GAICF-p4-c1",
            "document_id": "GAICF",
            "title": "Governing GenAI Across Financial Institutions (SR 26-2)",
            "source": "genai_governance.pdf",
            "page": 4,
            "rerank_score": 6.0,
            "content": "Generative AI systems require human oversight and output monitoring.",
        }
    ],
}


class FakeRetriever:
    def __init__(self) -> None:
        self.queries: list[str] = []

    def __call__(self, query: str) -> list[dict[str, Any]]:
        self.queries.append(query)
        lowered = query.lower()
        if "shap" in lowered or "sharc" in lowered:
            return [dict(c) for c in CHUNKS["shap"]]
        if "genai" in lowered or "generative" in lowered:
            return [dict(c) for c in CHUNKS["genai"]]
        return []


class ScriptedModel:
    """Streams scripted replies word by word and records every prompt it receives."""

    def __init__(self, *replies: str, fail: bool = False) -> None:
        self.name = "fake/scripted"
        self.replies = list(replies)
        self.fail = fail
        self.prompts: list[list[dict[str, str]]] = []

    def stream(self, messages: list[dict[str, str]]) -> Iterator[str]:
        self.prompts.append(messages)
        if self.fail:
            raise LLMUnavailable("primary and fallbacks timed out")
        reply = self.replies.pop(0) if self.replies else "SHAP explains predictions [SHARC-p9-c1]."
        words = reply.split(" ")
        for index, word in enumerate(words):
            yield word + (" " if index < len(words) - 1 else "")
