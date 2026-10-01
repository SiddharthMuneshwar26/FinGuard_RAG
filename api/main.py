"""FastAPI service for the FinGuard RAG pipeline."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from src.config import RERANK_TOP_N
from src.rag_chain import FinGuardRAG


class QueryRequest(BaseModel):
    """Request body for the RAG query endpoint."""

    question: str = Field(..., min_length=1, max_length=1000)
    top_k: int = Field(default=RERANK_TOP_N, ge=1, le=10)
    generate: bool = True


class LatencyResponse(BaseModel):
    """Pipeline latency measurements in milliseconds."""

    retrieval: float
    rerank: float
    generation: float
    total: float


class CitationResponse(BaseModel):
    """Citation metadata returned by the RAG pipeline."""

    title: str
    filename: str
    file: str
    page: int
    chunk_id: str
    rerank_score: float


class ChunkResponse(BaseModel):
    """Retrieved/reranked evidence chunk."""

    chunk_id: str
    filename: str
    page: int
    content: str
    retrieval_distance: float | None = None
    rerank_score: float | None = None
    rerank_position: int | None = None


class QueryResponse(BaseModel):
    """Response returned by the RAG query endpoint."""

    answer: str | None
    citations: list[CitationResponse]
    chunks: list[ChunkResponse]
    latency_ms: LatencyResponse


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the RAG pipeline once when the API process starts."""
    app.state.rag = FinGuardRAG()
    yield


app = FastAPI(
    title="FinGuard RAG API",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def health() -> dict[str, str]:
    """Return API health status."""
    return {"status": "ok"}


def _serialize_chunks(documents: list[Any]) -> list[ChunkResponse]:
    """Convert LangChain documents into JSON-safe API objects."""
    chunks: list[ChunkResponse] = []

    for document in documents:
        metadata = document.metadata

        chunks.append(
            ChunkResponse(
                chunk_id=str(metadata.get("chunk_id", "")),
                filename=str(
                    metadata.get(
                        "filename",
                        metadata.get("source", "Unknown"),
                    )
                ),
                page=int(metadata.get("page", 0)),
                content=str(document.page_content),
                retrieval_distance=(
                    float(metadata["retrieval_distance"])
                    if "retrieval_distance" in metadata
                    else None
                ),
                rerank_score=(
                    float(metadata["rerank_score"])
                    if "rerank_score" in metadata
                    else None
                ),
                rerank_position=(
                    int(metadata["rerank_position"])
                    if "rerank_position" in metadata
                    else None
                ),
            )
        )

    return chunks


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest, http_request: Request) -> QueryResponse:
    """Run retrieval, reranking, and optionally local LLM generation."""
    rag: FinGuardRAG = http_request.app.state.rag

    try:
        result = rag.answer(
            request.question,
            top_k=request.top_k,
            generate=request.generate,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Internal server error",
        ) from exc

    return QueryResponse(
        answer=result["answer"],
        citations=[
            CitationResponse(**citation)
            for citation in result["sources"]
        ],
        chunks=_serialize_chunks(result["documents"]),
        latency_ms=LatencyResponse(**result["latency_ms"]),
    )