"""Backward-compatible retrieval facade.

New code should import :func:`src.retrieval.retrieve_candidates` and
:func:`src.reranker.rerank` directly.
"""

from __future__ import annotations

from typing import Any

from .config import RERANK_TOP_N, RETRIEVAL_K
from .reranker import rerank
from .retrieval import retrieve_candidates
from .vectorstore import load_vectorstore


def retrieve(
    query: str,
    top_k: int = RETRIEVAL_K,
    final_k: int = RERANK_TOP_N,
) -> list[dict[str, Any]]:
    """Return reranked results in the legacy dictionary format."""
    candidates = retrieve_candidates(load_vectorstore(), query, top_k)
    results = rerank(
        query,
        [(document, float(document.metadata["retrieval_distance"])) for document in candidates],
        final_k,
    )
    return [
        {"text": document.page_content, "metadata": document.metadata, "score": score}
        for document, score in results
    ]
