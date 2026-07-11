"""Cross-encoder reranking using the model proven in the notebook."""
from functools import lru_cache

import numpy as np
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

from .config import (
    RERANK_TOP_N,
    RERANKER_DEVICE,
    RERANKER_MODEL_NAME,
)


class DocumentReranker:
    """Rerank retrieved documents using a CrossEncoder."""

    def __init__(
        self,
        model_name: str = RERANKER_MODEL_NAME,
        device: str = RERANKER_DEVICE,
    ) -> None:
        print(f"Loading reranker: {model_name}")
        print(f"Reranker device: {device}")

        self.model = CrossEncoder(
            model_name,
            device=device,
            max_length=512,
        )

    def rerank(
        self,
        query: str,
        documents: list[Document],
        top_n: int = RERANK_TOP_N,
    ) -> list[Document]:
        """Score query-document pairs and return the best documents."""
        if not documents:
            return []

        pairs = [
            (query, document.page_content)
            for document in documents
        ]

        raw_scores = self.model.predict(
            pairs,
            batch_size=8,
            show_progress_bar=False,
        )

        scores = np.asarray(raw_scores).reshape(-1)
        ranked_indices = np.argsort(scores)[::-1][:top_n]

        reranked_documents: list[Document] = []

        for rank, index in enumerate(ranked_indices, start=1):
            original = documents[int(index)]

            reranked_documents.append(
                Document(
                    page_content=original.page_content,
                    metadata={
                        **original.metadata,
                        "rerank_score": float(scores[index]),
                        "rerank_position": rank,
                    },
                )
            )

        return reranked_documents


@lru_cache(maxsize=1)
def get_reranker() -> DocumentReranker:
    """Return a cached reranker instance."""
    return DocumentReranker()


def rerank(
    query: str,
    candidates: list[tuple[Document, float]],
    top_n: int = RERANK_TOP_N,
) -> list[tuple[Document, float]]:
    """Rerank dense candidates while retaining the chain's scored interface."""
    if top_n <= 0:
        raise ValueError("Rerank result count must be positive")
    documents = get_reranker().rerank(query, [document for document, _ in candidates], top_n)
    return [(document, float(document.metadata["rerank_score"])) for document in documents]
