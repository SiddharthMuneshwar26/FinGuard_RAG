"""Dense candidate retrieval from the local FAISS index."""
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from .config import RETRIEVAL_K
from .embeddings import prepare_query


def retrieve_candidates(
    vectorstore: FAISS,
    query: str,
    k: int = RETRIEVAL_K,
) -> list[Document]:
    """Retrieve candidate documents from FAISS."""
    prepared_query = prepare_query(query)

    results = vectorstore.similarity_search_with_score(
        query=prepared_query,
        k=k,
    )

    candidates: list[Document] = []

    for document, distance in results:
        candidates.append(
            Document(
                page_content=document.page_content,
                metadata={
                    **document.metadata,
                    # FAISS L2 distance: lower usually means closer.
                    "retrieval_distance": float(distance),
                },
            )
        )

    return candidates
