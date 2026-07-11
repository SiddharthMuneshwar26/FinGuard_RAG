"""Recursive page-aware chunking for FinGuard documents."""

from __future__ import annotations

import logging
from collections import defaultdict

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import CHUNK_OVERLAP, CHUNK_SEPARATORS, CHUNK_SIZE

logger = logging.getLogger(__name__)


def create_chunks(
    documents: list[Document],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[Document]:
    """Split page documents and attach stable, page-level chunk identifiers."""
    if not documents:
        raise ValueError("No documents provided for chunking")
    if chunk_size <= 0 or chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("Chunk size must be positive and overlap must be smaller than size")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=CHUNK_SEPARATORS,
        length_function=len,
        add_start_index=True,
    )
    chunks = splitter.split_documents(documents)
    page_counts: defaultdict[tuple[str, int], int] = defaultdict(int)

    for chunk in chunks:
        document_id = str(chunk.metadata["document_id"])
        page = int(chunk.metadata["page"])
        key = (document_id, page)
        chunk_number = page_counts[key]
        page_counts[key] += 1
        chunk.metadata.update(
            {
                "chunk_number": chunk_number,
                "chunk_id": f"{document_id}-p{page}-c{chunk_number}",
                "character_count": len(chunk.page_content),
            }
        )

    logger.info("Created %d chunks from %d pages", len(chunks), len(documents))
    return chunks
