"""Load PDF files as page-level LangChain documents with citation metadata."""

from __future__ import annotations

import logging
import re
from pathlib import Path

from langchain_core.documents import Document
from pypdf import PdfReader
from pypdf.errors import PdfReadError

from .config import DATA_DIR, PAPER_METADATA

logger = logging.getLogger(__name__)


def clean_text(text: str) -> str:
    """Normalize extraction noise while retaining paragraph boundaries."""
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def load_pdf(pdf_path: Path) -> list[Document]:
    """Load a PDF into one document per non-empty page.

    Page numbers are stored as human-readable, one-based values. The original
    filename is retained separately from the stable document identifier.
    """
    if not pdf_path.is_file():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    try:
        reader = PdfReader(str(pdf_path))
    except (PdfReadError, OSError) as exc:
        raise RuntimeError(f"Could not read PDF: {pdf_path}") from exc

    configured = PAPER_METADATA.get(pdf_path.name.lower(), {})
    raw_metadata = reader.metadata or {}
    document_id = configured.get("document_id", pdf_path.stem.upper())
    title = configured.get("title") or raw_metadata.get("/Title") or pdf_path.stem
    author = raw_metadata.get("/Author") or "Unknown"
    documents: list[Document] = []

    for page_index, page in enumerate(reader.pages):
        try:
            text = clean_text(page.extract_text() or "")
        except Exception as exc:  # pypdf raises several parser-specific errors
            logger.warning("Skipping unreadable page %d in %s: %s", page_index + 1, pdf_path.name, exc)
            continue
        if not text:
            logger.warning("Skipping empty page %d in %s", page_index + 1, pdf_path.name)
            continue
        documents.append(
            Document(
                page_content=text,
                metadata={
                    "document_id": document_id,
                    "title": str(title),
                    "filename": pdf_path.name,
                    "source": pdf_path.name,
                    "page": page_index + 1,
                    "page_index": page_index,
                    "total_pages": len(reader.pages),
                    "author": str(author),
                },
            )
        )

    if not documents:
        raise RuntimeError(f"No text could be extracted from {pdf_path}")
    logger.info("Loaded %d pages from %s", len(documents), pdf_path.name)
    return documents


def load_all_documents(data_dir: Path = DATA_DIR) -> list[Document]:
    """Load every PDF in ``data_dir`` in deterministic filename order."""
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    pdf_files = sorted(data_dir.glob("*.pdf"), key=lambda path: path.name.lower())
    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in {data_dir}")

    documents = [document for path in pdf_files for document in load_pdf(path)]
    logger.info("Loaded %d pages from %d PDFs", len(documents), len(pdf_files))
    return documents
