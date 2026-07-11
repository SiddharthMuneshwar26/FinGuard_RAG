"""Command-line entry point for rebuilding the local FAISS index."""

from __future__ import annotations

import argparse
import logging

from .chunking import create_chunks
from .config import VECTORSTORE_DIR
from .ingest import load_all_documents
from .vectorstore import build_vectorstore, index_exists


def build_index(force: bool = False) -> int:
    """Load PDFs, create chunks, and build the FAISS index."""
    if index_exists() and not force:
        raise FileExistsError(
            f"An index already exists at {VECTORSTORE_DIR}. Use --force to replace generated index files."
        )
    documents = load_all_documents()
    chunks = create_chunks(documents)
    store = build_vectorstore(chunks)
    return int(store.index.ntotal)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="replace an existing generated FAISS index")
    parser.add_argument("--verbose", action="store_true", help="enable detailed logging")
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    try:
        count = build_index(force=args.force)
    except Exception as exc:
        logging.getLogger(__name__).error("Index build failed: %s", exc)
        return 1
    print(f"Indexed {count} chunks in {VECTORSTORE_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
