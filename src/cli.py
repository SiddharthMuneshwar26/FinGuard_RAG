"""CLI for querying the local FinGuard-RAG backend."""

from __future__ import annotations

import argparse
import logging
import sys

from .rag_chain import answer_question


def main() -> int:
    # Windows may inherit a legacy code page that cannot print PDF/model Unicode.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question", nargs="+", help="question to answer from the indexed PDFs")
    parser.add_argument("--verbose", action="store_true", help="enable detailed logging")
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    try:
        result = answer_question(" ".join(args.question))
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        logging.getLogger(__name__).error("Query failed: %s", exc)
        return 1

    print(result["answer"])
    if result["sources"]:
        print("\nSources:")
        for source in result["sources"]:
            print(
                f"- {source['title']} ({source['filename']}), "
                f"p.{source['page']} [{source['chunk_id']}]"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
