"""Citation grounding: remove citations that point at evidence which was never
retrieved, and drop sentences whose only support was a fabricated citation.

Framework-free; shared by the ADK agent's grounding_guardrail and the LangGraph
conversation service.
"""

from __future__ import annotations

import re
from typing import Optional


_BRACKET_RE = re.compile(r"\[([^\[\]\n]{1,200})\](?!\()")
_CHUNK_TOKEN_RE = re.compile(r"^[A-Za-z0-9_]+-p\d+-c\d+$")
_PAGE_TOKEN_RE = re.compile(r"^([A-Za-z0-9_]+)[,\s]+(?:p\.?|pp\.?|page)\s*(\d+)$", re.IGNORECASE)
_TOOL_TOKEN_RE = re.compile(r"^(?:drive|sql):[A-Za-z0-9_\-]+$")


def _split_sentences(line: str) -> list[str]:
    """Split on whitespace after . ! ? -- but never inside [...] (e.g. "[SHARC, p. 9]")."""
    sentences, start, depth = [], 0, 0
    for index, char in enumerate(line):
        if char == "[":
            depth += 1
        elif char == "]":
            depth = max(0, depth - 1)
        elif char.isspace() and depth == 0 and index > start and line[index - 1] in ".!?":
            sentences.append(line[start:index])
            start = index + 1
    sentences.append(line[start:])
    return [s for s in sentences if s.strip()] or [line]


def _classify(token: str) -> Optional[str]:
    if _CHUNK_TOKEN_RE.match(token):
        return "chunk"
    if _PAGE_TOKEN_RE.match(token):
        return "page"
    if _TOOL_TOKEN_RE.match(token):
        return "tool"
    return None


def _is_valid(token: str, kind: str, citations: set[str], pages: set[str]) -> bool:
    if kind in {"chunk", "tool"}:
        return token in citations
    match = _PAGE_TOKEN_RE.match(token)
    assert match is not None
    return f"{match.group(1).upper()}:{int(match.group(2))}" in pages


def strip_ungrounded_citations(text: str, citations: set[str], pages: set[str]) -> tuple[str, int]:
    """Remove citations not backed by this turn's tool results.

    A sentence whose citations are *all* invalid is dropped (it is an ungrounded
    claim). Returns the cleaned text and the number of valid citations kept.
    """
    pages = {page.upper() for page in pages}
    valid_total = 0
    out_lines: list[str] = []

    for line in text.split("\n"):
        kept_sentences: list[str] = []
        for sentence in _split_sentences(line):
            found = valid = 0

            def replace(match: re.Match[str]) -> str:
                nonlocal found, valid
                tokens = [token.strip() for token in re.split(r"[;,](?![\s]*(?:p\.?|pp\.?|page)\s*\d)", match.group(1))]
                kinds = [_classify(token) for token in tokens]
                if not any(kinds):
                    return match.group(0)  # not a citation, leave untouched
                good = [t for t, k in zip(tokens, kinds) if k and _is_valid(t, k, citations, pages)]
                found += sum(1 for k in kinds if k)
                valid += len(good)
                return f"[{', '.join(good)}]" if good else ""

            cleaned = _BRACKET_RE.sub(replace, sentence)
            if found and not valid:
                continue  # every citation in this sentence was fabricated
            valid_total += valid
            kept_sentences.append(re.sub(r"[ \t]+([.,;:!?])", r"\1", cleaned).rstrip())
        out_lines.append(" ".join(s for s in kept_sentences if s))

    cleaned_text = re.sub(r"\n{3,}", "\n\n", "\n".join(out_lines)).strip()
    return cleaned_text, valid_total
