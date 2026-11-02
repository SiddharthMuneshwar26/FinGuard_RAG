"""Framework-free input-safety helpers: prompt-injection detection, PII detection
and PII masking.

Shared by the ADK agent guardrails (agents/finguard_agent/guardrails.py) and the
LangGraph conversation service (conversation/). Pure functions: no model, no
network, no framework imports, so they are cheap to unit-test.
"""

from __future__ import annotations

import re

_INJECTION_PATTERNS = [
    r"\b(ignore|disregard|forget|override)\b[^.\n]{0,40}\b(previous|prior|above|earlier|all|your|system)\b[^.\n]{0,30}\b(instructions?|prompts?|rules?|guidelines?|directions?)\b",
    r"\b(reveal|show|print|repeat|leak|output)\b[^.\n]{0,30}\b(system|hidden|initial|original)\s+(prompt|instructions?|message)\b",
    r"\byou\s+are\s+now\b",
    r"\b(act|pretend|behave)\s+as\b[^.\n]{0,40}\b(unrestricted|without\s+(any\s+)?(rules|restrictions|limits|filters))\b",
    r"\b(developer|god|dan|jailbreak)\s+mode\b",
    r"\bjailbreak\b",
    r"\bdo\s+anything\s+now\b",
    r"<\|?(im_start|im_end|system|endoftext)\|?>",
    r"^\s*#{2,}\s*(system|instruction)s?\b",
    r"\bnew\s+(system\s+)?instructions?\s*:",
]
_INJECTION_RE = [re.compile(p, re.IGNORECASE | re.MULTILINE) for p in _INJECTION_PATTERNS]

_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_PAN_RE = re.compile(r"\b[A-Z]{5}[0-9]{4}[A-Z]\b")
_AADHAAR_RE = re.compile(r"(?<![\d])[2-9]\d{3}[\s-]?\d{4}[\s-]?\d{4}(?![\d])")
_CARD_CANDIDATE_RE = re.compile(r"(?<![\d])(?:\d[ -]?){12,18}\d(?![\d])")
_PHONE_RE = re.compile(r"(?<![\w+])(?:\+\d{1,3}[\s.-]?)?\(?\d{2,5}\)?[\s.-]?\d{3,4}[\s.-]?\d{3,5}(?![\w])")

Span = tuple[int, int]


def _luhn_ok(digits: str) -> bool:
    total, parity = 0, len(digits) % 2
    for index, char in enumerate(digits):
        value = int(char)
        if index % 2 == parity:
            value *= 2
            if value > 9:
                value -= 9
        total += value
    return total % 10 == 0


def _inside(span: Span, spans: list[Span]) -> bool:
    return any(start <= span[0] and span[1] <= end for start, end in spans)


def detect_prompt_injection(text: str) -> bool:
    return any(pattern.search(text) for pattern in _INJECTION_RE)


def find_pii(text: str) -> list[tuple[str, Span]]:
    """Return (kind, span) for every piece of personal data found, in text order.

    Overlaps are resolved by priority: card > Aadhaar > phone, so a 16-digit card
    number is never also reported as a phone number.
    """
    found: list[tuple[str, Span]] = []
    found += [("email address", m.span()) for m in _EMAIL_RE.finditer(text)]
    found += [("PAN-like ID", m.span()) for m in _PAN_RE.finditer(text)]

    card_spans: list[Span] = []
    for match in _CARD_CANDIDATE_RE.finditer(text):
        digits = re.sub(r"\D", "", match.group())
        if 13 <= len(digits) <= 19 and _luhn_ok(digits):
            card_spans.append(match.span())
    found += [("card-like number", span) for span in card_spans]

    aadhaar_spans = [m.span() for m in _AADHAAR_RE.finditer(text) if not _inside(m.span(), card_spans)]
    found += [("Aadhaar-like ID", span) for span in aadhaar_spans]

    taken = card_spans + aadhaar_spans
    for match in _PHONE_RE.finditer(text):
        digits = re.sub(r"\D", "", match.group())
        if 10 <= len(digits) <= 13 and not _inside(match.span(), taken):
            found.append(("phone number", match.span()))

    return sorted(found, key=lambda item: item[1][0])


def detect_pii(text: str) -> list[str]:
    """Return the distinct kinds of personal data found (empty list if none)."""
    kinds: list[str] = []
    for kind, _ in find_pii(text):
        if kind not in kinds:
            kinds.append(kind)
    return kinds


_MASK_LABELS = {
    "email address": "EMAIL",
    "PAN-like ID": "PAN",
    "card-like number": "CARD",
    "Aadhaar-like ID": "AADHAAR",
    "phone number": "PHONE",
}


def mask_pii(text: str) -> str:
    """Replace every detected piece of personal data with a typed placeholder,
    e.g. "mail me at a@b.com" -> "mail me at [EMAIL]".

    Used before anything is logged, audited, cached or sent to a model.
    """
    if not text:
        return text
    spans = find_pii(text)
    if not spans:
        return text
    out: list[str] = []
    cursor = 0
    for kind, (start, end) in spans:
        if start < cursor:  # overlapping match already masked
            continue
        out.append(text[cursor:start])
        out.append(f"[{_MASK_LABELS.get(kind, 'PII')}]")
        cursor = end
    out.append(text[cursor:])
    return "".join(out)
