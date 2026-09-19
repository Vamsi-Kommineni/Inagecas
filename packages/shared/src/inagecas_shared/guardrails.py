"""Deterministic guardrails that run before any model call.

These are fast, explainable checks: spotting personal data in text (so it can
be flagged and masked before a hosted model sees it) and blocking obvious
prompt-injection attempts. They do not depend on the LLM, so they are reliable
whatever model is behind the gateway, and they run on every piece of text that
came from outside: the question, the conversation history and the screenshot.
"""

from __future__ import annotations

import re

_PII_PATTERNS = {
    "email": re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"),
    "credit_card": re.compile(r"\b(?:\d[ -]?){13,16}\b"),
    "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "phone": re.compile(r"\b\+?\d{1,3}[ -]?\(?\d{3}\)?[ -]?\d{3}[ -]?\d{4}\b"),
}

# Override / injection attempts we refuse outright.
_BLOCKED_PATTERNS = {
    "prompt_injection": re.compile(
        r"ignore (?:all |the )?(?:previous|prior|above) instructions", re.IGNORECASE
    ),
    "system_prompt_probe": re.compile(
        r"\b(?:system prompt|reveal your (?:prompt|instructions))\b", re.IGNORECASE
    ),
    "role_override": re.compile(r"\byou are now\b", re.IGNORECASE),
}


def detect_pii(text: str) -> list[str]:
    """Return the names of PII categories found in ``text`` (may be empty)."""
    return [name for name, pattern in _PII_PATTERNS.items() if pattern.search(text)]


def redact_pii(text: str) -> str:
    """Mask personal data with its category, so ``bob@x.com`` becomes ``[email]``.

    The masked text still reads naturally enough to classify, retrieve and
    answer from, and it is what leaves the machine when a hosted model is used.
    """
    for name, pattern in _PII_PATTERNS.items():
        text = pattern.sub(f"[{name}]", text)
    return text


def matched_block(text: str | None) -> str | None:
    """Return the first blocked-pattern name that matches, or ``None``."""
    if not text:
        return None
    for name, pattern in _BLOCKED_PATTERNS.items():
        if pattern.search(text):
            return name
    return None
