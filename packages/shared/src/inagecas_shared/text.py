"""Small helpers for the text that goes to and comes back from the models.

A grounded answer carries two things: prose and ``[n]`` citation markers. Telling
them apart matters, because an answer made only of markers cites everything and
says nothing -- it has to fail grounding checks rather than sail through them.
"""

from __future__ import annotations

import re

CITATION_MARKER = re.compile(r"\[(\d+)\]")
# Some models cite in the full-width style they were trained on: 【1】 or 【1†L3】.
_WIDE_MARKER = re.compile(r"【(\d+)(?:†[^】]*)?】")


def normalize_citations(text: str) -> str:
    """Rewrite full-width citation markers to the ``[n]`` form the checks read."""
    return _WIDE_MARKER.sub(r"[\1]", text)


def strip_citations(text: str) -> str:
    """The prose left once citation markers are removed."""
    return CITATION_MARKER.sub(" ", text).strip()


def has_prose(text: str | None) -> bool:
    """True if the text says anything beyond citing its sources."""
    return bool(text and strip_citations(text))


def cited_indices(text: str) -> set[int]:
    """The passage numbers a piece of text cites."""
    return {int(n) for n in CITATION_MARKER.findall(text)}


def with_context(title: str | None, heading: str | None, text: str) -> str:
    """A passage under its title and heading, as the embedder and reranker see it.

    "Troubleshooting > Rate limit reached" in front of a short section is what
    lets a query about 429s find it. The stored passage stays bare.
    """
    labels = [part.strip() for part in (title, heading) if part and part.strip()]
    return f"{' > '.join(labels)}\n{text}" if labels else text
