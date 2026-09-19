"""Split documents into overlapping, section-aware chunks.

Markdown headings start a new section so chunks keep their context (a heading
plus nearby text). Long sections are split into overlapping word windows so a
single chunk stays small enough for the embedding model while neighbouring
chunks share some text to avoid cutting an answer in half.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_HEADING = re.compile(r"^(#{1,6})\s+(.*)$")


@dataclass(slots=True)
class Chunk:
    ordinal: int
    heading: str | None
    text: str

    @property
    def token_estimate(self) -> int:
        # Rough heuristic; good enough for budgeting and metrics.
        return max(1, len(self.text) // 4)


def _iter_sections(text: str, source_type: str) -> list[tuple[str | None, str]]:
    if source_type != "markdown":
        body = text.strip()
        return [(None, body)] if body else []

    sections: list[tuple[str | None, str]] = []
    heading: str | None = None
    buffer: list[str] = []

    def flush() -> None:
        body = "\n".join(buffer).strip()
        if body:
            sections.append((heading, body))

    for line in text.splitlines():
        match = _HEADING.match(line)
        if match:
            flush()
            heading = match.group(2).strip()
            buffer = []
        else:
            buffer.append(line)
    flush()
    return sections


def _window_words(body: str, max_words: int, overlap_words: int) -> list[str]:
    words = body.split()
    if len(words) <= max_words:
        return [" ".join(words)] if words else []

    step = max(1, max_words - overlap_words)
    windows = []
    for start in range(0, len(words), step):
        window = words[start : start + max_words]
        if window:
            windows.append(" ".join(window))
        if start + max_words >= len(words):
            break
    return windows


def chunk_document(
    text: str,
    source_type: str,
    *,
    max_words: int = 180,
    overlap_words: int = 30,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    ordinal = 0
    for heading, body in _iter_sections(text, source_type):
        for piece in _window_words(body, max_words, overlap_words):
            chunks.append(Chunk(ordinal=ordinal, heading=heading, text=piece))
            ordinal += 1
    return chunks
