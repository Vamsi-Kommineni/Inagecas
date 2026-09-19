"""Turn OCR lines into structured facts.

These heuristics are deliberately simple and explainable: is there enough
readable text, does any line look like an error, and which line is likely the
screen title (the tallest text tends to be a header).
"""

from __future__ import annotations

import re

from inagecas_shared.config import Settings
from inagecas_shared.schemas import ImageLine

_ERROR = re.compile(
    r"\b(error|failed|failure|exception|denied|invalid|unable|cannot|"
    r"not found|timed? out|[45]\d\d)\b",
    re.IGNORECASE,
)


def assess_readability(lines: list[ImageLine], settings: Settings) -> tuple[bool, float]:
    if not lines:
        return False, 0.0
    char_count = sum(len(line.text.strip()) for line in lines)
    confidence = sum(line.confidence for line in lines) / len(lines)
    readable = (
        char_count >= settings.vision_min_chars and confidence >= settings.vision_min_confidence
    )
    return readable, round(confidence, 4)


def detect_error(lines: list[ImageLine]) -> str | None:
    for line in lines:
        if _ERROR.search(line.text):
            return line.text.strip()
    return None


def _box_height(box: list[list[float]]) -> float:
    if not box:
        return 0.0
    ys = [point[1] for point in box]
    return max(ys) - min(ys)


def screen_title(lines: list[ImageLine]) -> str | None:
    if not lines:
        return None
    # The tallest text on screen is usually a header or title.
    tallest = max(lines, key=lambda line: _box_height(line.box))
    return tallest.text.strip() or None
