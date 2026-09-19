from __future__ import annotations

from inagecas_shared.config import get_settings
from inagecas_shared.schemas import ImageLine
from vision_service.facts import assess_readability, detect_error, screen_title

settings = get_settings()


def _line(text: str, confidence: float = 0.95, height: int = 20) -> ImageLine:
    return ImageLine(
        text=text, confidence=confidence, box=[[0, 0], [100, 0], [100, height], [0, height]]
    )


def test_detect_error_finds_error_line():
    lines = [_line("Welcome back"), _line("Error 500: server failed")]
    assert detect_error(lines) == "Error 500: server failed"


def test_detect_error_none_when_clean():
    assert detect_error([_line("Dashboard"), _line("Settings")]) is None


def test_screen_title_picks_tallest_line():
    lines = [_line("small print", height=10), _line("Big Title", height=44)]
    assert screen_title(lines) == "Big Title"


def test_assess_readability_true_for_clear_text():
    readable, confidence = assess_readability([_line("A clear line of text", 0.95)], settings)
    assert readable is True
    assert confidence > 0.9


def test_assess_readability_false_for_low_confidence():
    readable, _ = assess_readability([_line("blurry", 0.2)], settings)
    assert readable is False


def test_assess_readability_empty():
    assert assess_readability([], settings) == (False, 0.0)
