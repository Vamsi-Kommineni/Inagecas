from __future__ import annotations

import base64
from unittest.mock import AsyncMock, MagicMock

from factories import FakeGateway
from inagecas_shared.config import get_settings
from inagecas_shared.schemas import ImageLine
from vision_service.extract import extract
from vision_service.vlm import parse_reading

with_model = get_settings().model_copy(update={"vision_model": "vision"})
settings = with_model.model_copy(update={"vision_model": ""})

_READING = (
    '{"readable": true, "screen_title": "API Settings", "error_message": "Invalid API key", '
    '"text": ["API Settings", "Error: Invalid API key", "Status 401"]}'
)


class FailingGateway(FakeGateway):
    async def describe_image(self, instructions, image_base64, *, model=None, max_tokens=None):
        raise RuntimeError("429 rate limited")


class FakeOcr:
    def __init__(self, lines: list[ImageLine]) -> None:
        self._lines = lines

    def read(self, image_bytes: bytes) -> list[ImageLine]:
        return self._lines


def _session() -> MagicMock:
    session = MagicMock()
    session.add = MagicMock()
    session.commit = AsyncMock()
    return session


def _valid_b64() -> str:
    return base64.b64encode(b"pretend-image-bytes").decode()


async def test_extract_builds_facts_from_readable_image():
    lines = [
        ImageLine(text="Error 500", confidence=0.98, box=[[0, 0], [120, 0], [120, 30], [0, 30]])
    ]
    facts = await extract(_session(), FakeOcr(lines), settings, _valid_b64())
    assert facts.readable is True
    assert facts.detected_error == "Error 500"
    assert "Error 500" in facts.ocr_text


async def test_extract_unreadable_when_no_text():
    facts = await extract(_session(), FakeOcr([]), settings, _valid_b64())
    assert facts.readable is False
    assert facts.ocr_text == ""


async def test_extract_handles_invalid_base64():
    facts = await extract(_session(), FakeOcr([]), settings, "!!!not-base64!!!")
    assert facts.readable is False


async def test_a_vision_model_reads_the_screenshot_when_configured():
    gateway = FakeGateway(_READING)
    facts = await extract(_session(), FakeOcr([]), with_model, _valid_b64(), gateway=gateway)
    assert facts.readable is True
    assert facts.detected_error == "Invalid API key"
    assert facts.screen_title == "API Settings"
    assert "Status 401" in facts.ocr_text
    assert gateway.models_used == ["vision"]


async def test_ocr_takes_over_when_the_vision_model_fails():
    """A provider rate limit must not turn a readable screenshot into a refusal."""
    lines = [ImageLine(text="Error 500 on checkout", confidence=0.98)]
    facts = await extract(
        _session(), FakeOcr(lines), with_model, _valid_b64(), gateway=FailingGateway()
    )
    assert facts.readable is True
    assert facts.detected_error == "Error 500 on checkout"


async def test_ocr_takes_over_when_the_vision_model_does_not_answer_in_json():
    lines = [ImageLine(text="Error 500 on checkout", confidence=0.98)]
    gateway = FakeGateway("The screenshot shows an error page.")
    facts = await extract(_session(), FakeOcr(lines), with_model, _valid_b64(), gateway=gateway)
    assert facts.readable is True
    assert facts.detected_error == "Error 500 on checkout"


async def test_vision_model_is_not_called_without_the_setting():
    gateway = FakeGateway(_READING)
    facts = await extract(_session(), FakeOcr([]), settings, _valid_b64(), gateway=gateway)
    assert facts.readable is False
    assert gateway.models_used == []


def test_parse_reading_tolerates_code_fences_and_unreadable_images():
    fenced = "```json\n" + _READING + "\n```"
    reading = parse_reading(fenced)
    assert reading is not None and reading.readable and len(reading.lines) == 3
    blank = parse_reading('{"readable": false, "text": []}')
    assert blank is not None and blank.readable is False
    assert parse_reading("not json") is None
