"""Read a screenshot with a vision model.

The model is asked for a small JSON record, not a description: readability,
the visible text, a likely title and an error message if one is shown. A reply
that is missing or not JSON is reported as no reading, and the caller falls
back to OCR rather than guessing.
"""

from __future__ import annotations

from dataclasses import dataclass

from inagecas_shared.logging import get_logger
from inagecas_shared.model_gateway import ModelGateway
from inagecas_shared.schemas import ImageLine
from inagecas_shared.structured import parse_json

log = get_logger("vision")

INSTRUCTIONS = (
    "You read screenshots sent to customer support. Reply with JSON only, no prose:\n"
    '{"readable": true or false, "screen_title": "..." or null, '
    '"error_message": "..." or null, "text": ["each line of visible text, in order"]}\n'
    "readable is false when the image shows no legible text. Copy text exactly; "
    "do not summarise or translate it."
)


@dataclass(frozen=True)
class Reading:
    readable: bool
    lines: list[ImageLine]
    error_message: str | None
    screen_title: str | None


def _optional_text(value: object) -> str | None:
    text = str(value).strip() if value is not None else ""
    return text or None


def parse_reading(raw: str) -> Reading | None:
    data = parse_json(raw)
    if data is None:
        return None
    raw_lines = data.get("text")
    if not isinstance(raw_lines, list):
        raw_lines = []
    lines = [ImageLine(text=str(t).strip(), confidence=1.0) for t in raw_lines if str(t).strip()]
    return Reading(
        readable=bool(data.get("readable")) and bool(lines),
        lines=lines,
        error_message=_optional_text(data.get("error_message")),
        screen_title=_optional_text(data.get("screen_title")),
    )


async def read_screenshot(gateway: ModelGateway, model: str, image_base64: str) -> Reading | None:
    try:
        # Asked for JSON in the prompt, not through response_format: the vision
        # providers reject that parameter, and the prompt alone is honoured.
        raw = await gateway.describe_image(INSTRUCTIONS, image_base64, model=model, max_tokens=800)
    except Exception as exc:
        # A rate limit or an outage at the provider must not lose the ticket.
        log.warning("vision_model_failed", model=model, error=str(exc))
        return None
    reading = parse_reading(raw)
    if reading is None:
        log.warning("vision_model_unparseable", model=model, reply=raw[:200])
    return reading
