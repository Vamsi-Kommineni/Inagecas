"""Extract image facts from a base64 screenshot and record the run.

A vision model reads the image when one is configured; OCR reads it otherwise,
and also whenever the model fails, so a screenshot is never dropped because a
provider was down.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
from dataclasses import dataclass
from time import perf_counter

from sqlalchemy.ext.asyncio import AsyncSession

from inagecas_shared import metrics
from inagecas_shared.config import Settings
from inagecas_shared.ids import new_id
from inagecas_shared.model_gateway import ModelGateway
from inagecas_shared.models import ImageRun
from inagecas_shared.schemas import ImageFacts, ImageLine

from .facts import assess_readability, detect_error, screen_title
from .ocr import OcrEngine
from .vlm import read_screenshot


@dataclass(frozen=True)
class _Seen:
    """What was read off the image, whichever reader produced it."""

    lines: list[ImageLine]
    readable: bool
    confidence: float
    error: str | None
    title: str | None


async def _read(
    ocr_engine: OcrEngine,
    gateway: ModelGateway | None,
    settings: Settings,
    image_base64: str,
    image_bytes: bytes,
) -> _Seen:
    if gateway is not None and settings.vision_model and image_bytes:
        reading = await read_screenshot(gateway, settings.vision_model, image_base64)
        if reading is not None:
            chars = sum(len(line.text) for line in reading.lines)
            readable = reading.readable and chars >= settings.vision_min_chars
            error = reading.error_message or detect_error(reading.lines)
            return _Seen(reading.lines, readable, float(readable), error, reading.screen_title)

    lines = await asyncio.to_thread(ocr_engine.read, image_bytes)
    readable, confidence = assess_readability(lines, settings)
    return _Seen(lines, readable, confidence, detect_error(lines), screen_title(lines))


async def extract(
    session: AsyncSession,
    ocr_engine: OcrEngine,
    settings: Settings,
    image_base64: str,
    gateway: ModelGateway | None = None,
) -> ImageFacts:
    started = perf_counter()
    try:
        image_bytes = base64.b64decode(image_base64, validate=True)
    except (binascii.Error, ValueError):
        image_bytes = b""
    if len(image_bytes) > settings.image_max_bytes:
        image_bytes = b""  # oversized images are treated as unreadable

    seen = await _read(ocr_engine, gateway, settings, image_base64, image_bytes)
    metrics.record_ocr(seen.readable, seen.confidence)
    ocr_text = "\n".join(line.text for line in seen.lines)
    latency_ms = int((perf_counter() - started) * 1000)

    run = ImageRun(
        id=new_id(),
        char_count=len(ocr_text),
        line_count=len(seen.lines),
        readable=seen.readable,
        visual_confidence=seen.confidence,
        detected_error=seen.error[:512] if seen.error else None,
        screen_title=seen.title[:512] if seen.title else None,
        latency_ms=latency_ms,
    )
    session.add(run)
    await session.commit()

    return ImageFacts(
        image_run_id=run.id,
        readable=seen.readable,
        visual_confidence=seen.confidence,
        ocr_text=ocr_text,
        detected_error=seen.error,
        screen_title=seen.title,
        lines=seen.lines,
    )
