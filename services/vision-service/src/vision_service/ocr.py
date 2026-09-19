"""OCR wrapper around RapidOCR.

RapidOCR runs a small ONNX PP-OCR model on CPU and returns each detected line
with a bounding box and confidence. The engine is loaded lazily and a load or
decode failure yields no lines (treated downstream as an unreadable image)
rather than raising.
"""

from __future__ import annotations

import io
from typing import Any

from inagecas_shared.logging import get_logger
from inagecas_shared.schemas import ImageLine

log = get_logger("vision")


class OcrEngine:
    def __init__(self) -> None:
        self._engine: Any | None = None
        self._failed = False

    def _load(self) -> Any | None:
        if self._engine is None and not self._failed:
            try:
                from rapidocr_onnxruntime import RapidOCR

                self._engine = RapidOCR()
            except Exception as exc:
                self._failed = True
                log.warning("ocr_engine_load_failed", error=str(exc))
        return self._engine

    def read(self, image_bytes: bytes) -> list[ImageLine]:
        engine = self._load()
        if engine is None or not image_bytes:
            return []
        try:
            import numpy as np
            from PIL import Image, UnidentifiedImageError

            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except (UnidentifiedImageError, OSError, ValueError):
            return []

        result, _ = engine(np.asarray(image))
        lines: list[ImageLine] = []
        for box, text, score in result or []:
            lines.append(
                ImageLine(
                    text=str(text),
                    confidence=float(score),
                    box=[[float(x), float(y)] for x, y in box],
                )
            )
        return lines
