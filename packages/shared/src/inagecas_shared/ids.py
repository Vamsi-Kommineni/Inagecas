"""Identifier and hashing helpers."""

from __future__ import annotations

import hashlib
import uuid


def new_id() -> uuid.UUID:
    return uuid.uuid4()


def content_hash(text: str) -> str:
    """Stable SHA-256 of normalized content, used to detect duplicate versions."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
