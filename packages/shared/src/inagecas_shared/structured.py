"""Read a model reply that should be JSON but might not be.

Hosted models honour a JSON schema; the local fallbacks mostly do, and a model
in a bad mood answers in prose. Callers name the field they want and get the
raw reply back when there is no JSON, so their old keyword matching still runs.
"""

from __future__ import annotations

import json
import re
from typing import Any

_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$")


def parse_json(raw: str) -> dict[str, Any] | None:
    """The JSON object in ``raw``, with a code fence stripped, or ``None``."""
    try:
        data = json.loads(_FENCE.sub("", raw.strip()))
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def parse_field(raw: str, field: str) -> str:
    """``field`` from a JSON reply, or the whole reply when it is not JSON."""
    data = parse_json(raw)
    if data is None:
        return raw
    value = data.get(field)
    return str(value) if value is not None else ""
