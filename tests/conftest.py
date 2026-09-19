"""Test configuration: set safe defaults before any service code is imported."""

from __future__ import annotations

import os

os.environ.setdefault("INAGECAS_API_KEYS", "test-key")
os.environ.setdefault("INAGECAS_ANSWER_MIN_CONFIDENCE", "0.5")
os.environ.setdefault("INAGECAS_INTERNAL_TOKEN", "")
# Tests never export telemetry.
os.environ.pop("OTEL_EXPORTER_OTLP_ENDPOINT", None)
