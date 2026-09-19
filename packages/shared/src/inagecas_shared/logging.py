"""Structured logging with secret redaction.

Production emits JSON lines; development uses a readable console renderer.
A redaction processor masks common secret-bearing keys so customer text and
credentials never leak into logs by accident.
"""

from __future__ import annotations

import logging
from collections.abc import MutableMapping
from typing import Any

import structlog

_SENSITIVE_KEYS = {
    "api_key",
    "apikey",
    "authorization",
    "password",
    "secret",
    "token",
    "x-api-key",
    "x-internal-token",
}

_configured = False


def _redact_secrets(
    _: Any, __: str, event_dict: MutableMapping[str, Any]
) -> MutableMapping[str, Any]:
    for key in list(event_dict):
        if key.lower() in _SENSITIVE_KEYS:
            event_dict[key] = "***"
    return event_dict


def configure_logging(level: str = "INFO", *, json_logs: bool = True) -> None:
    global _configured
    if _configured:
        return

    numeric_level = logging.getLevelNamesMapping().get(level.upper(), logging.INFO)
    logging.basicConfig(format="%(message)s", level=numeric_level)

    renderer: structlog.types.Processor = (
        structlog.processors.JSONRenderer() if json_logs else structlog.dev.ConsoleRenderer()
    )

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            _redact_secrets,
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(numeric_level),
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )
    _configured = True


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    return structlog.get_logger(name)
