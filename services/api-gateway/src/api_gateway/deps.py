"""Shared dependencies and the rate limiter for the gateway."""

from __future__ import annotations

from slowapi import Limiter
from slowapi.util import get_remote_address

from inagecas_shared.config import get_settings

_settings = get_settings()

# Per-client rate limit, backed by Redis so it holds across gateway replicas.
# It applies to the routes decorated with ``@limiter.limit(RATE_LIMIT)``.
limiter = Limiter(key_func=get_remote_address, storage_uri=_settings.redis_url)

RATE_LIMIT = _settings.rate_limit
INGEST_TASK = "ingest_document"
