"""Authentication dependencies.

The public gateway requires a client API key. Internal services optionally
require a shared token so they cannot be called directly even from inside the
network; the check is disabled when no token is configured.
"""

from __future__ import annotations

import secrets

from fastapi import Depends, Header, HTTPException, status

from .config import Settings, get_settings


def _matches_any(provided: str, allowed: set[str]) -> bool:
    # compare_digest against every key keeps the check constant-time.
    return any(secrets.compare_digest(provided, key) for key in allowed)


async def require_api_key(
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    settings: Settings = Depends(get_settings),
) -> str:
    allowed = settings.api_key_set
    if not allowed:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="API authentication is not configured",
        )
    if not x_api_key or not _matches_any(x_api_key, allowed):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
            headers={"WWW-Authenticate": "ApiKey"},
        )
    return x_api_key


async def require_internal_token(
    x_internal_token: str | None = Header(default=None, alias="X-Internal-Token"),
    settings: Settings = Depends(get_settings),
) -> None:
    expected = settings.internal_token
    if not expected:
        return
    if not x_internal_token or not secrets.compare_digest(x_internal_token, expected):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid internal token",
        )
