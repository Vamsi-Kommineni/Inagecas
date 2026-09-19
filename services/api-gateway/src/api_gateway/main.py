"""API gateway: the only public service. Owns auth, rate limiting, and orchestration."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import httpx
from arq import create_pool
from arq.connections import RedisSettings
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from inagecas_shared.config import get_settings
from inagecas_shared.db import dispose_engine, get_engine
from inagecas_shared.logging import get_logger
from inagecas_shared.model_gateway import ModelGateway
from inagecas_shared.service import create_service_app
from inagecas_shared.telemetry import instrument_engine
from inagecas_shared.vectorstore import VectorStore

from .cache import RetrievalCache
from .clients import InternalClient
from .deps import limiter
from .routers import admin, ask, ingest

log = get_logger("api-gateway")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    app.state.settings = settings
    app.state.internal = InternalClient(settings)
    app.state.store = VectorStore(settings)
    app.state.redis_pool = await create_pool(RedisSettings.from_dsn(settings.redis_url))
    app.state.cache = RetrievalCache(app.state.redis_pool, settings.retrieval_cache_ttl)
    # The gateway's own model use: restating follow-ups.
    app.state.models = ModelGateway(settings)
    instrument_engine(get_engine())
    try:
        yield
    finally:
        await app.state.models.aclose()
        await app.state.internal.aclose()
        await app.state.store.aclose()
        await app.state.redis_pool.aclose()
        await dispose_engine()


async def _readiness(app: FastAPI) -> None:
    await app.state.internal.health()


app = create_service_app("api-gateway", public=True, lifespan=lifespan, readiness=_readiness)
app.state.limiter = limiter
# slowapi's handler is typed for its own exception; Starlette wants a broader type.
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)  # type: ignore[arg-type]


@app.exception_handler(httpx.HTTPError)
async def _upstream_error(request: Request, exc: httpx.HTTPError) -> JSONResponse:
    # Never surface internal service details to clients.
    log.warning("upstream_error", error=str(exc))
    return JSONResponse({"detail": "Upstream service unavailable"}, status_code=502)


app.include_router(ask.router)
app.include_router(ingest.router)
app.include_router(admin.router)
