"""Shared FastAPI app factory.

Every HTTP service is built from ``create_service_app`` so they all get the
same logging, tracing, metrics, security headers, request-id propagation, and
health endpoints. Services only add their own routers and lifespan.
"""

from __future__ import annotations

import uuid
from collections.abc import Awaitable, Callable
from contextlib import AbstractAsyncContextManager

import structlog
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from prometheus_fastapi_instrumentator import Instrumentator

from .config import get_settings
from .logging import configure_logging, get_logger
from .telemetry import instrument_app, setup_tracing

_SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
    "Cache-Control": "no-store",
}

ReadinessCheck = Callable[[FastAPI], Awaitable[None]]
Lifespan = Callable[[FastAPI], AbstractAsyncContextManager[None]]


def create_service_app(
    service_name: str,
    *,
    public: bool = False,
    lifespan: Lifespan | None = None,
    readiness: ReadinessCheck | None = None,
) -> FastAPI:
    settings = get_settings()
    configure_logging(settings.log_level, json_logs=settings.is_production)
    setup_tracing(service_name)
    log = get_logger(service_name)

    app = FastAPI(
        title=service_name,
        version="0.1.0",
        lifespan=lifespan,
        docs_url=None if settings.is_production else "/docs",
        redoc_url=None,
    )

    @app.middleware("http")
    async def context_and_headers(request: Request, call_next: Callable) -> Response:
        request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex
        structlog.contextvars.bind_contextvars(request_id=request_id, service=service_name)

        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > settings.request_max_bytes:
            structlog.contextvars.clear_contextvars()
            return JSONResponse({"detail": "Request body too large"}, status_code=413)

        try:
            response = await call_next(request)
        finally:
            structlog.contextvars.clear_contextvars()

        response.headers["X-Request-ID"] = request_id
        for key, value in _SECURITY_HEADERS.items():
            response.headers[key] = value
        return response

    if public and settings.cors_origin_list:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.cors_origin_list,
            allow_methods=["GET", "POST"],
            allow_headers=["Authorization", "Content-Type", "X-API-Key", "X-Request-ID"],
            max_age=600,
        )

    @app.get("/health/live", tags=["health"])
    async def health_live() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/health/ready", tags=["health"])
    async def health_ready() -> Response:
        if readiness is not None:
            try:
                await readiness(app)
            except Exception as exc:
                log.warning("readiness_failed", error=str(exc))
                return JSONResponse({"status": "not_ready"}, status_code=503)
        return JSONResponse({"status": "ready"})

    instrument_app(app)
    Instrumentator().instrument(app).expose(app, endpoint="/metrics", include_in_schema=False)

    return app
