"""arq worker that runs ingestion jobs enqueued by the gateway."""

from __future__ import annotations

import uuid
from typing import Any

from arq.connections import RedisSettings

from inagecas_shared.config import get_settings
from inagecas_shared.db import dispose_engine, get_session_factory
from inagecas_shared.logging import configure_logging
from inagecas_shared.model_gateway import ModelGateway
from inagecas_shared.sparse import SparseEmbedder
from inagecas_shared.telemetry import setup_tracing
from inagecas_shared.vectorstore import VectorStore

from .pipeline import run_ingestion


async def ingest_document(
    ctx: dict[str, Any],
    *,
    ingestion_run_id: str,
    title: str | None,
    source_type: str,
    source_uri: str,
    content: str,
) -> None:
    async with get_session_factory()() as session:
        await run_ingestion(
            session,
            ctx["gateway"],
            ctx["store"],
            ctx["sparse"],
            ingestion_run_id=uuid.UUID(ingestion_run_id),
            title=title,
            source_type=source_type,
            source_uri=source_uri,
            content=content,
        )


async def startup(ctx: dict[str, Any]) -> None:
    settings = get_settings()
    configure_logging(settings.log_level, json_logs=settings.is_production)
    setup_tracing("ingestion-worker")
    ctx["gateway"] = ModelGateway(settings)
    ctx["store"] = VectorStore(settings)
    ctx["sparse"] = SparseEmbedder(settings.sparse_model)


async def shutdown(ctx: dict[str, Any]) -> None:
    await ctx["gateway"].aclose()
    await ctx["store"].aclose()
    await dispose_engine()


class WorkerSettings:
    functions = [ingest_document]
    redis_settings = RedisSettings.from_dsn(get_settings().redis_url)
    on_startup = startup
    on_shutdown = shutdown
    max_tries = 2
    job_timeout = 600
    keep_result = 3600
