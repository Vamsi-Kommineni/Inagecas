"""Public /ingest endpoints: enqueue a document and check its status."""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy.ext.asyncio import AsyncSession

from inagecas_shared.db import get_session
from inagecas_shared.ids import new_id
from inagecas_shared.models import IngestionRun
from inagecas_shared.schemas import (
    IngestionStatus,
    IngestRequest,
    IngestResponse,
    SourceType,
)
from inagecas_shared.security import require_api_key

from ..deps import INGEST_TASK, RATE_LIMIT, limiter

router = APIRouter()

# PDFs are binary; ingest them with the seed CLI rather than inline JSON.
_INLINE_TYPES = {SourceType.markdown, SourceType.html, SourceType.text}


@router.post(
    "/ingest",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=IngestResponse,
    dependencies=[Depends(require_api_key)],
)
@limiter.limit(RATE_LIMIT)
async def ingest(
    payload: IngestRequest,
    request: Request,
    session: AsyncSession = Depends(get_session),
) -> IngestResponse:
    if payload.source_type not in _INLINE_TYPES:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Inline ingestion supports markdown, html, or text. Use the seed CLI for PDFs.",
        )

    run = IngestionRun(
        id=new_id(),
        status="queued",
        source_type=payload.source_type.value,
        source_uri=payload.source_uri,
    )
    session.add(run)
    await session.commit()

    await request.app.state.redis_pool.enqueue_job(
        INGEST_TASK,
        ingestion_run_id=str(run.id),
        title=payload.title,
        source_type=payload.source_type.value,
        source_uri=payload.source_uri,
        content=payload.content,
    )
    await request.app.state.cache.clear()
    return IngestResponse(ingestion_run_id=run.id, status="queued")


@router.get(
    "/ingest/{run_id}",
    response_model=IngestionStatus,
    dependencies=[Depends(require_api_key)],
)
async def ingest_status(
    run_id: uuid.UUID,
    session: AsyncSession = Depends(get_session),
) -> IngestionStatus:
    run = await session.get(IngestionRun, run_id)
    if run is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Ingestion run not found")
    return IngestionStatus(
        ingestion_run_id=run.id,
        status=run.status,
        document_id=run.document_id,
        chunks_indexed=run.chunks_indexed,
        error=run.error,
    )
