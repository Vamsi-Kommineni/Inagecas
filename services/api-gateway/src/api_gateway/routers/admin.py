"""Admin endpoints: documents, the escalation queue, and run inspection."""

from __future__ import annotations

import uuid
from collections.abc import Awaitable
from datetime import datetime

import httpx
from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response, status
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from inagecas_shared.db import get_session
from inagecas_shared.models import (
    AnswerRun,
    Document,
    Escalation,
    ImageRun,
    PolicyRun,
    RetrievalRun,
)
from inagecas_shared.schemas import (
    ClaimRequest,
    EscalationDetail,
    EscalationSummary,
    FeedbackRequest,
    ResolveRequest,
)
from inagecas_shared.security import require_api_key

router = APIRouter(prefix="/admin", dependencies=[Depends(require_api_key)])


class DocumentSummary(BaseModel):
    id: uuid.UUID
    title: str | None
    source_uri: str
    source_type: str
    status: str
    created_at: datetime


@router.get("/documents", response_model=list[DocumentSummary])
async def list_documents(
    session: AsyncSession = Depends(get_session),
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
) -> list[DocumentSummary]:
    rows = await session.scalars(
        select(Document).order_by(Document.created_at.desc()).limit(limit).offset(offset)
    )
    return [DocumentSummary.model_validate(doc, from_attributes=True) for doc in rows]


@router.delete("/documents/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document(
    document_id: uuid.UUID,
    request: Request,
    session: AsyncSession = Depends(get_session),
) -> Response:
    document = await session.get(Document, document_id)
    if document is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")

    await request.app.state.store.delete_document(document_id)
    await session.delete(document)
    await session.commit()
    await request.app.state.cache.clear()
    return Response(status_code=status.HTTP_204_NO_CONTENT)


# --- Escalation queue (proxied to the escalation service) ------------------


async def _proxy(call: Awaitable[EscalationDetail]) -> EscalationDetail:
    try:
        return await call
    except httpx.HTTPStatusError as exc:
        raise HTTPException(
            status_code=exc.response.status_code, detail="Escalation error"
        ) from exc


@router.get("/escalations", response_model=list[EscalationSummary])
async def list_escalations(
    request: Request,
    status: str | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
) -> list[EscalationSummary]:
    return await request.app.state.internal.list_escalations(
        status=status, limit=limit, offset=offset
    )


@router.get("/escalations/{escalation_id}", response_model=EscalationDetail)
async def get_escalation(escalation_id: uuid.UUID, request: Request) -> EscalationDetail:
    return await _proxy(request.app.state.internal.get_escalation(escalation_id))


@router.post("/escalations/{escalation_id}/claim", response_model=EscalationDetail)
async def claim_escalation(
    escalation_id: uuid.UUID, payload: ClaimRequest, request: Request
) -> EscalationDetail:
    return await _proxy(request.app.state.internal.claim_escalation(escalation_id, payload))


@router.post("/escalations/{escalation_id}/resolve", response_model=EscalationDetail)
async def resolve_escalation(
    escalation_id: uuid.UUID, payload: ResolveRequest, request: Request
) -> EscalationDetail:
    return await _proxy(request.app.state.internal.resolve_escalation(escalation_id, payload))


@router.post("/escalations/{escalation_id}/feedback", response_model=EscalationDetail)
async def feedback_escalation(
    escalation_id: uuid.UUID, payload: FeedbackRequest, request: Request
) -> EscalationDetail:
    return await _proxy(request.app.state.internal.feedback_escalation(escalation_id, payload))


# --- Run inspection (why a request answered, refused, or escalated) --------


class RunTrace(BaseModel):
    answer_run_id: uuid.UUID
    question: str
    status: str
    refusal_reason: str | None = None
    confidence: float | None = None
    coverage: float | None = None
    support: str | None = None
    route: str | None = None
    intent: str | None = None
    pii_detected: bool | None = None
    retrieval_top_score: float | None = None
    retrieval_result_count: int | None = None
    image_readable: bool | None = None
    image_detected_error: str | None = None
    escalation_id: uuid.UUID | None = None
    escalation_status: str | None = None
    created_at: datetime


@router.get("/runs/{answer_run_id}", response_model=RunTrace)
async def get_run_trace(
    answer_run_id: uuid.UUID,
    session: AsyncSession = Depends(get_session),
) -> RunTrace:
    answer = await session.get(AnswerRun, answer_run_id)
    if answer is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Run not found")

    policy = await session.get(PolicyRun, answer.policy_run_id) if answer.policy_run_id else None
    retrieval = (
        await session.get(RetrievalRun, answer.retrieval_run_id)
        if answer.retrieval_run_id
        else None
    )
    image = await session.get(ImageRun, answer.image_run_id) if answer.image_run_id else None
    escalation = await session.scalar(
        select(Escalation).where(Escalation.answer_run_id == answer_run_id)
    )

    return RunTrace(
        answer_run_id=answer.id,
        question=answer.question,
        status=answer.status,
        refusal_reason=answer.refusal_reason,
        confidence=answer.confidence,
        coverage=answer.coverage,
        support=answer.support,
        route=policy.route if policy else None,
        intent=policy.intent if policy else None,
        pii_detected=policy.pii_detected if policy else None,
        retrieval_top_score=retrieval.top_score if retrieval else None,
        retrieval_result_count=retrieval.result_count if retrieval else None,
        image_readable=image.readable if image else None,
        image_detected_error=image.detected_error if image else None,
        escalation_id=escalation.id if escalation else None,
        escalation_status=escalation.status if escalation else None,
        created_at=answer.created_at,
    )
