"""Escalation service: a queue of human work items with agent actions."""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Query, Request
from sqlalchemy.ext.asyncio import AsyncSession

from inagecas_shared.config import get_settings
from inagecas_shared.db import dispose_engine, get_engine, get_session
from inagecas_shared.model_gateway import ModelGateway
from inagecas_shared.models import Escalation
from inagecas_shared.schemas import (
    AgentActionOut,
    ClaimRequest,
    EscalationCreate,
    EscalationCreated,
    EscalationDetail,
    EscalationStatus,
    EscalationSummary,
    FeedbackRequest,
    ResolveRequest,
    Turn,
)
from inagecas_shared.security import require_internal_token
from inagecas_shared.service import create_service_app
from inagecas_shared.telemetry import instrument_engine

from . import store
from .summary import build_summary, narrate


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    app.state.gateway = ModelGateway(settings) if settings.escalation_llm_summary else None
    instrument_engine(get_engine())
    try:
        yield
    finally:
        if app.state.gateway is not None:
            await app.state.gateway.aclose()
        await dispose_engine()


def _to_summary(escalation: Escalation) -> EscalationSummary:
    return EscalationSummary(
        id=escalation.id,
        status=EscalationStatus(escalation.status),
        reason=escalation.reason,
        question=escalation.question,
        assignee=escalation.assignee,
        created_at=escalation.created_at,
    )


def _to_detail(escalation: Escalation) -> EscalationDetail:
    return EscalationDetail(
        id=escalation.id,
        status=EscalationStatus(escalation.status),
        reason=escalation.reason,
        channel=escalation.channel,
        customer_ref=escalation.customer_ref,
        question=escalation.question,
        history=[Turn.model_validate(turn) for turn in escalation.history],
        draft_answer=escalation.draft_answer,
        summary=escalation.summary,
        image_note=escalation.image_note,
        screenshot_base64=escalation.screenshot,
        citations=escalation.citations,
        assignee=escalation.assignee,
        resolution=escalation.resolution,
        policy_run_id=escalation.policy_run_id,
        retrieval_run_id=escalation.retrieval_run_id,
        answer_run_id=escalation.answer_run_id,
        image_run_id=escalation.image_run_id,
        created_at=escalation.created_at,
        resolved_at=escalation.resolved_at,
        actions=[
            AgentActionOut(
                agent=action.agent,
                action=action.action,
                feedback_kind=action.feedback_kind,
                note=action.note,
                created_at=action.created_at,
            )
            for action in escalation.actions
        ],
    )


def _require(escalation: Escalation | None) -> Escalation:
    if escalation is None:
        raise HTTPException(status_code=404, detail="Escalation not found")
    return escalation


router = APIRouter(dependencies=[Depends(require_internal_token)])


@router.post("/escalations", response_model=EscalationCreated, status_code=201)
async def create(
    payload: EscalationCreate,
    request: Request,
    session: AsyncSession = Depends(get_session),
) -> EscalationCreated:
    summary = build_summary(payload)
    gateway = request.app.state.gateway
    if gateway and (narrative := await narrate(gateway, summary)):
        summary = f"{narrative}\n\n{summary}"
    escalation = await store.create_escalation(session, payload, summary=summary)
    return EscalationCreated(
        escalation_id=escalation.id, status=EscalationStatus(escalation.status)
    )


@router.get("/escalations", response_model=list[EscalationSummary])
async def list_all(
    session: AsyncSession = Depends(get_session),
    status: str | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
) -> list[EscalationSummary]:
    rows = await store.list_escalations(session, status=status, limit=limit, offset=offset)
    return [_to_summary(row) for row in rows]


@router.get("/escalations/{escalation_id}", response_model=EscalationDetail)
async def detail(
    escalation_id: uuid.UUID,
    session: AsyncSession = Depends(get_session),
) -> EscalationDetail:
    return _to_detail(_require(await store.get_escalation(session, escalation_id)))


@router.post("/escalations/{escalation_id}/claim", response_model=EscalationDetail)
async def claim(
    escalation_id: uuid.UUID,
    payload: ClaimRequest,
    session: AsyncSession = Depends(get_session),
) -> EscalationDetail:
    return _to_detail(_require(await store.claim(session, escalation_id, payload)))


@router.post("/escalations/{escalation_id}/resolve", response_model=EscalationDetail)
async def resolve(
    escalation_id: uuid.UUID,
    payload: ResolveRequest,
    session: AsyncSession = Depends(get_session),
) -> EscalationDetail:
    return _to_detail(_require(await store.resolve(session, escalation_id, payload)))


@router.post("/escalations/{escalation_id}/feedback", response_model=EscalationDetail)
async def feedback(
    escalation_id: uuid.UUID,
    payload: FeedbackRequest,
    session: AsyncSession = Depends(get_session),
) -> EscalationDetail:
    return _to_detail(_require(await store.add_feedback(session, escalation_id, payload)))


app = create_service_app("escalation-service", lifespan=lifespan)
app.include_router(router)
