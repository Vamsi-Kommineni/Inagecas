"""Queue operations for escalations and agent actions."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from inagecas_shared.ids import new_id
from inagecas_shared.models import AgentAction, Escalation
from inagecas_shared.schemas import (
    ClaimRequest,
    EscalationCreate,
    EscalationStatus,
    FeedbackRequest,
    ResolveRequest,
)


async def create_escalation(
    session: AsyncSession, payload: EscalationCreate, *, summary: str
) -> Escalation:
    escalation = Escalation(
        id=new_id(),
        status=EscalationStatus.open.value,
        reason=payload.reason.value,
        channel=payload.channel,
        customer_ref=payload.customer_ref,
        question=payload.question,
        history=[turn.model_dump() for turn in payload.history],
        draft_answer=payload.draft_answer,
        summary=summary,
        image_note=payload.image_note,
        screenshot=payload.screenshot_base64,
        citations=[c.model_dump(mode="json") for c in payload.citations],
        policy_run_id=payload.policy_run_id,
        retrieval_run_id=payload.retrieval_run_id,
        answer_run_id=payload.answer_run_id,
        image_run_id=payload.image_run_id,
    )
    session.add(escalation)
    await session.commit()
    return escalation


async def list_escalations(
    session: AsyncSession, *, status: str | None, limit: int, offset: int
) -> list[Escalation]:
    stmt = select(Escalation).order_by(Escalation.created_at.desc()).limit(limit).offset(offset)
    if status:
        stmt = stmt.where(Escalation.status == status)
    return list(await session.scalars(stmt))


async def get_escalation(session: AsyncSession, escalation_id: uuid.UUID) -> Escalation | None:
    stmt = (
        select(Escalation)
        .where(Escalation.id == escalation_id)
        .options(selectinload(Escalation.actions))
    )
    return await session.scalar(stmt)


async def _record(
    session: AsyncSession, escalation: Escalation, action: AgentAction
) -> Escalation | None:
    session.add(action)
    await session.commit()
    # ``actions`` was loaded before this insert; expire it so the reload picks
    # up the new row instead of returning the copy already in the session.
    session.expire(escalation, ["actions"])
    return await get_escalation(session, escalation.id)


async def claim(
    session: AsyncSession, escalation_id: uuid.UUID, request: ClaimRequest
) -> Escalation | None:
    escalation = await get_escalation(session, escalation_id)
    if escalation is None:
        return None
    escalation.assignee = request.agent
    if escalation.status == EscalationStatus.open.value:
        escalation.status = EscalationStatus.in_progress.value
    action = AgentAction(
        id=new_id(), escalation_id=escalation.id, agent=request.agent, action="claim"
    )
    return await _record(session, escalation, action)


async def resolve(
    session: AsyncSession, escalation_id: uuid.UUID, request: ResolveRequest
) -> Escalation | None:
    escalation = await get_escalation(session, escalation_id)
    if escalation is None:
        return None
    escalation.status = EscalationStatus.resolved.value
    escalation.assignee = escalation.assignee or request.agent
    escalation.resolution = request.resolution
    # Store naive UTC to match the timezone-naive timestamp column.
    escalation.resolved_at = datetime.now(UTC).replace(tzinfo=None)
    action = AgentAction(
        id=new_id(),
        escalation_id=escalation.id,
        agent=request.agent,
        action="resolve",
        note=request.resolution,
    )
    return await _record(session, escalation, action)


async def add_feedback(
    session: AsyncSession, escalation_id: uuid.UUID, request: FeedbackRequest
) -> Escalation | None:
    escalation = await get_escalation(session, escalation_id)
    if escalation is None:
        return None
    action = AgentAction(
        id=new_id(),
        escalation_id=escalation.id,
        agent=request.agent,
        action="feedback",
        feedback_kind=request.kind.value,
        note=request.note,
    )
    return await _record(session, escalation, action)
