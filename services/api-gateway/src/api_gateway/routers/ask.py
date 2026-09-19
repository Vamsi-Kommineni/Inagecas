"""Public /ask endpoint: image OCR, policy gate, retrieve, answer, escalate."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from functools import partial

import httpx
from fastapi import APIRouter, Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession

from inagecas_shared import metrics
from inagecas_shared.config import Settings, get_settings
from inagecas_shared.db import get_session
from inagecas_shared.guardrails import redact_pii
from inagecas_shared.ids import new_id
from inagecas_shared.logging import get_logger
from inagecas_shared.model_gateway import ModelGateway
from inagecas_shared.models import AnswerRun
from inagecas_shared.schemas import (
    AnswerStatus,
    AskRequest,
    AskResponse,
    Citation,
    EscalationCreate,
    EscalationReason,
    GenerateRequest,
    GenerateResponse,
    ImageExtractRequest,
    ImageFacts,
    PolicyDecision,
    PolicyEvaluateRequest,
    RefusalReason,
    RetrieveRequest,
    RetrieveResponse,
    Route,
    Turn,
)
from inagecas_shared.security import require_api_key
from inagecas_shared.telemetry import set_span_attributes, traced

from ..cache import RetrievalCache
from ..clients import InternalClient
from ..deps import RATE_LIMIT, limiter
from ..rewrite import standalone_question

log = get_logger("api-gateway")
router = APIRouter()

# Refusals that mean "the AI could not help" and should reach a human.
_ESCALATE_REASONS = {
    RefusalReason.no_evidence,
    RefusalReason.low_confidence,
    RefusalReason.unsupported,
    RefusalReason.contradicted,
}


def _augmented_query(question: str, facts: ImageFacts | None) -> str:
    """Merge image facts into the retrieval query so the screenshot steers search."""
    if facts is None:
        return question
    parts = [question]
    if facts.detected_error:
        parts.append(facts.detected_error)
    if facts.screen_title:
        parts.append(facts.screen_title)
    if facts.ocr_text:
        parts.append(facts.ocr_text[:1000])
    return "\n".join(parts)


def _image_note(facts: ImageFacts | None) -> str | None:
    if facts is None:
        return None
    parts = []
    if facts.detected_error:
        parts.append(f"error: {facts.detected_error}")
    if facts.screen_title:
        parts.append(f"screen: {facts.screen_title}")
    return "; ".join(parts) or None


def _refusal_message(escalated: bool) -> str:
    note = "We could not find a confident answer to this in our documentation."
    if escalated:
        note += " A support agent will follow up."
    return note


def _escalation_reason(route: Route, reason: RefusalReason | None) -> EscalationReason | None:
    if route == Route.escalate:
        return EscalationReason.sensitive
    if reason in _ESCALATE_REASONS:
        return EscalationReason(reason.value)
    return None


def _refused(reason: RefusalReason | None) -> GenerateResponse:
    """A refusal decided before any answer was drafted."""
    return GenerateResponse(status=AnswerStatus.refused, refusal_reason=reason, confidence=0.0)


@dataclass(slots=True)
class _Trail:
    """What the stages before the answer produced, for the record and the hand-off."""

    facts: ImageFacts | None = None
    policy: PolicyDecision | None = None
    retrieval: RetrieveResponse | None = None

    @property
    def image_run_id(self) -> uuid.UUID | None:
        return self.facts.image_run_id if self.facts else None

    @property
    def policy_run_id(self) -> uuid.UUID | None:
        return self.policy.policy_run_id if self.policy else None

    @property
    def retrieval_run_id(self) -> uuid.UUID | None:
        return self.retrieval.retrieval_run_id if self.retrieval else None


async def _maybe_escalate(
    internal: InternalClient,
    settings: Settings,
    payload: AskRequest,
    trail: _Trail,
    generation: GenerateResponse,
    *,
    route: Route,
    answer_run_id: uuid.UUID,
) -> uuid.UUID | None:
    if not settings.escalation_enabled:
        return None
    reason = _escalation_reason(route, generation.refusal_reason)
    if reason is None:
        return None
    evidence = trail.retrieval.evidence if trail.retrieval else []
    try:
        created = await internal.create_escalation(
            EscalationCreate(
                question=payload.question,
                history=payload.history,
                reason=reason,
                channel=payload.channel,
                customer_ref=payload.customer_ref,
                draft_answer=generation.draft,
                image_note=_image_note(trail.facts),
                screenshot_base64=payload.image_base64,
                citations=[Citation.from_evidence(item) for item in evidence],
                policy_run_id=trail.policy_run_id,
                retrieval_run_id=trail.retrieval_run_id,
                answer_run_id=answer_run_id,
                image_run_id=trail.image_run_id,
            )
        )
    except httpx.HTTPError as exc:
        # The customer's request must not fail over this, but a lost hand-off
        # is an incident: count it so the alert fires.
        metrics.record_escalation_failure(reason.value)
        log.error("escalation_create_failed", reason=reason.value, error=str(exc))
        return None
    metrics.record_escalation(reason.value)
    return created.escalation_id


async def _finish(
    session: AsyncSession,
    internal: InternalClient,
    settings: Settings,
    payload: AskRequest,
    trail: _Trail,
    generation: GenerateResponse,
    *,
    route: Route,
    started: float,
    message: str | None = None,
) -> AskResponse:
    """Record the run, hand a refusal to a human when it warrants one, and reply."""
    latency_ms = int((time.perf_counter() - started) * 1000)
    refused = generation.status == AnswerStatus.refused
    reason = generation.refusal_reason.value if generation.refusal_reason else None

    run = AnswerRun(
        id=new_id(),
        retrieval_run_id=trail.retrieval_run_id,
        policy_run_id=trail.policy_run_id,
        image_run_id=trail.image_run_id,
        question=payload.question,
        status=generation.status.value,
        refusal_reason=reason,
        answer=generation.answer or generation.clarification,
        confidence=generation.confidence,
        coverage=generation.coverage,
        support=generation.support,
        model=settings.chat_model,
        citations=[c.model_dump(mode="json") for c in generation.citations],
        latency_ms=latency_ms,
    )
    session.add(run)
    await session.commit()

    escalation_id: uuid.UUID | None = None
    if refused:
        escalation_id = await _maybe_escalate(
            internal, settings, payload, trail, generation, route=route, answer_run_id=run.id
        )

    metrics.record_answer(generation.status.value, reason, generation.confidence)
    set_span_attributes(
        route=route.value,
        status=generation.status.value,
        refusal_reason=reason,
        escalation_id=escalation_id,
    )
    log.info(
        "ask_completed",
        status=generation.status.value,
        route=route.value,
        refusal_reason=reason,
        answer_run_id=str(run.id),
        escalation_id=str(escalation_id) if escalation_id else None,
        latency_ms=latency_ms,
    )
    if refused:
        message = message or _refusal_message(escalation_id is not None)
    return AskResponse(
        status=generation.status,
        answer=generation.answer,
        refusal_reason=generation.refusal_reason,
        message=message,
        confidence=generation.confidence,
        citations=generation.citations,
        coverage=generation.coverage,
        support=generation.support,
        clarification=generation.clarification,
        route=route,
        retrieval_run_id=trail.retrieval_run_id,
        answer_run_id=run.id,
        policy_run_id=trail.policy_run_id,
        image_run_id=trail.image_run_id,
        escalation_id=escalation_id,
    )


async def _model_question(
    models: ModelGateway | None, settings: Settings, payload: AskRequest
) -> str:
    """The question as the models get it: personal data masked, and a
    follow-up restated so it stands on its own."""
    question = redact_pii(payload.question)
    if models is None or not payload.history:
        return question
    history = [Turn(role=t.role, content=redact_pii(t.content)) for t in payload.history]
    return await standalone_question(models, question, history, model=settings.judge_model)


async def _retrieve(
    internal: InternalClient, cache: RetrievalCache | None, request: RetrieveRequest
) -> RetrieveResponse:
    if cache is not None and (hit := await cache.get(request)) is not None:
        return hit
    response = await internal.retrieve(request)
    if cache is not None:
        await cache.put(request, response)
    return response


async def answer_question(
    session: AsyncSession,
    internal: InternalClient,
    settings: Settings,
    payload: AskRequest,
    *,
    models: ModelGateway | None = None,
    cache: RetrievalCache | None = None,
) -> AskResponse:
    trail = _Trail()
    finish = partial(
        _finish, session, internal, settings, payload, trail, started=time.perf_counter()
    )

    if payload.image_base64:
        trail.facts = await internal.extract(ImageExtractRequest(image_base64=payload.image_base64))
        if not trail.facts.readable:
            return await finish(
                _refused(RefusalReason.image_unreadable),
                route=Route.refuse,
                message="The attached image could not be read. Please attach a clearer screenshot.",
            )
    facts = trail.facts

    if settings.policy_enabled:
        trail.policy = await internal.evaluate(
            PolicyEvaluateRequest(
                question=payload.question,
                history=payload.history,
                has_image=facts is not None,
                screenshot_text=facts.ocr_text[:4000] if facts else None,
            )
        )
        if trail.policy.route != Route.answer:
            return await finish(
                _refused(trail.policy.refusal_reason),
                route=trail.policy.route,
                message=trail.policy.message,
            )

    # From here on the text may leave the machine, so it is masked first.
    question = await _model_question(models, settings, payload)
    query = redact_pii(_augmented_query(question, facts))
    with traced("retrieve"):
        trail.retrieval = await _retrieve(
            internal, cache, RetrieveRequest(query=query, top_k=payload.top_k)
        )
    with traced("generate"):
        generation = await internal.generate(
            GenerateRequest(
                question=question,
                evidence=trail.retrieval.evidence,
                top_score=trail.retrieval.top_score,
                screenshot_text=redact_pii(facts.ocr_text[:1000]) if facts else None,
            )
        )
    return await finish(generation, route=Route.answer)


@router.post("/ask", response_model=AskResponse, dependencies=[Depends(require_api_key)])
@limiter.limit(RATE_LIMIT)
async def ask(
    payload: AskRequest,
    request: Request,
    session: AsyncSession = Depends(get_session),
    settings: Settings = Depends(get_settings),
) -> AskResponse:
    state = request.app.state
    return await answer_question(
        session, state.internal, settings, payload, models=state.models, cache=state.cache
    )
