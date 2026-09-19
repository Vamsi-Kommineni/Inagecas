"""Policy service: POST /evaluate returns a route decision for a question."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import APIRouter, Depends, FastAPI, Request
from prometheus_client import Counter
from sqlalchemy.ext.asyncio import AsyncSession

from inagecas_shared.config import get_settings
from inagecas_shared.db import dispose_engine, get_engine, get_session
from inagecas_shared.model_gateway import ModelGateway
from inagecas_shared.schemas import PolicyDecision, PolicyEvaluateRequest
from inagecas_shared.security import require_internal_token
from inagecas_shared.service import create_service_app
from inagecas_shared.telemetry import instrument_engine

from .policy import evaluate

_ROUTE_DECISIONS = Counter(
    "policy_route_decisions_total",
    "Policy route decisions by route and intent.",
    ["route", "intent"],
)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    app.state.gateway = ModelGateway(settings)
    app.state.settings = settings
    instrument_engine(get_engine())
    try:
        yield
    finally:
        await app.state.gateway.aclose()
        await dispose_engine()


router = APIRouter()


@router.post(
    "/evaluate",
    response_model=PolicyDecision,
    dependencies=[Depends(require_internal_token)],
)
async def evaluate_endpoint(
    payload: PolicyEvaluateRequest,
    request: Request,
    session: AsyncSession = Depends(get_session),
) -> PolicyDecision:
    state = request.app.state
    decision = await evaluate(
        session,
        state.gateway,
        state.settings,
        payload.question,
        payload.has_image,
        payload.screenshot_text,
        payload.history,
    )
    _ROUTE_DECISIONS.labels(route=decision.route.value, intent=decision.intent.value).inc()
    return decision


app = create_service_app("policy-service", lifespan=lifespan)
app.include_router(router)
