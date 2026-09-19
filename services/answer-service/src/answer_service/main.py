"""Answer service: POST /generate returns a grounded answer or a refusal."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import APIRouter, Depends, FastAPI, Request

from inagecas_shared.config import get_settings
from inagecas_shared.model_gateway import ModelGateway
from inagecas_shared.schemas import GenerateRequest, GenerateResponse
from inagecas_shared.security import require_internal_token
from inagecas_shared.service import create_service_app

from .generate import generate


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    app.state.gateway = ModelGateway(settings)
    app.state.settings = settings
    try:
        yield
    finally:
        await app.state.gateway.aclose()


router = APIRouter()


@router.post(
    "/generate",
    response_model=GenerateResponse,
    dependencies=[Depends(require_internal_token)],
)
async def generate_endpoint(payload: GenerateRequest, request: Request) -> GenerateResponse:
    state = request.app.state
    return await generate(state.gateway, state.settings, payload)


app = create_service_app("answer-service", lifespan=lifespan)
app.include_router(router)
