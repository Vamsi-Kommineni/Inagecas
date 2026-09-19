"""Vision service: POST /extract turns a screenshot into structured facts."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import APIRouter, Depends, FastAPI, Request
from sqlalchemy.ext.asyncio import AsyncSession

from inagecas_shared.config import get_settings
from inagecas_shared.db import dispose_engine, get_engine, get_session
from inagecas_shared.model_gateway import ModelGateway
from inagecas_shared.schemas import ImageExtractRequest, ImageFacts
from inagecas_shared.security import require_internal_token
from inagecas_shared.service import create_service_app
from inagecas_shared.telemetry import instrument_engine

from .extract import extract
from .ocr import OcrEngine


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    app.state.ocr = OcrEngine()
    app.state.gateway = ModelGateway(settings) if settings.vision_model else None
    app.state.settings = settings
    instrument_engine(get_engine())
    try:
        yield
    finally:
        if app.state.gateway is not None:
            await app.state.gateway.aclose()
        await dispose_engine()


router = APIRouter()


@router.post(
    "/extract",
    response_model=ImageFacts,
    dependencies=[Depends(require_internal_token)],
)
async def extract_endpoint(
    payload: ImageExtractRequest,
    request: Request,
    session: AsyncSession = Depends(get_session),
) -> ImageFacts:
    state = request.app.state
    return await extract(
        session, state.ocr, state.settings, payload.image_base64, gateway=state.gateway
    )


app = create_service_app("vision-service", lifespan=lifespan)
app.include_router(router)
