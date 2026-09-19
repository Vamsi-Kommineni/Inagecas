"""Retrieval service: POST /retrieve returns evidence for a query."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import APIRouter, Depends, FastAPI, Request
from sqlalchemy.ext.asyncio import AsyncSession

from inagecas_shared.config import get_settings
from inagecas_shared.db import dispose_engine, get_engine, get_session
from inagecas_shared.model_gateway import ModelGateway
from inagecas_shared.schemas import RetrieveRequest, RetrieveResponse
from inagecas_shared.security import require_internal_token
from inagecas_shared.service import create_service_app
from inagecas_shared.sparse import SparseEmbedder
from inagecas_shared.telemetry import instrument_engine
from inagecas_shared.vectorstore import VectorStore

from .rerank import Reranker
from .retriever import retrieve


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    gateway = ModelGateway(settings)
    store = VectorStore(settings)
    sparse = SparseEmbedder(settings.sparse_model)
    await store.ensure_collection()
    instrument_engine(get_engine())

    app.state.gateway = gateway
    app.state.store = store
    app.state.sparse = sparse
    app.state.reranker = Reranker(settings, gateway)
    app.state.settings = settings
    try:
        yield
    finally:
        await gateway.aclose()
        await store.aclose()
        await dispose_engine()


async def _readiness(app: FastAPI) -> None:
    await app.state.store.health()


router = APIRouter()


@router.post(
    "/retrieve",
    response_model=RetrieveResponse,
    dependencies=[Depends(require_internal_token)],
)
async def retrieve_endpoint(
    payload: RetrieveRequest,
    request: Request,
    session: AsyncSession = Depends(get_session),
) -> RetrieveResponse:
    state = request.app.state
    return await retrieve(
        session,
        state.gateway,
        state.store,
        state.sparse,
        state.reranker,
        state.settings,
        payload,
    )


app = create_service_app("retrieval-service", lifespan=lifespan, readiness=_readiness)
app.include_router(router)
