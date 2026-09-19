"""HTTP clients for the internal services the gateway orchestrates."""

from __future__ import annotations

import uuid

import httpx
from pydantic import BaseModel

from inagecas_shared.config import Settings
from inagecas_shared.schemas import (
    ClaimRequest,
    EscalationCreate,
    EscalationCreated,
    EscalationDetail,
    EscalationSummary,
    FeedbackRequest,
    GenerateRequest,
    GenerateResponse,
    ImageExtractRequest,
    ImageFacts,
    PolicyDecision,
    PolicyEvaluateRequest,
    ResolveRequest,
    RetrieveRequest,
    RetrieveResponse,
)


async def _post[T: BaseModel](
    client: httpx.AsyncClient, path: str, request: BaseModel, reply: type[T]
) -> T:
    response = await client.post(path, json=request.model_dump(mode="json"))
    response.raise_for_status()
    return reply.model_validate(response.json())


class InternalClient:
    def __init__(self, settings: Settings) -> None:
        headers = {}
        if settings.internal_token:
            headers["X-Internal-Token"] = settings.internal_token

        def client(base_url: str, timeout: float) -> httpx.AsyncClient:
            return httpx.AsyncClient(base_url=base_url, headers=headers, timeout=timeout)

        # The services that call a model get the longer timeouts.
        self._policy = client(settings.policy_service_url, 60.0)
        self._vision = client(settings.vision_service_url, 60.0)
        self._escalation = client(settings.escalation_service_url, 30.0)
        self._retrieval = client(settings.retrieval_service_url, 30.0)
        self._answer = client(settings.answer_service_url, 120.0)
        self._clients = (
            self._policy,
            self._vision,
            self._escalation,
            self._retrieval,
            self._answer,
        )

    async def evaluate(self, request: PolicyEvaluateRequest) -> PolicyDecision:
        return await _post(self._policy, "/evaluate", request, PolicyDecision)

    async def extract(self, request: ImageExtractRequest) -> ImageFacts:
        return await _post(self._vision, "/extract", request, ImageFacts)

    async def retrieve(self, request: RetrieveRequest) -> RetrieveResponse:
        return await _post(self._retrieval, "/retrieve", request, RetrieveResponse)

    async def generate(self, request: GenerateRequest) -> GenerateResponse:
        return await _post(self._answer, "/generate", request, GenerateResponse)

    async def create_escalation(self, request: EscalationCreate) -> EscalationCreated:
        try:
            return await _post(self._escalation, "/escalations", request, EscalationCreated)
        except httpx.TransportError:
            # A hand-off must not be lost to a dropped connection; try once more.
            return await _post(self._escalation, "/escalations", request, EscalationCreated)

    async def list_escalations(
        self, *, status: str | None, limit: int, offset: int
    ) -> list[EscalationSummary]:
        params: dict[str, str | int] = {"limit": limit, "offset": offset}
        if status:
            params["status"] = status
        response = await self._escalation.get("/escalations", params=params)
        response.raise_for_status()
        return [EscalationSummary.model_validate(item) for item in response.json()]

    async def get_escalation(self, escalation_id: uuid.UUID) -> EscalationDetail:
        response = await self._escalation.get(f"/escalations/{escalation_id}")
        response.raise_for_status()
        return EscalationDetail.model_validate(response.json())

    async def claim_escalation(
        self, escalation_id: uuid.UUID, request: ClaimRequest
    ) -> EscalationDetail:
        path = f"/escalations/{escalation_id}/claim"
        return await _post(self._escalation, path, request, EscalationDetail)

    async def resolve_escalation(
        self, escalation_id: uuid.UUID, request: ResolveRequest
    ) -> EscalationDetail:
        path = f"/escalations/{escalation_id}/resolve"
        return await _post(self._escalation, path, request, EscalationDetail)

    async def feedback_escalation(
        self, escalation_id: uuid.UUID, request: FeedbackRequest
    ) -> EscalationDetail:
        path = f"/escalations/{escalation_id}/feedback"
        return await _post(self._escalation, path, request, EscalationDetail)

    async def health(self) -> None:
        for client in self._clients:
            (await client.get("/health/live")).raise_for_status()

    async def aclose(self) -> None:
        for client in self._clients:
            await client.aclose()
