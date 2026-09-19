from __future__ import annotations

import httpx
from asgi_lifespan import LifespanManager

from answer_service.main import app
from factories import FakeGateway, make_evidence
from inagecas_shared.schemas import GenerateRequest


async def test_health_live():
    async with LifespanManager(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/health/live")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


async def test_generate_returns_grounded_answer():
    async with LifespanManager(app):
        app.state.gateway = FakeGateway(replies=["Paris is the capital [1].", "SUPPORTED"])
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            request = GenerateRequest(
                question="capital of France?",
                evidence=[make_evidence(score=0.9)],
                top_score=0.9,
            )
            response = await client.post("/generate", json=request.model_dump(mode="json"))
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "answered"
    assert body["citations"]
    # Security headers come from the shared middleware.
    assert response.headers["X-Content-Type-Options"] == "nosniff"
