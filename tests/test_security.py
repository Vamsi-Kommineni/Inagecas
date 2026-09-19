from __future__ import annotations

import httpx
from fastapi import Depends, FastAPI

from inagecas_shared.security import require_api_key


def _build_app() -> FastAPI:
    app = FastAPI()

    @app.get("/protected", dependencies=[Depends(require_api_key)])
    async def protected() -> dict[str, bool]:
        return {"ok": True}

    return app


async def _client(app: FastAPI) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")


async def test_missing_key_is_unauthorized():
    async with await _client(_build_app()) as client:
        response = await client.get("/protected")
    assert response.status_code == 401


async def test_wrong_key_is_unauthorized():
    async with await _client(_build_app()) as client:
        response = await client.get("/protected", headers={"X-API-Key": "nope"})
    assert response.status_code == 401


async def test_valid_key_is_authorized():
    async with await _client(_build_app()) as client:
        response = await client.get("/protected", headers={"X-API-Key": "test-key"})
    assert response.status_code == 200
    assert response.json() == {"ok": True}
