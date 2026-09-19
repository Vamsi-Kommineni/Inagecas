from __future__ import annotations

import uuid

import httpx
import respx

from api_gateway.clients import InternalClient
from inagecas_shared.config import get_settings
from inagecas_shared.schemas import EscalationCreate, EscalationReason, EscalationStatus


@respx.mock
async def test_filing_an_escalation_retries_once_after_a_dropped_connection():
    settings = get_settings().model_copy(
        update={"escalation_service_url": "http://escalation.test"}
    )
    created = {"escalation_id": str(uuid.uuid4()), "status": "open"}
    route = respx.post("http://escalation.test/escalations").mock(
        side_effect=[httpx.ConnectError("reset"), httpx.Response(201, json=created)]
    )
    client = InternalClient(settings)
    try:
        result = await client.create_escalation(
            EscalationCreate(question="q", reason=EscalationReason.no_evidence)
        )
    finally:
        await client.aclose()

    assert result.status == EscalationStatus.open
    assert route.call_count == 2
