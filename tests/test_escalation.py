from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, MagicMock

from api_gateway.routers.ask import _escalation_reason
from escalation_service.store import create_escalation, resolve
from escalation_service.summary import build_summary
from factories import make_evidence
from inagecas_shared.models import Escalation
from inagecas_shared.schemas import (
    Citation,
    EscalationCreate,
    EscalationReason,
    RefusalReason,
    ResolveRequest,
    Route,
)


def _citation(title: str = "Billing") -> Citation:
    return Citation.from_evidence(make_evidence(title=title))


def test_summary_includes_question_reason_and_evidence():
    payload = EscalationCreate(
        question="Why was I charged twice?",
        reason=EscalationReason.low_confidence,
        citations=[_citation()],
    )
    summary = build_summary(payload)
    assert "Question: Why was I charged twice?" in summary
    assert "Escalated because: low_confidence" in summary
    assert "Billing" in summary


def test_summary_notes_an_attached_screenshot():
    payload = EscalationCreate(
        question="Why this error?",
        reason=EscalationReason.unsupported,
        image_note="error: Invalid API key",
        screenshot_base64="Zm9v",
    )
    summary = build_summary(payload)
    assert "Image: error: Invalid API key" in summary
    assert "Screenshot: attached" in summary


def test_summary_notes_missing_draft_and_evidence():
    summary = build_summary(EscalationCreate(question="Hi", reason=EscalationReason.no_evidence))
    assert "Draft answer: none" in summary
    assert "Evidence: none retrieved" in summary


async def test_create_escalation_persists_open_item():
    session = MagicMock()
    session.add = MagicMock()
    session.commit = AsyncMock()

    payload = EscalationCreate(question="Q", reason=EscalationReason.unsupported)
    escalation = await create_escalation(session, payload, summary=build_summary(payload))

    session.add.assert_called_once()
    session.commit.assert_awaited_once()
    assert escalation.status == "open"
    assert escalation.reason == "unsupported"
    assert "Question: Q" in escalation.summary


async def test_resolve_reloads_actions_so_the_response_shows_the_new_one():
    """The resolve action used to appear only on the next GET."""
    escalation = Escalation(
        id=uuid.uuid4(), status="open", reason="unsupported", question="Q", summary="s"
    )
    session = MagicMock()
    session.add = MagicMock()
    session.commit = AsyncMock()
    session.expire = MagicMock()
    session.scalar = AsyncMock(return_value=escalation)

    result = await resolve(session, escalation.id, ResolveRequest(agent="a", resolution="done"))

    assert result is escalation
    assert escalation.status == "resolved"
    session.expire.assert_called_once_with(escalation, ["actions"])


def test_escalation_reason_maps_only_unhelpful_outcomes():
    assert _escalation_reason(Route.escalate, None) == EscalationReason.sensitive
    assert (
        _escalation_reason(Route.answer, RefusalReason.unsupported) == EscalationReason.unsupported
    )
    assert (
        _escalation_reason(Route.answer, RefusalReason.no_evidence) == EscalationReason.no_evidence
    )
    # Correct rejections are not escalated.
    assert _escalation_reason(Route.block, RefusalReason.policy_blocked) is None
    assert _escalation_reason(Route.refuse, RefusalReason.off_topic) is None
    assert _escalation_reason(Route.refuse, RefusalReason.image_unreadable) is None


def test_summary_opens_with_the_conversation():
    from inagecas_shared.schemas import Turn

    payload = EscalationCreate(
        question="And annual?",
        history=[
            Turn(role="customer", content="Refund on monthly?"),
            Turn(role="assistant", content="Monthly plans are non-refundable."),
        ],
        reason=EscalationReason.low_confidence,
    )
    summary = build_summary(payload)
    assert summary.startswith("customer: Refund on monthly?\nassistant: Monthly plans")
    assert "Question: And annual?" in summary


async def test_a_failed_narrative_leaves_the_notes_alone():
    from escalation_service.summary import narrate
    from factories import FakeGateway

    class Broken(FakeGateway):
        async def chat(self, *args, **kwargs):
            raise RuntimeError("provider down")

    assert await narrate(Broken(), "notes") is None
    assert await narrate(FakeGateway("  Customer wants a refund.  "), "notes") == (
        "Customer wants a refund."
    )


async def test_the_narrative_model_gets_masked_notes():
    from escalation_service.summary import narrate
    from factories import FakeGateway

    gateway = FakeGateway("Customer wants a refund.")
    await narrate(gateway, "Question: refund bob@example.com, card 4111 1111 1111 1111")

    sent = gateway.chat_calls[0][-1]["content"]
    assert "bob@example.com" not in sent and "4111" not in sent
    assert "[email]" in sent
