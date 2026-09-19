from __future__ import annotations

from api_gateway.rewrite import standalone_question
from factories import FakeGateway
from inagecas_shared.schemas import Turn

_HISTORY = [
    Turn(role="customer", content="Can I get a refund on a monthly plan?"),
    Turn(role="assistant", content="Monthly plans are non-refundable [1]."),
]


async def test_a_first_message_is_not_rewritten():
    gateway = FakeGateway('{"question": "never used"}')
    assert await standalone_question(gateway, "How do I upgrade?", [], model="judge") == (
        "How do I upgrade?"
    )
    assert gateway.chat_calls == []


async def test_a_follow_up_is_restated_from_the_conversation():
    gateway = FakeGateway('{"question": "Can I get a refund on an annual plan?"}')
    result = await standalone_question(gateway, "And on the annual one?", _HISTORY, model="judge")
    assert result == "Can I get a refund on an annual plan?"
    assert gateway.models_used == ["judge"]
    assert "Monthly plans are non-refundable" in gateway.chat_calls[0][-1]["content"]


async def test_an_empty_rewrite_keeps_the_original():
    gateway = FakeGateway('{"question": ""}')
    assert (
        await standalone_question(gateway, "And annual?", _HISTORY, model="judge") == "And annual?"
    )
