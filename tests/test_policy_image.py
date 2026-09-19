from __future__ import annotations

from inagecas_shared.config import get_settings
from inagecas_shared.schemas import Intent, RefusalReason, Route
from policy_service.policy import decide

settings = get_settings()


def test_image_required_refused_without_image():
    decision = decide(
        "see my screenshot", Intent.image_required, False, None, settings, has_image=False
    )
    assert decision.route == Route.refuse
    assert decision.refusal_reason == RefusalReason.image_unsupported


def test_image_required_answered_with_image():
    decision = decide(
        "see my screenshot", Intent.image_required, False, None, settings, has_image=True
    )
    assert decision.route == Route.answer
    assert decision.refusal_reason is None
