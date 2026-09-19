"""Combine guardrails and classification into a route decision."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from time import perf_counter

from sqlalchemy.ext.asyncio import AsyncSession

from inagecas_shared.config import Settings
from inagecas_shared.guardrails import detect_pii, matched_block, redact_pii
from inagecas_shared.ids import new_id
from inagecas_shared.model_gateway import ModelGateway
from inagecas_shared.models import PolicyRun
from inagecas_shared.schemas import Intent, PolicyDecision, RefusalReason, Route, Turn

from .classifier import classify

_ANSWERABLE = {Intent.how_to, Intent.troubleshooting, Intent.billing_account, Intent.unknown}


@dataclass(slots=True)
class Decision:
    route: Route
    intent: Intent
    complexity: int
    pii_detected: bool
    refusal_reason: RefusalReason | None
    matched_rule: str | None
    message: str | None


def estimate_complexity(question: str) -> int:
    words = len(question.split())
    for threshold, score in ((12, 1), (30, 2), (60, 3), (100, 4)):
        if words <= threshold:
            return score
    return 5


def _route_for_intent(
    intent: Intent, settings: Settings, has_image: bool = False
) -> tuple[Route, RefusalReason | None, str | None]:
    if intent in _ANSWERABLE:
        return Route.answer, None, None
    if intent == Intent.sensitive:
        if settings.policy_sensitive_action == "answer":
            return Route.answer, None, None
        return (
            Route.escalate,
            RefusalReason.sensitive,
            "This request involves sensitive data and is being routed to a human agent.",
        )
    if intent == Intent.image_required:
        if has_image:
            return Route.answer, None, None
        return (
            Route.refuse,
            RefusalReason.image_unsupported,
            "This question needs a screenshot; please attach one.",
        )
    return (
        Route.refuse,
        RefusalReason.off_topic,
        "This question is outside the scope of the supported documentation.",
    )


def decide(
    question: str,
    intent: Intent,
    pii_detected: bool,
    blocked_rule: str | None,
    settings: Settings,
    has_image: bool = False,
) -> Decision:
    complexity = estimate_complexity(question)
    if blocked_rule is not None:
        return Decision(
            route=Route.block,
            intent=intent,
            complexity=complexity,
            pii_detected=pii_detected,
            refusal_reason=RefusalReason.policy_blocked,
            matched_rule=blocked_rule,
            message="This request was blocked by policy.",
        )
    route, reason, message = _route_for_intent(intent, settings, has_image)
    return Decision(
        route=route,
        intent=intent,
        complexity=complexity,
        pii_detected=pii_detected,
        refusal_reason=reason,
        matched_rule=None,
        message=message,
    )


async def evaluate(
    session: AsyncSession,
    gateway: ModelGateway,
    settings: Settings,
    question: str,
    has_image: bool = False,
    screenshot_text: str | None = None,
    history: Sequence[Turn] = (),
) -> PolicyDecision:
    started = perf_counter()
    pii = detect_pii(question)
    # A screenshot and the earlier turns are outside text too: "ignore previous
    # instructions" there is the same attack as in the question.
    outside = [question, screenshot_text, *(turn.content for turn in history)]
    blocked_rule = next((rule for text in outside if (rule := matched_block(text))), None)
    # Don't spend an LLM call classifying a request we are about to block.
    if blocked_rule is not None or not settings.policy_classify:
        intent = Intent.unknown
    else:
        intent = await classify(gateway, redact_pii(question), model=settings.judge_model)

    decision = decide(question, intent, bool(pii), blocked_rule, settings, has_image)
    latency_ms = int((perf_counter() - started) * 1000)

    run = PolicyRun(
        id=new_id(),
        question=question,
        route=decision.route.value,
        intent=decision.intent.value,
        complexity=decision.complexity,
        pii_detected=decision.pii_detected,
        refusal_reason=decision.refusal_reason.value if decision.refusal_reason else None,
        matched_rule=decision.matched_rule,
        latency_ms=latency_ms,
    )
    session.add(run)
    await session.commit()

    return PolicyDecision(
        policy_run_id=run.id,
        route=decision.route,
        intent=decision.intent,
        complexity=decision.complexity,
        pii_detected=decision.pii_detected,
        refusal_reason=decision.refusal_reason,
        message=decision.message,
    )
