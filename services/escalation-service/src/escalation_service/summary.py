"""Assemble a human-readable context summary for an escalation.

Deterministic first: an agent gets the key facts (conversation, question, why
it escalated, the draft, the evidence, and trace ids) without a model call.
With ``INAGECAS_ESCALATION_LLM_SUMMARY`` on, a short narrative from the chat
model goes on top; if that call fails the agent still gets the facts.
"""

from __future__ import annotations

from inagecas_shared.guardrails import redact_pii
from inagecas_shared.logging import get_logger
from inagecas_shared.model_gateway import ModelGateway
from inagecas_shared.schemas import EscalationCreate

log = get_logger("escalation")

_NARRATE_SYSTEM = (
    "You brief a human support agent taking over from an AI assistant. In at most "
    "three plain sentences say what the customer needs, what the assistant found "
    "and why it handed over. Facts only, from the hand-off notes; no greeting."
)


def build_summary(payload: EscalationCreate) -> str:
    lines = [f"{turn.role}: {turn.content.strip()}" for turn in payload.history]
    lines += [
        f"Question: {payload.question.strip()}",
        f"Escalated because: {payload.reason.value}",
    ]
    if payload.channel:
        lines.append(f"Channel: {payload.channel}")
    lines.append(
        f"Draft answer: {payload.draft_answer.strip() if payload.draft_answer else 'none'}"
    )
    if payload.image_note:
        lines.append(f"Image: {payload.image_note}")
    if payload.screenshot_base64:
        lines.append("Screenshot: attached")

    if payload.citations:
        sources = sorted({c.title or c.source_uri for c in payload.citations})
        lines.append(f"Evidence: {len(payload.citations)} passage(s) from {', '.join(sources)}")
    else:
        lines.append("Evidence: none retrieved")

    trace = [
        f"{name}={value}"
        for name, value in (
            ("answer_run", payload.answer_run_id),
            ("retrieval_run", payload.retrieval_run_id),
            ("policy_run", payload.policy_run_id),
            ("image_run", payload.image_run_id),
        )
        if value is not None
    ]
    if trace:
        lines.append("Trace: " + " ".join(trace))
    return "\n".join(lines)


async def narrate(gateway: ModelGateway, notes: str) -> str | None:
    """A three-sentence briefing written from the deterministic notes.

    The agent sees the notes as the customer wrote them; the model gets them
    with personal data masked, like every other prompt that leaves the machine.
    """
    try:
        text = await gateway.chat(
            [
                {"role": "system", "content": _NARRATE_SYSTEM},
                {"role": "user", "content": redact_pii(notes)},
            ],
            temperature=0.1,
            max_tokens=200,
        )
    except Exception as exc:
        log.warning("escalation_narrative_failed", error=str(exc))
        return None
    return text.strip() or None
