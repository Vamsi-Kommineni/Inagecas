"""Grounded answer generation with validation and refusal control.

An answer is only returned when there is evidence, retrieval clears the
confidence floor, the model does not signal insufficient context, and the draft
passes validation (support + coverage) with a composed confidence above the
threshold. Borderline cases become a clarifying question; contradicted or
unsupported drafts become a refusal.
"""

from __future__ import annotations

from inagecas_shared import metrics
from inagecas_shared.config import Settings
from inagecas_shared.model_gateway import ModelGateway
from inagecas_shared.schemas import (
    AnswerStatus,
    Citation,
    Evidence,
    GenerateRequest,
    GenerateResponse,
    RefusalReason,
)
from inagecas_shared.text import cited_indices, has_prose, normalize_citations

from .prompts import build_messages, signals_insufficient
from .validation import (
    Verdict,
    check_text,
    citation_coverage,
    compose_confidence,
    generate_clarification,
    settle_verdict,
    support_score,
    verify_support,
)


def _refuse(
    reason: RefusalReason,
    confidence: float,
    *,
    coverage: float = 0.0,
    verdict: Verdict | None = None,
    draft: str | None = None,
) -> GenerateResponse:
    return GenerateResponse(
        status=AnswerStatus.refused,
        refusal_reason=reason,
        confidence=confidence,
        coverage=coverage,
        support=verdict.value if verdict else None,
        draft=draft,
    )


def _citations(answer: str, evidence: list[Evidence]) -> list[Citation]:
    indices = sorted(i for i in cited_indices(answer) if 1 <= i <= len(evidence))
    return [Citation.from_evidence(evidence[i - 1]) for i in indices]


async def generate(
    gateway: ModelGateway,
    settings: Settings,
    request: GenerateRequest,
) -> GenerateResponse:
    if not request.evidence:
        return _refuse(RefusalReason.no_evidence, 0.0)

    top_score = request.top_score if request.top_score is not None else request.evidence[0].score
    retrieval_confidence = round(min(max(top_score, 0.0), 1.0), 4)
    if retrieval_confidence < settings.answer_min_confidence:
        return _refuse(RefusalReason.low_confidence, retrieval_confidence)

    raw = await gateway.chat(
        build_messages(request.question, request.evidence, request.screenshot_text),
        temperature=settings.answer_temperature,
        max_tokens=settings.answer_max_tokens,
    )
    # An answer of nothing but citation markers cites every passage and states
    # nothing, so it has to be refused rather than scored.
    answer = normalize_citations(raw).strip()
    if not has_prose(answer) or signals_insufficient(answer):
        return _refuse(RefusalReason.unsupported, 0.0)

    coverage = citation_coverage(answer, request.evidence)
    # An answer that does not say where its claims come from is not grounded,
    # however good the passages were.
    if coverage < settings.answer_min_coverage:
        return _refuse(RefusalReason.unsupported, 0.0, coverage=coverage, draft=answer)

    source_count = len({item.document_id for item in request.evidence})
    verdict = Verdict.supported
    if settings.validation_enabled:
        judged = await verify_support(gateway, answer, request.evidence, model=settings.judge_model)
        text = check_text(
            answer,
            request.evidence,
            context=f"{request.question}\n{request.screenshot_text or ''}",
        )
        verdict = settle_verdict(judged, text)
    metrics.record_validation(verdict.value)

    confidence = compose_confidence(
        retrieval_confidence, source_count, coverage, support_score(verdict)
    )

    # A contradicted or unsupported draft is refused regardless of retrieval score.
    if verdict in (Verdict.contradicted, Verdict.unsupported):
        return _refuse(
            RefusalReason(verdict.value),
            confidence,
            coverage=coverage,
            verdict=verdict,
            draft=answer,
        )

    if confidence < settings.answer_min_confidence:
        if settings.clarification_enabled and confidence >= settings.clarification_band_low:
            clarification = await generate_clarification(gateway, request.question)
            return GenerateResponse(
                status=AnswerStatus.clarification,
                clarification=clarification,
                confidence=confidence,
                coverage=coverage,
                support=verdict.value,
            )
        return _refuse(
            RefusalReason.low_confidence,
            confidence,
            coverage=coverage,
            verdict=verdict,
            draft=answer,
        )

    return GenerateResponse(
        status=AnswerStatus.answered,
        answer=answer,
        confidence=confidence,
        citations=_citations(answer, request.evidence),
        coverage=coverage,
        support=verdict.value,
    )
