"""Domain metrics for the pipeline.

These live in the default Prometheus registry, so they are exposed on each
service's ``/metrics`` endpoint alongside the HTTP metrics. Each service only
records the ones it owns (the gateway records answers/escalations, retrieval
records scores, and so on).
"""

from __future__ import annotations

from prometheus_client import Counter, Histogram

_UNIT_BUCKETS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
_COUNT_BUCKETS = (0, 1, 2, 3, 5, 8, 13, 21)

answers_total = Counter(
    "inagecas_answers_total",
    "Answer outcomes by status and refusal reason.",
    ["status", "refusal_reason"],
)
answer_confidence = Histogram(
    "inagecas_answer_confidence", "Composed answer confidence.", buckets=_UNIT_BUCKETS
)
escalations_total = Counter(
    "inagecas_escalations_total", "Escalations created, by reason.", ["reason"]
)
escalation_failures_total = Counter(
    "inagecas_escalation_failures_total",
    "Escalations that could not be filed, by reason. Each one is a lost hand-off.",
    ["reason"],
)
retrieval_top_score = Histogram(
    "inagecas_retrieval_top_score", "Top retrieval score per query.", buckets=_UNIT_BUCKETS
)
retrieval_results = Histogram(
    "inagecas_retrieval_results", "Evidence passages returned per query.", buckets=_COUNT_BUCKETS
)
validation_verdict_total = Counter(
    "inagecas_validation_verdict_total", "Answer validation verdicts.", ["verdict"]
)
ocr_total = Counter("inagecas_ocr_total", "OCR extractions, by readability.", ["readable"])
ocr_confidence = Histogram(
    "inagecas_ocr_confidence", "Mean OCR confidence per image.", buckets=_UNIT_BUCKETS
)


def record_answer(status: str, refusal_reason: str | None, confidence: float) -> None:
    answers_total.labels(status=status, refusal_reason=refusal_reason or "none").inc()
    answer_confidence.observe(confidence)


def record_escalation(reason: str) -> None:
    escalations_total.labels(reason=reason).inc()


def record_escalation_failure(reason: str) -> None:
    escalation_failures_total.labels(reason=reason).inc()


def record_retrieval(top_score: float | None, result_count: int) -> None:
    if top_score is not None:
        retrieval_top_score.observe(top_score)
    retrieval_results.observe(result_count)


def record_validation(verdict: str) -> None:
    validation_verdict_total.labels(verdict=verdict).inc()


def record_ocr(readable: bool, confidence: float) -> None:
    ocr_total.labels(readable=str(readable).lower()).inc()
    ocr_confidence.observe(confidence)
