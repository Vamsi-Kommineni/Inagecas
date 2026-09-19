from __future__ import annotations

from prometheus_client import REGISTRY

from inagecas_shared import metrics
from inagecas_shared.telemetry import set_span_attributes, traced


def _value(name: str, labels: dict[str, str]) -> float:
    return REGISTRY.get_sample_value(name, labels) or 0.0


def test_record_answer_increments_counter():
    labels = {"status": "answered", "refusal_reason": "none"}
    before = _value("inagecas_answers_total", labels)
    metrics.record_answer("answered", None, 0.9)
    assert _value("inagecas_answers_total", labels) == before + 1


def test_record_refusal_uses_reason_label():
    labels = {"status": "refused", "refusal_reason": "unsupported"}
    before = _value("inagecas_answers_total", labels)
    metrics.record_answer("refused", "unsupported", 0.2)
    assert _value("inagecas_answers_total", labels) == before + 1


def test_domain_counters_record():
    metrics.record_escalation("sensitive")
    metrics.record_validation("supported")
    metrics.record_ocr(True, 0.95)
    metrics.record_retrieval(0.8, 3)
    assert _value("inagecas_escalations_total", {"reason": "sensitive"}) >= 1
    assert _value("inagecas_validation_verdict_total", {"verdict": "supported"}) >= 1
    assert _value("inagecas_ocr_total", {"readable": "true"}) >= 1
    assert _value("inagecas_retrieval_results_count", {}) >= 1


def test_tracing_helpers_are_safe_without_a_provider():
    with traced("stage", route="answer", missing=None) as span:
        assert span is not None
    # Should be a no-op, not raise, when there is no active recording span.
    set_span_attributes(status="answered", nothing=None)
