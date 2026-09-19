"""OpenTelemetry tracing setup.

Tracing only turns on when an OTLP endpoint is configured, so the services run
cleanly without the observability stack (for example in tests or a bare local
run). Metrics are exposed separately via Prometheus in ``service.py``.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

_configured = False


def telemetry_enabled() -> bool:
    return bool(os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT"))


def setup_tracing(service_name: str) -> None:
    global _configured
    if _configured or not telemetry_enabled():
        return

    provider = TracerProvider(resource=Resource.create({"service.name": service_name}))
    provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
    trace.set_tracer_provider(provider)
    _configured = True


def instrument_app(app: Any) -> None:
    if not telemetry_enabled():
        return
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
    from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor

    FastAPIInstrumentor.instrument_app(app)
    HTTPXClientInstrumentor().instrument()


def instrument_engine(engine: Any) -> None:
    if not telemetry_enabled():
        return
    from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor

    SQLAlchemyInstrumentor().instrument(engine=engine.sync_engine)


@contextmanager
def traced(name: str, **attributes: Any) -> Iterator[Any]:
    """Open a child span for a pipeline stage. A no-op when tracing is off."""
    tracer = trace.get_tracer("inagecas")
    with tracer.start_as_current_span(name) as span:
        _apply(span, attributes)
        yield span


def set_span_attributes(**attributes: Any) -> None:
    """Annotate the current span with decision context (route, status, ...)."""
    _apply(trace.get_current_span(), attributes)


def _apply(span: Any, attributes: dict[str, Any]) -> None:
    for key, value in attributes.items():
        if value is not None:
            span.set_attribute(key, str(value))
