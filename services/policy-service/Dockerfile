# syntax=docker/dockerfile:1
# Build context is the repository root so the shared package is available.
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=0
WORKDIR /app

COPY pyproject.toml uv.lock ./
COPY packages/shared/pyproject.toml packages/shared/pyproject.toml
COPY services/api-gateway/pyproject.toml services/api-gateway/pyproject.toml
COPY services/answer-service/pyproject.toml services/answer-service/pyproject.toml
COPY services/retrieval-service/pyproject.toml services/retrieval-service/pyproject.toml
COPY services/ingestion-worker/pyproject.toml services/ingestion-worker/pyproject.toml
COPY services/policy-service/pyproject.toml services/policy-service/pyproject.toml
COPY services/vision-service/pyproject.toml services/vision-service/pyproject.toml
COPY services/escalation-service/pyproject.toml services/escalation-service/pyproject.toml
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-workspace --package policy-service

COPY packages/shared packages/shared
COPY services/policy-service services/policy-service
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-editable --package policy-service

FROM python:3.12-slim-bookworm AS runtime
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1
RUN groupadd --system app && useradd --system --gid app --home /app app
WORKDIR /app

COPY --from=builder /app/.venv /app/.venv

USER app
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=5s --start-period=40s --retries=5 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health/live').status==200 else 1)"
CMD ["uvicorn", "policy_service.main:app", "--host", "0.0.0.0", "--port", "8000"]
