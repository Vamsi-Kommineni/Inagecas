"""Typed configuration shared by every service.

All settings come from environment variables prefixed with ``INAGECAS_`` (or a
local ``.env`` file during development). Secrets are never hard-coded; the
defaults here are safe placeholders for local runs only.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

Environment = Literal["development", "staging", "production"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="INAGECAS_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    environment: Environment = "development"
    log_level: str = "INFO"

    # Authentication. ``api_keys`` and ``cors_origins`` are comma-separated.
    api_keys: str = ""
    internal_token: str = ""
    cors_origins: str = ""

    # Datastores.
    database_url: str = "postgresql+asyncpg://inagecas:inagecas@localhost:5432/inagecas"
    redis_url: str = "redis://localhost:6379/0"
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str = ""
    qdrant_collection: str = "inagecas_chunks"

    # Model gateway (LiteLLM, OpenAI-compatible) and model aliases. What each
    # alias resolves to is LiteLLM's business (infra/litellm/config.yaml).
    model_gateway_url: str = "http://localhost:4000"
    model_gateway_api_key: str = ""
    chat_model: str = "chat"
    # The side model for short verdicts: grounding judge, intent classifier.
    # Pointing it at a different model than ``chat`` gives an independent judge.
    judge_model: str = "judge"
    # Vision model that reads screenshots. Empty means OCR only.
    vision_model: str = "vision"
    embed_model: str = "embed"
    embed_dim: int = 768
    rerank_strategy: Literal["none", "llm", "cross_encoder"] = "cross_encoder"
    cross_encoder_model: str = "Xenova/ms-marco-MiniLM-L-6-v2"

    # Hybrid retrieval and staleness.
    hybrid_enabled: bool = True
    sparse_model: str = "Qdrant/bm25"
    retrieval_candidate_k: int = 30
    staleness_days: int = 0

    # Retrieval and answering behaviour.
    retrieval_top_k: int = 5
    retrieval_min_score: float = 0.45
    answer_temperature: float = 0.1
    answer_max_tokens: int = 512
    answer_min_confidence: float = 0.5

    # Answer validation.
    validation_enabled: bool = True
    # A draft whose cited share of prose sentences is below this is refused.
    answer_min_coverage: float = 0.5
    clarification_enabled: bool = True
    # Below min_confidence but at or above this band, ask a clarifying question
    # instead of refusing outright.
    clarification_band_low: float = 0.35

    # Escalation to a human queue. The LLM summary is a short
    # narrative for the agent on top of the deterministic one; off by default.
    escalation_enabled: bool = True
    escalation_llm_summary: bool = False

    # Repeated questions reuse the retrieval result for this long (0 disables).
    retrieval_cache_ttl: int = 300

    # Policy and routing.
    policy_enabled: bool = True
    policy_classify: bool = True
    policy_sensitive_action: Literal["escalate", "answer"] = "escalate"

    # Screenshots.
    vision_min_confidence: float = 0.5
    vision_min_chars: int = 8
    image_max_bytes: int = 6_291_456

    # Internal service URLs used by the gateway.
    retrieval_service_url: str = "http://localhost:8001"
    answer_service_url: str = "http://localhost:8002"
    policy_service_url: str = "http://localhost:8003"
    vision_service_url: str = "http://localhost:8004"
    escalation_service_url: str = "http://localhost:8005"

    # Request hardening. The body cap is generous enough for a base64 screenshot.
    request_max_bytes: int = 8_388_608
    rate_limit: str = "60/minute"

    @property
    def api_key_set(self) -> set[str]:
        return {key.strip() for key in self.api_keys.split(",") if key.strip()}

    @property
    def cors_origin_list(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]

    @property
    def is_production(self) -> bool:
        return self.environment == "production"


@lru_cache
def get_settings() -> Settings:
    return Settings()
