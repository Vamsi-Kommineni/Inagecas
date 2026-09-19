"""Pydantic schemas for public API requests and internal service contracts."""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field


class SourceType(StrEnum):
    markdown = "markdown"
    html = "html"
    pdf = "pdf"
    text = "text"


class AnswerStatus(StrEnum):
    answered = "answered"
    refused = "refused"
    clarification = "clarification"


class RefusalReason(StrEnum):
    no_evidence = "no_evidence"
    low_confidence = "low_confidence"
    unsupported = "unsupported"
    contradicted = "contradicted"
    policy_blocked = "policy_blocked"
    off_topic = "off_topic"
    sensitive = "sensitive"
    image_unsupported = "image_unsupported"
    image_unreadable = "image_unreadable"


class Route(StrEnum):
    answer = "answer"
    refuse = "refuse"
    escalate = "escalate"
    block = "block"


class Intent(StrEnum):
    how_to = "how_to"
    troubleshooting = "troubleshooting"
    billing_account = "billing_account"
    sensitive = "sensitive"
    image_required = "image_required"
    off_topic = "off_topic"
    unknown = "unknown"


class Turn(BaseModel):
    """One earlier message in the conversation: who said it and what."""

    role: Literal["customer", "assistant"]
    content: str = Field(min_length=1, max_length=4000)


# --- Ingestion -------------------------------------------------------------


class IngestRequest(BaseModel):
    title: str | None = Field(default=None, max_length=512)
    source_type: SourceType = SourceType.markdown
    source_uri: str = Field(default="inline", max_length=1024)
    content: str = Field(min_length=1, max_length=2_000_000)


class IngestResponse(BaseModel):
    ingestion_run_id: uuid.UUID
    status: str


class IngestionStatus(BaseModel):
    ingestion_run_id: uuid.UUID
    status: str
    document_id: uuid.UUID | None = None
    chunks_indexed: int = 0
    error: str | None = None


# --- Retrieval (internal) --------------------------------------------------


class RetrieveRequest(BaseModel):
    query: str = Field(min_length=1, max_length=8192)
    top_k: int | None = Field(default=None, ge=1, le=50)
    min_score: float | None = Field(default=None, ge=0.0, le=1.0)
    filters: dict[str, str] = Field(default_factory=dict)


class Evidence(BaseModel):
    chunk_id: uuid.UUID
    document_id: uuid.UUID
    source_uri: str
    title: str | None = None
    ordinal: int
    score: float
    text: str
    stale: bool = False


class RetrieveResponse(BaseModel):
    retrieval_run_id: uuid.UUID
    query: str
    evidence: list[Evidence]
    top_score: float | None = None
    source_count: int = 0


# --- Answering (internal) --------------------------------------------------


class GenerateRequest(BaseModel):
    question: str = Field(min_length=1, max_length=8192)
    evidence: list[Evidence]
    top_score: float | None = None
    # Text read off a screenshot the customer attached, if any.
    screenshot_text: str | None = Field(default=None, max_length=4000)


class Citation(BaseModel):
    chunk_id: uuid.UUID
    document_id: uuid.UUID
    source_uri: str
    title: str | None = None
    ordinal: int
    score: float

    @classmethod
    def from_evidence(cls, item: Evidence) -> Citation:
        return cls(
            chunk_id=item.chunk_id,
            document_id=item.document_id,
            source_uri=item.source_uri,
            title=item.title,
            ordinal=item.ordinal,
            score=item.score,
        )


class GenerateResponse(BaseModel):
    status: AnswerStatus
    answer: str | None = None
    refusal_reason: RefusalReason | None = None
    confidence: float
    citations: list[Citation] = Field(default_factory=list)
    coverage: float = 0.0
    support: str | None = None
    clarification: str | None = None
    # What the model wrote before validation refused it. For the human queue,
    # never for the customer.
    draft: str | None = None


# --- Vision (internal) -----------------------------------------------------


class ImageExtractRequest(BaseModel):
    image_base64: str = Field(min_length=1, max_length=15_000_000)


class ImageLine(BaseModel):
    text: str
    confidence: float
    box: list[list[float]] = Field(default_factory=list)


class ImageFacts(BaseModel):
    image_run_id: uuid.UUID
    readable: bool
    visual_confidence: float
    ocr_text: str = ""
    detected_error: str | None = None
    screen_title: str | None = None
    lines: list[ImageLine] = Field(default_factory=list)


# --- Policy (internal) -----------------------------------------------------


class PolicyEvaluateRequest(BaseModel):
    question: str = Field(min_length=1, max_length=8192)
    has_image: bool = False
    # Earlier turns and text read off the screenshot: screened for injection
    # like the question.
    history: list[Turn] = Field(default_factory=list, max_length=20)
    screenshot_text: str | None = Field(default=None, max_length=4000)


class PolicyDecision(BaseModel):
    policy_run_id: uuid.UUID
    route: Route
    intent: Intent
    complexity: int = Field(ge=1, le=5)
    pii_detected: bool = False
    refusal_reason: RefusalReason | None = None
    message: str | None = None


# --- Ask (public) ----------------------------------------------------------


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=8192)
    # Earlier turns of the same conversation, oldest first, so a follow-up like
    # "and on the annual plan?" can be understood.
    history: list[Turn] = Field(default_factory=list, max_length=20)
    top_k: int | None = Field(default=None, ge=1, le=50)
    image_base64: str | None = Field(default=None, max_length=15_000_000)
    channel: str | None = Field(default=None, max_length=64)
    customer_ref: str | None = Field(default=None, max_length=128)


class AskResponse(BaseModel):
    status: AnswerStatus
    answer: str | None = None
    refusal_reason: RefusalReason | None = None
    # A short note for the customer explaining a refusal.
    message: str | None = None
    confidence: float
    citations: list[Citation] = Field(default_factory=list)
    route: Route | None = None
    coverage: float = 0.0
    support: str | None = None
    clarification: str | None = None
    retrieval_run_id: uuid.UUID | None = None
    answer_run_id: uuid.UUID | None = None
    policy_run_id: uuid.UUID | None = None
    image_run_id: uuid.UUID | None = None
    escalation_id: uuid.UUID | None = None


# --- Escalation ------------------------------------------------------------


class EscalationStatus(StrEnum):
    open = "open"
    in_progress = "in_progress"
    resolved = "resolved"


class EscalationReason(StrEnum):
    sensitive = "sensitive"
    no_evidence = "no_evidence"
    low_confidence = "low_confidence"
    unsupported = "unsupported"
    contradicted = "contradicted"


class FeedbackKind(StrEnum):
    answer_useful = "answer_useful"
    bad_retrieval = "bad_retrieval"
    missing_docs = "missing_docs"
    other = "other"


class EscalationCreate(BaseModel):
    question: str
    history: list[Turn] = Field(default_factory=list)
    reason: EscalationReason
    channel: str | None = None
    customer_ref: str | None = None
    draft_answer: str | None = None
    image_note: str | None = None
    screenshot_base64: str | None = Field(default=None, max_length=15_000_000)
    citations: list[Citation] = Field(default_factory=list)
    policy_run_id: uuid.UUID | None = None
    retrieval_run_id: uuid.UUID | None = None
    answer_run_id: uuid.UUID | None = None
    image_run_id: uuid.UUID | None = None


class EscalationCreated(BaseModel):
    escalation_id: uuid.UUID
    status: EscalationStatus


class AgentActionOut(BaseModel):
    agent: str
    action: str
    feedback_kind: FeedbackKind | None = None
    note: str | None = None
    created_at: datetime


class EscalationSummary(BaseModel):
    id: uuid.UUID
    status: EscalationStatus
    reason: EscalationReason
    question: str
    assignee: str | None = None
    created_at: datetime


class EscalationDetail(BaseModel):
    id: uuid.UUID
    status: EscalationStatus
    reason: EscalationReason
    channel: str | None = None
    customer_ref: str | None = None
    question: str
    history: list[Turn] = Field(default_factory=list)
    draft_answer: str | None = None
    summary: str
    image_note: str | None = None
    screenshot_base64: str | None = None
    citations: list[Citation] = Field(default_factory=list)
    assignee: str | None = None
    resolution: str | None = None
    policy_run_id: uuid.UUID | None = None
    retrieval_run_id: uuid.UUID | None = None
    answer_run_id: uuid.UUID | None = None
    image_run_id: uuid.UUID | None = None
    created_at: datetime
    resolved_at: datetime | None = None
    actions: list[AgentActionOut] = Field(default_factory=list)


class ClaimRequest(BaseModel):
    agent: str = Field(min_length=1, max_length=128)


class ResolveRequest(BaseModel):
    agent: str = Field(min_length=1, max_length=128)
    resolution: str = Field(min_length=1, max_length=8192)


class FeedbackRequest(BaseModel):
    agent: str = Field(min_length=1, max_length=128)
    kind: FeedbackKind
    note: str | None = Field(default=None, max_length=8192)
