"""SQLAlchemy ORM models.

Each row records part of a request's journey so any answer or refusal can be
explained and replayed later: documents and their versions, the chunks we
indexed, and a run record for ingestion, retrieval, and answering.
"""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import (
    BigInteger,
    Boolean,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


def _pk() -> Mapped[uuid.UUID]:
    return mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)


class Document(Base):
    __tablename__ = "documents"

    id: Mapped[uuid.UUID] = _pk()
    source_type: Mapped[str] = mapped_column(String(32))
    source_uri: Mapped[str] = mapped_column(String(1024))
    title: Mapped[str | None] = mapped_column(String(512))
    status: Mapped[str] = mapped_column(String(32), default="active", index=True)
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(server_default=func.now(), onupdate=func.now())

    versions: Mapped[list[DocumentVersion]] = relationship(
        back_populates="document", cascade="all, delete-orphan"
    )


class DocumentVersion(Base):
    __tablename__ = "document_versions"
    __table_args__ = (UniqueConstraint("document_id", "version", name="uq_document_version"),)

    id: Mapped[uuid.UUID] = _pk()
    document_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"), index=True
    )
    version: Mapped[int] = mapped_column(Integer)
    content_hash: Mapped[str] = mapped_column(String(64), index=True)
    byte_size: Mapped[int] = mapped_column(BigInteger)
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())

    document: Mapped[Document] = relationship(back_populates="versions")


class Chunk(Base):
    __tablename__ = "chunks"

    # The chunk id doubles as the Qdrant point id, keeping the two stores 1:1.
    id: Mapped[uuid.UUID] = _pk()
    document_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"), index=True
    )
    document_version_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("document_versions.id", ondelete="CASCADE"), index=True
    )
    ordinal: Mapped[int] = mapped_column(Integer)
    heading: Mapped[str | None] = mapped_column(String(512))
    text: Mapped[str] = mapped_column(Text)
    token_estimate: Mapped[int] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())


class IngestionRun(Base):
    __tablename__ = "ingestion_runs"

    id: Mapped[uuid.UUID] = _pk()
    document_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("documents.id", ondelete="SET NULL")
    )
    status: Mapped[str] = mapped_column(String(32), default="queued", index=True)
    source_type: Mapped[str] = mapped_column(String(32))
    source_uri: Mapped[str] = mapped_column(String(1024))
    chunks_indexed: Mapped[int] = mapped_column(Integer, default=0)
    error: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())
    finished_at: Mapped[datetime | None] = mapped_column()


class RetrievalRun(Base):
    __tablename__ = "retrieval_runs"

    id: Mapped[uuid.UUID] = _pk()
    query: Mapped[str] = mapped_column(Text)
    top_k: Mapped[int] = mapped_column(Integer)
    min_score: Mapped[float] = mapped_column(Float)
    result_count: Mapped[int] = mapped_column(Integer)
    top_score: Mapped[float | None] = mapped_column(Float)
    latency_ms: Mapped[int | None] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())


class PolicyRun(Base):
    __tablename__ = "policy_runs"

    id: Mapped[uuid.UUID] = _pk()
    question: Mapped[str] = mapped_column(Text)
    route: Mapped[str] = mapped_column(String(32), index=True)
    intent: Mapped[str] = mapped_column(String(32))
    complexity: Mapped[int] = mapped_column(Integer)
    pii_detected: Mapped[bool] = mapped_column(Boolean, default=False)
    refusal_reason: Mapped[str | None] = mapped_column(String(64))
    matched_rule: Mapped[str | None] = mapped_column(String(128))
    latency_ms: Mapped[int | None] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())


class ImageRun(Base):
    __tablename__ = "image_runs"

    id: Mapped[uuid.UUID] = _pk()
    char_count: Mapped[int] = mapped_column(Integer, default=0)
    line_count: Mapped[int] = mapped_column(Integer, default=0)
    readable: Mapped[bool] = mapped_column(Boolean, default=False)
    visual_confidence: Mapped[float] = mapped_column(Float, default=0.0)
    detected_error: Mapped[str | None] = mapped_column(String(512))
    screen_title: Mapped[str | None] = mapped_column(String(512))
    latency_ms: Mapped[int | None] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())


class AnswerRun(Base):
    __tablename__ = "answer_runs"

    id: Mapped[uuid.UUID] = _pk()
    retrieval_run_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("retrieval_runs.id", ondelete="SET NULL")
    )
    policy_run_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("policy_runs.id", ondelete="SET NULL")
    )
    image_run_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("image_runs.id", ondelete="SET NULL")
    )
    question: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(32), index=True)
    refusal_reason: Mapped[str | None] = mapped_column(String(64))
    answer: Mapped[str | None] = mapped_column(Text)
    confidence: Mapped[float | None] = mapped_column(Float)
    coverage: Mapped[float | None] = mapped_column(Float)
    support: Mapped[str | None] = mapped_column(String(32))
    model: Mapped[str] = mapped_column(String(128))
    citations: Mapped[list[dict]] = mapped_column(JSONB, default=list)
    latency_ms: Mapped[int | None] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())


class Escalation(Base):
    __tablename__ = "escalations"

    id: Mapped[uuid.UUID] = _pk()
    status: Mapped[str] = mapped_column(String(32), default="open", index=True)
    reason: Mapped[str] = mapped_column(String(32))
    channel: Mapped[str | None] = mapped_column(String(64))
    customer_ref: Mapped[str | None] = mapped_column(String(128))
    question: Mapped[str] = mapped_column(Text)
    # Earlier turns of the conversation, so the agent never asks the customer
    # to repeat themselves.
    history: Mapped[list[dict]] = mapped_column(JSONB, default=list)
    draft_answer: Mapped[str | None] = mapped_column(Text)
    summary: Mapped[str] = mapped_column(Text)
    image_note: Mapped[str | None] = mapped_column(String(1024))
    # The customer's screenshot, base64, so the agent sees what the AI saw.
    screenshot: Mapped[str | None] = mapped_column(Text)
    citations: Mapped[list[dict]] = mapped_column(JSONB, default=list)
    assignee: Mapped[str | None] = mapped_column(String(128))
    resolution: Mapped[str | None] = mapped_column(Text)
    # Trace references to the AI runs that led here (not enforced as FKs).
    policy_run_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True))
    retrieval_run_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True))
    answer_run_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True))
    image_run_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True))
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(server_default=func.now(), onupdate=func.now())
    resolved_at: Mapped[datetime | None] = mapped_column()

    actions: Mapped[list[AgentAction]] = relationship(
        back_populates="escalation",
        cascade="all, delete-orphan",
        order_by="AgentAction.created_at",
    )


class AgentAction(Base):
    __tablename__ = "agent_actions"

    id: Mapped[uuid.UUID] = _pk()
    escalation_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("escalations.id", ondelete="CASCADE"), index=True
    )
    agent: Mapped[str] = mapped_column(String(128))
    action: Mapped[str] = mapped_column(String(32))
    feedback_kind: Mapped[str | None] = mapped_column(String(32))
    note: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())

    escalation: Mapped[Escalation] = relationship(back_populates="actions")
