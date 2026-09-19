"""initial schema

Revision ID: 0001
Revises:
Create Date: 2026-06-24
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "documents",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("source_type", sa.String(32), nullable=False),
        sa.Column("source_uri", sa.String(1024), nullable=False),
        sa.Column("title", sa.String(512), nullable=True),
        sa.Column("status", sa.String(32), nullable=False, server_default="active"),
        sa.Column("created_at", sa.DateTime(), server_default=sa.text("now()"), nullable=False),
        sa.Column("updated_at", sa.DateTime(), server_default=sa.text("now()"), nullable=False),
    )
    op.create_index("ix_documents_status", "documents", ["status"])

    op.create_table(
        "document_versions",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("document_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column("content_hash", sa.String(64), nullable=False),
        sa.Column("byte_size", sa.BigInteger(), nullable=False),
        sa.Column("created_at", sa.DateTime(), server_default=sa.text("now()"), nullable=False),
        sa.ForeignKeyConstraint(["document_id"], ["documents.id"], ondelete="CASCADE"),
        sa.UniqueConstraint("document_id", "version", name="uq_document_version"),
    )
    op.create_index("ix_document_versions_document_id", "document_versions", ["document_id"])
    op.create_index("ix_document_versions_content_hash", "document_versions", ["content_hash"])

    op.create_table(
        "chunks",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("document_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("document_version_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("ordinal", sa.Integer(), nullable=False),
        sa.Column("heading", sa.String(512), nullable=True),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("token_estimate", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(), server_default=sa.text("now()"), nullable=False),
        sa.ForeignKeyConstraint(["document_id"], ["documents.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(
            ["document_version_id"], ["document_versions.id"], ondelete="CASCADE"
        ),
    )
    op.create_index("ix_chunks_document_id", "chunks", ["document_id"])
    op.create_index("ix_chunks_document_version_id", "chunks", ["document_version_id"])

    op.create_table(
        "ingestion_runs",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("document_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("status", sa.String(32), nullable=False, server_default="queued"),
        sa.Column("source_type", sa.String(32), nullable=False),
        sa.Column("source_uri", sa.String(1024), nullable=False),
        sa.Column("chunks_indexed", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(), server_default=sa.text("now()"), nullable=False),
        sa.Column("finished_at", sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(["document_id"], ["documents.id"], ondelete="SET NULL"),
    )
    op.create_index("ix_ingestion_runs_status", "ingestion_runs", ["status"])

    op.create_table(
        "retrieval_runs",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("query", sa.Text(), nullable=False),
        sa.Column("top_k", sa.Integer(), nullable=False),
        sa.Column("min_score", sa.Float(), nullable=False),
        sa.Column("result_count", sa.Integer(), nullable=False),
        sa.Column("top_score", sa.Float(), nullable=True),
        sa.Column("latency_ms", sa.Integer(), nullable=True),
        sa.Column("created_at", sa.DateTime(), server_default=sa.text("now()"), nullable=False),
    )

    op.create_table(
        "answer_runs",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("retrieval_run_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("question", sa.Text(), nullable=False),
        sa.Column("status", sa.String(32), nullable=False),
        sa.Column("refusal_reason", sa.String(64), nullable=True),
        sa.Column("answer", sa.Text(), nullable=True),
        sa.Column("confidence", sa.Float(), nullable=True),
        sa.Column("model", sa.String(128), nullable=False),
        sa.Column("citations", postgresql.JSONB(), nullable=False, server_default="[]"),
        sa.Column("latency_ms", sa.Integer(), nullable=True),
        sa.Column("created_at", sa.DateTime(), server_default=sa.text("now()"), nullable=False),
        sa.ForeignKeyConstraint(["retrieval_run_id"], ["retrieval_runs.id"], ondelete="SET NULL"),
    )
    op.create_index("ix_answer_runs_status", "answer_runs", ["status"])


def downgrade() -> None:
    op.drop_table("answer_runs")
    op.drop_table("retrieval_runs")
    op.drop_table("ingestion_runs")
    op.drop_table("chunks")
    op.drop_table("document_versions")
    op.drop_table("documents")
