"""escalations and agent actions

Revision ID: 0005
Revises: 0004
Create Date: 2026-06-24
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0005"
down_revision: str | None = "0004"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "escalations",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("status", sa.String(32), nullable=False, server_default="open"),
        sa.Column("reason", sa.String(32), nullable=False),
        sa.Column("channel", sa.String(64), nullable=True),
        sa.Column("customer_ref", sa.String(128), nullable=True),
        sa.Column("question", sa.Text(), nullable=False),
        sa.Column("draft_answer", sa.Text(), nullable=True),
        sa.Column("summary", sa.Text(), nullable=False),
        sa.Column("image_note", sa.String(1024), nullable=True),
        sa.Column("citations", postgresql.JSONB(), nullable=False, server_default="[]"),
        sa.Column("assignee", sa.String(128), nullable=True),
        sa.Column("resolution", sa.Text(), nullable=True),
        sa.Column("policy_run_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("retrieval_run_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("answer_run_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("image_run_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("created_at", sa.DateTime(), server_default=sa.text("now()"), nullable=False),
        sa.Column("updated_at", sa.DateTime(), server_default=sa.text("now()"), nullable=False),
        sa.Column("resolved_at", sa.DateTime(), nullable=True),
    )
    op.create_index("ix_escalations_status", "escalations", ["status"])

    op.create_table(
        "agent_actions",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("escalation_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("agent", sa.String(128), nullable=False),
        sa.Column("action", sa.String(32), nullable=False),
        sa.Column("feedback_kind", sa.String(32), nullable=True),
        sa.Column("note", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(), server_default=sa.text("now()"), nullable=False),
        sa.ForeignKeyConstraint(["escalation_id"], ["escalations.id"], ondelete="CASCADE"),
    )
    op.create_index("ix_agent_actions_escalation_id", "agent_actions", ["escalation_id"])


def downgrade() -> None:
    op.drop_table("agent_actions")
    op.drop_table("escalations")
