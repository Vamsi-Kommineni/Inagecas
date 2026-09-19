"""policy runs and routing

Revision ID: 0002
Revises: 0001
Create Date: 2026-06-24
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0002"
down_revision: str | None = "0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "policy_runs",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("question", sa.Text(), nullable=False),
        sa.Column("route", sa.String(32), nullable=False),
        sa.Column("intent", sa.String(32), nullable=False),
        sa.Column("complexity", sa.Integer(), nullable=False),
        sa.Column("pii_detected", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("refusal_reason", sa.String(64), nullable=True),
        sa.Column("matched_rule", sa.String(128), nullable=True),
        sa.Column("latency_ms", sa.Integer(), nullable=True),
        sa.Column("created_at", sa.DateTime(), server_default=sa.text("now()"), nullable=False),
    )
    op.create_index("ix_policy_runs_route", "policy_runs", ["route"])

    op.add_column(
        "answer_runs",
        sa.Column("policy_run_id", postgresql.UUID(as_uuid=True), nullable=True),
    )
    op.create_foreign_key(
        "fk_answer_runs_policy_run_id",
        "answer_runs",
        "policy_runs",
        ["policy_run_id"],
        ["id"],
        ondelete="SET NULL",
    )


def downgrade() -> None:
    op.drop_constraint("fk_answer_runs_policy_run_id", "answer_runs", type_="foreignkey")
    op.drop_column("answer_runs", "policy_run_id")
    op.drop_table("policy_runs")
