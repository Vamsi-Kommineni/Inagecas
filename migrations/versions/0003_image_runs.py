"""image runs for screenshots

Revision ID: 0003
Revises: 0002
Create Date: 2026-06-24
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0003"
down_revision: str | None = "0002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "image_runs",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("char_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("line_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("readable", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("visual_confidence", sa.Float(), nullable=False, server_default="0"),
        sa.Column("detected_error", sa.String(512), nullable=True),
        sa.Column("screen_title", sa.String(512), nullable=True),
        sa.Column("latency_ms", sa.Integer(), nullable=True),
        sa.Column("created_at", sa.DateTime(), server_default=sa.text("now()"), nullable=False),
    )

    op.add_column(
        "answer_runs",
        sa.Column("image_run_id", postgresql.UUID(as_uuid=True), nullable=True),
    )
    op.create_foreign_key(
        "fk_answer_runs_image_run_id",
        "answer_runs",
        "image_runs",
        ["image_run_id"],
        ["id"],
        ondelete="SET NULL",
    )


def downgrade() -> None:
    op.drop_constraint("fk_answer_runs_image_run_id", "answer_runs", type_="foreignkey")
    op.drop_column("answer_runs", "image_run_id")
    op.drop_table("image_runs")
