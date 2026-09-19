"""answer validation columns

Revision ID: 0004
Revises: 0003
Create Date: 2026-06-24
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0004"
down_revision: str | None = "0003"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("answer_runs", sa.Column("coverage", sa.Float(), nullable=True))
    op.add_column("answer_runs", sa.Column("support", sa.String(32), nullable=True))


def downgrade() -> None:
    op.drop_column("answer_runs", "support")
    op.drop_column("answer_runs", "coverage")
