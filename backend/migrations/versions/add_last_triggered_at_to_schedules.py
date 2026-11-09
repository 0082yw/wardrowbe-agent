"""add last_triggered_at to schedules

Revision ID: 9a1b2c3d4e5f
Revises: 534aab99f47c
Create Date: 2026-01-17

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "eb355c4bc653"
down_revision: str | None = "534aab99f47c"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # Add last_triggered_at column to track when schedule last sent a notification
    # This prevents duplicate notifications if the worker runs multiple times
    op.add_column(
        "schedules", sa.Column("last_triggered_at", sa.DateTime(timezone=True), nullable=True)
    )


def downgrade() -> None:
    op.drop_column("schedules", "last_triggered_at")
