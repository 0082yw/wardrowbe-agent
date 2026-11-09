"""add_ai_endpoints_to_preferences

Revision ID: 99f268e84ada
Revises: eb355c4bc653
Create Date: 2026-01-16 23:16:51.585119

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "99f268e84ada"
down_revision: str | None = "eb355c4bc653"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # Add ai_endpoints column to user_preferences
    op.add_column(
        "user_preferences",
        sa.Column(
            "ai_endpoints",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=True,
            server_default="[]",
        ),
    )


def downgrade() -> None:
    op.drop_column("user_preferences", "ai_endpoints")
