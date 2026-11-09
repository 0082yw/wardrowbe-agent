"""add_ai_tagging_fields

Revision ID: 534aab99f47c
Revises: 001_initial_schema
Create Date: 2026-01-16 13:42:21.905887

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "534aab99f47c"
down_revision: str | None = "001_initial_schema"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # Add new columns for AI tagging
    op.add_column("clothing_items", sa.Column("pattern", sa.String(length=50), nullable=True))
    op.add_column("clothing_items", sa.Column("material", sa.String(length=50), nullable=True))
    op.add_column(
        "clothing_items",
        sa.Column("style", postgresql.ARRAY(sa.String()), server_default="{}", nullable=True),
    )
    op.add_column("clothing_items", sa.Column("formality", sa.String(length=50), nullable=True))
    op.add_column(
        "clothing_items",
        sa.Column("season", postgresql.ARRAY(sa.String()), server_default="{}", nullable=True),
    )
    op.add_column(
        "clothing_items",
        sa.Column("ai_processed", sa.Boolean(), server_default="false", nullable=False),
    )


def downgrade() -> None:
    op.drop_column("clothing_items", "ai_processed")
    op.drop_column("clothing_items", "season")
    op.drop_column("clothing_items", "formality")
    op.drop_column("clothing_items", "style")
    op.drop_column("clothing_items", "material")
    op.drop_column("clothing_items", "pattern")
