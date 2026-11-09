"""add_ai_description_to_items

Revision ID: df5d193d2b23
Revises: 99f268e84ada
Create Date: 2026-01-17 04:18:55.067969

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "df5d193d2b23"
down_revision: str | None = "99f268e84ada"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("clothing_items", sa.Column("ai_description", sa.Text(), nullable=True))


def downgrade() -> None:
    op.drop_column("clothing_items", "ai_description")
