"""add_image_hash_column

Revision ID: add_image_hash
Revises: df5d193d2b23
Create Date: 2026-01-17

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "add_image_hash"
down_revision: str | None = "df5d193d2b23"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("clothing_items", sa.Column("image_hash", sa.String(length=16), nullable=True))
    op.create_index("idx_clothing_items_image_hash", "clothing_items", ["image_hash"])


def downgrade() -> None:
    op.drop_index("idx_clothing_items_image_hash", "clothing_items")
    op.drop_column("clothing_items", "image_hash")
