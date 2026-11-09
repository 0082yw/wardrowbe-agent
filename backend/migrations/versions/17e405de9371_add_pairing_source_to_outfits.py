"""add_pairing_source_to_outfits

Revision ID: 17e405de9371
Revises: 2a26ad24d04a
Create Date: 2026-01-18 10:18:06.639646

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import UUID

# revision identifiers, used by Alembic.
revision: str = "17e405de9371"
down_revision: str | None = "2a26ad24d04a"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # Add 'pairing' to outfit_source enum
    op.execute("ALTER TYPE outfit_source ADD VALUE IF NOT EXISTS 'pairing'")

    # Add source_item_id column (nullable FK to clothing_items)
    op.add_column("outfits", sa.Column("source_item_id", UUID(as_uuid=True), nullable=True))

    # Add foreign key constraint
    op.create_foreign_key(
        "fk_outfits_source_item_id",
        "outfits",
        "clothing_items",
        ["source_item_id"],
        ["id"],
        ondelete="SET NULL",
    )

    # Add index for efficient queries
    op.create_index("ix_outfits_source_item_id", "outfits", ["source_item_id"])


def downgrade() -> None:
    # Remove index
    op.drop_index("ix_outfits_source_item_id", table_name="outfits")

    # Remove foreign key
    op.drop_constraint("fk_outfits_source_item_id", "outfits", type_="foreignkey")

    # Remove column
    op.drop_column("outfits", "source_item_id")

    # Note: Cannot remove enum value in PostgreSQL, so 'pairing' will remain
