from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import UUID

# revision identifiers, used by Alembic.
revision: str = "a1b2c3d4e5f6"
down_revision: str | None = "9a4d3f6e8c12"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # Add wash tracking columns to clothing_items
    op.add_column(
        "clothing_items",
        sa.Column("wears_since_wash", sa.Integer(), server_default="0", nullable=False),
    )
    op.add_column("clothing_items", sa.Column("last_washed_at", sa.Date(), nullable=True))
    op.add_column("clothing_items", sa.Column("wash_interval", sa.Integer(), nullable=True))
    op.add_column(
        "clothing_items",
        sa.Column("needs_wash", sa.Boolean(), server_default="false", nullable=False),
    )

    # Create wash_history table
    op.create_table(
        "wash_history",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "item_id",
            UUID(as_uuid=True),
            sa.ForeignKey("clothing_items.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("washed_at", sa.Date(), nullable=False),
        sa.Column("method", sa.String(50), nullable=True),
        sa.Column("notes", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_wash_history_item_washed", "wash_history", ["item_id", "washed_at"])


def downgrade() -> None:
    op.drop_index("ix_wash_history_item_washed", table_name="wash_history")
    op.drop_table("wash_history")
    op.drop_column("clothing_items", "needs_wash")
    op.drop_column("clothing_items", "wash_interval")
    op.drop_column("clothing_items", "last_washed_at")
    op.drop_column("clothing_items", "wears_since_wash")
