"""Initial database schema.

Revision ID: 001_initial_schema
Revises:
Create Date: 2024-01-16

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "001_initial_schema"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # Create enum types
    op.execute("CREATE TYPE item_status AS ENUM ('processing', 'ready', 'error', 'archived')")
    op.execute(
        "CREATE TYPE outfit_status AS ENUM ('pending', 'sent', 'viewed', 'accepted', 'rejected', 'expired')"
    )
    op.execute("CREATE TYPE outfit_source AS ENUM ('scheduled', 'on_demand', 'manual')")
    op.execute(
        "CREATE TYPE notification_status AS ENUM ('pending', 'sent', 'delivered', 'failed', 'retrying')"
    )

    # Create families table
    op.create_table(
        "families",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("name", sa.String(100), nullable=False),
        sa.Column("created_by", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("invite_code", sa.String(20), nullable=False),
        sa.Column("settings", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("invite_code"),
    )
    op.create_index("idx_families_invite_code", "families", ["invite_code"])
    op.create_index("idx_families_created_by", "families", ["created_by"])

    # Create users table
    op.create_table(
        "users",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("family_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("external_id", sa.String(255), nullable=False),
        sa.Column("email", sa.String(255), nullable=False),
        sa.Column("display_name", sa.String(100), nullable=False),
        sa.Column("avatar_url", sa.String(500), nullable=True),
        sa.Column("role", sa.String(20), nullable=False, server_default="member"),
        sa.Column("timezone", sa.String(50), nullable=False, server_default="UTC"),
        sa.Column("location_lat", sa.Numeric(10, 8), nullable=True),
        sa.Column("location_lon", sa.Numeric(11, 8), nullable=True),
        sa.Column("location_name", sa.String(100), nullable=True),
        sa.Column("is_active", sa.Boolean, nullable=False, server_default="true"),
        sa.Column("last_login_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("onboarding_completed", sa.Boolean, nullable=False, server_default="false"),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("external_id"),
        sa.UniqueConstraint("email"),
        sa.ForeignKeyConstraint(["family_id"], ["families.id"], ondelete="SET NULL"),
    )
    op.create_index("idx_users_family_id", "users", ["family_id"])
    op.create_index("idx_users_external_id", "users", ["external_id"])
    op.create_index("idx_users_email", "users", ["email"])

    # Add foreign key from families.created_by to users.id
    op.create_foreign_key(
        "fk_families_created_by",
        "families",
        "users",
        ["created_by"],
        ["id"],
    )

    # Create user_preferences table
    op.create_table(
        "user_preferences",
        sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("color_favorites", postgresql.ARRAY(sa.String), nullable=True),
        sa.Column("color_avoid", postgresql.ARRAY(sa.String), nullable=True),
        sa.Column("style_profile", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("default_occasion", sa.String(50), nullable=True, server_default="casual"),
        sa.Column("occasion_preferences", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("temperature_sensitivity", sa.String(20), nullable=True, server_default="normal"),
        sa.Column("cold_threshold", sa.Integer, nullable=True, server_default="10"),
        sa.Column("hot_threshold", sa.Integer, nullable=True, server_default="25"),
        sa.Column("layering_preference", sa.String(20), nullable=True, server_default="moderate"),
        sa.Column("avoid_repeat_days", sa.Integer, nullable=True, server_default="7"),
        sa.Column("prefer_underused_items", sa.Boolean, nullable=True, server_default="true"),
        sa.Column("variety_level", sa.String(20), nullable=True, server_default="moderate"),
        sa.Column(
            "excluded_item_ids", postgresql.ARRAY(postgresql.UUID(as_uuid=True)), nullable=True
        ),
        sa.Column("excluded_combinations", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("user_id"),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
    )

    # Create notification_settings table
    op.create_table(
        "notification_settings",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("channel", sa.String(20), nullable=False),
        sa.Column("enabled", sa.Boolean, nullable=False, server_default="true"),
        sa.Column("config", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("priority", sa.Integer, nullable=False, server_default="1"),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.UniqueConstraint("user_id", "channel"),
    )
    op.create_index("idx_notification_settings_user_id", "notification_settings", ["user_id"])

    # Create schedules table
    op.create_table(
        "schedules",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("day_of_week", sa.Integer, nullable=False),
        sa.Column("notification_time", sa.Time, nullable=False),
        sa.Column("occasion", sa.String(50), nullable=False, server_default="casual"),
        sa.Column("enabled", sa.Boolean, nullable=False, server_default="true"),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.UniqueConstraint("user_id", "day_of_week"),
    )
    op.create_index("idx_schedules_user_day", "schedules", ["user_id", "day_of_week"])

    # Create clothing_items table
    op.create_table(
        "clothing_items",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("image_path", sa.String(500), nullable=False),
        sa.Column("thumbnail_path", sa.String(500), nullable=True),
        sa.Column("medium_path", sa.String(500), nullable=True),
        sa.Column("type", sa.String(50), nullable=False),
        sa.Column("subtype", sa.String(50), nullable=True),
        sa.Column("tags", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("colors", postgresql.ARRAY(sa.String), nullable=True),
        sa.Column("primary_color", sa.String(50), nullable=True),
        sa.Column(
            "status",
            postgresql.ENUM(
                "processing", "ready", "error", "archived", name="item_status", create_type=False
            ),
            nullable=False,
            server_default="processing",
        ),
        sa.Column("ai_confidence", sa.Numeric(3, 2), nullable=True),
        sa.Column("ai_raw_response", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("wear_count", sa.Integer, nullable=False, server_default="0"),
        sa.Column("last_worn_at", sa.Date, nullable=True),
        sa.Column("last_suggested_at", sa.Date, nullable=True),
        sa.Column("suggestion_count", sa.Integer, nullable=False, server_default="0"),
        sa.Column("acceptance_count", sa.Integer, nullable=False, server_default="0"),
        sa.Column("name", sa.String(100), nullable=True),
        sa.Column("brand", sa.String(100), nullable=True),
        sa.Column("purchase_date", sa.Date, nullable=True),
        sa.Column("purchase_price", sa.Numeric(10, 2), nullable=True),
        sa.Column("notes", sa.Text, nullable=True),
        sa.Column("favorite", sa.Boolean, nullable=False, server_default="false"),
        sa.Column("is_archived", sa.Boolean, nullable=False, server_default="false"),
        sa.Column("archived_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("archive_reason", sa.String(50), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
    )
    op.create_index("idx_clothing_items_user_id", "clothing_items", ["user_id"])
    op.create_index("idx_clothing_items_user_type", "clothing_items", ["user_id", "type"])
    op.create_index("idx_clothing_items_user_status", "clothing_items", ["user_id", "status"])
    op.create_index(
        "idx_clothing_items_colors", "clothing_items", ["colors"], postgresql_using="gin"
    )
    op.create_index("idx_clothing_items_tags", "clothing_items", ["tags"], postgresql_using="gin")

    # Create outfits table
    op.create_table(
        "outfits",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("weather_data", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("occasion", sa.String(50), nullable=False),
        sa.Column("scheduled_for", sa.Date, nullable=False),
        sa.Column("reasoning", sa.Text, nullable=True),
        sa.Column("style_notes", sa.Text, nullable=True),
        sa.Column("ai_raw_response", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column(
            "status",
            postgresql.ENUM(
                "pending",
                "sent",
                "viewed",
                "accepted",
                "rejected",
                "expired",
                name="outfit_status",
                create_type=False,
            ),
            nullable=False,
            server_default="pending",
        ),
        sa.Column(
            "source",
            postgresql.ENUM(
                "scheduled",
                "on_demand",
                "manual",
                name="outfit_source",
                create_type=False,
            ),
            nullable=False,
            server_default="scheduled",
        ),
        sa.Column("sent_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("viewed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("responded_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
    )
    op.create_index("idx_outfits_user_id", "outfits", ["user_id"])
    op.create_index("idx_outfits_user_date", "outfits", ["user_id", "scheduled_for"])
    op.create_index("idx_outfits_user_status", "outfits", ["user_id", "status"])

    # Create outfit_items table
    op.create_table(
        "outfit_items",
        sa.Column("outfit_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("item_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("position", sa.Integer, nullable=False),
        sa.Column("layer_type", sa.String(20), nullable=True),
        sa.PrimaryKeyConstraint("outfit_id", "item_id"),
        sa.ForeignKeyConstraint(["outfit_id"], ["outfits.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["item_id"], ["clothing_items.id"], ondelete="CASCADE"),
    )
    op.create_index("idx_outfit_items_item_id", "outfit_items", ["item_id"])

    # Create user_feedback table
    op.create_table(
        "user_feedback",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("outfit_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("accepted", sa.Boolean, nullable=True),
        sa.Column("rating", sa.Integer, nullable=True),
        sa.Column("comfort_rating", sa.Integer, nullable=True),
        sa.Column("style_rating", sa.Integer, nullable=True),
        sa.Column("comment", sa.Text, nullable=True),
        sa.Column("worn_at", sa.Date, nullable=True),
        sa.Column("worn_with_modifications", sa.Boolean, nullable=True, server_default="false"),
        sa.Column("modification_notes", sa.Text, nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(["outfit_id"], ["outfits.id"], ondelete="CASCADE"),
    )
    op.create_index("idx_user_feedback_outfit_id", "user_feedback", ["outfit_id"], unique=True)

    # Create item_history table
    op.create_table(
        "item_history",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("item_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("outfit_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("worn_at", sa.Date, nullable=False),
        sa.Column("occasion", sa.String(50), nullable=True),
        sa.Column("notes", sa.Text, nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(["item_id"], ["clothing_items.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["outfit_id"], ["outfits.id"], ondelete="SET NULL"),
    )
    op.create_index("idx_item_history_item_id", "item_history", ["item_id"])
    op.create_index("idx_item_history_worn_at", "item_history", ["item_id", "worn_at"])
    op.create_index("idx_item_history_outfit_id", "item_history", ["outfit_id"])

    # Create notifications table
    op.create_table(
        "notifications",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("outfit_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("channel", sa.String(20), nullable=False),
        sa.Column(
            "status",
            postgresql.ENUM(
                "pending",
                "sent",
                "delivered",
                "failed",
                "retrying",
                name="notification_status",
                create_type=False,
            ),
            nullable=False,
            server_default="pending",
        ),
        sa.Column("attempts", sa.Integer, nullable=False, server_default="0"),
        sa.Column("max_attempts", sa.Integer, nullable=False, server_default="3"),
        sa.Column("last_attempt_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("sent_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("delivered_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("payload", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("error_message", sa.Text, nullable=True),
        sa.Column("error_details", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["outfit_id"], ["outfits.id"], ondelete="SET NULL"),
    )
    op.create_index("idx_notifications_user_id", "notifications", ["user_id"])
    op.create_index("idx_notifications_outfit_id", "notifications", ["outfit_id"])
    op.create_index("idx_notifications_user_created", "notifications", ["user_id", "created_at"])

    # Create family_invites table
    op.create_table(
        "family_invites",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("family_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("email", sa.String(255), nullable=False),
        sa.Column("token", sa.String(100), nullable=False),
        sa.Column("invited_by", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("role", sa.String(20), nullable=False, server_default="member"),
        sa.Column("accepted_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("token"),
        sa.ForeignKeyConstraint(["family_id"], ["families.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["invited_by"], ["users.id"]),
    )
    op.create_index("idx_family_invites_token", "family_invites", ["token"])
    op.create_index("idx_family_invites_email", "family_invites", ["email"])
    op.create_index("idx_family_invites_family_id", "family_invites", ["family_id"])

    # Create update_updated_at function
    op.execute("""
        CREATE OR REPLACE FUNCTION update_updated_at_column()
        RETURNS TRIGGER AS $$
        BEGIN
            NEW.updated_at = NOW();
            RETURN NEW;
        END;
        $$ language 'plpgsql';
    """)

    # Apply triggers to tables with updated_at
    tables_with_updated_at = [
        "families",
        "users",
        "user_preferences",
        "notification_settings",
        "schedules",
        "clothing_items",
        "outfits",
        "notifications",
    ]
    for table in tables_with_updated_at:
        op.execute(f"""
            CREATE TRIGGER update_{table}_updated_at
            BEFORE UPDATE ON {table}
            FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
        """)


def downgrade() -> None:
    # Drop tables in reverse order
    op.drop_table("family_invites")
    op.drop_table("notifications")
    op.drop_table("item_history")
    op.drop_table("user_feedback")
    op.drop_table("outfit_items")
    op.drop_table("outfits")
    op.drop_table("clothing_items")
    op.drop_table("schedules")
    op.drop_table("notification_settings")
    op.drop_table("user_preferences")

    # Drop the foreign key from families before dropping users
    op.drop_constraint("fk_families_created_by", "families", type_="foreignkey")
    op.drop_table("users")
    op.drop_table("families")

    # Drop function
    op.execute("DROP FUNCTION IF EXISTS update_updated_at_column()")

    # Drop enum types
    op.execute("DROP TYPE IF EXISTS notification_status")
    op.execute("DROP TYPE IF EXISTS outfit_source")
    op.execute("DROP TYPE IF EXISTS outfit_status")
    op.execute("DROP TYPE IF EXISTS item_status")
