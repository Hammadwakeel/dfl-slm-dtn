"""Persist SLM assistant conversations and replies."""

from alembic import op
import sqlalchemy as sa


revision = "0005_slm_conversation_history"
down_revision = "0004_apg_run_audit_context"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "slm_conversations",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("title", sa.String(length=120), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_slm_conversations_updated_at",
        "slm_conversations",
        ["updated_at"],
    )
    op.create_table(
        "slm_messages",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("conversation_id", sa.String(length=36), nullable=False),
        sa.Column("role", sa.String(length=16), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("response_data", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["conversation_id"],
            ["slm_conversations.id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_slm_messages_conversation_id",
        "slm_messages",
        ["conversation_id"],
    )


def downgrade() -> None:
    op.drop_index("ix_slm_messages_conversation_id", table_name="slm_messages")
    op.drop_table("slm_messages")
    op.drop_index(
        "ix_slm_conversations_updated_at",
        table_name="slm_conversations",
    )
    op.drop_table("slm_conversations")
