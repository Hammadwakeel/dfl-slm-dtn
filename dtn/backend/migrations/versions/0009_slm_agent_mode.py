"""Persist SLM conversation mode."""

from alembic import op
import sqlalchemy as sa


revision = "0009_slm_agent_mode"
down_revision = "0008_remove_autonomy_mode"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "slm_conversations",
        sa.Column(
            "mode",
            sa.String(length=12),
            nullable=False,
            server_default="human",
        ),
    )


def downgrade() -> None:
    op.drop_column("slm_conversations", "mode")
