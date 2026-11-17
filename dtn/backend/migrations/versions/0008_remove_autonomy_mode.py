"""Remove the approval-mode state switch."""

from alembic import op
import sqlalchemy as sa


revision = "0008_remove_autonomy_mode"
down_revision = "0007_dynamic_training_workers"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.drop_column("autonomy_state", "mode")


def downgrade() -> None:
    op.add_column(
        "autonomy_state",
        sa.Column("mode", sa.String(length=16), nullable=False, server_default="FULL_ACCESS"),
    )
