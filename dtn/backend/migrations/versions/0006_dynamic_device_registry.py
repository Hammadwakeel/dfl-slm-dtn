"""Support managed device registration from the DTN dashboard."""

from alembic import op
import sqlalchemy as sa


revision = "0006_dynamic_device_registry"
down_revision = "0005_slm_conversation_history"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "devices",
        sa.Column(
            "is_managed",
            sa.Boolean(),
            nullable=False,
            server_default=sa.false(),
        ),
    )
    op.add_column(
        "devices",
        sa.Column("api_token", sa.Text(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("devices", "api_token")
    op.drop_column("devices", "is_managed")
