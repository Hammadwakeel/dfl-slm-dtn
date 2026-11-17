"""Track training endpoints and dynamic-worker readiness."""

from alembic import op
import sqlalchemy as sa


revision = "0007_dynamic_training_workers"
down_revision = "0006_dynamic_device_registry"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("devices", sa.Column("training_port", sa.Integer(), nullable=True))
    op.add_column(
        "devices",
        sa.Column(
            "training_ready",
            sa.Boolean(),
            nullable=False,
            server_default=sa.false(),
        ),
    )
    op.add_column("devices", sa.Column("training_error", sa.String(length=512), nullable=True))


def downgrade() -> None:
    op.drop_column("devices", "training_error")
    op.drop_column("devices", "training_ready")
    op.drop_column("devices", "training_port")
