"""Create device registry and status history tables."""

from alembic import op
import sqlalchemy as sa

revision = "0001_device_registry_history"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "devices",
        sa.Column("node_id", sa.String(length=32), primary_key=True),
        sa.Column("role", sa.String(length=16), nullable=False),
        sa.Column("hostname", sa.String(length=128), nullable=False),
        sa.Column("ip", sa.String(length=64), nullable=False),
        sa.Column("port", sa.Integer(), nullable=False),
        sa.Column("online", sa.Boolean(), nullable=False),
        sa.Column("last_seen", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_error", sa.String(length=512), nullable=True),
        sa.Column("latest_status", sa.JSON(), nullable=True),
    )
    op.create_table(
        "device_status_history",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column(
            "node_id",
            sa.String(length=32),
            sa.ForeignKey("devices.node_id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("sampled_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("cpu_utilization_percent", sa.Float(), nullable=True),
        sa.Column("gpu_utilization_percent", sa.Float(), nullable=True),
        sa.Column("memory", sa.JSON(), nullable=False),
        sa.Column("temperatures_celsius", sa.JSON(), nullable=False),
        sa.Column("payload", sa.JSON(), nullable=False),
    )
    op.create_index(
        "ix_device_status_history_node_id",
        "device_status_history",
        ["node_id"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("ix_device_status_history_node_id", table_name="device_status_history")
    op.drop_table("device_status_history")
    op.drop_table("devices")
