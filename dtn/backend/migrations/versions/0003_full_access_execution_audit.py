"""Add persistent audit records for automatic FULL_ACCESS executions."""

from alembic import op
import sqlalchemy as sa


revision = "0003_full_access_execution_audit"
down_revision = "0002_autonomy_control"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "autonomy_execution_history",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("action", sa.String(length=64), nullable=False),
        sa.Column("mode", sa.String(length=16), nullable=False),
        sa.Column("timestamp", sa.DateTime(timezone=True), nullable=False),
        sa.Column("result_status", sa.String(length=16), nullable=False),
        sa.Column("dataset_name", sa.String(length=64), nullable=True),
    )


def downgrade() -> None:
    op.drop_table("autonomy_execution_history")