"""Add run identifiers and configuration details to execution audit events."""

from alembic import op
import sqlalchemy as sa


revision = "0004_apg_run_audit_context"
down_revision = "0003_full_access_execution_audit"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "autonomy_execution_history",
        sa.Column("run_id", sa.String(length=64), nullable=True),
    )
    op.add_column(
        "autonomy_execution_history",
        sa.Column("details", sa.JSON(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("autonomy_execution_history", "details")
    op.drop_column("autonomy_execution_history", "run_id")
