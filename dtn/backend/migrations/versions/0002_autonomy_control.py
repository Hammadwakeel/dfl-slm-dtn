"""Add persistent autonomy state, decisions, history, and loops."""

from alembic import op
import sqlalchemy as sa

revision = "0002_autonomy_control"
down_revision = "0001_device_registry_history"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "autonomy_state",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("mode", sa.String(length=16), nullable=False),
        sa.Column("stop_requested", sa.Boolean(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_table(
        "autonomy_decisions",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column("action", sa.String(length=64), nullable=False),
        sa.Column("parameters", sa.JSON(), nullable=False),
        sa.Column("reason", sa.Text(), nullable=False),
        sa.Column("evidence", sa.JSON(), nullable=False),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index(
        "ix_autonomy_decisions_status",
        "autonomy_decisions",
        ["status"],
        unique=False,
    )
    op.create_table(
        "autonomy_decision_history",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column(
            "decision_id",
            sa.String(length=36),
            sa.ForeignKey("autonomy_decisions.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("event", sa.String(length=32), nullable=False),
        sa.Column("details", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index(
        "ix_autonomy_decision_history_decision_id",
        "autonomy_decision_history",
        ["decision_id"],
        unique=False,
    )
    op.create_table(
        "autonomy_loops",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("max_iterations", sa.Integer(), nullable=False),
        sa.Column("iteration", sa.Integer(), nullable=False),
        sa.Column("time_limit_seconds", sa.Integer(), nullable=False),
        sa.Column("stop_rule", sa.String(length=32), nullable=False),
        sa.Column("target_metric", sa.Float(), nullable=True),
        sa.Column("best_metric", sa.Float(), nullable=True),
        sa.Column("no_improvement_count", sa.Integer(), nullable=False),
        sa.Column("failure_reason", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("deadline_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_autonomy_loops_status", "autonomy_loops", ["status"], unique=False)


def downgrade() -> None:
    op.drop_index("ix_autonomy_loops_status", table_name="autonomy_loops")
    op.drop_table("autonomy_loops")
    op.drop_index(
        "ix_autonomy_decision_history_decision_id",
        table_name="autonomy_decision_history",
    )
    op.drop_table("autonomy_decision_history")
    op.drop_index("ix_autonomy_decisions_status", table_name="autonomy_decisions")
    op.drop_table("autonomy_decisions")
    op.drop_table("autonomy_state")
