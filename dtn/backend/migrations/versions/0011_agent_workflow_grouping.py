"""Associate Agent conversations and autonomy loops with a workflow run."""

from alembic import op
import sqlalchemy as sa


revision = "0011_agent_workflow_grouping"
down_revision = "0010_agent_runs"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "slm_conversations",
        sa.Column("workflow_id", sa.String(length=36), nullable=True),
    )
    op.create_index(
        "ix_slm_conversations_workflow_id",
        "slm_conversations",
        ["workflow_id"],
    )
    op.add_column(
        "autonomy_loops",
        sa.Column("workflow_id", sa.String(length=36), nullable=True),
    )
    op.create_index(
        "ix_autonomy_loops_workflow_id",
        "autonomy_loops",
        ["workflow_id"],
    )


def downgrade() -> None:
    op.drop_index("ix_autonomy_loops_workflow_id", table_name="autonomy_loops")
    op.drop_column("autonomy_loops", "workflow_id")
    op.drop_index("ix_slm_conversations_workflow_id", table_name="slm_conversations")
    op.drop_column("slm_conversations", "workflow_id")
