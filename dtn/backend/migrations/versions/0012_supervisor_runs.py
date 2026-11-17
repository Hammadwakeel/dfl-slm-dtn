"""Store the supervisor agent's goal and resumable workflow state on agent runs."""

from alembic import op
import sqlalchemy as sa


revision = "0012_supervisor_runs"
down_revision = "0011_agent_workflow_grouping"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("agent_runs") as batch:
        batch.add_column(sa.Column("goal", sa.JSON(), nullable=True))
        batch.add_column(sa.Column("state", sa.JSON(), nullable=True))
    # Earlier browser-driven loops cannot resume under the server-side supervisor.
    op.execute(
        "UPDATE autonomy_loops SET status = 'STOPPED', "
        "failure_reason = 'Superseded by the server-side supervisor (v2)' "
        "WHERE status = 'RUNNING'"
    )


def downgrade() -> None:
    with op.batch_alter_table("agent_runs") as batch:
        batch.drop_column("state")
        batch.drop_column("goal")
