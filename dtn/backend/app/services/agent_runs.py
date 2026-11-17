"""Persistence helpers and checkpoint access for long-running Agent workflows."""

import sqlite3
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any
from urllib.parse import unquote

from langgraph.checkpoint.sqlite import SqliteSaver
from sqlalchemy import select
from sqlalchemy.engine import make_url
from sqlalchemy.orm import Session, sessionmaker

from app.db.models import AgentRun, AgentRunEvent


def _now() -> datetime:
    return datetime.now(timezone.utc)


@contextmanager
def open_checkpointer(database_url: str) -> Iterator[SqliteSaver]:
    url = make_url(database_url)
    if not url.drivername.startswith("sqlite"):
        raise ValueError("LangGraph Agent checkpoints require a SQLite database")
    database = unquote(url.database or ":memory:")
    connection = sqlite3.connect(database, check_same_thread=False, timeout=30)
    try:
        saver = SqliteSaver(connection)
        saver.setup()
        yield saver
    finally:
        connection.close()


def add_run_event(
    session: Session,
    run_id: str,
    event_type: str,
    message: str,
    details: dict[str, Any] | None = None,
) -> None:
    session.add(
        AgentRunEvent(
            run_id=run_id,
            event_type=event_type,
            message=message,
            details=details,
            created_at=_now(),
        )
    )


class AgentRunManager:
    def __init__(
        self,
        sessions: sessionmaker[Session],
        database_url: str,
    ) -> None:
        self._sessions = sessions
        self._database_url = database_url
        self._lock = threading.Lock()
        self._threads: dict[str, threading.Thread] = {}

    def start(self, run_id: str, target: Callable[[], None]) -> None:
        with self._lock:
            for finished_id, thread in list(self._threads.items()):
                if not thread.is_alive():
                    del self._threads[finished_id]
            current = self._threads.get(run_id)
            if current and current.is_alive():
                raise RuntimeError(f"Agent run {run_id} is already active")
            thread = threading.Thread(
                target=target,
                name=f"agent-run-{run_id}",
                daemon=True,
            )
            self._threads[run_id] = thread
            thread.start()

    def recover_interrupted_runs(self) -> None:
        timestamp = _now()
        with self._sessions.begin() as session:
            runs = session.scalars(
                select(AgentRun).where(
                    AgentRun.status.in_(("queued", "running", "stop_requested"))
                )
            ).all()
            for run in runs:
                run.status = "interrupted"
                run.stage = "interrupted"
                run.error = (
                    "The coordinator stopped before this run completed. "
                    "Review the last checkpoint and event log before resuming."
                )
                run.updated_at = timestamp
                add_run_event(
                    session,
                    run.id,
                    "interrupted",
                    "Coordinator restart detected. Checkpoint retained; explicit resume is required.",
                )

    def has_checkpoint(self, run_id: str) -> bool:
        with open_checkpointer(self._database_url) as checkpointer:
            return (
                checkpointer.get_tuple(
                    {"configurable": {"thread_id": run_id}}
                )
                is not None
            )

    def recovery_fields(self, run: AgentRun) -> dict[str, bool]:
        if run.status not in {"paused", "interrupted", "failed"}:
            return {"can_resume": False, "checkpoint_available": False}
        checkpoint_available = self.has_checkpoint(run.id)
        can_resume = checkpoint_available or run.status in {"interrupted", "failed"}
        return {
            "can_resume": can_resume,
            "checkpoint_available": checkpoint_available,
        }

    def get_run(self, run_id: str, *, include_events: bool) -> dict[str, Any] | None:
        with self._sessions() as session:
            run = session.get(AgentRun, run_id)
            if run is None:
                return None
            result = serialize_run(run)
            result.update(self.recovery_fields(run))
            if include_events:
                result["events"] = [
                    serialize_event(event)
                    for event in session.scalars(
                        select(AgentRunEvent)
                        .where(AgentRunEvent.run_id == run_id)
                        .order_by(AgentRunEvent.id)
                    )
                ]
            return result

    def list_runs(self, limit: int = 50) -> list[dict[str, Any]]:
        with self._sessions() as session:
            runs = session.scalars(
                select(AgentRun)
                .order_by(AgentRun.created_at.desc())
                .limit(limit)
            )
            return [
                {
                    **serialize_run(run),
                    **self.recovery_fields(run),
                }
                for run in runs
            ]


def serialize_event(event: AgentRunEvent) -> dict[str, Any]:
    return {
        "id": event.id,
        "event_type": event.event_type,
        "message": event.message,
        "details": event.details,
        "created_at": event.created_at.isoformat(),
    }


def serialize_run(run: AgentRun) -> dict[str, Any]:
    return {
        "id": run.id,
        "task": run.task,
        "status": run.status,
        "stage": run.stage,
        "progress": run.progress,
        "selected_task": run.selected_task,
        "selected_reason": run.selected_reason,
        "result": run.result,
        "error": run.error,
        "created_at": run.created_at.isoformat(),
        "updated_at": run.updated_at.isoformat(),
        "started_at": run.started_at.isoformat() if run.started_at else None,
        "completed_at": run.completed_at.isoformat() if run.completed_at else None,
    }
