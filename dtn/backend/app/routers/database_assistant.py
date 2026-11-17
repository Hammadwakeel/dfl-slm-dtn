"""Database-grounded answers about recorded coordinator state."""

import re
from datetime import datetime
from typing import Any

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel, Field
from sqlalchemy import select

from app.core.security import require_api_token
from app.db.models import (
    AgentRun,
    AgentRunEvent,
    AutonomyExecutionEvent,
    Device,
)

router = APIRouter(
    prefix="/api/database-assistant",
    tags=["database-assistant"],
    dependencies=[Depends(require_api_token)],
)


class DatabaseQuestion(BaseModel):
    question: str = Field(min_length=1, max_length=1000)


def _time(value: datetime | None) -> str:
    return value.isoformat() if value else "time not recorded"


def _worker_answer(question: str, request: Request) -> dict[str, Any]:
    worker_match = re.search(r"\bworker[-_ ]?([1-9][0-9]*)\b", question, re.IGNORECASE)
    with request.app.state.session_factory() as session:
        query = select(Device).where(Device.role == "worker").order_by(Device.node_id)
        if worker_match:
            query = query.where(Device.node_id == f"worker{worker_match.group(1)}")
        devices = session.scalars(query).all()

    if not devices:
        target = f"worker{worker_match.group(1)}" if worker_match else "workers"
        return {
            "answer": f"The coordinator database has no recorded device entries for {target}.",
            "sources": [],
        }

    lines: list[str] = []
    sources: list[dict[str, str]] = []
    for device in devices:
        status = "online" if device.online else "offline"
        detail = f"{device.node_id} is recorded as {status}; last seen {_time(device.last_seen)}."
        if device.last_error:
            detail += f" Recorded error: {device.last_error}."
        elif device.online:
            detail += " No device error is recorded."
        metrics = device.latest_status or {}
        cpu = metrics.get("cpu_utilization_percent")
        if isinstance(cpu, (int, float)):
            detail += f" Latest recorded CPU utilization: {cpu:.1f}%."
        lines.append(detail)
        sources.append({
            "record_type": "device",
            "record_id": device.node_id,
            "recorded_at": _time(device.last_seen),
        })
    return {"answer": "\n".join(lines), "sources": sources}


ACTIVE_RUN_STATUSES = ("queued", "running", "stopping")


def _run_lines(run: AgentRun) -> list[str]:
    state = run.state or {}
    lines = [
        f"Supervisor run {run.id}: {run.status} at stage {run.stage} ({run.progress}%); goal: {run.task}."
    ]
    best = state.get("best")
    if isinstance(best, dict) and isinstance(best.get("validation_accuracy"), (int, float)):
        lines.append(
            f" Best mean worker validation accuracy: {best['validation_accuracy'] * 100:.2f}% "
            f"(from {best.get('step')}, experiment {best.get('group_id')})."
        )
    active = state.get("active")
    if isinstance(active, dict) and active.get("group_id"):
        progress = active.get("progress")
        lines.append(
            f" Active experiment: {active.get('step')} (group {active['group_id']})"
            + (f", {progress * 100:.0f}% of rounds done." if isinstance(progress, (int, float)) else ".")
        )
    if state.get("waiting_reason") and run.status in ACTIVE_RUN_STATUSES:
        lines.append(f" Waiting: {state['waiting_reason']}.")
    if run.selected_task:
        detail = f" Current step: {run.selected_task}."
        if run.selected_reason:
            detail += f" Recorded reason: {run.selected_reason}"
        lines.append(detail)
    result = run.result or {}
    if result.get("outcome"):
        lines.append(f" Outcome: {result['outcome']} - {result.get('summary', '')}")
    elif run.error:
        lines.append(f" Recorded error: {run.error}")
    return lines


def _activity_answer(request: Request, *, active_only: bool) -> dict[str, Any]:
    with request.app.state.session_factory() as session:
        run_query = select(AgentRun).where(AgentRun.goal.is_not(None)).order_by(AgentRun.created_at.desc())
        run_query = (
            run_query.where(AgentRun.status.in_(ACTIVE_RUN_STATUSES))
            if active_only
            else run_query.limit(5)
        )
        runs = session.scalars(run_query).all()
        events: list[AgentRunEvent] = []
        if runs:
            events = session.scalars(
                select(AgentRunEvent)
                .where(AgentRunEvent.run_id.in_([run.id for run in runs]))
                .order_by(AgentRunEvent.created_at.desc(), AgentRunEvent.id.desc())
                .limit(12)
            ).all()
        execution_query = select(AutonomyExecutionEvent).order_by(
            AutonomyExecutionEvent.timestamp.desc(),
            AutonomyExecutionEvent.id.desc(),
        )
        if active_only:
            run_ids = [run.id for run in runs]
            execution_query = execution_query.where(
                AutonomyExecutionEvent.details["evidence"]["run_id"].as_string().in_(run_ids)
            ) if run_ids else None
        executions = session.scalars(execution_query.limit(8)).all() if execution_query is not None else []
        for run in runs:
            session.expunge(run)

    lines: list[str] = []
    sources: list[dict[str, str]] = []
    for run in runs:
        lines.extend(_run_lines(run))
        sources.append({"record_type": "supervisor_run", "record_id": run.id, "recorded_at": _time(run.updated_at)})
    for event in executions:
        detail = f"Recorded action {event.action} ({event.result_status}) on {_time(event.timestamp)}"
        if event.dataset_name:
            detail += f" for {event.dataset_name}"
        if event.run_id:
            detail += f"; experiment/group ID {event.run_id}"
        parameters = (event.details or {}).get("parameters")
        if isinstance(parameters, dict):
            settings = [f"{key}={parameters[key]}" for key in ("rounds", "epochs", "topology") if key in parameters]
            if settings:
                detail += "; " + ", ".join(settings)
        lines.append(detail + ".")
        sources.append({
            "record_type": "autonomy_execution",
            "record_id": str(event.id),
            "recorded_at": _time(event.timestamp),
        })
    for event in events:
        lines.append(f"Supervisor log ({_time(event.created_at)}): {event.message}")
        sources.append({
            "record_type": "agent_run_event",
            "record_id": str(event.id),
            "recorded_at": _time(event.created_at),
        })
    if not lines:
        answer = (
            "The coordinator database has no active supervisor runs."
            if active_only
            else "The coordinator database has no supervisor-run or action history recorded yet."
        )
    else:
        answer = "\n".join(lines)
    return {"answer": answer, "sources": sources}


@router.post("/ask")
def ask_database(question: DatabaseQuestion, request: Request) -> dict[str, Any]:
    """Answer supported status and history questions using coordinator DB rows only."""
    normalized = question.question.casefold()
    if any(term in normalized for term in ("worker", "device", "online", "offline", "cpu")):
        return _worker_answer(normalized, request)
    current_status_phrases = (
        "happening",
        "right now",
        "currently",
        "active",
        "running",
        "progress",
    )
    if any(phrase in normalized for phrase in current_status_phrases):
        return _activity_answer(request, active_only=True)
    if any(term in normalized for term in ("agent", "run", "checkpoint", "progress", "history", "happened", "action", "experiment")):
        return _activity_answer(request, active_only=False)
    return {
        "answer": (
            "I can only answer from records stored in the coordinator database. "
            "Ask about worker/device status, the supervisor run, or recent actions and experiments."
        ),
        "sources": [],
    }
