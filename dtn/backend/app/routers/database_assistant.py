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
    AutonomyLoop,
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


def _activity_answer(request: Request, *, active_only: bool) -> dict[str, Any]:
    with request.app.state.session_factory() as session:
        run_query = select(AgentRun).order_by(AgentRun.created_at.desc()).limit(8)
        if active_only:
            run_query = select(AgentRun).where(
                AgentRun.status.in_(("queued", "running", "stop_requested"))
            ).order_by(AgentRun.created_at.desc())
        runs = session.scalars(run_query).all()
        loop_query = select(AutonomyLoop).order_by(AutonomyLoop.created_at.desc()).limit(8)
        if active_only:
            loop_query = select(AutonomyLoop).where(
                AutonomyLoop.status == "RUNNING"
            ).order_by(AutonomyLoop.created_at.desc())
        loops = session.scalars(loop_query).all()
        events: list[AgentRunEvent] = []
        if runs:
            events = session.scalars(
                select(AgentRunEvent)
                .where(AgentRunEvent.run_id.in_([run.id for run in runs]))
                .order_by(AgentRunEvent.created_at.desc())
                .limit(16)
            ).all()
        if active_only:
            active_workflow_ids = [
                loop.workflow_id for loop in loops if loop.workflow_id is not None
            ]
            executions = (
                session.scalars(
                    select(AutonomyExecutionEvent)
                    .where(
                        AutonomyExecutionEvent.details["evidence"]["workflow_id"]
                        .as_string()
                        .in_(active_workflow_ids)
                    )
                    .order_by(
                        AutonomyExecutionEvent.timestamp.desc(),
                        AutonomyExecutionEvent.id.desc(),
                    )
                    .limit(24)
                ).all()
                if active_workflow_ids
                else []
            )
        else:
            executions = session.scalars(
                select(AutonomyExecutionEvent)
                .order_by(AutonomyExecutionEvent.timestamp.desc(), AutonomyExecutionEvent.id.desc())
                .limit(8)
            ).all()

    lines: list[str] = []
    sources: list[dict[str, str]] = []
    for run in runs:
        lines.append(
            f"Agent run {run.id}: {run.status} at stage {run.stage} "
            f"({run.progress}%); task: {run.task}."
        )
        if run.selected_task:
            detail = f" Selected action: {run.selected_task}."
            if run.selected_reason:
                detail += f" Recorded reason: {run.selected_reason}"
            lines.append(detail)
        if run.error:
            lines.append(f" Recorded error: {run.error}")
        sources.append({
            "record_type": "agent_run",
            "record_id": run.id,
            "recorded_at": _time(run.updated_at),
        })
    for loop in loops:
        rule = loop.stop_rule.replace("_", " ")
        line = (
            f"Autonomy loop {loop.id} is {loop.status}; "
            f"{loop.iteration} experiment(s) recorded, stop criterion: {rule}."
        )
        if loop.target_metric is not None:
            line += f" Target accuracy: {loop.target_metric * 100:.2f}%."
        if loop.best_metric is not None:
            line += f" Best recorded accuracy: {loop.best_metric * 100:.2f}%."
        if loop.failure_reason:
            line += f" Recorded reason: {loop.failure_reason}."
        lines.append(line)
        sources.append({
            "record_type": "autonomy_loop",
            "record_id": loop.id,
            "recorded_at": _time(loop.updated_at),
        })
    if active_only:
        workflow_events = executions[:8]
        for event in workflow_events:
            action_detail = f"Latest recorded workflow action: {event.action} ({event.result_status})"
            if event.dataset_name:
                action_detail += f" for {event.dataset_name}"
            if event.run_id:
                action_detail += f"; experiment/group ID {event.run_id}"
            parameters = (event.details or {}).get("parameters")
            if isinstance(parameters, dict):
                settings = [
                    f"{key}={parameters[key]}"
                    for key in ("rounds", "epochs", "topology")
                    if key in parameters
                ]
                if settings:
                    action_detail += f"; " + ", ".join(settings)
            reason = (event.details or {}).get("reason")
            if isinstance(reason, str) and reason:
                action_detail += f". Agent rationale: {reason[:300]}"
            if event.result_status == "executed":
                action_detail += (
                    ". The action was accepted; this record alone does not confirm "
                    "that its experiment is still running or has completed."
                )
            lines.append(action_detail)
            sources.append({
                "record_type": "autonomy_execution",
                "record_id": str(event.id),
                "recorded_at": _time(event.timestamp),
            })
        if loops:
            lines.append(
                "The coordinator database records loop counts and actions, but not live per-worker "
                "epochs or experiment status. Use the Agent Runs page for live worker progress."
            )
            if not workflow_events:
                lines.append("No workflow-linked action record is available yet.")
    for event in events:
        lines.append(f"Agent log ({_time(event.created_at)}): {event.message}")
        sources.append({
            "record_type": "agent_run_event",
            "record_id": str(event.id),
            "recorded_at": _time(event.created_at),
        })
    if not active_only:
        for event in executions:
            lines.append(
                f"Recorded action {event.action} ({event.result_status}) "
                f"on {_time(event.timestamp)}"
                f"{f' for {event.dataset_name}' if event.dataset_name else ''}."
            )
            sources.append({
                "record_type": "autonomy_execution",
                "record_id": str(event.id),
                "recorded_at": _time(event.timestamp),
            })
    if not lines:
        answer = (
            "The coordinator database has no active Agent runs or autonomy loops."
            if active_only
            else "The coordinator database has no Agent-run or autonomy activity recorded yet."
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
            "Ask about worker/device status, active Agent runs, checkpoints, or recent actions and experiments."
        ),
        "sources": [],
    }
