"""Server-side action authorization and durable autonomy state."""

import json
import logging
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Callable

from jsonschema import Draft202012Validator, ValidationError
from sqlalchemy import or_, select, update
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import REPO_ROOT
from app.db.models import (
    AutonomyDecision,
    AutonomyDecisionEvent,
    AutonomyExecutionEvent,
    AutonomyLoop,
    AutonomyState,
    SlmConversation,
)

READ_ONLY_ACTIONS = frozenset(
    {
        "get_status",
        "get_logs",
        "get_results",
        "compare_runs",
        "suggest_next_action",
        "clarify",
        "refuse",
        "redirect",
    }
)
ACTION_SCHEMA_DOCUMENT = json.loads(
    (REPO_ROOT / "common" / "slm_action_schema.json").read_text(encoding="utf-8")
)
ACTION_VALIDATOR = Draft202012Validator(
    {
        **ACTION_SCHEMA_DOCUMENT,
        **ACTION_SCHEMA_DOCUMENT["$defs"]["action"],
    }
)
STOP_RULES = frozenset({"max_iterations", "target_metric", "no_improvement_2"})
LOGGER = logging.getLogger(__name__)


class AutonomyError(RuntimeError):
    def __init__(self, message: str, status_code: int = 409) -> None:
        super().__init__(message)
        self.status_code = status_code


def validate_action(action: Any) -> dict[str, Any]:
    try:
        ACTION_VALIDATOR.validate(action)
    except ValidationError as error:
        candidates = [error]
        leaf_errors: list[ValidationError] = []
        while candidates:
            candidate = candidates.pop()
            if candidate.context:
                candidates.extend(candidate.context)
            else:
                leaf_errors.append(candidate)
        most_specific = max(
            leaf_errors or [error],
            key=lambda candidate: (
                len(candidate.absolute_path),
                not (candidate.absolute_path and candidate.absolute_path[-1] == "model_type"),
            ),
        )
        location = ".".join(str(part) for part in most_specific.absolute_path) or "action"
        raise AutonomyError(
            f"Action does not match the allowed action schema at {location}: {most_specific.message}",
            422,
        ) from error
    return action


class AutonomyController:
    def __init__(
        self,
        sessions: sessionmaker[Session],
        execute: Callable[[dict[str, Any]], dict[str, Any]],
        *,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._sessions = sessions
        self._execute = execute
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self.initialize()

    def initialize(self) -> None:
        with self._sessions() as session:
            state = session.get(AutonomyState, 1)
            if state is None:
                state = AutonomyState(
                    id=1,
                    stop_requested=False,
                    updated_at=self._clock(),
                )
                session.add(state)
            else:
                state.updated_at = self._clock()
            session.execute(
                update(AutonomyDecision)
                .where(AutonomyDecision.status == "PENDING")
                .values(status="CANCELLED", updated_at=self._clock())
            )
            session.commit()

    def state(self) -> dict[str, Any]:
        with self._sessions() as session:
            state = session.get(AutonomyState, 1)
            if state is None:
                raise RuntimeError("Autonomy state singleton was not initialized")
            loops = session.scalars(
                select(AutonomyLoop).order_by(AutonomyLoop.created_at.desc())
            ).all()
            return {
                "stop_requested": state.stop_requested,
                "loops": [self._loop_dict(loop) for loop in loops],
            }

    def dispatch(
        self,
        action: Any,
        *,
        reason: str,
        evidence: dict[str, Any] | None = None,
        mode: str = "FULL_ACCESS",
    ) -> dict[str, Any]:
        validated = validate_action(action)
        if mode not in {"FULL_ACCESS", "AGENT"}:
            raise AutonomyError("Unsupported action execution mode", 422)
        if validated["action"] in READ_ONLY_ACTIONS:
            return {"status": "executed", "result": self._execute(validated)}

        with self._sessions.begin() as session:
            state = session.get(AutonomyState, 1)
            if state is None:
                raise RuntimeError("Autonomy state singleton was not initialized")
            if state.stop_requested:
                raise AutonomyError("Global STOP is active; reset it before state changes", 423)
        try:
            result = self._execute(validated)
        except Exception:
            if mode == "AGENT":
                self._record_execution_event(
                    validated,
                    "failed",
                    mode=mode,
                    reason=reason,
                    evidence=evidence,
                )
            else:
                self._record_execution_event_safely(
                    validated,
                    "failed",
                    mode=mode,
                    reason=reason,
                    evidence=evidence,
                )
            raise
        if mode == "AGENT":
            self._record_execution_event(
                validated,
                "executed",
                mode=mode,
                result=result,
                reason=reason,
                evidence=evidence,
            )
        else:
            self._record_execution_event_safely(
                validated,
                "executed",
                mode=mode,
                result=result,
                reason=reason,
                evidence=evidence,
            )
        return {"status": "executed", "result": result}

    def _record_execution_event_safely(
        self,
        action: dict[str, Any],
        result_status: str,
        *,
        mode: str,
        result: dict[str, Any] | None = None,
        reason: str = "",
        evidence: dict[str, Any] | None = None,
    ) -> None:
        try:
            self._record_execution_event(
                action,
                result_status,
                mode=mode,
                result=result,
                reason=reason,
                evidence=evidence,
            )
        except Exception:
            LOGGER.warning(
                "Could not persist action execution audit event for action=%s",
                action["action"],
            )

    def _record_execution_event(
        self,
        action: dict[str, Any],
        result_status: str,
        *,
        mode: str,
        result: dict[str, Any] | None = None,
        reason: str = "",
        evidence: dict[str, Any] | None = None,
    ) -> None:
        parameters = action.get("parameters", {})
        dataset_name = parameters.get("dataset_name", parameters.get("model_name"))
        if not isinstance(dataset_name, str):
            dataset_name = None
        run_id = result.get("run_id", result.get("group_id")) if result else None
        if not isinstance(run_id, str):
            run_id = None
        details: dict[str, Any] = {
            "parameters": parameters,
            "reason": reason[:4000],
            "evidence": evidence or {},
        }
        if result is not None:
            details["result"] = result
        with self._sessions.begin() as session:
            session.add(
                AutonomyExecutionEvent(
                    action=action["action"],
                    mode=mode,
                    timestamp=self._clock(),
                    result_status=result_status,
                    dataset_name=dataset_name,
                    run_id=run_id,
                    details=details,
                )
            )

    def list_decisions(self) -> list[dict[str, Any]]:
        with self._sessions() as session:
            decisions = session.scalars(
                select(AutonomyDecision).order_by(AutonomyDecision.created_at.desc())
            ).all()
            return [self._decision_dict(item) for item in decisions]

    def history(self, limit: int = 250) -> list[dict[str, Any]]:
        with self._sessions() as session:
            events = session.scalars(
                select(AutonomyDecisionEvent)
                .order_by(AutonomyDecisionEvent.id.desc())
                .limit(limit)
            ).all()
            history = [
                {
                    "id": event.id,
                    "decision_id": event.decision_id,
                    "event": event.event,
                    "details": event.details,
                    "created_at": event.created_at.isoformat(),
                }
                for event in events
            ]
            executions = session.scalars(
                select(AutonomyExecutionEvent).order_by(
                    AutonomyExecutionEvent.timestamp.desc(),
                    AutonomyExecutionEvent.id.desc(),
                ).limit(limit)
            ).all()
            history.extend(
                {
                    "id": f"execution-{event.id}",
                    "decision_id": None,
                    "event": (
                        "user-initiated run"
                        if event.action == "run_apg_simulation"
                        else "executed automatically"
                        if event.result_status == "executed"
                        else "automatic execution failed"
                    ),
                    "action": event.action,
                    "mode": event.mode,
                    "timestamp": event.timestamp.isoformat(),
                    "result_status": event.result_status,
                    "dataset_name": event.dataset_name,
                    "details": {
                        "action": event.action,
                        "mode": event.mode,
                        "result_status": event.result_status,
                        "dataset_name": event.dataset_name,
                        "run_id": event.run_id,
                        **(event.details or {}),
                    },
                    "created_at": event.timestamp.isoformat(),
                }
                for event in executions
            )
            return sorted(
                history,
                key=lambda row: row["created_at"],
                reverse=True,
            )[:limit]

    def delete_workflows(
        self,
        workflow_ids: list[str],
        conversation_ids: list[str],
    ) -> dict[str, int]:
        selected_workflows = set(workflow_ids)
        selected_conversations = set(conversation_ids)
        delete_legacy_unlinked = "legacy-unlinked" in selected_workflows
        selected_workflows.discard("legacy-unlinked")
        if not selected_workflows and not selected_conversations:
            if not delete_legacy_unlinked:
                raise AutonomyError("Select at least one workflow or conversation to delete", 422)

        with self._sessions.begin() as session:
            loops = []
            if selected_workflows:
                running_loop = session.scalar(
                    select(AutonomyLoop.id).where(
                        AutonomyLoop.workflow_id.in_(selected_workflows),
                        AutonomyLoop.status == "RUNNING",
                    )
                )
                if running_loop is not None:
                    raise AutonomyError(
                        "A running workflow cannot be deleted. Stop or finish it first."
                    )
                loops = session.scalars(
                    select(AutonomyLoop).where(
                        AutonomyLoop.workflow_id.in_(selected_workflows)
                    )
                ).all()

            conversation_filters = []
            if selected_workflows:
                conversation_filters.append(
                    SlmConversation.workflow_id.in_(selected_workflows)
                )
            if selected_conversations:
                conversation_filters.append(
                    SlmConversation.id.in_(selected_conversations)
                )
            conversations = (
                session.scalars(
                    select(SlmConversation).where(
                        SlmConversation.mode == "agent",
                        or_(*conversation_filters),
                    )
                ).all()
                if conversation_filters
                else []
            )

            events = session.scalars(
                select(AutonomyExecutionEvent)
            ).all()
            matching_events = []
            for event in events:
                evidence = (event.details or {}).get("evidence", {})
                if not isinstance(evidence, dict):
                    continue
                if (
                    evidence.get("workflow_id") in selected_workflows
                    or evidence.get("conversation_id") in selected_conversations
                    or (
                        delete_legacy_unlinked
                        and evidence.get("source") == "agent_workflow"
                        and not evidence.get("workflow_id")
                        and not evidence.get("conversation_id")
                    )
                ):
                    matching_events.append(event)

            for item in [*loops, *conversations, *matching_events]:
                session.delete(item)

            return {
                "deleted_workflows": len(loops),
                "deleted_conversations": len(conversations),
                "deleted_actions": len(matching_events),
            }

    def stop(self) -> dict[str, Any]:
        now = self._clock()
        with self._sessions.begin() as session:
            state = session.get(AutonomyState, 1)
            if state is None:
                raise RuntimeError("Autonomy state singleton was not initialized")
            state.stop_requested = True
            state.updated_at = now
            loops = session.scalars(
                select(AutonomyLoop).where(AutonomyLoop.status == "RUNNING")
            ).all()
            for loop in loops:
                loop.status = "STOPPED"
                loop.failure_reason = "global STOP requested"
                loop.updated_at = now
        return self.state()

    def reset_stop(self) -> dict[str, Any]:
        with self._sessions.begin() as session:
            state = session.get(AutonomyState, 1)
            if state is None:
                raise RuntimeError("Autonomy state singleton was not initialized")
            state.stop_requested = False
            state.updated_at = self._clock()
        return self.state()

    def start_loop(
        self,
        *,
        max_iterations: int = 10,
        time_limit_seconds: int = 3600,
        stop_rule: str = "max_iterations",
        target_metric: float | None = None,
        workflow_id: str | None = None,
    ) -> dict[str, Any]:
        if not 1 <= max_iterations <= 100:
            raise AutonomyError("max_iterations must be between 1 and 100", 422)
        if not 1 <= time_limit_seconds <= 86400:
            raise AutonomyError("time_limit_seconds must be between 1 and 86400", 422)
        if stop_rule not in STOP_RULES:
            raise AutonomyError("Unsupported loop stop rule", 422)
        if stop_rule == "target_metric" and target_metric is None:
            raise AutonomyError("target_metric stop rule requires target_metric", 422)
        if stop_rule != "target_metric" and target_metric is not None:
            raise AutonomyError("target_metric is only valid for target_metric stop rule", 422)
        now = self._clock()
        with self._sessions.begin() as session:
            state = session.get(AutonomyState, 1)
            if state is None:
                raise RuntimeError("Autonomy state singleton was not initialized")
            if state.stop_requested:
                raise AutonomyError("Global STOP is active; reset it before starting a loop", 423)
            existing = session.scalar(
                select(AutonomyLoop.id).where(AutonomyLoop.status == "RUNNING")
            )
            if existing is not None:
                raise AutonomyError("A loop is already running", 409)
            loop = AutonomyLoop(
                id=str(uuid.uuid4()),
                status="RUNNING",
                max_iterations=max_iterations,
                iteration=0,
                time_limit_seconds=time_limit_seconds,
                stop_rule=stop_rule,
                target_metric=target_metric,
                workflow_id=workflow_id,
                best_metric=None,
                no_improvement_count=0,
                failure_reason=None,
                created_at=now,
                deadline_at=now + timedelta(seconds=time_limit_seconds),
                updated_at=now,
            )
            session.add(loop)
            session.flush()
            return self._loop_dict(loop)

    def record_iteration(
        self,
        loop_id: str,
        *,
        metric: float | None = None,
        worker_failure: str | None = None,
    ) -> dict[str, Any]:
        now = self._clock()
        with self._sessions.begin() as session:
            loop = session.get(AutonomyLoop, loop_id)
            if loop is None:
                raise AutonomyError("Loop not found", 404)
            state = session.get(AutonomyState, 1)
            if loop.status != "RUNNING":
                raise AutonomyError(f"Loop is {loop.status.lower()}, not running", 409)
            if state is not None and state.stop_requested:
                loop.status = "STOPPED"
                loop.failure_reason = "global STOP requested"
            elif now >= self._normalize_datetime(loop.deadline_at):
                loop.status = "STOPPED"
                loop.failure_reason = "time limit reached"
            elif loop.iteration >= loop.max_iterations:
                loop.status = "STOPPED"
                loop.failure_reason = "maximum iterations reached"
            elif worker_failure:
                loop.status = "PAUSED"
                loop.failure_reason = worker_failure[:1000]
                linked_decision = next(
                    (
                        decision
                        for decision in session.scalars(
                            select(AutonomyDecision).order_by(
                                AutonomyDecision.created_at.desc()
                            )
                        )
                        if decision.status == "APPROVED"
                        and decision.evidence.get("loop_id") == loop_id
                    ),
                    None,
                )
                if linked_decision is not None:
                    session.add(
                        AutonomyDecisionEvent(
                            decision_id=linked_decision.id,
                            event="loop_paused",
                            details={
                                "loop_id": loop.id,
                                "reason": loop.failure_reason,
                            },
                            created_at=now,
                        )
                    )
            else:
                loop.iteration += 1
                if metric is not None:
                    improved = loop.best_metric is None or metric > loop.best_metric
                    if improved:
                        loop.best_metric = metric
                        loop.no_improvement_count = 0
                    else:
                        loop.no_improvement_count += 1
                if loop.stop_rule == "target_metric" and metric is not None:
                    if loop.target_metric is not None and metric >= loop.target_metric:
                        loop.status = "STOPPED"
                        loop.failure_reason = "target metric reached"
                if loop.stop_rule == "no_improvement_2" and loop.no_improvement_count >= 2:
                    loop.status = "STOPPED"
                    loop.failure_reason = "no improvement for two iterations"
                if loop.iteration >= loop.max_iterations:
                    loop.status = "STOPPED"
                    loop.failure_reason = "maximum iterations reached"
            loop.updated_at = now
            return self._loop_dict(loop)

    def stop_loop(self, loop_id: str, reason: str) -> dict[str, Any]:
        if not reason.strip():
            raise AutonomyError("A loop stop reason is required", 422)
        now = self._clock()
        with self._sessions.begin() as session:
            loop = session.get(AutonomyLoop, loop_id)
            if loop is None:
                raise AutonomyError("Loop not found", 404)
            if loop.status != "RUNNING":
                raise AutonomyError(f"Loop is {loop.status.lower()}, not running", 409)
            loop.status = "STOPPED"
            loop.failure_reason = reason.strip()[:1000]
            loop.updated_at = now
            return self._loop_dict(loop)

    @staticmethod
    def _decision_dict(decision: AutonomyDecision) -> dict[str, Any]:
        return {
            "id": decision.id,
            "action": decision.action,
            "parameters": decision.parameters,
            "reason": decision.reason,
            "evidence": decision.evidence,
            "status": decision.status,
            "created_at": decision.created_at.isoformat(),
            "expires_at": decision.expires_at.isoformat(),
            "updated_at": decision.updated_at.isoformat(),
        }

    @staticmethod
    def _loop_dict(loop: AutonomyLoop) -> dict[str, Any]:
        return {
            "id": loop.id,
            "status": loop.status,
            "iteration": loop.iteration,
            "max_iterations": loop.max_iterations,
            "time_limit_seconds": loop.time_limit_seconds,
            "stop_rule": loop.stop_rule,
            "target_metric": loop.target_metric,
            "best_metric": loop.best_metric,
            "no_improvement_count": loop.no_improvement_count,
            "failure_reason": loop.failure_reason,
            "workflow_id": loop.workflow_id,
            "created_at": loop.created_at.isoformat(),
            "deadline_at": loop.deadline_at.isoformat(),
            "updated_at": loop.updated_at.isoformat(),
        }

    @staticmethod
    def _normalize_datetime(value: datetime) -> datetime:
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value
