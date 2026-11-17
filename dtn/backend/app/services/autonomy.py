"""Server-side action authorization and durable autonomy state."""

import json
import logging
from datetime import datetime, timezone
from typing import Any, Callable

from jsonschema import Draft202012Validator, ValidationError
from sqlalchemy import select, update
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import REPO_ROOT
from app.db.models import (
    AutonomyDecision,
    AutonomyDecisionEvent,
    AutonomyExecutionEvent,
    AutonomyState,
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
            return {"stop_requested": state.stop_requested}

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

    def stop(self) -> dict[str, Any]:
        now = self._clock()
        with self._sessions.begin() as session:
            state = session.get(AutonomyState, 1)
            if state is None:
                raise RuntimeError("Autonomy state singleton was not initialized")
            state.stop_requested = True
            state.updated_at = now
        return self.state()

    def reset_stop(self) -> dict[str, Any]:
        with self._sessions.begin() as session:
            state = session.get(AutonomyState, 1)
            if state is None:
                raise RuntimeError("Autonomy state singleton was not initialized")
            state.stop_requested = False
            state.updated_at = self._clock()
        return self.state()

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
    def _normalize_datetime(value: datetime) -> datetime:
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value
