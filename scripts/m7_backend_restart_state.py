"""Prepare and verify real SQLite autonomy state across a backend restart."""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "dtn" / "backend"))
sys.path.insert(0, str(ROOT / "scripts"))

from app.core.config import Settings  # noqa: E402
from m5_feedback_loop import BackendClient  # noqa: E402

DB_PATH = ROOT / "dtn" / "backend" / "data" / "dtn.sqlite3"


class RestartStateError(RuntimeError):
    pass


def _client(settings: Settings) -> BackendClient:
    return BackendClient(f"http://127.0.0.1:{settings.port}", settings.api_token)


def _prepare(backend: BackendClient) -> None:
    state = backend.request("GET", "/api/autonomy/state")
    if (
        state.get("mode") != "ASK_HUMAN"
        or state.get("stop_requested") is not False
        or state.get("pending_decisions") != 0
    ):
        raise RestartStateError("Precondition failed: expected ASK_HUMAN, STOP off, no pending decision")
    paused = next(
        (loop for loop in state.get("loops", []) if loop.get("status") == "PAUSED"),
        None,
    )
    if not isinstance(paused, dict) or not isinstance(paused.get("id"), str):
        raise RestartStateError("M7 worker-failure paused loop is missing")

    dispatched = backend.request(
        "POST",
        "/api/autonomy/dispatch",
        {
            "action": {"action": "set_rounds", "parameters": {"rounds": 4}},
            "reason": "M7 backend restart persistence test",
            "evidence": {"loop_id": paused["id"], "test": "backend_restart"},
        },
    )
    decision = dispatched.get("decision")
    if (
        dispatched.get("status") != "pending"
        or not isinstance(decision, dict)
        or decision.get("status") != "PENDING"
        or not isinstance(decision.get("id"), str)
    ):
        raise RestartStateError("ASK_HUMAN did not create the M7 pending decision")
    decision_id = decision["id"]

    stopped = backend.request("POST", "/api/autonomy/stop")
    if stopped.get("stop_requested") is not True:
        raise RestartStateError("Global STOP did not activate before restart")

    expired_at = (datetime.now(timezone.utc) - timedelta(minutes=1)).replace(tzinfo=None)
    with sqlite3.connect(DB_PATH) as connection:
        cursor = connection.execute(
            "UPDATE autonomy_decisions SET expires_at = ? "
            "WHERE id = ? AND status = 'PENDING'",
            (expired_at.strftime("%Y-%m-%d %H:%M:%S.%f"), decision_id),
        )
        if cursor.rowcount != 1:
            raise RestartStateError("Could not set the test decision expiry in the backed-up database")
        connection.commit()
        row = connection.execute(
            "SELECT status, expires_at FROM autonomy_decisions WHERE id = ?",
            (decision_id,),
        ).fetchone()
    if row is None or row[0] != "PENDING":
        raise RestartStateError("Test decision was not pending immediately before restart")
    print(
        json.dumps(
            {
                "decision_id": decision_id,
                "decision_status_before_restart": row[0],
                "decision_expires_at": row[1],
                "loop_id": paused["id"],
                "loop_status": paused["status"],
                "global_stop_before_restart": True,
            },
            sort_keys=True,
        )
    )


def _verify(backend: BackendClient, decision_id: str, loop_id: str) -> None:
    state = backend.request("GET", "/api/autonomy/state")
    decisions = backend.request("GET", "/api/autonomy/decisions")
    history = backend.request("GET", "/api/autonomy/history")
    decision = next(
        (item for item in decisions if item.get("id") == decision_id),
        None,
    )
    loop = next(
        (item for item in state.get("loops", []) if item.get("id") == loop_id),
        None,
    )
    decision_events = [
        row.get("event")
        for row in history
        if row.get("decision_id") == decision_id
    ]
    if not isinstance(decision, dict) or decision.get("status") != "EXPIRED":
        raise RestartStateError("Expired decision did not reload as EXPIRED")
    if "approved" in decision_events or "expired" not in decision_events:
        raise RestartStateError("Expired decision was approved or expiry history is missing")
    if not isinstance(loop, dict) or loop.get("status") != "PAUSED":
        raise RestartStateError("Paused loop did not reload as PAUSED")
    if state.get("stop_requested") is not True:
        raise RestartStateError("Global STOP did not persist across backend restart")
    if state.get("pending_decisions") != 0:
        raise RestartStateError("Expired decision remained pending after backend restart")
    print(
        json.dumps(
            {
                "decision_id": decision_id,
                "decision_status_after_restart": decision.get("status"),
                "decision_history_events": decision_events,
                "loop_id": loop_id,
                "loop_status_after_restart": loop.get("status"),
                "loop_failure_reason": loop.get("failure_reason"),
                "global_stop_after_restart": state.get("stop_requested"),
                "pending_decisions": state.get("pending_decisions"),
            },
            sort_keys=True,
        )
    )
    reset = backend.request("POST", "/api/autonomy/reset-stop")
    if reset.get("stop_requested") is not False:
        raise RestartStateError("Global STOP did not reset after persistence verification")
    final = backend.request("GET", "/api/autonomy/state")
    if final.get("mode") != "ASK_HUMAN" or final.get("stop_requested") is not False:
        raise RestartStateError("Backend did not return to safe ASK_HUMAN/STOP-off state")
    print("global STOP reset; mode=ASK_HUMAN; stop_requested=False")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("prepare", "verify"))
    parser.add_argument("--decision-id")
    parser.add_argument("--loop-id")
    args = parser.parse_args()
    settings = Settings.from_env()
    backend = _client(settings)
    if args.mode == "prepare":
        _prepare(backend)
        return
    if not args.decision_id or not args.loop_id:
        raise SystemExit("verify requires --decision-id and --loop-id")
    _verify(backend, args.decision_id, args.loop_id)


if __name__ == "__main__":
    main()
