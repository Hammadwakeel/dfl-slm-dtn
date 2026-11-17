"""Run bounded dataset-specific feedback loops through the authenticated backend."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import re
import shlex
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "dtn" / "backend"))

from app.core.config import Settings  # noqa: E402
from app.services.feedback import build_training_feedback_summary  # noqa: E402
from app.services.worker_shards import (  # noqa: E402
    get_worker_dataset_metadata,
    get_worker_training_status,
)


class FeedbackLoopError(RuntimeError):
    pass


RAM_PATTERN = re.compile(r"\bRAM\s+(\d+)/(\d+)MB\b")
SWAP_PATTERN = re.compile(r"\bSWAP\s+(\d+)/(\d+)MB\b")


def _redact_tokens(value: Any, settings: Settings) -> Any:
    tokens = (
        settings.api_token,
        settings.master_slm_token,
        *(node.token for node in settings.nodes),
    )
    if isinstance(value, str):
        redacted = value
        for token in tokens:
            if token:
                redacted = redacted.replace(token, "[REDACTED]")
        return re.sub(
            r"(?i)\bBearer\s+[A-Za-z0-9._~+/-]+=*",
            "Bearer [REDACTED]",
            redacted,
        )
    if isinstance(value, dict):
        return {key: _redact_tokens(item, settings) for key, item in value.items()}
    if isinstance(value, list):
        return [_redact_tokens(item, settings) for item in value]
    return value


class BackendClient:
    def __init__(self, base_url: str, token: str) -> None:
        self.base_url = base_url.rstrip("/")
        self.headers = {
            "Authorization": f"Bearer {token}",
            "Accept": "application/json",
            "Content-Type": "application/json",
        }

    def request(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
        *,
        timeout: float = 30,
    ) -> Any:
        body = json.dumps(payload).encode("utf-8") if payload is not None else None
        request = Request(self.base_url + path, data=body, headers=self.headers, method=method)
        try:
            with urlopen(request, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")[:500]
            raise FeedbackLoopError(
                f"Backend {method} {path} returned HTTP {error.code}: {detail}"
            ) from error
        except (URLError, TimeoutError, OSError) as error:
            raise FeedbackLoopError(
                f"Backend {method} {path} failed: {type(error).__name__}: {error}"
            ) from error
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise FeedbackLoopError(f"Backend {method} {path} returned invalid JSON") from error


def _ssh(user: str, host: str, command: str, timeout: int = 45) -> str:
    process = subprocess.run(
        [
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=8",
            f"{user}@{host}",
            command,
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        check=False,
    )
    if process.returncode:
        raise FeedbackLoopError(
            f"SSH to {host} failed (exit {process.returncode}): "
            f"{process.stderr.strip()[-500:]}"
        )
    return process.stdout


@contextlib.contextmanager
def _sample_tegrastats(settings: Settings, label: str):
    nodes = sorted(
        (node for node in settings.nodes if node.role in {"master", "worker"}),
        key=lambda node: node.node_id,
    )
    if [node.node_id for node in nodes] != ["master", "worker1", "worker2", "worker3"]:
        raise FeedbackLoopError("Expected master and exactly worker1, worker2, worker3")
    ssh_user = os.environ.get("M4_SSH_USER", "hammad")
    run_id = f"{label}-{uuid.uuid4().hex[:8]}"
    prefixes: dict[str, str] = {}
    try:
        for node in nodes:
            prefix = f"/tmp/dfl-m5-tegrastats-{run_id}-{node.node_id}"
            log_path, pid_path = f"{prefix}.log", f"{prefix}.pid"
            command = (
                "sh -c "
                + shlex.quote(
                    "command -v tegrastats >/dev/null 2>&1 || exit 127; "
                    f"nohup tegrastats --interval 1000 > {shlex.quote(log_path)} 2>&1 "
                    f"</dev/null & echo $! > {shlex.quote(pid_path)}; "
                    f"sleep 1; kill -0 \"$(cat {shlex.quote(pid_path)})\""
                )
            )
            _ssh(ssh_user, node.ip, command)
            prefixes[node.node_id] = prefix
            print(f"{label} {node.node_id}: tegrastats sampling started")
        yield
    finally:
        failures: list[str] = []
        for node in nodes:
            prefix = prefixes.get(node.node_id)
            if prefix is None:
                continue
            log_path, pid_path = f"{prefix}.log", f"{prefix}.pid"
            command = (
                "sh -c "
                + shlex.quote(
                    f"if test -f {shlex.quote(pid_path)}; then "
                    f"kill \"$(cat {shlex.quote(pid_path)})\" 2>/dev/null || true; fi; "
                    "sleep 1; "
                    f"if test -f {shlex.quote(log_path)}; then cat {shlex.quote(log_path)}; fi; "
                    f"rm -f -- {shlex.quote(pid_path)} {shlex.quote(log_path)}"
                )
            )
            try:
                output = _ssh(ssh_user, node.ip, command)
            except FeedbackLoopError as error:
                failures.append(f"{node.node_id}: {error}")
                continue
            ram = [int(match.group(1)) for match in RAM_PATTERN.finditer(output)]
            swap = [int(match.group(1)) for match in SWAP_PATTERN.finditer(output)]
            if not ram or not swap:
                failures.append(f"{node.node_id}: tegrastats returned no RAM/SWAP samples")
                continue
            print(
                f"{label} {node.node_id}: peak_RAM_MB={max(ram)}; "
                f"peak_swap_used_MB={max(swap)}; samples={min(len(ram), len(swap))}"
            )
        if failures:
            raise FeedbackLoopError("; ".join(failures))


def _action_data(envelope: Any) -> dict[str, Any]:
    if not isinstance(envelope, dict):
        raise FeedbackLoopError("Action result was not an object")
    current: Any = envelope.get("result", envelope)
    for _ in range(4):
        if not isinstance(current, dict):
            break
        if isinstance(current.get("data"), dict):
            return current["data"]
        if isinstance(current.get("result"), dict):
            current = current["result"]
            continue
        break
    raise FeedbackLoopError("Action result did not contain a data object")


def _check_preflight(
    backend: BackendClient,
    settings: Settings,
    expected_mode: str,
    dataset_name: str,
    *,
    allowed_loop_id: str | None = None,
) -> None:
    state = backend.request("GET", "/api/autonomy/state")
    if state.get("mode") != expected_mode or state.get("stop_requested") is not False:
        raise FeedbackLoopError(
            f"Expected autonomy={expected_mode} and global STOP inactive"
        )
    if state.get("pending_decisions"):
        raise FeedbackLoopError("Existing pending decisions must be resolved first")
    if any(
        loop.get("status") == "RUNNING" and loop.get("id") != allowed_loop_id
        for loop in state.get("loops", [])
    ):
        raise FeedbackLoopError("An autonomy loop is already running")
    results = backend.request("GET", "/api/slm/results")
    if results.get("active_group_id") is not None:
        raise FeedbackLoopError("The master already has an active training or NAS group")
    workers = sorted(
        (node for node in settings.nodes if node.role == "worker"),
        key=lambda node: node.node_id,
    )
    if [node.node_id for node in workers] != ["worker1", "worker2", "worker3"]:
        raise FeedbackLoopError("Expected worker1, worker2, and worker3")
    for worker in workers:
        status = get_worker_training_status(worker)
        if status.get("active_run_id") is not None or status.get("scheduled_run_id") is not None:
            raise FeedbackLoopError(
                f"{worker.node_id} is not idle: "
                f"active={status.get('active_run_id')}; "
                f"scheduled={status.get('scheduled_run_id')}"
            )
        dataset = get_worker_dataset_metadata(worker, dataset_name)
        print(
            f"{worker.node_id}: authenticated and idle; "
            f"dataset={dataset_name}; sample_count={dataset.get('sample_count')}"
        )
    print(f"autonomy={expected_mode}; global STOP inactive; master idle")


def _dispatch(
    backend: BackendClient,
    mode: str,
    action: dict[str, Any],
    reason: str,
    evidence: dict[str, Any],
) -> dict[str, Any]:
    dispatched = backend.request(
        "POST",
        "/api/autonomy/dispatch",
        {"action": action, "reason": reason, "evidence": evidence},
    )
    if mode == "ASK_HUMAN":
        decision = dispatched.get("decision")
        if (
            dispatched.get("status") != "pending"
            or not isinstance(decision, dict)
            or decision.get("status") != "PENDING"
            or not isinstance(decision.get("id"), str)
        ):
            raise FeedbackLoopError("ASK_HUMAN did not create a pending decision")
        decision_id = decision["id"]
        print(
            f"decision_id={decision_id}; status=PENDING; "
            f"reason={decision.get('reason')}; "
            "evidence="
            + json.dumps(decision.get("evidence", {}), sort_keys=True)
        )
        approved = backend.request(
            "POST",
            f"/api/autonomy/decisions/{decision_id}/approve",
            timeout=900,
        )
        if approved.get("status") != "approved":
            raise FeedbackLoopError(
                f"Decision {decision_id} approval ended as {approved.get('status')}"
            )
        matching = [
            row
            for row in backend.request("GET", "/api/autonomy/history")
            if row.get("decision_id") == decision_id
        ]
        if not {"created", "approved"}.issubset({row.get("event") for row in matching}):
            raise FeedbackLoopError(f"Decision history is incomplete for {decision_id}")
        print(
            "decision_history="
            + json.dumps(
                [
                    {
                        "decision_id": row.get("decision_id"),
                        "event": row.get("event"),
                        "details": row.get("details"),
                        "created_at": row.get("created_at"),
                    }
                    for row in matching
                ],
                sort_keys=True,
            )
        )
        return approved
    if dispatched.get("status") != "executed":
        raise FeedbackLoopError(
            f"FULL_ACCESS dispatch did not execute: {dispatched.get('status')}"
        )
    return dispatched


def _wait_for_group(
    backend: BackendClient,
    settings: Settings,
    group_id: str,
    timeout_seconds: int = 1800,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_seconds
    last_status = "unknown"
    while time.monotonic() < deadline:
        results = backend.request("GET", "/api/slm/results")
        groups = results.get("groups")
        if not isinstance(groups, list):
            raise FeedbackLoopError("Master results omitted its groups")
        group = next(
            (
                row
                for row in groups
                if isinstance(row, dict) and row.get("group_id") == group_id
            ),
            None,
        )
        if group is None:
            raise FeedbackLoopError(f"Master results omitted group {group_id}")
        last_status = str(group.get("status", "unknown"))
        if last_status in {"completed", "failed", "stopped"}:
            if last_status != "completed":
                raise FeedbackLoopError(
                    f"Group {group_id} ended {last_status}: {group.get('error')}"
                )
            for worker in settings.nodes:
                if worker.role != "worker":
                    continue
                status = get_worker_training_status(worker)
                if status.get("active_run_id") is not None or status.get("scheduled_run_id") is not None:
                    raise FeedbackLoopError(
                        f"{worker.node_id} did not return idle after group {group_id}"
                    )
            return group
        time.sleep(3)
    raise FeedbackLoopError(
        f"Timed out waiting for group {group_id}; last status={last_status}"
    )


def _run_training(
    backend: BackendClient,
    settings: Settings,
    mode: str,
    *,
    dataset_name: str,
    rounds: int,
    epochs: int,
    loop_id: str,
    iteration: int,
) -> dict[str, Any]:
    _check_preflight(
        backend,
        settings,
        mode,
        dataset_name,
        allowed_loop_id=loop_id,
    )
    action = {
        "action": "start_training",
        "parameters": {
            "dataset_name": dataset_name,
            "rounds": rounds,
            "epochs": epochs,
        },
    }
    with _sample_tegrastats(settings, f"{mode.lower()}_iter{iteration}_training"):
        execution = _dispatch(
            backend,
            mode,
            action,
            f"M5 dataset feedback loop iteration {iteration}",
            {"loop_id": loop_id, "iteration": iteration, "dataset_name": dataset_name},
        )
        data = _action_data(execution)
        group_id = data.get("group_id")
        if not isinstance(group_id, str):
            raise FeedbackLoopError("Training action did not return a group_id")
        group = _wait_for_group(backend, settings, group_id)
    print(
        f"iteration={iteration}; group_id={group_id}; status=completed; "
        f"rounds={group.get('rounds')}; epochs={group.get('epochs')}; "
        f"duration_seconds={group.get('completed_at', 0) - group.get('started_at', 0):.2f}"
    )
    return group


def _suggest_and_execute(
    backend: BackendClient,
    settings: Settings,
    mode: str,
    *,
    group: dict[str, Any],
    loop_id: str,
    iteration: int,
    prior_accuracy: float | None,
    rounds: int,
    epochs: int,
) -> tuple[dict[str, Any], int, int, dict[str, Any] | None, bool]:
    prestarted_group: dict[str, Any] | None = None
    with _sample_tegrastats(settings, f"{mode.lower()}_iter{iteration}_feedback"):
        suggestion = backend.request(
            "POST",
            "/api/slm/suggest-next-action",
            {
                "group_id": group["group_id"],
                "iteration": iteration,
                "previous_accuracy": prior_accuracy,
            },
            timeout=settings.slm_timeout_seconds + 60,
        )
        attempts = suggestion.get("attempts", [])
        if isinstance(attempts, list):
            for attempt in attempts:
                if isinstance(attempt, dict):
                    print(
                        f"SLM attempt={attempt.get('attempt')}; "
                        f"validator={attempt.get('validator_result')}; "
                        "raw_output="
                        + json.dumps(
                            _redact_tokens(attempt.get("raw_output"), settings),
                            sort_keys=True,
                        )
                    )
        if suggestion.get("status") == "rejected":
            raise FeedbackLoopError(
                f"SLM suggestion rejected: {suggestion.get('reason')}"
            )
        if suggestion.get("status") == "needs_human":
            raise FeedbackLoopError(
                f"SLM suggestions were invalid twice; needs_human: "
                f"{suggestion.get('reason')}"
            )
        summary = suggestion.get("summary")
        expected_summary = build_training_feedback_summary(
            group,
            iteration=iteration,
            previous_accuracy=prior_accuracy,
        )
        if not isinstance(summary, dict) or summary != expected_summary:
            raise FeedbackLoopError("Backend feedback summary did not match run results")
        print("summary_sent_to_slm=" + json.dumps(summary, sort_keys=True))

        action = suggestion.get("action")
        if not isinstance(action, dict):
            raise FeedbackLoopError("Feedback endpoint omitted its validated action")
        params = action.get("parameters", {})
        print(
            f"source={suggestion.get('source')}; "
            f"SLM suggestion status={suggestion.get('status')}; "
            f"action={action.get('action')}; parameters="
            + json.dumps(params, sort_keys=True)
        )
        if action.get("action") == "start_training" and (
            params.get("rounds") != rounds or params.get("epochs") != epochs
        ):
            raise FeedbackLoopError(
                f"SLM training suggestion must preserve the required {rounds}x{epochs} run budget"
            )
        elif action.get("action") == "stop_training":
            backend.request(
                "POST",
                f"/api/autonomy/loops/{loop_id}/stop",
                {"reason": "The SLM recommended stopping after measured validation."},
            )
            print("loop stopped by validated SLM recommendation")
            return suggestion, rounds, epochs, None, True

        result = suggestion.get("result")
        if suggestion.get("status") == "pending":
            decision = result.get("decision") if isinstance(result, dict) else None
            if not isinstance(decision, dict) or not isinstance(decision.get("id"), str):
                raise FeedbackLoopError("Feedback suggestion has no pending decision id")
            decision_id = decision["id"]
            print(
                f"decision_id={decision_id}; status=PENDING; "
                f"reason={decision.get('reason')}; "
                "evidence="
                + json.dumps(decision.get("evidence", {}), sort_keys=True)
            )
            approved = backend.request(
                "POST",
                f"/api/autonomy/decisions/{decision_id}/approve",
                timeout=900,
            )
            if approved.get("status") != "approved":
                raise FeedbackLoopError(
                    f"SLM decision {decision_id} ended as {approved.get('status')}"
                )
            result = approved
        elif suggestion.get("status") != "executed":
            raise FeedbackLoopError(
                f"Unexpected SLM suggestion state: {suggestion.get('status')}"
            )
        else:
            print(
                "decision_history=[]; FULL_ACCESS executed the validated action "
                "without creating a pending approval decision"
            )

        if suggestion.get("status") == "pending":
            decision_id = suggestion["result"]["decision"]["id"]
            matching = [
                row
                for row in backend.request("GET", "/api/autonomy/history")
                if row.get("decision_id") == decision_id
            ]
            events = {row.get("event") for row in matching}
            if not {"created", "approved"}.issubset(events):
                raise FeedbackLoopError(f"Decision history is incomplete for {decision_id}")
            print(
                "decision_history="
                + json.dumps(
                    [
                        {
                            "decision_id": row.get("decision_id"),
                            "event": row.get("event"),
                            "details": row.get("details"),
                            "created_at": row.get("created_at"),
                        }
                        for row in matching
                    ],
                    sort_keys=True,
                )
            )

        if action.get("action") == "start_training":
            data = _action_data(result)
            next_group_id = data.get("group_id")
            if not isinstance(next_group_id, str):
                raise FeedbackLoopError("Suggested training action omitted group_id")
            prestarted_group = _wait_for_group(backend, settings, next_group_id)
        elif action.get("action") == "enable_nas" and params.get("enabled") is True:
            nas_data = _action_data(result)
            nas_group_id = nas_data.get("group_id")
            if not isinstance(nas_group_id, str):
                raise FeedbackLoopError("Suggested NAS action omitted group_id")
            _wait_for_group(backend, settings, nas_group_id)
    return suggestion, rounds, epochs, prestarted_group, False


def _run_loop(
    backend: BackendClient,
    settings: Settings,
    mode: str,
    dataset_name: str,
) -> None:
    state = backend.request("GET", "/api/autonomy/state")
    if state.get("mode") != mode:
        backend.request("PUT", "/api/autonomy/mode", {"mode": mode})
    _check_preflight(backend, settings, mode, dataset_name)
    loop = backend.request(
        "POST",
        "/api/autonomy/loops",
        {"max_iterations": 2, "time_limit_seconds": 1800, "stop_rule": "max_iterations"},
    )
    loop_id = loop["id"]
    previous_accuracy: float | None = None
    rounds, epochs = 3, 2
    prestarted_group: dict[str, Any] | None = None
    try:
        for iteration in range(1, 3):
            if prestarted_group is not None:
                group = prestarted_group
                prestarted_group = None
            else:
                group = _run_training(
                    backend,
                    settings,
                    mode,
                    dataset_name=dataset_name,
                    rounds=rounds,
                    epochs=epochs,
                    loop_id=loop_id,
                    iteration=iteration,
                )

            prior_accuracy = previous_accuracy
            summary = build_training_feedback_summary(
                group,
                iteration=iteration,
                previous_accuracy=prior_accuracy,
            )
            previous_accuracy = float(summary["accuracy"])
            print(
                f"run_metrics iteration={iteration}; rounds={summary['rounds']}; "
                f"epochs={summary['epochs']}; topology={summary['topology']}; "
                f"accuracy={summary['accuracy']:.6f}; "
                f"macro_f1={summary['macro_f1']:.6f}; "
                f"communication_bytes={summary['communication_bytes']}; "
                f"duration_seconds={summary['duration_seconds']:.2f}"
            )
            print("feedback_summary=" + json.dumps(summary, sort_keys=True))
            loop_state = backend.request(
                "POST",
                f"/api/autonomy/loops/{loop_id}/iterations",
                {"metric": previous_accuracy},
            )
            print(
                f"loop_id={loop_id}; iteration={loop_state['iteration']}; "
                f"status={loop_state['status']}; accuracy={previous_accuracy:.6f}"
            )
            if loop_state["status"] != "RUNNING":
                break

            if iteration < 2:
                _, rounds, epochs, prestarted_group, stop_recommended = (
                    _suggest_and_execute(
                        backend,
                        settings,
                        mode,
                        group=group,
                        loop_id=loop_id,
                        iteration=iteration,
                        prior_accuracy=prior_accuracy,
                        rounds=rounds,
                        epochs=epochs,
                    )
                )
                if stop_recommended:
                    return
    except Exception as error:
        state = backend.request("GET", "/api/autonomy/state")
        loop_state = next(
            (row for row in state.get("loops", []) if row.get("id") == loop_id),
            None,
        )
        if loop_state is not None and loop_state.get("status") == "RUNNING":
            backend.request(
                "POST",
                f"/api/autonomy/loops/{loop_id}/iterations",
                {"worker_failure": str(error)[:1000]},
            )
        raise


def _verify_global_stop(
    backend: BackendClient,
    settings: Settings,
    dataset_name: str,
) -> None:
    _check_preflight(backend, settings, "ASK_HUMAN", dataset_name)
    loop = backend.request(
        "POST",
        "/api/autonomy/loops",
        {"max_iterations": 2, "time_limit_seconds": 1800, "stop_rule": "max_iterations"},
    )
    loop_id = loop.get("id")
    if not isinstance(loop_id, str) or loop.get("status") != "RUNNING":
        raise FeedbackLoopError("Global STOP test did not start a running loop")
    print(f"global_STOP test: created running loop {loop_id}")
    try:
        stopped = backend.request("POST", "/api/autonomy/stop")
        matching = next(
            (row for row in stopped.get("loops", []) if row.get("id") == loop_id),
            None,
        )
        if (
            stopped.get("stop_requested") is not True
            or not isinstance(matching, dict)
            or matching.get("status") != "STOPPED"
        ):
            raise FeedbackLoopError("Global STOP did not stop the running loop")
        print(
            f"global_STOP test: loop_status={matching['status']}; "
            f"failure_reason={matching.get('failure_reason')}"
        )
        try:
            backend.request(
                "POST",
                "/api/autonomy/dispatch",
                {
                    "action": {"action": "set_rounds", "parameters": {"rounds": 1}},
                    "reason": "Verify global STOP blocks state-changing actions",
                    "evidence": {"test": "M5 global STOP acceptance"},
                },
            )
        except FeedbackLoopError as error:
            if "HTTP 423" not in str(error) or "Global STOP is active" not in str(error):
                raise
            print("global_STOP test: state-changing action rejected with HTTP 423")
        else:
            raise FeedbackLoopError("Global STOP did not block a state-changing action")
    finally:
        state = backend.request("GET", "/api/autonomy/state")
        if state.get("stop_requested") is True:
            backend.request("POST", "/api/autonomy/reset-stop")
        state = backend.request("GET", "/api/autonomy/state")
        if state.get("mode") != "ASK_HUMAN":
            backend.request("PUT", "/api/autonomy/mode", {"mode": "ASK_HUMAN"})
        final_state = backend.request("GET", "/api/autonomy/state")
        if final_state.get("stop_requested") is not False:
            raise FeedbackLoopError("Global STOP test could not reset global STOP")
        print(
            f"global_STOP test cleanup: mode={final_state.get('mode')}; "
            f"global_STOP={final_state.get('stop_requested')}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-name", default="cifar_10k_demo")
    parser.add_argument("--stop-test-only", action="store_true")
    args = parser.parse_args()
    settings = Settings.from_env()
    backend = BackendClient("http://127.0.0.1:8000", settings.api_token)
    initial_state = backend.request("GET", "/api/autonomy/state")
    if initial_state.get("mode") != "ASK_HUMAN":
        raise SystemExit("M5 runner requires ASK_HUMAN as the initial autonomy mode")
    if args.stop_test_only:
        _verify_global_stop(backend, settings, args.dataset_name)
        return
    try:
        _run_loop(backend, settings, "ASK_HUMAN", args.dataset_name)
        _run_loop(backend, settings, "FULL_ACCESS", args.dataset_name)
        state = backend.request("GET", "/api/autonomy/state")
        if state.get("mode") != "ASK_HUMAN":
            backend.request("PUT", "/api/autonomy/mode", {"mode": "ASK_HUMAN"})
        _verify_global_stop(backend, settings, args.dataset_name)
    finally:
        state = backend.request("GET", "/api/autonomy/state")
        if state.get("mode") != "ASK_HUMAN":
            backend.request("PUT", "/api/autonomy/mode", {"mode": "ASK_HUMAN"})
        final_state = backend.request("GET", "/api/autonomy/state")
        if final_state.get("stop_requested") is not False:
            raise FeedbackLoopError("Autonomy global STOP is active after M5 cleanup")
        print(
            f"cleanup: mode={final_state.get('mode')}; "
            f"global_STOP={final_state.get('stop_requested')}"
        )


if __name__ == "__main__":
    main()
