"""Run the bounded two-iteration MNIST acceptance under temporary FULL_ACCESS."""

from __future__ import annotations

import json
import argparse
import os
import re
import shlex
import subprocess
import sys
import time
import uuid
from datetime import datetime
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "dtn" / "backend"))

from app.core.config import Settings  # noqa: E402
from app.services.autonomy import validate_action  # noqa: E402
from app.services.feedback import build_training_feedback_summary  # noqa: E402
from app.services.worker_shards import (  # noqa: E402
    get_worker_dataset_initial_model,
    get_worker_training_status,
)


BACKEND_URL = "http://127.0.0.1:8000"
DATASET_NAME = "mnist_demo"
EXPECTED_WORKERS = ("worker1", "worker2", "worker3")
EXPECTED_TOPOLOGY = "bidirectional_ring"
TRAINING_ROUNDS = 5
TRAINING_EPOCHS = 20
TRAINING_TIMEOUT_SECONDS = 15 * 60
LOOP_TIMEOUT_SECONDS = 20 * 60
ITERATION_LIMIT = 2
ACCURACY_ESTIMATE_SECONDS = 360.0
RAM_PATTERN = re.compile(r"\bRAM\s+(\d+)/(\d+)MB\b")
SWAP_PATTERN = re.compile(r"\bSWAP\s+(\d+)/(\d+)MB\b")


class AcceptanceError(RuntimeError):
    pass


class TransientBackendError(AcceptanceError):
    pass


class BackendClient:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.base_url = BACKEND_URL

    def request(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
        *,
        deadline: float | None = None,
        timeout: float = 30.0,
    ) -> Any:
        if deadline is not None:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"Overall loop deadline reached before {method} {path}")
            timeout = min(timeout, remaining)
        body = json.dumps(payload).encode("utf-8") if payload is not None else None
        request = Request(
            self.base_url + path,
            data=body,
            headers={
                "Authorization": f"Bearer {self.settings.api_token}",
                "Accept": "application/json",
                "Content-Type": "application/json",
            },
            method=method,
        )
        try:
            with urlopen(request, timeout=timeout) as response:
                result = json.loads(response.read().decode("utf-8"))
        except HTTPError as error:
            if error.code == 408 or error.code >= 500:
                raise TransientBackendError(
                    f"Backend {method} {path} returned transient HTTP {error.code}"
                ) from error
            raise AcceptanceError(f"Backend {method} {path} returned HTTP {error.code}") from error
        except (URLError, TimeoutError, OSError) as error:
            raise TransientBackendError(
                f"Backend {method} {path} failed: {type(error).__name__}"
            ) from error
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise AcceptanceError(f"Backend {method} {path} returned invalid JSON") from error
        if not isinstance(result, (dict, list)):
            raise AcceptanceError(f"Backend {method} {path} returned an unexpected JSON value")
        return result


def _redact(value: Any, settings: Settings) -> Any:
    tokens = (
        settings.api_token,
        settings.master_slm_token,
        *(node.token for node in settings.nodes),
    )
    if isinstance(value, str):
        result = value
        for token in tokens:
            if token:
                result = result.replace(token, "[REDACTED]")
        return re.sub(
            r"(?i)\bBearer\s+[A-Za-z0-9._~+/-]+=*",
            "Bearer [REDACTED]",
            result,
        )
    if isinstance(value, dict):
        return {key: _redact(item, settings) for key, item in value.items()}
    if isinstance(value, list):
        return [_redact(item, settings) for item in value]
    return value


def _safe_error(error: BaseException, settings: Settings) -> str:
    return str(_redact(f"{type(error).__name__}: {error}", settings))


def _check_deadline(deadline: float, step: str) -> float:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError(f"20-minute loop deadline reached at {step}")
    return remaining


def _worker_nodes(settings: Settings) -> list[Any]:
    workers = sorted(
        (node for node in settings.nodes if node.role == "worker"),
        key=lambda node: node.node_id,
    )
    if tuple(node.node_id for node in workers) != EXPECTED_WORKERS:
        raise AcceptanceError("Expected exactly worker1, worker2, and worker3")
    return workers


def _preflight(
    backend: BackendClient,
    settings: Settings,
    *,
    expected_mode: str,
    require_no_pending: bool = True,
) -> dict[str, Any]:
    state = backend.request("GET", "/api/autonomy/state")
    if state.get("mode") != expected_mode or state.get("stop_requested") is not False:
        raise AcceptanceError(
            f"Expected mode={expected_mode} and global STOP inactive"
        )
    if require_no_pending and state.get("pending_decisions") != 0:
        raise AcceptanceError("Pending autonomy decisions exist")
    if any(loop.get("status") == "RUNNING" for loop in state.get("loops", [])):
        raise AcceptanceError("An autonomy loop is already running")

    results = backend.request("GET", "/api/slm/results")
    if results.get("active_group_id") is not None:
        raise AcceptanceError("Master already has an active training or NAS group")
    master = backend.request("GET", "/api/slm/status")
    if master.get("node_id") != "master":
        raise AcceptanceError("Authenticated master status returned an unexpected node")

    for worker in _worker_nodes(settings):
        status = get_worker_training_status(worker)
        if status.get("active_run_id") is not None or status.get("scheduled_run_id") is not None:
            raise AcceptanceError(f"{worker.node_id} is not idle")
    print(
        f"preflight: mode={state['mode']}; STOP={state['stop_requested']}; "
        f"pending={state['pending_decisions']}; running_loops=0; "
        "master and all workers idle"
    )
    return state


def _verify_initial_models(settings: Settings) -> None:
    expected_architecture = (
        "b66616d37317a211cd9dc2cb8d02b6213cfc292d049a2258dc7711d1e7f73bb2"
    )
    expected_weights = "d23e3d771bb065c347655ac932ab90cc3a8ce12ea01345c533ffca004be78d75"
    for worker in _worker_nodes(settings):
        model = get_worker_dataset_initial_model(worker, DATASET_NAME)
        architecture = model.get("architecture_sha256")
        weights = model.get("weights_sha256")
        print(
            f"{worker.node_id}: architecture_sha256={architecture}; "
            f"weights_sha256={weights}"
        )
        if architecture != expected_architecture or weights != expected_weights:
            raise AcceptanceError(f"{worker.node_id} MNIST initial-model hash mismatch")


def _check_listener_source_mtime() -> None:
    source_path = ROOT / "dtn" / "backend" / "app" / "services" / "autonomy.py"
    powershell = r"""
$ErrorActionPreference = 'Stop'
$listener = Get-NetTCPConnection -LocalPort 8000 -State Listen | Select-Object -First 1
if (-not $listener) { throw 'No listener on port 8000' }
$process = Get-CimInstance Win32_Process -Filter "ProcessId=$($listener.OwningProcess)"
if (-not $process -or $process.Name -notmatch '^python(?:\.exe)?$' -or $process.CommandLine -notmatch '(?i)-m\s+app\.main') {
    throw 'Port 8000 listener is not python -m app.main'
}
$start = (Get-Process -Id $process.ProcessId -ErrorAction Stop).StartTime.ToString('yyyy-MM-ddTHH:mm:ss.ffffffzzz')
[pscustomobject]@{ PID = $process.ProcessId; StartTime = $start } | ConvertTo-Json -Compress
"""
    try:
        result = subprocess.run(
            [
                "powershell.exe",
                "-NoLogo",
                "-NoProfile",
                "-NonInteractive",
                "-Command",
                powershell,
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=15,
            check=False,
        )
    except subprocess.TimeoutExpired as error:
        raise AcceptanceError("Timed out inspecting backend listener start time") from error
    if result.returncode:
        raise AcceptanceError("Could not verify the port-8000 backend listener")
    try:
        listener = json.loads(result.stdout)
        listener_start = datetime.fromisoformat(listener["StartTime"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise AcceptanceError("Listener start-time query returned invalid data") from error
    if listener_start.tzinfo is None:
        raise AcceptanceError("Listener StartTime did not include a timezone offset")
    source_write = datetime.fromtimestamp(source_path.stat().st_mtime).astimezone()
    print(
        f"listener_source_check: pid={listener['PID']}; "
        f"StartTime={listener_start.isoformat()}; "
        f"autonomy.py LastWriteTime={source_write.isoformat()}"
    )
    if listener_start.astimezone() <= source_write:
        raise AcceptanceError("Backend listener predates autonomy.py LastWriteTime")


def _ssh(user: str, host: str, command: str, timeout: float) -> str:
    try:
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
            timeout=max(1.0, timeout),
            check=False,
        )
    except subprocess.TimeoutExpired as error:
        raise TimeoutError(f"SSH to {host} timed out") from error
    if process.returncode:
        raise AcceptanceError(f"SSH to {host} failed with exit {process.returncode}")
    return process.stdout


@dataclass
class TegraStatsSampler:
    settings: Settings
    deadline: float
    verbose: bool = True
    nodes: list[Any] = field(init=False)
    ssh_user: str = field(init=False)
    run_id: str = field(init=False)
    prefixes: dict[str, str] = field(init=False, default_factory=dict)
    failures: list[str] = field(init=False, default_factory=list)

    def __post_init__(self) -> None:
        self.nodes = sorted(
            (node for node in self.settings.nodes if node.role in {"master", "worker"}),
            key=lambda node: node.node_id,
        )
        if tuple(node.node_id for node in self.nodes) != ("master", *EXPECTED_WORKERS):
            raise AcceptanceError("Expected master and exactly three configured workers")
        self.ssh_user = os.environ.get("M4_SSH_USER", "hammad")
        self.run_id = uuid.uuid4().hex[:10]

    def __enter__(self) -> "TegraStatsSampler":
        try:
            for node in self.nodes:
                remaining = _check_deadline(self.deadline, f"starting tegrastats on {node.node_id}")
                prefix = f"/tmp/dfl-mnist-full-access-{self.run_id}-{node.node_id}"
                log_path, pid_path = f"{prefix}.log", f"{prefix}.pid"
                shell = (
                    "command -v tegrastats >/dev/null 2>&1 || exit 127; "
                    f"nohup tegrastats --interval 1000 > {shlex.quote(log_path)} 2>&1 "
                    f"</dev/null & echo $! > {shlex.quote(pid_path)}; "
                    f"sleep 1; kill -0 \"$(cat {shlex.quote(pid_path)})\""
                )
                self.prefixes[node.node_id] = prefix
                try:
                    self._ssh_node(
                        node,
                        "sh -c " + shlex.quote(shell),
                        min(30.0, remaining),
                    )
                    if self.verbose:
                        print(f"tegrastats: {node.node_id} sampling started")
                except Exception as error:
                    self.failures.append(
                        f"{node.node_id}: {_safe_error(error, self.settings)}"
                    )
            if self.failures:
                print("WARNING: tegrastats startup failed: " + "; ".join(self.failures))
            return self
        except BaseException:
            self._stop()
            raise

    def _ssh_node(self, node: Any, command: str, timeout: float) -> str:
        return _ssh(self.ssh_user, node.ip, command, timeout)

    def _stop(self) -> None:
        failures = list(self.failures)
        self.failures = []
        for node in self.nodes:
            prefix = self.prefixes.get(node.node_id)
            if prefix is None:
                continue
            log_path, pid_path = f"{prefix}.log", f"{prefix}.pid"
            shell = (
                f"if test -f {shlex.quote(pid_path)}; then "
                f"kill \"$(cat {shlex.quote(pid_path)})\" 2>/dev/null || true; fi; "
                "sleep 1; "
                f"if test -f {shlex.quote(log_path)}; then cat {shlex.quote(log_path)}; fi; "
                f"rm -f -- {shlex.quote(pid_path)} {shlex.quote(log_path)}"
            )
            try:
                output = self._ssh_node(node, "sh -c " + shlex.quote(shell), 45.0)
            except Exception as error:
                failures.append(f"{node.node_id}: {_safe_error(error, self.settings)}")
                continue
            ram = [int(match.group(1)) for match in RAM_PATTERN.finditer(output)]
            swap = [int(match.group(1)) for match in SWAP_PATTERN.finditer(output)]
            if not ram or not swap:
                failures.append(f"{node.node_id}: tegrastats produced no RAM/SWAP samples")
                continue
            print(
                f"tegrastats {node.node_id}: peak_RAM_MB={max(ram)}; "
                f"peak_swap_used_MB={max(swap)}"
            )
        self.prefixes.clear()
        self.failures = failures
        if failures:
            message = "; ".join(failures)
            self.failures = failures
            print(f"WARNING: tegrastats sampling/cleanup failed: {message}")

    def __exit__(self, error_type: Any, error: Any, traceback: Any) -> bool:
        try:
            self._stop()
        except Exception as cleanup_error:
            self.failures.append(_safe_error(cleanup_error, self.settings))
            print(f"WARNING: tegrastats cleanup failed: {self.failures[-1]}")
        return False


def _action_data(value: Any) -> dict[str, Any]:
    current = value
    for _ in range(5):
        if not isinstance(current, dict):
            break
        data = current.get("data")
        if isinstance(data, dict):
            return data
        nested = current.get("result")
        if isinstance(nested, dict):
            current = nested
            continue
        break
    raise AcceptanceError("Action result did not contain a data object")


def _find_group_id(value: Any) -> str | None:
    if isinstance(value, dict):
        candidate = value.get("group_id")
        if isinstance(candidate, str) and candidate:
            return candidate
        for nested in value.values():
            found = _find_group_id(nested)
            if found is not None:
                return found
    elif isinstance(value, list):
        for nested in value:
            found = _find_group_id(nested)
            if found is not None:
                return found
    return None


def _active_group_id(backend: BackendClient) -> str | None:
    results = backend.request("GET", "/api/slm/results", timeout=15.0)
    active = results.get("active_group_id")
    if active is not None and not isinstance(active, str):
        raise AcceptanceError("Master active_group_id is malformed")
    return active


def _cancel_active_group(backend: BackendClient) -> None:
    group_id = _active_group_id(backend)
    if group_id is None:
        return
    stop = backend.request(
        "POST",
        "/api/autonomy/dispatch",
        {
            "action": {"action": "stop_training", "parameters": {}},
            "reason": f"Abort acceptance and cancel active group {group_id}",
            "evidence": {"group_id": group_id, "abort_cleanup": True},
        },
        timeout=30.0,
    )
    if stop.get("status") != "executed":
        raise AcceptanceError(
            f"Abort cleanup stop_training returned {stop.get('status')}; not approving"
        )
    deadline = time.monotonic() + 60.0
    while time.monotonic() < deadline:
        results = backend.request("GET", "/api/slm/results", timeout=15.0)
        groups = results.get("groups", [])
        group = next(
            (
                row
                for row in groups
                if isinstance(row, dict) and row.get("group_id") == group_id
            ),
            None,
        )
        if group is not None and group.get("status") in {"completed", "failed", "stopped"}:
            print(f"abort cleanup: group={group_id}; terminal_status={group['status']}")
            return
        time.sleep(2.0)
    raise TimeoutError(f"Abort cleanup: group {group_id} did not reach terminal state within 60 seconds")


def _poll_group(
    backend: BackendClient,
    group_id: str,
    settings: Settings,
    *,
    loop_deadline: float,
) -> dict[str, Any]:
    run_deadline = min(loop_deadline, time.monotonic() + TRAINING_TIMEOUT_SECONDS)
    consecutive_failures = 0
    try:
        while True:
            remaining = _check_deadline(run_deadline, f"polling group {group_id}")
            try:
                results = backend.request(
                    "GET",
                    "/api/slm/results",
                    deadline=loop_deadline,
                    timeout=min(20.0, remaining),
                )
                consecutive_failures = 0
            except TransientBackendError as error:
                consecutive_failures += 1
                if consecutive_failures >= 3:
                    raise AcceptanceError(
                        f"Three consecutive transient errors polling {group_id}: {error}"
                    ) from error
                time.sleep(min(2.0, max(0.0, run_deadline - time.monotonic())))
                continue
            groups = results.get("groups")
            if not isinstance(groups, list):
                raise AcceptanceError("Master results omitted its groups list")
            group = next(
                (
                    row
                    for row in groups
                    if isinstance(row, dict) and row.get("group_id") == group_id
                ),
                None,
            )
            if group is None:
                raise AcceptanceError(f"Master results omitted group {group_id}")
            last_status = str(group.get("status", "unknown"))
            if last_status in {"completed", "failed", "stopped"}:
                if last_status != "completed":
                    raise AcceptanceError(
                        f"Group {group_id} ended {last_status}: "
                        f"{_redact(group.get('error'), settings)}"
                    )
                return group
            time.sleep(min(3.0, max(0.0, run_deadline - time.monotonic())))
    except TimeoutError:
        cancel = backend.request(
            "POST",
            "/api/autonomy/dispatch",
            {
                "action": {"action": "stop_training", "parameters": {}},
                "reason": f"Cancel MNIST group {group_id} after its polling deadline",
                "evidence": {"group_id": group_id, "timeout_seconds": TRAINING_TIMEOUT_SECONDS},
            },
            timeout=30.0,
        )
        if cancel.get("status") != "executed":
            raise AcceptanceError(
                f"Timed-out group {group_id} could not be cancelled; status={cancel.get('status')}"
            )
        cancel_deadline = time.monotonic() + 60.0
        while time.monotonic() < cancel_deadline:
            results = backend.request("GET", "/api/slm/results", timeout=15.0)
            groups = results.get("groups", [])
            group = next(
                (
                    row
                    for row in groups
                    if isinstance(row, dict) and row.get("group_id") == group_id
                ),
                None,
            )
            if group is not None and group.get("status") in {"completed", "failed", "stopped"}:
                raise TimeoutError(
                    f"Group {group_id} exceeded its deadline and was cancelled; "
                    f"terminal_status={group.get('status')}"
                )
            time.sleep(2.0)
        raise TimeoutError(f"Group {group_id} exceeded its deadline; cancellation did not reach terminal state")


def _group_metrics(group: dict[str, Any]) -> tuple[dict[str, Any], dict[str, float]]:
    if (
        group.get("status") != "completed"
        or group.get("dataset_name") != DATASET_NAME
        or group.get("topology") != EXPECTED_TOPOLOGY
    ):
        raise AcceptanceError("Completed group does not match expected dataset/topology")
    summary = build_training_feedback_summary(group, iteration=1)
    runs = group.get("runs")
    if not isinstance(runs, list) or not runs:
        raise AcceptanceError("Completed group has no runs")
    per_worker: dict[str, list[float]] = {node: [] for node in EXPECTED_WORKERS}
    for run in runs:
        rounds = run.get("rounds") if isinstance(run, dict) else None
        if not isinstance(rounds, list) or not rounds:
            raise AcceptanceError("Completed run is missing round metrics")
        workers = rounds[-1].get("workers") if isinstance(rounds[-1], dict) else None
        if not isinstance(workers, dict):
            raise AcceptanceError("Final round is missing per-worker metrics")
        for worker_id in EXPECTED_WORKERS:
            metrics = workers.get(worker_id)
            accuracy = metrics.get("validation_accuracy") if isinstance(metrics, dict) else None
            if not isinstance(accuracy, (int, float)):
                raise AcceptanceError(f"Final accuracy is missing for {worker_id}")
            per_worker[worker_id].append(float(accuracy))
    return summary, {node: sum(values) / len(values) for node, values in per_worker.items()}


def _history_identity(row: dict[str, Any]) -> tuple[str, str] | None:
    event = row.get("event")
    row_id = row.get("id")
    if not isinstance(event, str) or row_id is None:
        return None
    return event, str(row_id)


def _history_ids(backend: BackendClient) -> set[tuple[str, str]]:
    history = backend.request("GET", "/api/autonomy/history")
    if not isinstance(history, list):
        raise AcceptanceError("Autonomy history response is not a list")
    return {
        identity
        for row in history
        if isinstance(row, dict)
        and (identity := _history_identity(row)) is not None
    }


def _new_automatic_history(
    backend: BackendClient,
    old_ids: set[tuple[str, str]],
    *,
    action: str,
    deadline: float,
) -> list[dict[str, Any]]:
    history = backend.request("GET", "/api/autonomy/history", deadline=deadline)
    if not isinstance(history, list):
        raise AcceptanceError("Autonomy history response is not a list")
    new_rows = [
        row
        for row in history
        if isinstance(row, dict)
        and _history_identity(row) not in old_ids
        and row.get("event") == "executed automatically"
        and row.get("mode") == "FULL_ACCESS"
        and row.get("action") == action
    ]
    if not new_rows:
        raise AcceptanceError(f"No automatic FULL_ACCESS audit event found for {action}")
    return new_rows


def _set_topology(
    backend: BackendClient,
    old_history_ids: set[Any],
    deadline: float,
) -> None:
    action = {
        "action": "set_topology",
        "parameters": {"topology": EXPECTED_TOPOLOGY},
    }
    validate_action(action)
    result = backend.request(
        "POST",
        "/api/autonomy/dispatch",
        {
            "action": action,
            "reason": "Configure required topology before MNIST training",
            "evidence": {"dataset_name": DATASET_NAME},
        },
        deadline=deadline,
    )
    if result.get("status") != "executed":
        raise AcceptanceError(
            f"set_topology dispatch returned {result.get('status')}; not approving"
        )
    data = _action_data(result.get("result"))
    if data.get("topology") != EXPECTED_TOPOLOGY:
        raise AcceptanceError("set_topology result did not confirm bidirectional_ring")
    events = _new_automatic_history(
        backend,
        old_history_ids,
        action="set_topology",
        deadline=deadline,
    )
    print(f"topology_dispatch=executed; topology={data['topology']}")
    print("topology_automatic_history=" + json.dumps(events, sort_keys=True))


def _start_training(
    backend: BackendClient,
    settings: Settings,
    loop_id: str,
    loop_deadline: float,
    *,
    iteration: int,
    old_history_ids: set[Any],
    parameters: dict[str, Any],
) -> dict[str, Any]:
    if parameters.get("dataset_name") != DATASET_NAME:
        raise AcceptanceError("Training action selected an unexpected dataset")
    if parameters.get("rounds") != TRAINING_ROUNDS or parameters.get("epochs") != TRAINING_EPOCHS:
        raise AcceptanceError("Training action did not match the authorized 5x20 run")
    if "topology" in parameters:
        raise AcceptanceError("start_training schema unexpectedly accepted a topology parameter")

    # Topology is configured separately via the autonomy action and verified on the group.
    action = {
        "action": "start_training",
        "parameters": {
            "dataset_name": DATASET_NAME,
            "rounds": TRAINING_ROUNDS,
            "epochs": TRAINING_EPOCHS,
        },
    }
    validate_action(action)
    _check_deadline(loop_deadline, f"dispatching iteration {iteration}")
    with TegraStatsSampler(settings, loop_deadline):
        response = backend.request(
            "POST",
            "/api/autonomy/dispatch",
            {
                "action": action,
                "reason": f"MNIST FULL_ACCESS acceptance iteration {iteration}",
                "evidence": {"loop_id": loop_id, "iteration": iteration},
            },
            deadline=loop_deadline,
            timeout=60.0,
        )
        status = response.get("status")
        print(f"iteration={iteration}; dispatch_status={status}")
        if status == "pending":
            raise AcceptanceError("FULL_ACCESS dispatch unexpectedly returned pending; not approving")
        if status != "executed":
            raise AcceptanceError(f"Training dispatch returned unexpected status {status}")
        data = _action_data(response.get("result"))
        group_id = data.get("group_id")
        if not isinstance(group_id, str) or not group_id:
            raise AcceptanceError("Executed training action did not return a group_id")
        print(f"iteration={iteration}; group_id={group_id}; polling_timeout_seconds=900")
        group = _poll_group(
            backend,
            group_id,
            settings,
            loop_deadline=loop_deadline,
        )

    if group.get("rounds") != TRAINING_ROUNDS or group.get("epochs") != TRAINING_EPOCHS:
        raise AcceptanceError("Completed group did not use the authorized 5x20 run")
    summary, per_worker = _group_metrics(group)
    duration = float(group["completed_at"] - group["started_at"])
    print(
        f"iteration={iteration}; group_status=completed; "
        "validation split metric (160 imgs/worker): "
        f"accuracy={summary['accuracy']:.6f}; "
        f"macro_f1={summary['macro_f1']:.6f}; "
        f"per_worker_accuracy={json.dumps(per_worker, sort_keys=True)}; "
        f"communication_bytes={summary['communication_bytes']}; "
        f"duration_seconds={duration:.2f}; "
        f"360-second estimate comparison={duration - ACCURACY_ESTIMATE_SECONDS:+.2f}s "
        "(360 s is an estimate, not a measurement)"
    )
    automatic_events = _new_automatic_history(
        backend,
        old_history_ids,
        action="start_training",
        deadline=loop_deadline,
    )
    print("automatic_history=" + json.dumps(automatic_events, sort_keys=True))
    return {"group": group, "summary": summary, "per_worker_accuracy": per_worker}


def _check_recent_group(backend: BackendClient) -> None:
    results = backend.request("GET", "/api/slm/results")
    groups = results.get("groups")
    if not isinstance(groups, list):
        raise AcceptanceError("Master results omitted its groups list")
    candidates = [
        group
        for group in groups
        if isinstance(group, dict)
        and group.get("status") == "completed"
        and group.get("dataset_name") == DATASET_NAME
    ]
    if not candidates:
        raise AcceptanceError("No completed mnist_demo group is available for --check")
    group = max(candidates, key=lambda row: float(row.get("completed_at", 0)))
    summary, per_worker = _group_metrics(group)
    started_at = float(group["started_at"])
    completed_at = float(group["completed_at"])
    if completed_at < started_at:
        raise AcceptanceError("Latest completed MNIST group has invalid duration timestamps")
    print(
        f"check_recent_group={group['group_id']}; "
        "validation split metric (160 imgs/worker): "
        f"accuracy={summary['accuracy']:.6f}; macro_f1={summary['macro_f1']:.6f}; "
        f"per_worker_accuracy={json.dumps(per_worker, sort_keys=True)}; "
        f"communication_bytes={summary['communication_bytes']}; "
        f"duration_seconds={completed_at - started_at:.2f}"
    )


def _check_history_shape(backend: BackendClient) -> None:
    history = backend.request("GET", "/api/autonomy/history")
    if not isinstance(history, list):
        raise AcceptanceError("Autonomy history response is not a list")
    keys = sorted({key for row in history if isinstance(row, dict) for key in row})
    print(f"autonomy_history_shape: item_keys={json.dumps(keys)}")


def _run_read_only_check(settings: Settings, backend: BackendClient) -> None:
    _check_listener_source_mtime()
    _preflight(backend, settings, expected_mode="ASK_HUMAN")
    _verify_initial_models(settings)
    _check_recent_group(backend)

    sampler = TegraStatsSampler(
        settings,
        time.monotonic() + 120.0,
        verbose=False,
    )
    try:
        with sampler:
            time.sleep(3.0)
    except Exception as error:
        message = _safe_error(error, settings)
        print(f"tegrastats_non_sudo=unavailable; error={message}")
        raise AcceptanceError(f"Read-only tegrastats check failed without sudo: {message}") from error
    if sampler.failures:
        message = "; ".join(sampler.failures)
        print(f"tegrastats_non_sudo=unavailable; error={message}")
        raise AcceptanceError(f"Read-only tegrastats check failed without sudo: {message}")
    print("tegrastats_non_sudo=available; sampling_seconds=3; sudo_used=False")
    _check_history_shape(backend)
    print("--check=PASS; no mode change, dispatch, or training performed")


def _stop_loop(backend: BackendClient, loop_id: str, reason: str) -> None:
    state = backend.request("GET", "/api/autonomy/state")
    loop = next(
        (row for row in state.get("loops", []) if row.get("id") == loop_id),
        None,
    )
    if loop is not None and loop.get("status") == "RUNNING":
        stopped = backend.request(
            "POST",
            f"/api/autonomy/loops/{loop_id}/stop",
            {"reason": reason[:1000]},
        )
        if stopped.get("status") != "STOPPED":
            raise AcceptanceError("Could not stop the bounded autonomy loop")


def _run_feedback_iteration(
    backend: BackendClient,
    settings: Settings,
    loop_id: str,
    loop_deadline: float,
    first_group: dict[str, Any],
    first_summary: dict[str, Any],
    old_history_ids: set[Any],
) -> None:
    if settings.feedback_rule_fallback_enabled:
        raise AcceptanceError("Rule fallback must remain OFF for this acceptance")
    summary = build_training_feedback_summary(
        first_group,
        iteration=1,
        previous_accuracy=None,
    )
    print("feedback_summary=" + json.dumps(summary, sort_keys=True))
    _check_deadline(loop_deadline, "requesting feedback suggestion")
    # The backend validates before dispatch and returns needs_human without dispatch when invalid.
    with TegraStatsSampler(settings, loop_deadline):
        response = backend.request(
            "POST",
            "/api/slm/suggest-next-action",
            {
                "group_id": first_group["group_id"],
                "iteration": 1,
                "previous_accuracy": None,
            },
            deadline=loop_deadline,
            timeout=min(settings.slm_timeout_seconds + 60.0, 240.0),
        )
        attempts = response.get("attempts", [])
        if not isinstance(attempts, list) or not attempts:
            raise AcceptanceError("Feedback endpoint returned no SLM attempts")
        for attempt in attempts:
            if not isinstance(attempt, dict):
                raise AcceptanceError("Feedback attempt record is malformed")
            print(
                f"SLM attempt={attempt.get('attempt')}; validator_result="
                f"{_redact(attempt.get('validator_result'), settings)}; raw_output="
                + json.dumps(_redact(attempt.get("raw_output"), settings), sort_keys=True)
            )
        returned_summary = response.get("summary")
        if returned_summary != summary:
            raise AcceptanceError("Feedback endpoint summary differs from the measured group")
        source = response.get("source")
        status = response.get("status")
        print(f"feedback_source={source}; feedback_status={status}; rule_fallback=False")
        if source != "slm":
            raise AcceptanceError(f"Unexpected feedback source {source}")
        if status == "needs_human":
            print(f"needs_human: {_redact(response.get('reason'), settings)}")
            return
        if status == "pending":
            raise AcceptanceError("Feedback unexpectedly created a pending decision; not approving")
        if status != "executed":
            raise AcceptanceError(f"Feedback returned unexpected status {status}")

        action = response.get("action")
        if not isinstance(action, dict):
            raise AcceptanceError("Feedback omitted its schema-validated action")
        try:
            validate_action(action)
        except Exception as error:
            raise AcceptanceError("Feedback endpoint claimed execution of an invalid action") from error
        if attempts[-1].get("validator_result") != "valid":
            raise AcceptanceError("Executed feedback action lacks a valid validator result")
        action_name = action.get("action")
        automatic_events = _new_automatic_history(
            backend,
            old_history_ids,
            action=action_name,
            deadline=loop_deadline,
        )
        print("feedback_automatic_history=" + json.dumps(automatic_events, sort_keys=True))

        group_id = _find_group_id(response.get("result"))
        metric = float(first_summary["accuracy"])
        if group_id is not None:
            print(f"iteration=2; group_id={group_id}; polling_timeout_seconds=900")
            second_group = _poll_group(
                backend,
                group_id,
                settings,
                loop_deadline=loop_deadline,
            )
            print(f"iteration=2; group_status={second_group.get('status')}")
            if (
                second_group.get("kind") == "decentralized"
                and second_group.get("dataset_name") == DATASET_NAME
            ):
                second_summary, per_worker = _group_metrics(second_group)
                metric = float(second_summary["accuracy"])
                print(
                    "iteration=2; validation split metric (160 imgs/worker): "
                    f"accuracy={second_summary['accuracy']:.6f}; "
                    f"macro_f1={second_summary['macro_f1']:.6f}; "
                    f"per_worker_accuracy={json.dumps(per_worker, sort_keys=True)}; "
                    f"communication_bytes={second_summary['communication_bytes']}"
                )

    iteration_state = backend.request(
        "POST",
        f"/api/autonomy/loops/{loop_id}/iterations",
        {"metric": metric},
        deadline=loop_deadline,
    )
    if iteration_state.get("iteration") != ITERATION_LIMIT or iteration_state.get("status") != "STOPPED":
        raise AcceptanceError("Two-iteration autonomy loop did not stop at its configured limit")


def _main_run(settings: Settings, backend: BackendClient) -> None:
    if settings.feedback_rule_fallback_enabled:
        raise AcceptanceError("Rule fallback must be OFF before switching to FULL_ACCESS")
    _preflight(backend, settings, expected_mode="ASK_HUMAN")
    _verify_initial_models(settings)
    old_history_ids = _history_ids(backend)

    loop_id: str | None = None
    overall_deadline = time.monotonic() + LOOP_TIMEOUT_SECONDS
    try:
        full_state = backend.request(
            "PUT",
            "/api/autonomy/mode",
            {"mode": "FULL_ACCESS"},
            deadline=overall_deadline,
        )
        if full_state.get("mode") != "FULL_ACCESS" or full_state.get("stop_requested") is not False:
            raise AcceptanceError("Backend did not enter FULL_ACCESS with STOP inactive")
        print(
            f"mode_state: mode={full_state['mode']}; "
            f"STOP={full_state['stop_requested']}; "
            f"pending={full_state.get('pending_decisions')}"
        )
        _set_topology(backend, old_history_ids, overall_deadline)

        loop = backend.request(
            "POST",
            "/api/autonomy/loops",
            {
                "max_iterations": ITERATION_LIMIT,
                "time_limit_seconds": LOOP_TIMEOUT_SECONDS,
                "stop_rule": "max_iterations",
            },
            deadline=overall_deadline,
        )
        loop_id = loop.get("id")
        if (
            not isinstance(loop_id, str)
            or loop.get("status") != "RUNNING"
            or loop.get("max_iterations") != ITERATION_LIMIT
            or loop.get("time_limit_seconds") != LOOP_TIMEOUT_SECONDS
        ):
            raise AcceptanceError("Backend did not create the requested bounded autonomy loop")

        iteration_one = _start_training(
            backend,
            settings,
            loop_id,
            overall_deadline,
            iteration=1,
            old_history_ids=old_history_ids,
            parameters={
                "dataset_name": DATASET_NAME,
                "rounds": TRAINING_ROUNDS,
                "epochs": TRAINING_EPOCHS,
            },
        )
        _check_deadline(overall_deadline, "recording iteration 1")
        recorded = backend.request(
            "POST",
            f"/api/autonomy/loops/{loop_id}/iterations",
            {"metric": float(iteration_one["summary"]["accuracy"])},
            deadline=overall_deadline,
        )
        if recorded.get("iteration") != 1 or recorded.get("status") != "RUNNING":
            raise AcceptanceError("Autonomy loop did not record iteration 1 as RUNNING")

        _run_feedback_iteration(
            backend,
            settings,
            loop_id,
            overall_deadline,
            iteration_one["group"],
            iteration_one["summary"],
            old_history_ids,
        )
    except BaseException:
        try:
            autonomy = backend.request("GET", "/api/autonomy/state", timeout=15.0)
            if autonomy.get("mode") == "FULL_ACCESS":
                _cancel_active_group(backend)
        except Exception as cleanup_error:
            print("LOUD ACTIVE-GROUP CLEANUP ERROR: " + _safe_error(cleanup_error, settings))
        if loop_id is not None:
            try:
                _stop_loop(backend, loop_id, "Aborted after an unexpected acceptance result")
            except Exception as cleanup_error:
                print("LOUD LOOP CLEANUP ERROR: " + _safe_error(cleanup_error, settings))
        raise
    else:
        if loop_id is not None:
            _stop_loop(backend, loop_id, "Bounded two-iteration acceptance complete")
    finally:
        primary_error = sys.exc_info()[1]
        cleanup_errors: list[str] = []
        if loop_id is not None:
            try:
                _stop_loop(backend, loop_id, "Ensure bounded acceptance loop is not left running")
            except Exception as error:
                cleanup_errors.append("loop stop: " + _safe_error(error, settings))
        try:
            restored = backend.request(
                "PUT",
                "/api/autonomy/mode",
                {"mode": "ASK_HUMAN"},
                timeout=30.0,
            )
            state = backend.request("GET", "/api/autonomy/state", timeout=30.0)
            if (
                restored.get("mode") != "ASK_HUMAN"
                or state.get("mode") != "ASK_HUMAN"
                or state.get("stop_requested") is not False
            ):
                raise AcceptanceError("Could not verify ASK_HUMAN restoration and STOP inactive")
            print("finally: mode=ASK_HUMAN; STOP=False")
        except Exception as error:
            cleanup_errors.append("mode restore: " + _safe_error(error, settings))
            print("LOUD MODE RESTORATION ERROR: " + cleanup_errors[-1])
        try:
            _preflight(backend, settings, expected_mode="ASK_HUMAN")
        except Exception as error:
            cleanup_errors.append("final preflight: " + _safe_error(error, settings))
            print("LOUD FINAL PREFLIGHT ERROR: " + cleanup_errors[-1])
        if cleanup_errors and primary_error is None:
            raise AcceptanceError("Cleanup failed: " + "; ".join(cleanup_errors))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--check",
        action="store_true",
        help="Run only read-only preflight, model/group checks, and tegrastats sampling.",
    )
    args = parser.parse_args()
    settings = Settings.from_env()
    backend = BackendClient(settings)
    try:
        if args.check:
            _run_read_only_check(settings, backend)
        else:
            _main_run(settings, backend)
    except BaseException as error:
        print("ACCEPTANCE STOPPED: " + _safe_error(error, settings))
        raise


if __name__ == "__main__":
    main()