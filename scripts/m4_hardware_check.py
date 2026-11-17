"""Run the real ASK_HUMAN M4 model-install and dataset-NAS acceptance."""

from __future__ import annotations

import argparse
import hashlib
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
from app.services.worker_shards import (  # noqa: E402
    get_worker_dataset_initial_model,
    get_worker_training_status,
)


RAM_PATTERN = re.compile(r"\bRAM\s+(\d+)/(\d+)MB\b")
SWAP_PATTERN = re.compile(r"\bSWAP\s+(\d+)/(\d+)MB\b")
EXPECTED_WORKERS = ("worker1", "worker2", "worker3")


class HardwareCheckError(RuntimeError):
    pass


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
        request = Request(
            self.base_url + path,
            data=body,
            headers=self.headers,
            method=method,
        )
        try:
            with urlopen(request, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except HTTPError as error:
            raise HardwareCheckError(
                f"Backend {method} {path} returned HTTP {error.code}"
            ) from error
        except (URLError, TimeoutError, OSError) as error:
            raise HardwareCheckError(
                f"Backend {method} {path} failed: {type(error).__name__}: {error}"
            ) from error
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise HardwareCheckError(
                f"Backend {method} {path} returned invalid JSON"
            ) from error


def _action_schema(action_name: str) -> dict[str, Any]:
    schema = json.loads((ROOT / "common" / "slm_action_schema.json").read_text(encoding="utf-8"))
    for action in schema["$defs"]["action"]["oneOf"]:
        if action.get("properties", {}).get("action", {}).get("const") == action_name:
            return action
    raise HardwareCheckError(f"{action_name} is absent from the action schema")


def _dataset_worker_nodes(settings: Settings) -> list[Any]:
    workers = sorted(
        (node for node in settings.nodes if node.role == "worker"),
        key=lambda node: node.node_id,
    )
    if tuple(node.node_id for node in workers) != EXPECTED_WORKERS:
        raise HardwareCheckError("Configured workers are not exactly worker1, worker2, worker3")
    return workers


def _check_workers_idle(workers: list[Any]) -> None:
    for worker in workers:
        status = get_worker_training_status(worker)
        print(
            f"{worker.node_id}: authenticated status OK; "
            f"active_run_id={status.get('active_run_id')}; "
            f"scheduled_run_id={status.get('scheduled_run_id')}"
        )


def _approved_action_data(envelope: Any) -> dict[str, Any]:
    if not isinstance(envelope, dict):
        raise HardwareCheckError("Approval endpoint returned a non-object response")
    if envelope.get("status") != "approved":
        raise HardwareCheckError(
            f"Approval endpoint did not approve the decision: {envelope.get('status')}"
        )
    result = envelope.get("result")
    if not isinstance(result, dict):
        raise HardwareCheckError("Approved response omitted its execution result")
    if isinstance(result.get("data"), dict):
        return result["data"]
    nested = result.get("result")
    if isinstance(nested, dict) and isinstance(nested.get("data"), dict):
        return nested["data"]
    raise HardwareCheckError("Approved response omitted the action result data")


def _dispatch_and_approve(
    client: BackendClient,
    *,
    action: dict[str, Any],
    reason: str,
    evidence: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    state = client.request("GET", "/api/autonomy/state")
    if state.get("mode") != "ASK_HUMAN" or state.get("stop_requested") is not False:
        raise HardwareCheckError("Autonomy must be ASK_HUMAN with global STOP inactive")
    dispatched = client.request(
        "POST",
        "/api/autonomy/dispatch",
        {"action": action, "reason": reason, "evidence": evidence},
    )
    if dispatched.get("status") != "pending":
        raise HardwareCheckError(
            f"Expected a pending ASK_HUMAN decision, got {dispatched.get('status')}"
        )
    decision = dispatched.get("decision")
    if (
        not isinstance(decision, dict)
        or decision.get("status") != "PENDING"
        or not isinstance(decision.get("id"), str)
    ):
        raise HardwareCheckError("Dispatch response did not contain a PENDING decision")
    decision_id = decision["id"]
    print(f"decision_id={decision_id}; status=PENDING; action={decision.get('action')}")
    if decision.get("parameters") != action["parameters"]:
        raise HardwareCheckError("Pending decision parameters do not match the requested action")
    approved = client.request(
        "POST",
        f"/api/autonomy/decisions/{decision_id}/approve",
        timeout=600,
    )
    if approved.get("status") != "approved":
        raise HardwareCheckError(
            f"Decision {decision_id} did not complete approval: {approved.get('status')}"
        )
    history = client.request("GET", "/api/autonomy/history")
    matching = [
        row
        for row in history
        if isinstance(row, dict) and row.get("decision_id") == decision_id
    ]
    events = {row.get("event") for row in matching}
    if not {"created", "approved"}.issubset(events):
        raise HardwareCheckError(f"Decision history is incomplete for {decision_id}")
    print(
        "decision_history="
        + json.dumps(
            [
                {"decision_id": row["decision_id"], "event": row["event"], "created_at": row["created_at"]}
                for row in matching
            ],
            sort_keys=True,
        )
    )
    return decision_id, _approved_action_data(approved)


def _verify_initial_model(
    settings: Settings,
    workers: list[Any],
    dataset_name: str,
    result: dict[str, Any] | None,
) -> None:
    model_path = settings.dataset_root / dataset_name / "models" / "initial_model"
    architecture_path = model_path / "architecture.json"
    weights_path = model_path / "initial_weights.npz"
    if not architecture_path.is_file() or not weights_path.is_file():
        raise HardwareCheckError("The PC initial-model artifact files are missing")
    architecture_sha256 = hashlib.sha256(architecture_path.read_bytes()).hexdigest()
    weights_sha256 = hashlib.sha256(weights_path.read_bytes()).hexdigest()
    architecture = json.loads(architecture_path.read_text(encoding="utf-8"))
    parameter_count = architecture.get("parameter_count")
    if parameter_count is None or parameter_count > 12_000:
        raise HardwareCheckError(f"Initial model exceeds the 12,000-parameter cap: {parameter_count}")
    if result is not None and (
        result.get("architecture_sha256") != architecture_sha256
        or result.get("weights_sha256") != weights_sha256
        or result.get("parameter_count") != parameter_count
    ):
        raise HardwareCheckError("Backend result does not match the PC model artifact hashes")
    print(
        f"PC artifact: architecture_sha256={architecture_sha256}; "
        f"weights_sha256={weights_sha256}; parameter_count={parameter_count}"
    )
    for worker in workers:
        installed = get_worker_dataset_initial_model(worker, dataset_name)
        if (
            installed.get("architecture_sha256") != architecture_sha256
            or installed.get("weights_sha256") != weights_sha256
            or installed.get("parameter_count") != parameter_count
        ):
            raise HardwareCheckError(f"{worker.node_id} model hashes differ from the PC artifact")
        print(
            f"{worker.node_id}: architecture_sha256={installed['architecture_sha256']}; "
            f"weights_sha256={installed['weights_sha256']}; "
            f"parameter_count={installed['parameter_count']}"
        )


def _ssh(user: str, host: str, remote_command: str, *, timeout: float = 20) -> str:
    process = subprocess.run(
        [
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=8",
            f"{user}@{host}",
            remote_command,
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if process.returncode:
        raise HardwareCheckError(
            f"SSH to {host} failed with exit {process.returncode}: "
            f"{process.stderr.strip()[:400]}"
        )
    return process.stdout


def _start_tegrastats(
    nodes: list[Any],
    ssh_user: str,
    run_id: str,
    paths: dict[str, str],
) -> None:
    for node in nodes:
        prefix = f"/tmp/dfl-m4-tegrastats-{run_id}-{node.node_id}"
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
        paths[node.node_id] = prefix
        print(f"{node.node_id}: tegrastats sampling started")


def _stop_tegrastats(nodes: list[Any], prefixes: dict[str, str], ssh_user: str) -> None:
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
                f"rm -f {shlex.quote(pid_path)} {shlex.quote(log_path)}"
            )
        )
        output = _ssh(ssh_user, node.ip, command)
        values = [int(match.group(1)) for match in RAM_PATTERN.finditer(output)]
        swap_values = [int(match.group(1)) for match in SWAP_PATTERN.finditer(output)]
        if not values or not swap_values:
            raise HardwareCheckError(
                f"{node.node_id} tegrastats did not produce both RAM and SWAP samples"
            )
        print(
            f"{node.node_id}: tegrastats peak RAM={max(values)} MB; "
            f"peak swap used={max(swap_values)} MB; "
            f"samples RAM={len(values)}, swap={len(swap_values)}"
        )


def _await_nas_terminal(
    client: BackendClient,
    group_id: str | None,
    *,
    poll_interval_seconds: float = 2,
) -> dict[str, Any] | None:
    while True:
        results = client.request("GET", "/api/slm/results", timeout=20)
        active_group_id = results.get("active_group_id")
        if group_id is None:
            if active_group_id is None:
                return None
            group_id = active_group_id
        group = next(
            (
                item
                for item in results.get("groups", [])
                if isinstance(item, dict) and item.get("group_id") == group_id
            ),
            None,
        )
        if group is None:
            if active_group_id is None:
                raise HardwareCheckError(
                    f"Cannot verify terminal state for NAS group {group_id}"
                )
        elif group.get("status") in {"completed", "failed", "stopped"}:
            return group
        time.sleep(poll_interval_seconds)


def _run(args: argparse.Namespace) -> None:
    settings = Settings.from_env()
    client = BackendClient(args.backend_url, settings.api_token)
    workers = _dataset_worker_nodes(settings)
    nodes = sorted(settings.nodes, key=lambda node: node.node_id)

    schema = _action_schema("create_initial_model")
    parameter_schema = schema["properties"]["parameters"]
    print(
        "create_initial_model schema: "
        f"required={parameter_schema.get('required', [])}; "
        f"optional={sorted(set(parameter_schema.get('properties', {})) - set(parameter_schema.get('required', [])))}"
    )
    if "model_name" not in parameter_schema.get("properties", {}):
        raise HardwareCheckError("Schema does not expose the model_name dataset selector")
    if "dataset_name" in parameter_schema.get("properties", {}):
        dataset_field = "dataset_name"
    elif "model_name" in parameter_schema.get("properties", {}):
        dataset_field = "model_name"
    else:
        raise HardwareCheckError("create_initial_model cannot select the requested dataset")

    state = client.request("GET", "/api/autonomy/state")
    if state.get("mode") != "ASK_HUMAN" or state.get("stop_requested") is not False:
        raise HardwareCheckError("Autonomy must be ASK_HUMAN with global STOP inactive")
    if state.get("pending_decisions"):
        raise HardwareCheckError("Existing pending decisions must be reviewed before M4")
    if not args.dataset_name:
        raise HardwareCheckError("An explicit dataset name is required")
    _check_workers_idle(workers)

    if args.skip_model_check:
        if not args.nas_only:
            raise HardwareCheckError("--skip-model-check requires --nas-only")
    elif args.nas_only:
        _verify_initial_model(settings, workers, args.dataset_name, None)
    else:
        model_action = {
            "action": "create_initial_model",
            "parameters": {dataset_field: args.dataset_name},
        }
        _, model_result = _dispatch_and_approve(
            client,
            action=model_action,
            reason="Operator-approved M4 initial model installation for the explicitly selected dataset.",
            evidence={"dataset_name": args.dataset_name, "all_workers_idle_checked": True},
        )
        _verify_initial_model(settings, workers, args.dataset_name, model_result)
    if not args.skip_model_check:
        print("M4 initial-model installation and hash comparison: PASS")

    nas_schema = _action_schema("enable_nas")["properties"]["parameters"]
    if nas_schema.get("properties", {}).get("search_time_limit_seconds", {}).get("minimum") != 30:
        raise HardwareCheckError("NAS minimum search-time schema changed; review before execution")
    results_before = client.request("GET", "/api/slm/results")
    if results_before.get("active_group_id") is not None:
        raise HardwareCheckError("The master already has an active training/NAS group")
    _check_workers_idle(workers)

    run_id = uuid.uuid4().hex[:12]
    sampler_prefixes: dict[str, str] = {}
    group_id: str | None = None
    try:
        _start_tegrastats(nodes, args.ssh_user, run_id, sampler_prefixes)
        nas_action = {
            "action": "enable_nas",
            "parameters": {
                "enabled": True,
                "dataset_name": args.dataset_name,
                "search_time_limit_seconds": args.search_time_limit_seconds,
            },
        }
        _, nas_result = _dispatch_and_approve(
            client,
            action=nas_action,
            reason="Operator-approved minimum-budget dataset-specific M4 NAS acceptance.",
            evidence={
                "dataset_name": args.dataset_name,
                "search_time_limit_seconds": args.search_time_limit_seconds,
            },
        )
        group_id = nas_result.get("group_id")
        if nas_result.get("accepted") is not True or not isinstance(group_id, str):
            raise HardwareCheckError("Approved dataset NAS was not accepted by the master")
        print(
            f"dataset NAS accepted: group_id={group_id}; "
            f"search_time_limit_seconds={args.search_time_limit_seconds}"
        )
        group = _await_nas_terminal(client, group_id)
        if group is None or group.get("status") != "completed":
            raise HardwareCheckError(
                f"Dataset NAS ended with status={None if group is None else group.get('status')}; "
                f"error={None if group is None else group.get('error')}"
            )
        architecture = group.get("selected_architecture")
        searches = group.get("worker_search")
        if not isinstance(architecture, dict) or not isinstance(searches, list) or len(searches) != 3:
            raise HardwareCheckError("Completed NAS result is missing architecture or worker scores")
        selected_id = architecture.get("id")
        print(
            f"NAS selected architecture={json.dumps(architecture, sort_keys=True)}; "
            f"parameter_count={group.get('selected_parameter_count')}"
        )
        for worker_result in searches:
            candidates = worker_result.get("candidates", [])
            selected_rows = [
                item
                for item in candidates
                if item.get("architecture", {}).get("id") == selected_id
            ]
            if len(selected_rows) != 1:
                raise HardwareCheckError(
                    f"Expected one selected-architecture score from {worker_result.get('node_id')}"
                )
            for candidate in sorted(
                candidates,
                key=lambda item: item.get("architecture", {}).get("id", ""),
            ):
                print(
                    f"{worker_result['node_id']} candidate="
                    f"{candidate.get('architecture', {}).get('id')}; "
                    f"validation_accuracy={candidate.get('validation_accuracy')}; "
                    f"validation_samples={candidate.get('validation_samples')}"
                )
        print(
            f"NAS elapsed={group.get('search_elapsed_seconds')} seconds; "
            f"master-reported worker peak RSS={group.get('search_peak_rss_mb')} MB"
        )
    finally:
        terminal_group = _await_nas_terminal(client, group_id)
        if sampler_prefixes:
            _stop_tegrastats(nodes, sampler_prefixes, args.ssh_user)
    print("M4 dataset-specific NAS and tegrastats acceptance: PASS")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--backend-url", default="http://127.0.0.1:8000")
    parser.add_argument("--ssh-user", default=os.environ.get("M4_SSH_USER", "hammad"))
    parser.add_argument("--nas-only", action="store_true")
    parser.add_argument("--skip-model-check", action="store_true")
    parser.add_argument("--search-time-limit-seconds", type=int, choices=range(30, 301), default=30)
    args = parser.parse_args()
    try:
        _run(args)
    except HardwareCheckError as error:
        raise SystemExit(f"M4 hardware acceptance FAILED: {error}") from error


if __name__ == "__main__":
    main()
