"""Run MNIST initial-model, dataset NAS, training, and SLM feedback acceptance."""

from __future__ import annotations

import contextlib
import io
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "dtn" / "backend"))
sys.path.insert(0, str(ROOT / "scripts"))

from app.core.config import DeviceConfig, Settings  # noqa: E402
from app.services.autonomy import validate_action  # noqa: E402
from app.services.worker_shards import (  # noqa: E402
    get_worker_dataset_initial_model,
    get_worker_training_status,
    WorkerShardError,
)
from m4_hardware_check import (  # noqa: E402
    HardwareCheckError,
    _approved_action_data,
)
from m5_feedback_loop import (  # noqa: E402
    BackendClient,
    FeedbackLoopError,
    _redact_tokens,
    _sample_tegrastats,
)

DATASET_NAME = "mnist_demo"
OUTPUT_PATH = ROOT / "data" / "mnist_task_k_results.json"
TERMINAL_GROUP_STATES = {"completed", "failed", "stopped"}
TEGRA_PATTERN = re.compile(
    r"^(?P<label>\S+) (?P<node>master|worker[123]): peak_RAM_MB=(?P<ram>\d+); "
    r"peak_swap_used_MB=(?P<swap>\d+); samples=(?P<samples>\d+)$",
    re.MULTILINE,
)
TEGRA_NODES = {"master", "worker1", "worker2", "worker3"}


class TaskKError(RuntimeError):
    pass


def _parse_tegrastats_peaks(output: str, label: str) -> dict[str, dict[str, int]]:
    peaks = {
        match.group("node"): {
            "peak_ram_mb": int(match.group("ram")),
            "peak_swap_used_mb": int(match.group("swap")),
            "samples": int(match.group("samples")),
        }
        for match in TEGRA_PATTERN.finditer(output)
        if match.group("label") == label
    }
    if set(peaks) != TEGRA_NODES:
        raise TaskKError(f"Full-interval tegrastats peaks are incomplete for {label}")
    return peaks


def _workers(settings: Settings) -> list[DeviceConfig]:
    workers = sorted(
        (node for node in settings.nodes if node.role == "worker"),
        key=lambda node: node.node_id,
    )
    if [node.node_id for node in workers] != ["worker1", "worker2", "worker3"]:
        raise TaskKError("Expected exactly worker1, worker2, and worker3")
    return workers


def _preflight(
    client: BackendClient,
    settings: Settings,
    workers: list[DeviceConfig],
    *,
    allowed_pending_decision: str | None = None,
    require_dataset: bool = True,
) -> None:
    state = client.request("GET", "/api/autonomy/state")
    if state.get("mode") != "ASK_HUMAN" or state.get("stop_requested") is not False:
        raise TaskKError("Required autonomy state is ASK_HUMAN with global STOP inactive")
    if any(loop.get("status") == "RUNNING" for loop in state.get("loops", [])):
        raise TaskKError("An autonomy loop is running")
    pending = [
        item
        for item in client.request("GET", "/api/autonomy/decisions")
        if item.get("status") == "PENDING"
    ]
    expected_pending = [] if allowed_pending_decision is None else [allowed_pending_decision]
    if sorted(item.get("id") for item in pending) != sorted(expected_pending):
        raise TaskKError("Unexpected pending autonomy decisions")
    results = client.request("GET", "/api/slm/results")
    if results.get("active_group_id") is not None:
        raise TaskKError("A master training or NAS group is active")
    master_status = client.request("GET", "/api/slm/status")
    if master_status.get("node_id") != "master":
        raise TaskKError("Authenticated master status returned the wrong node")
    for worker in workers:
        status = get_worker_training_status(worker)
        if status.get("active_run_id") is not None or status.get("scheduled_run_id") is not None:
            raise TaskKError(f"{worker.node_id} is not idle")
        if require_dataset:
            from app.services.worker_shards import get_worker_dataset_metadata

            metadata = get_worker_dataset_metadata(worker, DATASET_NAME)
            if metadata.get("dataset_name") != DATASET_NAME:
                raise TaskKError(f"{worker.node_id} is missing {DATASET_NAME}")
        print(
            f"{worker.node_id}: authenticated status OK; idle"
            + (f"; dataset={DATASET_NAME}" if require_dataset else "")
        )
    print("master: authenticated status OK; idle; ASK_HUMAN; STOP inactive")


def _dispatch(
    client: BackendClient,
    action: dict[str, Any],
    reason: str,
    evidence: dict[str, Any],
) -> dict[str, Any]:
    validate_action(action)
    response = client.request(
        "POST",
        "/api/autonomy/dispatch",
        {"action": action, "reason": reason, "evidence": evidence},
    )
    decision = response.get("decision")
    if (
        response.get("status") != "pending"
        or not isinstance(decision, dict)
        or decision.get("status") != "PENDING"
        or decision.get("action") != action["action"]
        or decision.get("parameters") != action["parameters"]
        or not isinstance(decision.get("id"), str)
    ):
        raise TaskKError("ASK_HUMAN dispatch did not create the expected pending decision")
    return decision


def _approve(
    client: BackendClient,
    settings: Settings,
    workers: list[DeviceConfig],
    decision: dict[str, Any],
    *,
    timeout: int = 1200,
) -> dict[str, Any]:
    _preflight(
        client,
        settings,
        workers,
        allowed_pending_decision=decision["id"],
    )
    response = client.request(
        "POST",
        f"/api/autonomy/decisions/{decision['id']}/approve",
        timeout=timeout,
    )
    if response.get("status") != "approved":
        raise TaskKError(f"Decision {decision['id']} approval did not complete")
    history = client.request("GET", "/api/autonomy/history")
    entries = [item for item in history if item.get("decision_id") == decision["id"]]
    events = {item.get("event") for item in entries}
    if not {"created", "approved"}.issubset(events):
        raise TaskKError(f"Decision history is incomplete for {decision['id']}")
    decision["history"] = [
        {"event": item.get("event"), "created_at": item.get("created_at")}
        for item in entries
    ]
    return response


def _find_group(client: BackendClient, group_id: str) -> dict[str, Any]:
    results = client.request("GET", "/api/slm/results")
    groups = results.get("groups")
    if not isinstance(groups, list):
        raise TaskKError("Master results response omitted groups")
    group = next(
        (item for item in groups if isinstance(item, dict) and item.get("group_id") == group_id),
        None,
    )
    if group is None:
        raise TaskKError(f"Master results omitted group {group_id}")
    return group


def _wait_for_group(
    client: BackendClient,
    settings: Settings,
    workers: list[DeviceConfig],
    group_id: str,
    *,
    timeout_seconds: int,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_seconds
    last_state = "unknown"
    while time.monotonic() < deadline:
        group = _find_group(client, group_id)
        last_state = str(group.get("status", "unknown"))
        if last_state in TERMINAL_GROUP_STATES:
            if last_state != "completed":
                raise TaskKError(
                    f"Group {group_id} ended {last_state}: {group.get('error')}"
                )
            _preflight(client, settings, workers)
            return group
        time.sleep(5)
    raise TaskKError(f"Timed out waiting for {group_id}; last state={last_state}")


def _wait_for_approved_async_group(
    client: BackendClient,
    settings: Settings,
    workers: list[DeviceConfig],
    approval: dict[str, Any],
    *,
    baseline_group_ids: set[str],
    dataset_name: str,
    timeout_seconds: int,
) -> dict[str, Any]:
    try:
        action_data = _approved_action_data(approval)
    except HardwareCheckError as error:
        if "Approved response omitted the action result data" not in str(error):
            raise
        action_data = {}
    group_id = action_data.get("group_id")
    if group_id is not None and not isinstance(group_id, str):
        raise TaskKError("Approved async action returned an invalid group_id")

    deadline = time.monotonic() + timeout_seconds
    last_state = "not-created"
    while time.monotonic() < deadline:
        results = client.request("GET", "/api/slm/results")
        groups = results.get("groups")
        if not isinstance(groups, list):
            raise TaskKError("Master results response omitted groups")
        if group_id is not None:
            group = next(
                (item for item in groups if isinstance(item, dict) and item.get("group_id") == group_id),
                None,
            )
        else:
            candidates = [
                item
                for item in groups
                if isinstance(item, dict)
                and item.get("group_id") not in baseline_group_ids
                and item.get("dataset_name") == dataset_name
                and item.get("kind") in {"dataset_architecture_search", "decentralized"}
            ]
            if len(candidates) > 1:
                raise TaskKError("Multiple new dataset groups appeared after one approval")
            group = candidates[0] if candidates else None
            if group is not None:
                group_id = group.get("group_id")

        if group is not None:
            if group.get("dataset_name") != dataset_name:
                raise TaskKError("Approved async action started a different dataset")
            last_state = str(group.get("status", "unknown"))
            if last_state == "completed":
                _preflight(client, settings, workers)
                return group
            if last_state in TERMINAL_GROUP_STATES:
                raise TaskKError(
                    f"Group {group_id} ended {last_state}: {group.get('error')}"
                )
        time.sleep(5)
    raise TaskKError(
        f"Timed out waiting for approved {dataset_name} group; "
        f"group_id={group_id}; last state={last_state}"
    )


def _run_sampled_approval(
    client: BackendClient,
    settings: Settings,
    workers: list[DeviceConfig],
    decision: dict[str, Any],
    *,
    label: str,
    timeout_seconds: int,
) -> tuple[dict[str, Any], dict[str, Any] | None, str]:
    captured = io.StringIO()
    started = time.perf_counter()
    group: dict[str, Any] | None = None
    response: dict[str, Any]
    prior_results = client.request("GET", "/api/slm/results")
    prior_groups = prior_results.get("groups")
    if not isinstance(prior_groups, list):
        raise TaskKError("Master results response omitted groups before approval")
    baseline_group_ids = {
        item.get("group_id")
        for item in prior_groups
        if isinstance(item, dict) and isinstance(item.get("group_id"), str)
    }
    try:
        with contextlib.redirect_stdout(captured):
            with _sample_tegrastats(settings, label):
                response = _approve(client, settings, workers, decision)
                group = _wait_for_approved_async_group(
                    client,
                    settings,
                    workers,
                    response,
                    baseline_group_ids=baseline_group_ids,
                    dataset_name=DATASET_NAME,
                    timeout_seconds=timeout_seconds,
                )
    finally:
        telemetry_text = captured.getvalue()
        print(telemetry_text, end="")
    peaks = _parse_tegrastats_peaks(telemetry_text, label)
    return response, group, json.dumps(
        {"elapsed_seconds": round(time.perf_counter() - started, 3), "nodes": peaks},
        sort_keys=True,
    )


def _history_for(client: BackendClient, decision_ids: set[str]) -> list[dict[str, Any]]:
    return [
        {
            "decision_id": item.get("decision_id"),
            "event": item.get("event"),
            "details": item.get("details"),
            "created_at": item.get("created_at"),
        }
        for item in client.request("GET", "/api/autonomy/history")
        if item.get("decision_id") in decision_ids
    ]


def _model_acceptance(
    client: BackendClient,
    settings: Settings,
    workers: list[DeviceConfig],
) -> tuple[dict[str, Any], str]:
    _preflight(client, settings, workers)
    action = {"action": "create_initial_model", "parameters": {"model_name": DATASET_NAME}}
    decision = _dispatch(
        client,
        action,
        "Create the dataset-derived initial CNN for real MNIST grayscale shards.",
        {"dataset_name": DATASET_NAME, "input_shape": [28, 28, 1]},
    )
    approved = _approve(client, settings, workers, decision)
    report = _approved_action_data(approved)
    architecture = report.get("architecture")
    parameter_count = report.get("parameter_count")
    if (
        not isinstance(architecture, dict)
        or architecture.get("input_shape") != [28, 28, 1]
        or not isinstance(parameter_count, int)
        or parameter_count > 12_000
    ):
        raise TaskKError("Initial model shape or parameter cap did not match MNIST acceptance")

    local_root = settings.dataset_root / DATASET_NAME / "models" / "initial_model"
    metadata_path = local_root / "metadata.json"
    if not metadata_path.is_file():
        raise TaskKError("PC initial-model metadata artifact is missing")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    pc_hashes = {
        "architecture_sha256": metadata.get("architecture_sha256"),
        "weights_sha256": metadata.get("weights_sha256"),
    }
    if (
        pc_hashes["architecture_sha256"] != report.get("architecture_sha256")
        or pc_hashes["weights_sha256"] != report.get("weights_sha256")
    ):
        raise TaskKError("PC model metadata hashes disagree with the approved response")

    worker_hashes: dict[str, dict[str, Any]] = {}
    for worker in workers:
        status = get_worker_training_status(worker)
        if status.get("active_run_id") is not None or status.get("scheduled_run_id") is not None:
            raise TaskKError(f"{worker.node_id} is not idle before model verification")
        installed = get_worker_dataset_initial_model(worker, DATASET_NAME)
        if (
            installed.get("architecture_sha256") != pc_hashes["architecture_sha256"]
            or installed.get("weights_sha256") != pc_hashes["weights_sha256"]
            or installed.get("parameter_count") != parameter_count
        ):
            raise WorkerShardError(f"{worker.node_id} model hashes do not match the PC artifact")
        worker_hashes[worker.node_id] = {
            "input_shape": architecture["input_shape"],
            "parameter_count": installed["parameter_count"],
            "architecture_sha256": installed["architecture_sha256"],
            "weights_sha256": installed["weights_sha256"],
        }
        print(
            f"{worker.node_id}: shape={architecture['input_shape']}; "
            f"parameters={installed['parameter_count']}; "
            f"architecture_sha256={installed['architecture_sha256']}; "
            f"weights_sha256={installed['weights_sha256']}"
        )

    data = {
        "status": "PASS",
        "decision_id": decision["id"],
        "input_shape": architecture["input_shape"],
        "parameter_count": parameter_count,
        "pc": pc_hashes,
        "workers": worker_hashes,
    }
    return data, decision["id"]


def _nas_acceptance(
    client: BackendClient,
    settings: Settings,
    workers: list[DeviceConfig],
) -> tuple[dict[str, Any], str]:
    search_limit = 120
    action = {
        "action": "enable_nas",
        "parameters": {
            "enabled": True,
            "dataset_name": DATASET_NAME,
            "search_time_limit_seconds": search_limit,
        },
    }
    validate_action(action)
    _preflight(client, settings, workers)
    decision = _dispatch(
        client,
        action,
        "Run dataset-specific NAS on the real MNIST grayscale worker shards with a 120-second limit.",
        {"dataset_name": DATASET_NAME, "search_time_limit_seconds": search_limit},
    )
    _preflight(
        client,
        settings,
        workers,
        allowed_pending_decision=decision["id"],
    )
    approved, group, sampling = _run_sampled_approval(
        client,
        settings,
        workers,
        decision,
        label="mnist-nas",
        timeout_seconds=search_limit + 180,
    )
    if group is None:
        raise TaskKError("Dataset NAS approval did not return a group")
    if group.get("dataset_name") != DATASET_NAME or group.get("status") != "completed":
        raise TaskKError("Dataset NAS completed for an unexpected dataset or state")
    selected = group.get("selected_architecture")
    candidates = group.get("nas_candidates")
    worker_search = group.get("worker_search")
    if not isinstance(selected, dict) or not isinstance(candidates, list) or not isinstance(worker_search, list):
        raise TaskKError("Dataset NAS results omitted selection, candidates, or worker details")

    per_candidate: list[dict[str, Any]] = []
    for aggregate in candidates:
        architecture = aggregate.get("architecture", {})
        architecture_id = architecture.get("id")
        worker_rows = []
        for worker_result in worker_search:
            match = next(
                (
                    row
                    for row in worker_result.get("candidates", [])
                    if row.get("architecture", {}).get("id") == architecture_id
                ),
                None,
            )
            if match is None:
                raise TaskKError(f"NAS worker details omitted candidate {architecture_id}")
            worker_rows.append(
                {
                    "node_id": match.get("node_id"),
                    "validation_accuracy": match.get("validation_accuracy"),
                    "validation_samples": match.get("validation_samples"),
                }
            )
        per_candidate.append(
            {
                "architecture": architecture,
                "parameter_count": aggregate.get("parameter_count"),
                "mean_validation_accuracy": aggregate.get("mean_validation_accuracy"),
                "workers": worker_rows,
            }
        )
    scores = [
        row["mean_validation_accuracy"]
        for row in per_candidate
        if isinstance(row.get("mean_validation_accuracy"), (int, float))
    ]
    selected_mean = next(
        row["mean_validation_accuracy"]
        for row in per_candidate
        if row["architecture"].get("id") == selected.get("id")
    )
    data = {
        "status": "PASS",
        "decision_id": decision["id"],
        "group_id": group["group_id"],
        "search_time_limit_seconds": search_limit,
        "search_duration_seconds": group.get("search_elapsed_seconds"),
        "approval_wall_seconds": json.loads(sampling)["elapsed_seconds"],
        "selected_architecture": selected,
        "parameter_count": group.get("selected_parameter_count"),
        "selected_mean_validation_accuracy": selected_mean,
        "chance_accuracy": 0.1,
        "descriptively_above_chance": bool(selected_mean > 0.1),
        "candidate_scores_differ": len(set(scores)) > 1,
        "candidates": per_candidate,
        "tegrastats": json.loads(sampling)["nodes"],
    }
    print(
        "MNIST NAS summary="
        + json.dumps(_redact_tokens(data, settings), sort_keys=True)
    )
    return data, decision["id"]


def _training_metrics(group: dict[str, Any]) -> dict[str, Any]:
    runs = group.get("runs")
    if not isinstance(runs, list) or len(runs) != 1:
        raise TaskKError("Expected exactly one completed MNIST training seed")
    run = runs[0]
    round_rows = run.get("rounds", [])
    if run.get("status") != "completed" or len(round_rows) != 5:
        raise TaskKError("MNIST training did not complete five rounds")
    final_workers = round_rows[-1].get("workers", {})
    per_worker = {}
    total_bytes = 0
    for worker_id in ("worker1", "worker2", "worker3"):
        if worker_id not in final_workers:
            raise TaskKError(f"Final round omitted {worker_id}")
        metric = final_workers[worker_id]
        accuracy = metric.get("validation_accuracy", metric.get("accuracy"))
        macro_f1 = metric.get("validation_macro_f1", metric.get("macro_f1"))
        if not isinstance(accuracy, (int, float)) or not isinstance(macro_f1, (int, float)):
            raise TaskKError(f"Final validation metrics are missing for {worker_id}")
        sent = sum(
            row.get("workers", {}).get(worker_id, {}).get("bytes_sent", 0)
            for row in round_rows
        )
        received = sum(
            row.get("workers", {}).get(worker_id, {}).get("bytes_received", 0)
            for row in round_rows
        )
        total_bytes += sent + received
        per_worker[worker_id] = {
            "accuracy": accuracy,
            "macro_f1": macro_f1,
            "bytes_sent": sent,
            "bytes_received": received,
        }
    accuracy = sum(row["accuracy"] for row in per_worker.values()) / 3
    macro_f1 = sum(row["macro_f1"] for row in per_worker.values()) / 3
    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "per_worker": per_worker,
        "communication_bytes_total_sent_plus_received": total_bytes,
        "rounds": group.get("rounds"),
        "epochs": group.get("epochs"),
        "duration_seconds": group.get("completed_at", 0) - group.get("started_at", 0),
    }


def _training_acceptance(
    client: BackendClient,
    settings: Settings,
    workers: list[DeviceConfig],
) -> tuple[dict[str, Any], str, dict[str, Any]]:
    decisions: list[str] = []
    topology_action = {
        "action": "set_topology",
        "parameters": {"topology": "bidirectional_ring"},
    }
    _preflight(client, settings, workers)
    topology_decision = _dispatch(
        client,
        topology_action,
        "Set the requested MNIST training topology before the short training run.",
        {"dataset_name": DATASET_NAME, "topology": "bidirectional_ring"},
    )
    _approve(client, settings, workers, topology_decision)
    decisions.append(topology_decision["id"])
    if client.request("GET", "/api/slm/results").get("configuration", {}).get("topology") != "bidirectional_ring":
        raise TaskKError("Master topology did not become bidirectional_ring")

    action = {
        "action": "start_training",
        "parameters": {"dataset_name": DATASET_NAME, "rounds": 5, "epochs": 2},
    }
    _preflight(client, settings, workers)
    train_decision = _dispatch(
        client,
        action,
        "Run five rounds by two local epochs on the real MNIST grayscale dataset.",
        {"dataset_name": DATASET_NAME, "rounds": 5, "epochs": 2, "topology": "bidirectional_ring"},
    )
    decisions.append(train_decision["id"])
    _preflight(
        client,
        settings,
        workers,
        allowed_pending_decision=train_decision["id"],
    )
    approved, group, sampling = _run_sampled_approval(
        client,
        settings,
        workers,
        train_decision,
        label="mnist-train-5x2",
        timeout_seconds=1800,
    )
    if group is None or group.get("dataset_name") != DATASET_NAME:
        raise TaskKError("MNIST short-training approval did not return the expected group")
    metrics = _training_metrics(group)
    metrics.update(
        {
            "status": group["status"],
            "group_id": group["group_id"],
            "seed": group["runs"][0]["seed"],
            "decision_ids": decisions,
            "decision_history": _history_for(client, set(decisions)),
            "tegrastats": json.loads(sampling)["nodes"],
        }
    )
    metrics["nas_architecture_second_run"] = {
        "status": "UNSUPPORTED",
        "reason": "start_training schema and master action handler do not expose an architecture parameter.",
    }
    print("MNIST training summary=" + json.dumps(_redact_tokens(metrics, settings), sort_keys=True))
    return metrics, train_decision["id"], group


def _feedback_acceptance(
    client: BackendClient,
    settings: Settings,
    workers: list[DeviceConfig],
    group: dict[str, Any],
) -> tuple[dict[str, Any], str | None]:
    if settings.feedback_rule_fallback_enabled:
        raise TaskKError("Feedback rule fallback must remain disabled")
    _preflight(client, settings, workers)
    buffer = io.StringIO()
    decision_id: str | None = None
    approved_result: dict[str, Any] | None = None
    follow_up_group: dict[str, Any] | None = None
    with contextlib.redirect_stdout(buffer):
        with _sample_tegrastats(settings, "mnist-feedback"):
            suggestion = client.request(
                "POST",
                "/api/slm/suggest-next-action",
                {"group_id": group["group_id"], "iteration": 1},
                timeout=settings.slm_timeout_seconds + 60,
            )
            suggestion = _redact_tokens(suggestion, settings)
            attempts = suggestion.get("attempts", [])
            if not isinstance(attempts, list):
                raise TaskKError("Feedback response omitted attempts")
            valid_count = sum(
                1
                for item in attempts
                if isinstance(item, dict) and item.get("validator_result") == "valid"
            )
            print("feedback_raw_attempts=" + json.dumps(attempts, sort_keys=True))
            print(
                f"feedback_source={suggestion.get('source')}; "
                f"validator_valid={valid_count}/{len(attempts)}; "
                f"fallback_enabled={settings.feedback_rule_fallback_enabled}"
            )
            if suggestion.get("source") != "slm":
                raise TaskKError("Feedback source was not the SLM; fallback must stay off")
            if suggestion.get("status") == "needs_human":
                telemetry_text = buffer.getvalue()
                print(telemetry_text, end="", file=sys.__stdout__)
                return {
                    "status": "NEEDS_HUMAN",
                    "source": suggestion.get("source"),
                    "valid_suggestions": valid_count,
                    "total_suggestions": len(attempts),
                    "attempts": attempts,
                    "fallback_enabled": False,
                    "tegrastats": _parse_tegrastats_peaks(
                        telemetry_text, "mnist-feedback"
                    ),
                }, None
            action = suggestion.get("action")
            if not isinstance(action, dict):
                raise TaskKError("Feedback response omitted its action")
            validate_action(action)
            if not attempts or attempts[-1].get("validator_result") != "valid":
                raise TaskKError("Final SLM attempt was not schema-valid")
            envelope = suggestion.get("result")
            decision = envelope.get("decision") if isinstance(envelope, dict) else None
            if (
                suggestion.get("status") != "pending"
                or not isinstance(decision, dict)
                or decision.get("status") != "PENDING"
                or decision.get("action") != action.get("action")
                or decision.get("parameters") != action.get("parameters")
                or not isinstance(decision.get("id"), str)
            ):
                raise TaskKError("Validated SLM suggestion did not create the expected pending decision")
            decision_id = decision["id"]
            print(
                "feedback_pending_decision="
                + json.dumps(
                    {
                        "id": decision_id,
                        "action": decision.get("action"),
                        "parameters": decision.get("parameters"),
                        "reason": decision.get("reason"),
                        "evidence": decision.get("evidence"),
                    },
                    sort_keys=True,
                )
            )
            _preflight(
                client,
                settings,
                workers,
                allowed_pending_decision=decision_id,
            )
            prior_results = client.request("GET", "/api/slm/results")
            prior_groups = prior_results.get("groups")
            if not isinstance(prior_groups, list):
                raise TaskKError("Master results response omitted groups before feedback approval")
            baseline_group_ids = {
                item.get("group_id")
                for item in prior_groups
                if isinstance(item, dict) and isinstance(item.get("group_id"), str)
            }
            approved = client.request(
                "POST",
                f"/api/autonomy/decisions/{decision_id}/approve",
                timeout=1200,
            )
            if approved.get("status") != "approved":
                raise TaskKError("Validated SLM decision approval failed")
            approved_result = approved
            decision_history = _history_for(client, {decision_id})
            if not {"created", "approved"}.issubset(
                {row.get("event") for row in decision_history}
            ):
                raise TaskKError("SLM decision history omitted created/approved entries")
            print("feedback_decision_history=" + json.dumps(decision_history, sort_keys=True))
            if action.get("action") in {"enable_nas", "start_training"}:
                follow_up_group = _wait_for_approved_async_group(
                    client,
                    settings,
                    workers,
                    approved,
                    baseline_group_ids=baseline_group_ids,
                    dataset_name=DATASET_NAME,
                    timeout_seconds=1800,
                )
            else:
                approved_data = _approved_action_data(approved)
                group_id = approved_data.get("group_id")
                if isinstance(group_id, str):
                    follow_up_group = _wait_for_group(
                        client,
                        settings,
                        workers,
                        group_id,
                        timeout_seconds=1800,
                    )
    telemetry_text = buffer.getvalue()
    print(telemetry_text, end="")
    return {
        "status": "PASS",
        "source": suggestion.get("source"),
        "valid_suggestions": valid_count,
        "total_suggestions": len(attempts),
        "attempts": attempts,
        "action": suggestion.get("action"),
        "decision_id": decision_id,
        "approval_result": approved_result,
        "follow_up_group": follow_up_group,
        "fallback_enabled": False,
        "tegrastats": _parse_tegrastats_peaks(telemetry_text, "mnist-feedback"),
    }, decision_id


def main() -> None:
    settings = Settings.from_env()
    client = BackendClient(f"http://127.0.0.1:{settings.port}", settings.api_token)
    workers = _workers(settings)
    result: dict[str, Any] = {
        "dataset_name": DATASET_NAME,
        "status": "RUNNING",
        "fallback_enabled": settings.feedback_rule_fallback_enabled,
        "steps": {},
    }
    try:
        if settings.feedback_rule_fallback_enabled:
            raise TaskKError("Feedback rule fallback must be False")
        model, model_decision = _model_acceptance(client, settings, workers)
        result["steps"]["initial_model"] = model
        nas, nas_decision = _nas_acceptance(client, settings, workers)
        result["steps"]["dataset_nas"] = nas
        training, training_decision, group = _training_acceptance(client, settings, workers)
        result["steps"]["short_training"] = training
        feedback, feedback_decision = _feedback_acceptance(client, settings, workers, group)
        result["steps"]["feedback"] = feedback
        result["decision_history"] = _history_for(
            client,
            {
                model_decision,
                nas_decision,
                *training["decision_ids"],
                *({feedback_decision} if feedback_decision else set()),
            },
        )
        result["status"] = "PASS"
    except Exception as error:
        result["status"] = "FAIL"
        result["error_type"] = type(error).__name__
        result["error"] = str(_redact_tokens(str(error), settings))
        raise
    finally:
        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        safe_result = _redact_tokens(result, settings)
        OUTPUT_PATH.write_text(
            json.dumps(safe_result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"task_k_result_json={OUTPUT_PATH}")
        print(json.dumps(safe_result, sort_keys=True))


if __name__ == "__main__":
    main()
