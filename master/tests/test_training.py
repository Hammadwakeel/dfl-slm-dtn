import json
import sqlite3
import threading
import time
from dataclasses import replace
from pathlib import Path

import pytest

from common.dataset_nas import (
    CIFAR_INITIAL_POPULATION_SIZE,
    FINAL_EPOCHS,
    MEMORY_CONSTRAINED_MAX_PARAMETERS,
    INITIAL_EPOCHS,
    INITIAL_POPULATION_SIZE,
    initial_population,
    smoke_test_architecture,
)
from master.training import (
    TrainingConfig,
    TrainingCoordinator,
    TrainingError,
    TrainingWorker,
)


class FakeEvaluator:
    def evaluate(self, models, architecture=None):
        return {}

    def train_baseline(self, seed, rounds, epochs, architecture=None):
        return []

    def close(self):
        pass


def dataset_search_response(worker, payload, *, best_candidate_id=None):
    classes = ["cat", "dog"]
    dataset_name = payload["dataset_name"]
    shape = (32, 32, 3) if dataset_name == "cifar10" else (28, 28, 1)
    architectures = payload.get("architectures")
    if not architectures:
        architectures = (
            [smoke_test_architecture(input_shape=shape, num_classes=len(classes))]
            if payload.get("smoke_test")
            else initial_population(
                seed=payload["seed"],
                input_shape=shape,
                num_classes=len(classes),
                size=(
                    CIFAR_INITIAL_POPULATION_SIZE
                    if dataset_name == "cifar10"
                    else INITIAL_POPULATION_SIZE
                ),
                memory_constrained=dataset_name == "cifar10",
            )
        )
    return {
        "node_id": worker.node_id,
        "dataset_name": payload["dataset_name"],
        "dataset_classes": classes,
        "dataset_samples": 40,
        "dataset_image_shape": list(shape),
        "elapsed_seconds": 4.0,
        "peak_rss_mb": 1000.0,
        "fidelity_epochs": payload["epochs"],
        "search_method": (
            "single_candidate_smoke_test"
            if payload.get("smoke_test")
            else "pareto_evolutionary_multifidelity"
        ),
        "candidates": [
            {
                "node_id": worker.node_id,
                "architecture": architecture,
                "parameter_count": architecture["parameter_count"],
                "validation_accuracy": (
                    0.99
                    if architecture["id"] == best_candidate_id
                    else 0.5
                ),
                "validation_macro_precision": 0.5,
                "validation_macro_recall": 0.5,
                "validation_macro_f1": 0.5,
                "elapsed_seconds": 1.0,
                "peak_rss_mb": 1000.0,
            }
            for architecture in architectures
        ],
    }


def make_config(tmp_path: Path) -> TrainingConfig:
    return TrainingConfig(
        workers=tuple(
            TrainingWorker(f"worker{index}", f"http://127.0.0.1:{5003 + index}", "x" * 32)
            for index in range(1, 4)
        ),
        evaluator_python="python",
        evaluator_script="evaluator.py",
        test_data_path="test.npz",
        train_data_path="train.npz",
        evaluator_log_path=str(tmp_path / "evaluator.log"),
        results_path=str(tmp_path / "experiments.json"),
    )


def test_worker_results_poll_uses_configured_peer_timeout(
    tmp_path: Path,
) -> None:
    config = replace(make_config(tmp_path), peer_timeout_seconds=37)
    coordinator = TrainingCoordinator(config, evaluator=FakeEvaluator())
    observed_timeouts = []
    with coordinator._lock:
        coordinator._groups.append({
            "group_id": "group-1",
            "runs": [{"run_id": "run-1", "rounds": []}],
        })

    def fake_request(worker, path, *, timeout):
        observed_timeouts.append(timeout)
        return json.dumps({
            "runs": {"run-1": [{}]},
            "progress": {
                "run_id": "run-1",
                "round": 1,
                "epoch": 5,
                "epochs": 5,
                "phase": "peer_exchange",
            },
        }).encode(), {}

    coordinator._request = fake_request
    coordinator._parallel_workers = lambda operation: [
        operation(worker) for worker in config.workers
    ]

    coordinator._wait_for_worker_rounds("run-1", 0, "group-1")

    assert observed_timeouts == [37, 37, 37]
    run = coordinator.results()["groups"][0]["runs"][0]
    assert run["progress"]["round"] == 1
    assert run["progress"]["workers"]["worker1"] == {
        "epoch": 5,
        "epochs": 5,
        "phase": "peer_exchange",
    }


def test_run_results_retry_one_timeout(tmp_path: Path, monkeypatch) -> None:
    coordinator = TrainingCoordinator(make_config(tmp_path), evaluator=FakeEvaluator())
    worker = coordinator.config.workers[1]
    requests = []

    def fake_get_json(requested_worker, path, *, timeout=10):
        requests.append((requested_worker.node_id, path, timeout))
        if len(requests) == 1:
            raise TrainingError("Could not reach worker2: timed out")
        return {"runs": {"run-1": [{}]}}

    coordinator._get_json = fake_get_json
    monkeypatch.setattr("master.training.time.sleep", lambda _seconds: None)

    result = coordinator._get_run_results(worker, "run-1")

    assert result == {"runs": {"run-1": [{}]}}
    assert requests == [
        ("worker2", "/v1/results?run_id=run-1", 60.0),
        ("worker2", "/v1/results?run_id=run-1", 60.0),
    ]


def test_run_results_retry_transient_unreachable_worker(
    tmp_path: Path,
    monkeypatch,
) -> None:
    coordinator = TrainingCoordinator(make_config(tmp_path), evaluator=FakeEvaluator())
    worker = coordinator.config.workers[2]
    requests = []

    def fake_get_json(requested_worker, path, *, timeout=10):
        requests.append(requested_worker.node_id)
        if len(requests) == 1:
            raise TrainingError("Could not reach worker3: No route to host")
        return {"runs": {"run-1": [{}]}}

    coordinator._get_json = fake_get_json
    monkeypatch.setattr("master.training.time.sleep", lambda _seconds: None)

    result = coordinator._get_run_results(worker, "run-1")

    assert result == {"runs": {"run-1": [{}]}}
    assert requests == ["worker3", "worker3"]


def test_run_results_does_not_retry_non_timeout_errors(
    tmp_path: Path,
    monkeypatch,
) -> None:
    coordinator = TrainingCoordinator(make_config(tmp_path), evaluator=FakeEvaluator())
    worker = coordinator.config.workers[1]
    requests = []

    def fake_get_json(*_args, **_kwargs):
        requests.append(True)
        raise TrainingError("worker2 returned HTTP 401")

    coordinator._get_json = fake_get_json
    monkeypatch.setattr("master.training.time.sleep", lambda _seconds: None)

    with pytest.raises(TrainingError, match="HTTP 401"):
        coordinator._get_run_results(worker, "run-1")

    assert len(requests) == 1


def test_configure_workers_updates_runtime_inventory_and_all_peer_maps(
    tmp_path: Path,
) -> None:
    coordinator = TrainingCoordinator(make_config(tmp_path), evaluator=FakeEvaluator())
    workers = tuple(
        TrainingWorker(
            f"worker{index}",
            f"http://127.0.0.1:{5003 + index}",
            f"worker{index}-token-" + "x" * 32,
        )
        for index in range(1, 5)
    )
    updates = []

    def fake_post(worker, path, payload, *, timeout=30):
        updates.append((worker.node_id, path, payload, timeout))
        return {"configured_peers": [peer["node_id"] for peer in payload["peers"]]}

    coordinator._post = fake_post
    result = coordinator.configure_workers(workers)

    assert result == {"configured_workers": ["worker1", "worker2", "worker3", "worker4"]}
    assert coordinator.config.workers == workers
    assert len(updates) == 4
    for node_id, path, payload, _timeout in updates:
        assert path == "/v1/peers"
        assert len(payload["peers"]) == 3
        assert node_id not in {peer["node_id"] for peer in payload["peers"]}


def test_configure_workers_rolls_back_peers_if_any_worker_cannot_be_updated(
    tmp_path: Path,
) -> None:
    coordinator = TrainingCoordinator(make_config(tmp_path), evaluator=FakeEvaluator())
    previous_workers = coordinator.config.workers
    workers = (*previous_workers, TrainingWorker("worker4", "http://127.0.0.1:5007", "w" * 32))
    updates = []

    def fake_post(worker, path, payload, *, timeout=30):
        peer_count = len(payload["peers"])
        updates.append((worker.node_id, peer_count))
        if worker.node_id == "worker4":
            raise TrainingError("worker4 training API is unavailable")
        return {"configured_peers": []}

    coordinator._post = fake_post
    try:
        coordinator.configure_workers(workers)
    except TrainingError as error:
        assert "worker4 training API is unavailable" in str(error)
    else:
        raise AssertionError("A failed worker peer sync must fail the inventory update")

    assert coordinator.config.workers == previous_workers
    assert any(node_id != "worker4" and peer_count == 2 for node_id, peer_count in updates)


def test_distributed_seed_cleans_worker_memory_when_training_fails(tmp_path: Path) -> None:
    coordinator = TrainingCoordinator(make_config(tmp_path), evaluator=FakeEvaluator())
    run = {"run_id": "failed-run", "rounds": []}
    coordinator._groups.append({"group_id": "group-1", "runs": [run]})
    cleaned_workers = []
    worker_actions = []

    def fail_before_training(group_id, group, run):
        coordinator._started_worker_runs.add(run["run_id"])
        raise TrainingError("training failed")

    coordinator._run_distributed_seed_impl = fail_before_training
    coordinator._parallel_workers = lambda operation: [
        operation(worker) for worker in coordinator.config.workers
    ]
    coordinator._post = lambda worker, path, payload, *, timeout=30: (
        worker_actions.append((path, worker.node_id)) or {"accepted": True}
    )
    coordinator._delete = lambda worker, path, *, timeout=30: (
        cleaned_workers.append((worker.node_id, path))
        or worker_actions.append((path, worker.node_id))
        or {"cleaned": True}
    )

    try:
        coordinator._run_distributed_seed("group-1", {}, run)
    except TrainingError as error:
        assert str(error) == "training failed"
    else:
        raise AssertionError("Training failure was not propagated")

    assert len(cleaned_workers) == 3
    assert run["worker_cleanup"]["status"] == "completed"
    assert len([action for action in worker_actions if action[0] == "/v1/cancel"]) == 3
    first_cleanup = min(
        index for index, (path, _node_id) in enumerate(worker_actions)
        if path.startswith("/v1/runs/")
    )
    assert all(
        index < first_cleanup
        for index, (path, _node_id) in enumerate(worker_actions)
        if path == "/v1/cancel"
    )


def test_worker_round_wait_uses_configured_round_timeout(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config = replace(
        make_config(tmp_path),
        round_timeout_seconds=11,
    )
    coordinator = TrainingCoordinator(config, evaluator=FakeEvaluator())
    observed_times = iter([100, 100, 106, 112])
    poll_count = 0
    with coordinator._lock:
        coordinator._groups.append({
            "group_id": "group-1",
            "runs": [{"run_id": "run-1", "rounds": []}],
        })

    def fake_parallel_workers(operation):
        nonlocal poll_count
        poll_count += 1
        return [(worker.node_id, {"runs": {}}) for worker in config.workers]

    monkeypatch.setattr("master.training.time.monotonic", lambda: next(observed_times))
    monkeypatch.setattr("master.training.time.sleep", lambda _: None)
    coordinator._parallel_workers = fake_parallel_workers

    try:
        coordinator._wait_for_worker_rounds("run-1", 0, "group-1")
    except TrainingError as error:
        assert "Timed out waiting" in str(error)
    else:
        raise AssertionError("Expected the configured round deadline to expire")

    assert poll_count == 2


def test_failed_distributed_run_is_persisted_as_failed(tmp_path: Path) -> None:
    coordinator = TrainingCoordinator(make_config(tmp_path), evaluator=FakeEvaluator())

    def fail_before_rounds(group_id, group, run):
        raise RuntimeError("worker disconnected")

    coordinator._run_distributed_seed = fail_before_rounds
    accepted = coordinator.start_training(rounds=1, epochs=1)
    deadline = time.monotonic() + 3
    while coordinator.results()["active_group_id"] is not None and time.monotonic() < deadline:
        time.sleep(0.01)

    group = next(item for item in coordinator.results()["groups"] if item["group_id"] == accepted["group_id"])
    assert group["status"] == "failed"
    assert group["runs"][0]["status"] == "failed"
    assert group["runs"][0]["error"] == "RuntimeError: worker disconnected"

    with sqlite3.connect(tmp_path / "experiments.sqlite3") as connection:
        persisted = json.loads(
            connection.execute(
                "SELECT payload_json FROM experiment_groups WHERE group_id = ?",
                (accepted["group_id"],),
            ).fetchone()[0]
        )
    assert persisted["runs"][0]["status"] == "failed"


def test_legacy_json_results_are_imported_into_sqlite_without_being_removed(
    tmp_path: Path,
) -> None:
    legacy_path = tmp_path / "experiments.json"
    legacy_group = {"group_id": "legacy-group", "status": "completed", "runs": []}
    legacy_path.write_text(json.dumps([legacy_group]), encoding="utf-8")

    coordinator = TrainingCoordinator(make_config(tmp_path), evaluator=FakeEvaluator())

    assert coordinator.results()["groups"] == [legacy_group]
    assert json.loads(legacy_path.read_text(encoding="utf-8")) == [legacy_group]
    with sqlite3.connect(tmp_path / "experiments.sqlite3") as connection:
        stored = connection.execute(
            "SELECT payload_json FROM experiment_groups WHERE group_id = ?",
            ("legacy-group",),
        ).fetchone()
    assert json.loads(stored[0]) == legacy_group


def test_nas_summary_accepts_dataset_validation_metric_names() -> None:
    runs = [
        {
            "rounds": [
                {
                    "workers": {
                        "worker1": {
                            "validation_accuracy": 0.92,
                            "validation_macro_f1": 0.91,
                            "round_seconds": 2.0,
                            "peak_process_rss_mb": 1200,
                            "bytes_sent": 100,
                        },
                        "worker2": {
                            "validation_accuracy": 0.94,
                            "validation_macro_f1": 0.93,
                            "round_seconds": 2.5,
                            "peak_process_rss_mb": 1300,
                            "bytes_sent": 150,
                        },
                    },
                },
            ],
        },
    ]

    summary = TrainingCoordinator._summarize_nas_runs(
        runs,
        search_seconds=10,
        search_peak_rss_mb=1100,
    )

    assert abs(summary["mean_accuracy"] - 0.93) < 1e-9
    assert abs(summary["mean_macro_f1"] - 0.92) < 1e-9
    assert summary["mean_communication_bytes"] == 250


def test_nas_summary_retains_legacy_metric_name_compatibility() -> None:
    runs = [
        {
            "rounds": [
                {
                    "workers": {
                        "worker1": {
                            "accuracy": 0.9,
                            "macro_f1": 0.8,
                        },
                    },
                },
            ],
        },
    ]

    summary = TrainingCoordinator._summarize_nas_runs(
        runs,
        search_seconds=0,
        search_peak_rss_mb=0,
    )

    assert summary["mean_accuracy"] == 0.9
    assert summary["mean_macro_f1"] == 0.8


def test_running_groups_are_marked_interrupted_after_coordinator_restart(
    tmp_path: Path,
) -> None:
    legacy_path = tmp_path / "experiments.json"
    legacy_group = {
        "group_id": "interrupted-group",
        "status": "running",
        "completed_at": None,
        "error": None,
        "runs": [
            {
                "run_id": "interrupted-run",
                "status": "running",
                "progress": {"round": 1, "workers": {}},
                "rounds": [],
            }
        ],
    }
    legacy_path.write_text(json.dumps([legacy_group]), encoding="utf-8")

    coordinator = TrainingCoordinator(make_config(tmp_path), evaluator=FakeEvaluator())
    recovered = coordinator.results()["groups"][0]

    assert coordinator.results()["active_group_id"] is None
    assert recovered["status"] == "failed"
    assert recovered["completed_at"] is not None
    assert "master coordinator restarted" in recovered["error"]
    assert recovered["runs"][0]["status"] == "failed"
    assert recovered["runs"][0]["error"] == recovered["error"]
    assert "progress" not in recovered["runs"][0]

    with sqlite3.connect(tmp_path / "experiments.sqlite3") as connection:
        stored = connection.execute(
            "SELECT payload_json FROM experiment_groups WHERE group_id = ?",
            ("interrupted-group",),
        ).fetchone()
    assert json.loads(stored[0]) == recovered


def test_delete_completed_experiment_group_removes_persisted_result(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    coordinator = TrainingCoordinator(config, evaluator=FakeEvaluator())
    group = {
        "group_id": "nas-result-1",
        "kind": "dataset_architecture_search",
        "status": "completed",
        "runs": [],
    }
    with coordinator._lock:
        coordinator._groups.append(group)
        coordinator._persist_locked()

    assert coordinator.delete_group("nas-result-1") == {
        "deleted": True,
        "group_id": "nas-result-1",
    }
    assert coordinator.results()["groups"] == []
    with sqlite3.connect(tmp_path / "experiments.sqlite3") as connection:
        assert connection.execute("SELECT COUNT(*) FROM experiment_groups").fetchone()[0] == 0


def test_delete_active_experiment_group_is_rejected(tmp_path: Path) -> None:
    coordinator = TrainingCoordinator(make_config(tmp_path), evaluator=FakeEvaluator())
    group = {"group_id": "active-1", "kind": "decentralized", "status": "running", "runs": []}
    with coordinator._lock:
        coordinator._groups.append(group)
        coordinator._active_group_id = group["group_id"]

    try:
        coordinator.delete_group(group["group_id"])
    except RuntimeError as error:
        assert "active" in str(error)
    else:
        raise AssertionError("An active experiment group was deleted")


def test_dataset_training_records_dataset_and_uses_one_seed_by_default(tmp_path: Path) -> None:
    coordinator = TrainingCoordinator(make_config(tmp_path), evaluator=FakeEvaluator())
    coordinator._run_distributed_seed = lambda group_id, group, run: None

    accepted = coordinator.start_training(
        rounds=1,
        epochs=1,
        dataset_name="cifar_10k_demo",
    )
    deadline = time.monotonic() + 3
    while coordinator.results()["active_group_id"] is not None and time.monotonic() < deadline:
        time.sleep(0.01)

    group = next(
        item
        for item in coordinator.results()["groups"]
        if item["group_id"] == accepted["group_id"]
    )
    assert accepted["dataset_name"] == "cifar_10k_demo"
    assert group["dataset_name"] == "cifar_10k_demo"
    assert group["seeds"] == [2026]
    assert group["status"] == "completed"


def test_dataset_training_rejects_invalid_dataset_name(tmp_path: Path) -> None:
    coordinator = TrainingCoordinator(make_config(tmp_path), evaluator=FakeEvaluator())

    try:
        coordinator.start_training(
            rounds=1,
            epochs=1,
            dataset_name="../outside",
        )
    except ValueError as error:
        assert "dataset_name" in str(error)
    else:
        raise AssertionError("Invalid dataset name was accepted")


def test_dataset_training_accepts_validated_custom_model_manifest(tmp_path: Path) -> None:
    coordinator = TrainingCoordinator(make_config(tmp_path), evaluator=FakeEvaluator())
    coordinator._run_distributed_seed = lambda group_id, group, run: None
    classes = [str(index) for index in range(10)]
    architecture = {
        "model_type": "custom",
        "layers": [
            {"type": "conv2d", "filters": 8, "kernel_size": 3},
            {"type": "global_average_pooling2d"},
            {"type": "dense", "units": 10, "activation": "linear"},
        ],
        "training": {"optimizer": "adamw", "learning_rate": 0.0005, "batch_size": 64},
        "dataset_name": "mnist",
        "input_shape": [28, 28, 1],
        "classes": classes,
        "num_classes": 10,
        "parameter_count": 170,
        "initialization_seed": 42,
    }

    accepted = coordinator.start_training(
        rounds=1,
        epochs=1,
        dataset_name="mnist",
        architecture=architecture,
    )
    deadline = time.monotonic() + 3
    while coordinator.results()["active_group_id"] is not None and time.monotonic() < deadline:
        time.sleep(0.01)

    group = next(
        item
        for item in coordinator.results()["groups"]
        if item["group_id"] == accepted["group_id"]
    )
    assert group["status"] == "completed"
    assert group["architecture"]["model_type"] == "custom"
    assert group["architecture"]["layers"][0]["padding"] == "same"
    assert group["architecture"]["training"]["batch_size"] == 64
    assert group["architecture"]["parameter_count"] == 170


def test_dataset_training_rejects_invalid_custom_model_manifest(tmp_path: Path) -> None:
    coordinator = TrainingCoordinator(make_config(tmp_path), evaluator=FakeEvaluator())

    try:
        coordinator.start_training(
            rounds=1,
            epochs=1,
            dataset_name="mnist",
            architecture={
                "model_type": "custom",
                "layers": [
                    {"type": "flatten"},
                    {"type": "dense", "units": 10, "activation": "softmax"},
                ],
                "dataset_name": "mnist",
                "input_shape": [28, 28, 1],
                "classes": [str(index) for index in range(10)],
                "num_classes": 10,
                "parameter_count": 10,
            },
        )
    except ValueError as error:
        assert "linear Dense" in str(error)
    else:
        raise AssertionError("Invalid custom model manifest was accepted")


def test_dataset_training_forwards_named_shard_and_uses_worker_validation(
    tmp_path: Path,
) -> None:
    coordinator = TrainingCoordinator(make_config(tmp_path), evaluator=FakeEvaluator())
    group = {
        "group_id": "dataset-run",
        "kind": "decentralized",
        "dataset_name": "cifar_10k_demo",
        "topology": "bidirectional_ring",
        "rounds": 1,
        "epochs": 1,
        "dataset_split": "stratified",
        "architecture": {"id": "b1_c8_k3"},
        "aggregation": "metropolis",
        "sync_mode": "semi_sync",
        "sync_deadline_seconds": 120,
        "runs": [
            {
                "run_id": "dataset-run-seed-2026",
                "seed": 2026,
                "topology": "bidirectional_ring",
                "status": "running",
                "rounds": [],
            }
        ],
    }
    run = {
        "run_id": "dataset-run-seed-2026",
        "seed": 2026,
        "topology": "bidirectional_ring",
        "rounds": [],
    }
    coordinator._groups.append(group)
    config_payloads = []
    coordinator._parallel_workers = lambda callback: [
        callback(worker) for worker in coordinator.config.workers
    ]

    def fake_post(worker, path, payload, *, timeout=30):
        if path == "/v1/config":
            config_payloads.append(payload)
        return {"accepted": True}

    coordinator._post = fake_post
    coordinator._wait_for_worker_rounds = lambda *args: None
    coordinator._get_json = lambda worker, path, *, timeout=10: {
        "runs": {
            run["run_id"]: [
                {
                    "node_id": worker.node_id,
                    "training_accuracy": 0.4,
                    "validation_accuracy": 0.3,
                    "validation_macro_f1": 0.25,
                    "bytes_sent": 100,
                }
            ]
        }
    }
    downloaded_weights = []
    coordinator._get_bytes = lambda worker, path: downloaded_weights.append(path) or b"weights"
    cleaned_workers = []
    coordinator._delete = lambda worker, path, *, timeout=30: (
        cleaned_workers.append((worker.node_id, path)) or {"cleaned": True}
    )

    coordinator._run_distributed_seed("dataset-run", group, run)

    assert len(config_payloads) == 3
    assert all(
        config["dataset_name"] == "cifar_10k_demo"
        and config["aggregation"] == "metropolis"
        and config["sync_mode"] == "semi_sync"
        and config["sync_deadline_seconds"] == 120
        for config in config_payloads
    )
    # Dataset runs report worker-local validation metrics; weights are not downloaded.
    assert downloaded_weights == []
    stored_group = coordinator.results()["groups"][0]
    final_workers = stored_group["runs"][0]["rounds"][0]["workers"]
    assert set(final_workers) == {"worker1", "worker2", "worker3"}
    assert all(row["validation_accuracy"] == 0.3 for row in final_workers.values())
    assert all(row["validation_macro_f1"] == 0.25 for row in final_workers.values())
    assert all(row["accuracy"] == 0.3 for row in final_workers.values())
    assert all(row["macro_f1"] == 0.25 for row in final_workers.values())
    assert cleaned_workers == [
        (worker.node_id, "/v1/runs/dataset-run-seed-2026")
        for worker in coordinator.config.workers
    ]
    assert run["worker_cleanup"]["status"] == "completed"
    assert set(run["worker_cleanup"]["workers"]) == {
        worker.node_id for worker in coordinator.config.workers
    }


def test_optimizer_runs_real_candidate_sweep_and_persists_reward_selection(tmp_path: Path) -> None:
    coordinator = TrainingCoordinator(make_config(tmp_path), evaluator=FakeEvaluator())

    def record_candidate_run(group_id, group, run):
        accuracy = {
            "directed_ring": 0.54,
            "bidirectional_ring": 0.55,
            "mesh": 0.56,
            "chain": 0.57,
            "star": 0.58,
        }[group["topology"]]
        row = {
            "round": 1,
            "workers": {
                worker.node_id: {
                    "validation_accuracy": accuracy,
                    "bytes_sent": 100,
                    "exchange_seconds": 0.1,
                    "round_seconds": 2,
                }
                for worker in coordinator.config.workers
            },
        }
        run["rounds"].append(row)

    coordinator._run_distributed_seed = record_candidate_run
    accepted = coordinator.start_optimization(
        rounds=1,
        epochs=1,
        seeds=(17,),
        alphas=(0.3,),
    )
    deadline = time.monotonic() + 3
    while coordinator.results()["active_group_id"] is not None and time.monotonic() < deadline:
        time.sleep(0.01)

    group = next(
        item
        for item in coordinator.results()["groups"]
        if item["group_id"] == accepted["group_id"]
    )
    assert group["status"] == "completed"
    assert group["dataset_name"] == "mnist"
    assert group["reward_metric_source"] == "validation_accuracy"
    assert len(group["candidates"]) == 5
    # Reward v2: equal costs, so the most accurate topology wins; at 58% no worker
    # reaches the 90% quality gate, which the selection reports.
    assert group["selection"]["method"] == "quality_gated_reward_v2"
    assert group["selection"]["quality_gate_met"] is False
    assert group["selection"]["selected_candidate"]["topology"] == "star"
    assert group["selection"]["baselines"]["best_observed"]["topology"] == "star"
    assert group["reward_config"]["version"] == 2
    assert group["reward_config"]["weights"]["quality"] == 0.5
    assert all(candidate["reward_version"] == 2 for candidate in group["candidates"])
    assert all(candidate["std_reward"] == 0 for candidate in group["candidates"])
    assert coordinator.results()["configuration"]["dataset_split"] == "dirichlet_a03"


@pytest.mark.parametrize("dataset_split", ["dirichlet_a01", "dirichlet_a07"])
def test_training_configuration_accepts_new_dirichlet_splits(
    tmp_path: Path,
    dataset_split: str,
) -> None:
    coordinator = TrainingCoordinator(make_config(tmp_path), evaluator=FakeEvaluator())

    assert coordinator.configure(dataset_split=dataset_split)["dataset_split"] == dataset_split


def test_optimizer_publishes_candidate_metrics_after_each_completed_run(
    tmp_path: Path,
) -> None:
    coordinator = TrainingCoordinator(make_config(tmp_path), evaluator=FakeEvaluator())
    second_run_started = threading.Event()
    release_second_run = threading.Event()

    def record_candidate_run(group_id, group, run):
        accuracy = {
            "directed_ring": 0.54,
            "bidirectional_ring": 0.55,
            "mesh": 0.56,
            "chain": 0.57,
            "star": 0.58,
        }[group["topology"]]
        run["rounds"].append(
            {
                "round": 1,
                "workers": {
                    worker.node_id: {
                        "validation_accuracy": accuracy,
                        "bytes_sent": 100,
                        "exchange_seconds": 0.1,
                        "round_seconds": 2,
                    }
                    for worker in coordinator.config.workers
                },
            }
        )
        if group["topology"] == "bidirectional_ring":
            second_run_started.set()
            release_second_run.wait(timeout=3)

    coordinator._run_distributed_seed = record_candidate_run
    accepted = coordinator.start_optimization(
        rounds=1,
        epochs=1,
        seeds=(17,),
        alphas=(0.3,),
    )
    try:
        assert second_run_started.wait(timeout=3)
        group = next(
            item
            for item in coordinator.results()["groups"]
            if item["group_id"] == accepted["group_id"]
        )

        assert group["status"] == "running"
        assert len(group["candidates"]) == 5
        assert group["candidates"][0]["topology"] == "directed_ring"
        assert group["candidates"][0]["completed_runs"] == 1
        assert group["candidates"][0]["mean_reward"] >= 0
        assert group["candidates"][1]["topology"] == "bidirectional_ring"
        assert group["candidates"][1]["completed_runs"] == 0
        assert "mean_reward" not in group["candidates"][1]
        assert group.get("selection") is None
    finally:
        release_second_run.set()

    deadline = time.monotonic() + 3
    while coordinator.results()["active_group_id"] is not None and time.monotonic() < deadline:
        time.sleep(0.01)


def test_optimizer_defaults_to_only_dirichlet_alpha_03(tmp_path: Path) -> None:
    coordinator = TrainingCoordinator(make_config(tmp_path), evaluator=FakeEvaluator())
    coordinator._run_distributed_seed = lambda group_id, group, run: run["rounds"].append(
        {
            "round": 1,
            "workers": {
                worker.node_id: {"validation_accuracy": 0.6}
                for worker in coordinator.config.workers
            },
        }
    )

    accepted = coordinator.start_optimization(rounds=1, epochs=1, seeds=(17,))
    deadline = time.monotonic() + 3
    while coordinator.results()["active_group_id"] is not None and time.monotonic() < deadline:
        time.sleep(0.01)

    group = next(
        item
        for item in coordinator.results()["groups"]
        if item["group_id"] == accepted["group_id"]
    )
    assert accepted["alphas"] == [0.3]
    assert accepted["candidate_count"] == 5
    assert group["alphas"] == [0.3]
    assert len(group["runs"]) == 5
    assert {run["alpha"] for run in group["runs"]} == {0.3}


def test_optimizer_rejects_unsupported_or_multiple_alphas(tmp_path: Path) -> None:
    coordinator = TrainingCoordinator(make_config(tmp_path), evaluator=FakeEvaluator())

    for alphas in ((0.7,), (0.3, 0.5), ()):
        with pytest.raises(ValueError, match="exactly one Dirichlet alpha"):
            coordinator.start_optimization(alphas=alphas)
    with pytest.raises(ValueError, match="supports"):
        coordinator.start_optimization(dataset_name="imagenet")


def test_cifar_optimizer_uses_the_cifar_architecture_and_records_the_alpha(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    coordinator = TrainingCoordinator(make_config(tmp_path), evaluator=FakeEvaluator())
    seen: list[dict] = []

    def fake_seed(group_id, group, run):
        seen.append(group)
        run["rounds"].append(
            {
                "round": 1,
                "workers": {
                    worker.node_id: {"validation_accuracy": 0.6}
                    for worker in coordinator.config.workers
                },
            }
        )

    monkeypatch.setattr(coordinator, "_run_distributed_seed", fake_seed)

    accepted = coordinator.start_optimization(
        rounds=1, epochs=1, seeds=(17,), alphas=(0.1,), dataset_name="cifar10"
    )
    deadline = time.monotonic() + 3
    while coordinator.results()["active_group_id"] is not None and time.monotonic() < deadline:
        time.sleep(0.01)

    assert accepted["dataset_name"] == "cifar10"
    assert accepted["architecture_id"] == "nas-d3-w32x64x64-k3-conv-h64-drop20"
    assert len(seen) == 5
    assert all(group["dataset_name"] == "cifar10" for group in seen)
    assert all(group["dataset_split"] == "dirichlet_a01" for group in seen)
    assert all(
        group["architecture"]["channel_schedule"] == [32, 64, 64] for group in seen
    )


def test_dataset_nas_searches_selected_shards_and_persists_selection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    coordinator = TrainingCoordinator(make_config(tmp_path), evaluator=FakeEvaluator())
    calls = []
    initial = initial_population(
        seed=17,
        input_shape=(28, 28, 1),
        num_classes=2,
    )
    best_candidate_id = initial[0]["id"]

    def fake_post(worker, path, payload, timeout):
        calls.append((worker.node_id, path, payload, timeout))
        return dataset_search_response(
            worker,
            payload,
            best_candidate_id=best_candidate_id,
        )

    def run_workers(operation, *, on_result=None):
        results = []
        for worker in coordinator.config.workers:
            result = operation(worker)
            results.append(result)
            if on_result is not None:
                on_result(result)
        return results

    coordinator._parallel_workers = run_workers
    coordinator._post = fake_post
    accepted = coordinator.start_dataset_nas(
        dataset_name="cats-and-dogs",
        search_time_limit_seconds=600,
        seed=17,
    )
    deadline = time.monotonic() + 3
    while coordinator.results()["active_group_id"] is not None and time.monotonic() < deadline:
        time.sleep(0.01)

    group = next(
        item
        for item in coordinator.results()["groups"]
        if item["group_id"] == accepted["group_id"]
    )
    assert group["status"] == "completed"
    assert group["search_time_limit_seconds"] == 600
    assert group["dataset_name"] == "cats-and-dogs"
    assert group["dataset_classes"] == ["cat", "dog"]
    assert group["selected_architecture"]["id"] == best_candidate_id
    assert group["selected_architecture"]["dataset_name"] == "cats-and-dogs"
    assert group["selected_architecture"]["input_shape"] == [28, 28, 1]
    assert group["selected_architecture"]["classes"] == ["cat", "dog"]
    assert group["selected_architecture"]["num_classes"] == 2
    assert group["selected_parameter_count"] == group["selected_architecture"]["parameter_count"]
    assert group["search_method"] == "pareto_evolutionary_multifidelity"
    assert [stage["fidelity_epochs"] for stage in group["nas_generations"]] == [INITIAL_EPOCHS, FINAL_EPOCHS]
    assert [len(stage["candidates"]) for stage in group["nas_generations"]] == [8, 4]
    assert group["selection"]["selected_candidate"]["architecture"]["id"] == best_candidate_id
    assert (
        group["selection"]["selected_candidate"]["architecture"]
        == group["selected_architecture"]
    )
    assert [result["node_id"] for result in group["worker_search"]] == [
        "worker1",
        "worker2",
        "worker3",
    ]
    assert {
        worker: progress["status"]
        for worker, progress in group["worker_search_progress"].items()
    } == {
        "worker1": "completed",
        "worker2": "completed",
        "worker3": "completed",
    }
    assert len(calls) == 6
    assert all(
        path == "/v1/nas/search"
        and payload["dataset_name"] == "cats-and-dogs"
        for _, path, payload, _ in calls
    )
    assert [payload["epochs"] for _, _, payload, _ in calls[:3]] == [INITIAL_EPOCHS] * 3
    assert [payload["epochs"] for _, _, payload, _ in calls[3:]] == [FINAL_EPOCHS] * 3
    assert [payload["time_limit_seconds"] for _, _, payload, _ in calls[:3]] == [300, 300, 300]
    assert [payload["time_limit_seconds"] for _, _, payload, _ in calls[3:]] == [300, 300, 300]
    assert all("architectures" not in payload for _, _, payload, _ in calls[:3])
    assert all("architectures" in payload for _, _, payload, _ in calls[3:])

    monkeypatch.setattr(coordinator, "_run_group", lambda _group_id: None)
    retrain = coordinator.start_training(
        rounds=5,
        epochs=10,
        dataset_name="cats-and-dogs",
        architecture=group["selected_architecture"],
    )
    assert retrain["accepted"] is True
    assert retrain["rounds"] == 5
    assert retrain["epochs"] == 10


def test_cifar10_dataset_nas_uses_memory_constrained_candidates(tmp_path: Path) -> None:
    coordinator = TrainingCoordinator(make_config(tmp_path), evaluator=FakeEvaluator())
    calls = []

    def fake_post(worker, path, payload, timeout):
        calls.append(payload)
        return dataset_search_response(worker, payload)

    def run_workers(operation, *, on_result=None):
        results = []
        for worker in coordinator.config.workers:
            result = operation(worker)
            results.append(result)
            if on_result is not None:
                on_result(result)
        return results

    coordinator._parallel_workers = run_workers
    coordinator._post = fake_post
    accepted = coordinator.start_dataset_nas(
        dataset_name="cifar10",
        search_time_limit_seconds=600,
        seed=17,
    )
    deadline = time.monotonic() + 3
    while coordinator.results()["active_group_id"] is not None and time.monotonic() < deadline:
        time.sleep(0.01)

    group = next(
        item
        for item in coordinator.results()["groups"]
        if item["group_id"] == accepted["group_id"]
    )
    assert group["status"] == "completed"
    workers = len(coordinator.config.workers)
    # CIFAR-10 now evolves too: a one-epoch screen, then the survivors for FINAL_EPOCHS.
    assert len(calls) == 2 * workers
    assert group["search_method"] == "pareto_evolutionary_multifidelity"
    assert [stage["fidelity_epochs"] for stage in group["nas_generations"]] == [INITIAL_EPOCHS, FINAL_EPOCHS]
    assert len(group["nas_generations"][0]["candidates"]) == CIFAR_INITIAL_POPULATION_SIZE
    assert [payload["epochs"] for payload in calls[:workers]] == [INITIAL_EPOCHS] * workers
    assert [payload["epochs"] for payload in calls[workers:]] == [FINAL_EPOCHS] * workers
    assert all("architectures" not in payload for payload in calls[:workers])
    assert all(
        2 <= candidate["architecture"]["blocks"] <= 3
        and max(candidate["architecture"]["channel_schedule"]) <= 64
        and candidate["architecture"]["parameter_count"] <= MEMORY_CONSTRAINED_MAX_PARAMETERS
        and candidate["architecture"]["kernel_size"] == 3
        and candidate["architecture"]["attention"] is False
        for stage in group["nas_generations"]
        for candidate in stage["candidates"]
    )
    assert group["selection"]["method"] == "pareto_evolutionary_multifidelity"


def test_dataset_nas_publishes_worker_progress_as_each_worker_finishes(
    tmp_path: Path,
) -> None:
    coordinator = TrainingCoordinator(make_config(tmp_path), evaluator=FakeEvaluator())
    first_worker_finished = threading.Event()
    release_remaining_workers = threading.Event()

    def fake_post(worker, path, payload, timeout):
        response = dataset_search_response(worker, payload)
        response["dataset_classes"] = ["zero", "one"]
        return response

    def run_workers(operation, *, on_result=None):
        results = []
        for index, worker in enumerate(coordinator.config.workers):
            result = operation(worker)
            results.append(result)
            if on_result is not None:
                on_result(result)
            if index == 0:
                first_worker_finished.set()
                release_remaining_workers.wait(timeout=3)
        return results

    coordinator._post = fake_post
    coordinator._parallel_workers = run_workers
    accepted = coordinator.start_dataset_nas(
        dataset_name="mnist",
        search_time_limit_seconds=90,
        seed=17,
    )
    try:
        assert first_worker_finished.wait(timeout=3)
        group = next(
            item
            for item in coordinator.results()["groups"]
            if item["group_id"] == accepted["group_id"]
        )
        assert len(group["worker_search"]) == 1
        assert group["worker_search_progress"]["worker1"]["status"] == "completed"
        assert group["worker_search_progress"]["worker1"]["candidate_count"] > 0
        assert group["worker_search_progress"]["worker2"]["status"] == "searching"
        assert group["worker_search_progress"]["worker3"]["status"] == "searching"
    finally:
        release_remaining_workers.set()

    deadline = time.monotonic() + 3
    while coordinator.results()["active_group_id"] is not None and time.monotonic() < deadline:
        time.sleep(0.01)


def test_dataset_nas_smoke_test_runs_one_candidate_and_one_stage(tmp_path: Path) -> None:
    coordinator = TrainingCoordinator(make_config(tmp_path), evaluator=FakeEvaluator())
    calls = []

    def fake_post(worker, path, payload, timeout):
        calls.append((worker.node_id, path, payload))
        return dataset_search_response(worker, payload)

    def run_workers(operation, *, on_result=None):
        results = []
        for worker in coordinator.config.workers:
            result = operation(worker)
            results.append(result)
            if on_result is not None:
                on_result(result)
        return results

    coordinator._parallel_workers = run_workers
    coordinator._post = fake_post
    accepted = coordinator.start_dataset_nas(
        dataset_name="mnist",
        search_time_limit_seconds=90,
        smoke_test=True,
    )
    deadline = time.monotonic() + 3
    while coordinator.results()["active_group_id"] is not None and time.monotonic() < deadline:
        time.sleep(0.01)

    group = next(
        item
        for item in coordinator.results()["groups"]
        if item["group_id"] == accepted["group_id"]
    )
    assert group["status"] == "completed"
    assert group["smoke_test"] is True
    assert group["search_method"] == "single_candidate_smoke_test"
    assert [stage["fidelity_epochs"] for stage in group["nas_generations"]] == [1]
    assert [len(stage["candidates"]) for stage in group["nas_generations"]] == [1]
    assert len(calls) == 3
    assert all(payload["smoke_test"] is True for _, _, payload in calls)
    assert all(payload["epochs"] == 1 for _, _, payload in calls)
    assert all(payload["time_limit_seconds"] >= 30 for _, _, payload in calls)


def test_dataset_nas_rejects_missing_or_invalid_dataset_name(tmp_path: Path) -> None:
    coordinator = TrainingCoordinator(make_config(tmp_path), evaluator=FakeEvaluator())
    for invalid in ("", "../outside", "bad/name"):
        try:
            coordinator.start_dataset_nas(dataset_name=invalid)
        except ValueError as error:
            assert "dataset_name" in str(error)
        else:
            raise AssertionError(f"Invalid dataset name was accepted: {invalid}")


def test_results_topologies_include_all_configured_workers(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    workers = config.workers + (
        TrainingWorker("worker4", "http://127.0.0.1:5007", "x" * 32),
    )
    coordinator = TrainingCoordinator(
        replace(config, workers=workers),
        evaluator=FakeEvaluator(),
    )

    topologies = coordinator.results()["topologies"]

    assert topologies["ring"] == [
        "worker1",
        "worker2",
        "worker3",
        "worker4",
        "worker1",
    ]
    assert len(topologies["mesh"]) == 6
    assert topologies["chain"][-1] == ["worker3", "worker4"]
    assert topologies["star"] == [
        ["worker1", "worker2"],
        ["worker1", "worker3"],
        ["worker1", "worker4"],
    ]


def test_configure_validates_aggregation_and_sync_settings(tmp_path: Path) -> None:
    coordinator = TrainingCoordinator(make_config(tmp_path), evaluator=FakeEvaluator())

    settings = coordinator.configure(aggregation="trimmed_mean", sync_mode="semi_sync", sync_deadline_seconds=90)

    assert settings["aggregation"] == "trimmed_mean"
    assert settings["sync_mode"] == "semi_sync"
    assert settings["sync_deadline_seconds"] == 90
    with pytest.raises(ValueError, match="aggregation"):
        coordinator.configure(aggregation="median")
    with pytest.raises(ValueError, match="sync_mode"):
        coordinator.configure(sync_mode="async")
    with pytest.raises(ValueError, match="sync_deadline_seconds"):
        coordinator.configure(sync_deadline_seconds=1)


def test_hpo_tunes_training_settings_of_one_architecture_on_every_worker(
    tmp_path: Path,
) -> None:
    from common.hpo import HPO_EPOCHS

    coordinator = TrainingCoordinator(make_config(tmp_path), evaluator=FakeEvaluator())
    base = {
        **initial_population(seed=17, input_shape=(28, 28, 1), num_classes=2)[0],
        "dataset_name": "cats-and-dogs",
        "input_shape": [28, 28, 1],
        "classes": ["cat", "dog"],
        "num_classes": 2,
    }
    calls = []

    def fake_post(worker, path, payload, timeout):
        calls.append((path, payload))
        return dataset_search_response(worker, payload, best_candidate_id="hpo-adam-lr3e-3-bs64")

    def run_workers(operation, *, on_result=None):
        results = []
        for worker in coordinator.config.workers:
            result = operation(worker)
            results.append(result)
            if on_result is not None:
                on_result(result)
        return results

    coordinator._parallel_workers = run_workers
    coordinator._post = fake_post
    accepted = coordinator.start_hpo(dataset_name="cats-and-dogs", architecture=base, search_time_limit_seconds=300)
    deadline = time.monotonic() + 3
    while coordinator.results()["active_group_id"] is not None and time.monotonic() < deadline:
        time.sleep(0.01)

    group = next(item for item in coordinator.results()["groups"] if item["group_id"] == accepted["group_id"])
    assert group["status"] == "completed", group.get("error")
    assert group["kind"] == "hyperparameter_search"
    assert accepted["candidate_count"] == 6
    assert all(path == "/v1/nas/search" and payload["epochs"] == HPO_EPOCHS for path, payload in calls)
    assert len(calls[0][1]["architectures"]) == 6
    assert group["selected_hyperparameters"] == {"optimizer": "adam", "learning_rate": 0.003, "batch_size": 64}
    selected = group["selected_architecture"]
    assert selected["id"] == "hpo-adam-lr3e-3-bs64"
    assert selected["layers"] == base["layers"]
    assert selected["classes"] == ["cat", "dog"] and selected["input_shape"] == [28, 28, 1]


def test_hpo_rejects_architectures_without_dataset_shape(tmp_path: Path) -> None:
    coordinator = TrainingCoordinator(make_config(tmp_path), evaluator=FakeEvaluator())
    base = initial_population(seed=17, input_shape=(28, 28, 1), num_classes=2)[0]
    with pytest.raises(ValueError, match="input_shape and classes"):
        coordinator.start_hpo(dataset_name="cats-and-dogs", architecture=base)


def test_learned_topology_probes_a_mesh_then_trains_on_the_apg_graph(tmp_path: Path) -> None:
    coordinator = TrainingCoordinator(make_config(tmp_path), evaluator=FakeEvaluator())
    worker_ids = [worker.node_id for worker in coordinator.config.workers]
    seen: list[dict] = []

    def fake_seed(group_id, group, run):
        seen.append({"topology": group["topology"], "measure": group.get("measure_peers"), "senders": group.get("learned_senders")})
        row = {}
        for receiver in worker_ids:
            others = [worker for worker in worker_ids if worker != receiver]
            row[receiver] = {
                "validation_accuracy": 0.9,
                "peer_losses": {sender: (2.0 if sender == "worker3" else 0.3) for sender in others},
                "peer_arrival_seconds": {sender: 1.0 for sender in others},
                "class_histogram": [5, 5],
            }
        run["rounds"].append({"round": 1, "workers": row})

    coordinator._run_distributed_seed = fake_seed
    coordinator.configure(topology="learned")
    accepted = coordinator.start_training(rounds=2, epochs=1, dataset_name="cats-and-dogs")
    deadline = time.monotonic() + 3
    while coordinator.results()["active_group_id"] is not None and time.monotonic() < deadline:
        time.sleep(0.01)

    group = next(item for item in coordinator.results()["groups"] if item["group_id"] == accepted["group_id"])
    assert group["status"] == "completed", group.get("error")
    assert seen[0] == {"topology": "mesh", "measure": True, "senders": None}
    assert seen[1]["topology"] == "learned" and seen[1]["senders"] == group["learned_senders"]
    assert set(group["learned_senders"]) == set(worker_ids)
    assert group["runs"][0]["phase"] == "topology_probe"
    assert group["topology_learning"]["edges"] <= group["topology_learning"]["mesh_edges"]


def test_learned_topology_requires_a_dataset_run(tmp_path: Path) -> None:
    coordinator = TrainingCoordinator(make_config(tmp_path), evaluator=FakeEvaluator())
    coordinator.configure(topology="learned")
    with pytest.raises(ValueError, match="dataset"):
        coordinator.start_training(rounds=2, epochs=1)


def test_topology_summary_rescores_history_with_the_groups_own_reward_config(tmp_path: Path) -> None:
    coordinator = TrainingCoordinator(make_config(tmp_path), evaluator=FakeEvaluator())
    v1_config = json.loads(
        (Path(__file__).resolve().parents[2] / "topology_opt" / "reward_config_v1.json").read_text(
            encoding="utf-8"
        )
    )
    rounds = [
        {
            "workers": {
                f"worker{index}": {
                    "validation_accuracy": 0.6 + 0.1 * round_index,
                    "bytes_sent": 100,
                    "exchange_seconds": 0.1,
                    "round_seconds": 2,
                }
                for index in range(1, 4)
            }
        }
        for round_index in range(2)
    ]
    group = {
        "reward_config": v1_config,
        "runs": [{"topology": "mesh", "alpha": 0.3, "status": "completed", "rounds": rounds}],
    }

    (historical,) = coordinator._summarize_topology_candidates(group, [{"topology": "mesh", "alpha": 0.3}])

    assert set(historical["mean_components"]) == set(v1_config["weights"])
    assert "reward_version" not in historical
    group["reward_config"] = coordinator.reward_config.as_dict()
    (current,) = coordinator._summarize_topology_candidates(group, [{"topology": "mesh", "alpha": 0.3}])
    assert current["reward_version"] == 2
