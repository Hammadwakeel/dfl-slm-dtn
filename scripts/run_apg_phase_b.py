"""Run the approved all-PC APG Phase B experiment matrix; never accesses Jetsons."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ["MPLCONFIGDIR"] = str(ROOT / "results_apg" / ".mplconfig")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from topology_opt.apg import (
    APGParameters,
    adjacency_edges,
    count_edge_changes,
    map_real_worker_sender_edges,
    mesh_adjacency,
    normalize_off_diagonal,
    optimize_topology,
    ring_adjacency,
    uniform_mesh_weights,
)
from topology_opt.sharding import (
    constrained_dirichlet_indices,
    iid_stratified_indices,
    stratified_local_validation_split,
)
from worker.training_service import deserialize_weights, serialize_weights
from scripts.mnist_test_split_check import (
    _build_model,
    _installed_initial_weights,
)


DATASET_ROOT = ROOT / "data" / "datasets" / "mnist_demo"
EXPORT_ROOT = ROOT / "results_apg"
PROFILE_PATH = EXPORT_ROOT / "SYNTHETIC_PROFILES.json"
CONVERGENCE_CSV = EXPORT_ROOT / "convergence.csv"
FINAL_JSON = EXPORT_ROOT / "final_metrics.json"
CONFIG_JSON = EXPORT_ROOT / "simulation_config.json"
REPORT_PATH = EXPORT_ROOT / "PHASE_B_REPORT.md"
MATRIX_RUN_DIR = EXPORT_ROOT / "matrix_runs"
MATRIX_TIME_CAP_SECONDS = 3 * 60 * 60
NODE_IDS = tuple(f"node_{index}" for index in range(8))
EMULATED_WORKER_IDS = {"node_5": "worker1", "node_6": "worker2", "node_7": "worker3"}
TRAINING_SEEDS = (17, 42, 2026)
DATA_SETTINGS = ("iid_stratified", "constrained_dirichlet_a01")
TOPOLOGIES = ("ring", "mesh", "apg")
SIMULATION_MODES = ("all_simulated", "hybrid_shaped_simulation")
ROUNDS = 30
LOCAL_EPOCHS = 3
LEARNING_RATE = 0.01
BATCH_SIZE = 32
TRAINING_ROWS_PER_NODE = 240
VALIDATION_ROWS_PER_NODE = 60
CLASS_NAMES = tuple(str(value) for value in range(10))
EXPECTED_ARCHITECTURE_SHA256 = "b66616d37317a211cd9dc2cb8d02b6213cfc292d049a2258dc7711d1e7f73bb2"
EXPECTED_WEIGHTS_SHA256 = "d23e3d771bb065c347655ac932ab90cc3a8ce12ea01345c533ffca004be78d75"
PROFILE_CHOICES = {
    "node_0": (1.0, 100),
    "node_1": (0.75, 75),
    "node_2": (0.5, 50),
    "node_3": (0.35, 25),
    "node_4": (0.25, 10),
    "node_5": (1.0, 100),
    "node_6": (1.0, 100),
    "node_7": (1.0, 100),
}
OUTPUT_FILES = (
    "convergence.csv",
    "validation_convergence_summary.csv",
    "apg_round_diagnostics.csv",
    "test_confusion_matrices.csv",
    "final_metrics.json",
    "simulation_config.json",
    "PHASE_B_REPORT.md",
    "convergence_all_simulated_iid_stratified_ring.png",
    "convergence_all_simulated_iid_stratified_mesh.png",
    "convergence_all_simulated_iid_stratified_apg.png",
    "convergence_all_simulated_constrained_dirichlet_a01_ring.png",
    "convergence_all_simulated_constrained_dirichlet_a01_mesh.png",
    "convergence_all_simulated_constrained_dirichlet_a01_apg.png",
    "convergence_hybrid_shaped_simulation_iid_stratified_ring.png",
    "convergence_hybrid_shaped_simulation_iid_stratified_mesh.png",
    "convergence_hybrid_shaped_simulation_iid_stratified_apg.png",
    "convergence_hybrid_shaped_simulation_constrained_dirichlet_a01_ring.png",
    "convergence_hybrid_shaped_simulation_constrained_dirichlet_a01_mesh.png",
    "convergence_hybrid_shaped_simulation_constrained_dirichlet_a01_apg.png",
    "final_test_all_simulated_iid_stratified.png",
    "final_test_all_simulated_constrained_dirichlet_a01.png",
    "final_test_hybrid_shaped_simulation_iid_stratified.png",
    "final_test_hybrid_shaped_simulation_constrained_dirichlet_a01.png",
    "cm_MNIST_all_simulated_iid_stratified_ring_TEST.png",
    "cm_MNIST_all_simulated_iid_stratified_mesh_TEST.png",
    "cm_MNIST_all_simulated_iid_stratified_apg_TEST.png",
    "cm_MNIST_all_simulated_constrained_dirichlet_a01_ring_TEST.png",
    "cm_MNIST_all_simulated_constrained_dirichlet_a01_mesh_TEST.png",
    "cm_MNIST_all_simulated_constrained_dirichlet_a01_apg_TEST.png",
    "cm_MNIST_hybrid_shaped_simulation_iid_stratified_ring_TEST.png",
    "cm_MNIST_hybrid_shaped_simulation_iid_stratified_mesh_TEST.png",
    "cm_MNIST_hybrid_shaped_simulation_iid_stratified_apg_TEST.png",
    "cm_MNIST_hybrid_shaped_simulation_constrained_dirichlet_a01_ring_TEST.png",
    "cm_MNIST_hybrid_shaped_simulation_constrained_dirichlet_a01_mesh_TEST.png",
    "cm_MNIST_hybrid_shaped_simulation_constrained_dirichlet_a01_apg_TEST.png",
)
SMOKE_OUTPUT_FILES = (
    "smoke_apg_all_simulated_iid_seed17.json",
    "smoke_apg_all_simulated_iid_seed17_convergence.csv",
)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object at {path}")
    return value


def _latest_training_run() -> tuple[Path, dict[str, Any]]:
    reports = sorted(
        (DATASET_ROOT / "processed").glob("run-*/preprocessing_report.json"),
        key=lambda path: path.stat().st_mtime_ns,
    )
    if not reports:
        raise FileNotFoundError("Saved MNIST preprocessing report is missing")
    report_path = reports[-1]
    report = _read_json(report_path)
    if report.get("dataset_name") != "mnist_demo":
        raise ValueError("Latest saved preprocessing report is not mnist_demo")
    if report.get("split_sizes") != {"train": 2400, "validation": 300, "test": 300}:
        raise ValueError("Expected the saved 2400/300/300 MNIST split")
    if report.get("split_indices_disjoint") is not True:
        raise ValueError("Saved MNIST train/validation/test indices are not confirmed disjoint")
    return report_path, report


def _load_train_only(report_path: Path) -> tuple[np.ndarray, np.ndarray, str]:
    train_path = report_path.parent / "train.npz"
    train_hash = hashlib.sha256(train_path.read_bytes()).hexdigest()
    with np.load(train_path, allow_pickle=False) as archive:
        images = np.asarray(archive["images"], dtype=np.float32)
        labels = np.asarray(archive["labels"], dtype=np.int64).reshape(-1)
    if images.shape != (2400, 28, 28, 1) or labels.shape != (2400,):
        raise ValueError(f"Unexpected saved MNIST TRAIN shape: {images.shape}, {labels.shape}")
    if not np.array_equal(np.bincount(labels, minlength=10), np.full(10, 240)):
        raise ValueError("Saved MNIST TRAIN split must contain 240 examples per class")
    return images, labels, train_hash


def _load_initial_model() -> tuple[dict[str, Any], list[np.ndarray], dict[str, str]]:
    architecture, weights = _installed_initial_weights()
    model_root = DATASET_ROOT / "models" / "initial_model"
    architecture_hash = hashlib.sha256((model_root / "architecture.json").read_bytes()).hexdigest()
    weights_hash = hashlib.sha256((model_root / "initial_weights.npz").read_bytes()).hexdigest()
    if architecture_hash != EXPECTED_ARCHITECTURE_SHA256 or weights_hash != EXPECTED_WEIGHTS_SHA256:
        raise ValueError("Saved MNIST initial-model hashes differ from the approved installed model")
    return architecture, weights, {
        "architecture_sha256": architecture_hash,
        "weights_sha256": weights_hash,
    }


def _compile_training_model(model: tf.keras.Model) -> tf.keras.Model:
    # IMPLEMENTATION CHOICE: compile this driver-owned model with accuracy as a metric.
    # The shared MNIST evaluation helper is left unchanged; Adam/LR/loss stay identical.
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    actual_learning_rate = float(tf.keras.backend.get_value(model.optimizer.learning_rate))
    if not np.isclose(actual_learning_rate, LEARNING_RATE, rtol=1e-6):
        raise ValueError(f"APG simulation learning rate mismatch: {actual_learning_rate}")
    return model


def _class_counts(labels: np.ndarray) -> list[int]:
    return [int(value) for value in np.bincount(labels, minlength=10)]


def _make_data_setting(
    images: np.ndarray,
    labels: np.ndarray,
    setting: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if setting == "iid_stratified":
        shards = iid_stratified_indices(labels, seed=2026)
    elif setting == "constrained_dirichlet_a01":
        shards = constrained_dirichlet_indices(labels, alpha=0.1, seed=2026)
    else:
        raise ValueError(f"Unsupported data setting: {setting}")
    shard_records: list[dict[str, Any]] = []
    for node_index, node_id in enumerate(NODE_IDS):
        shard_indices = shards[node_index]
        shard_labels = labels[shard_indices]
        shard_records.append(
            {
                "node_id": node_id,
                "images": images[shard_indices],
                "labels": shard_labels,
                "source_train_indices": shard_indices,
                "shard_rows": int(len(shard_indices)),
                "shard_class_counts": _class_counts(shard_labels),
            }
        )
    all_indices = np.concatenate([record["source_train_indices"] for record in shard_records])
    if not np.array_equal(np.sort(all_indices), np.arange(len(labels))):
        raise ValueError(f"{setting} shards do not cover the MNIST TRAIN split exactly once")
    if any(record["shard_rows"] != 300 for record in shard_records):
        raise ValueError(f"{setting} constrained 300-row client capacities were not satisfied")

    seed_splits: list[dict[str, Any]] = []
    for training_seed in TRAINING_SEEDS:
        nodes = []
        for record in shard_records:
            split = stratified_local_validation_split(
                record["labels"],
                validation_rows=VALIDATION_ROWS_PER_NODE,
                seed=training_seed,
            )
            train_indices = split.train_indices
            validation_indices = split.validation_indices
            theta = np.asarray(split.train_class_counts, dtype=np.float64)
            theta /= theta.sum()
            nodes.append(
                {
                    "node_id": record["node_id"],
                    "images": record["images"],
                    "labels": record["labels"],
                    "train_images": record["images"][train_indices],
                    "train_labels": record["labels"][train_indices],
                    "validation_images": record["images"][validation_indices],
                    "validation_labels": record["labels"][validation_indices],
                    "shard_rows": record["shard_rows"],
                    "train_rows": int(len(train_indices)),
                    "validation_rows": int(len(validation_indices)),
                    "shard_class_counts": record["shard_class_counts"],
                    "train_class_counts": list(split.train_class_counts),
                    "validation_class_counts": list(split.validation_class_counts),
                    "theta": theta,
                }
            )
        if sum(node["train_rows"] for node in nodes) != 8 * 240:
            raise ValueError(f"{setting} training exposure is not exactly 1,920 rows per epoch")
        seed_splits.append({"seed": training_seed, "nodes": nodes})
    return seed_splits, shard_records


def _metrics_from_logits(labels: np.ndarray, logits: np.ndarray) -> dict[str, Any]:
    true = np.asarray(labels, dtype=np.int64).reshape(-1)
    scores = np.asarray(logits, dtype=np.float64)
    if scores.shape != (true.size, 10):
        raise ValueError("MNIST logits must have shape (sample_count, 10)")
    predicted = scores.argmax(axis=1)
    matrix = np.zeros((10, 10), dtype=np.int64)
    np.add.at(matrix, (true, predicted), 1)
    support = matrix.sum(axis=1)
    predicted_count = matrix.sum(axis=0)
    true_positive = np.diag(matrix).astype(np.float64)
    precision = np.divide(true_positive, predicted_count, out=np.zeros(10), where=predicted_count > 0)
    recall = np.divide(true_positive, support, out=np.zeros(10), where=support > 0)
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros(10),
        where=(precision + recall) > 0,
    )
    shifted = scores - scores.max(axis=1, keepdims=True)
    log_partition = np.log(np.exp(shifted).sum(axis=1))
    losses = log_partition - shifted[np.arange(true.size), true]
    return {
        "accuracy": float(np.mean(predicted == true)),
        "macro_precision": float(precision.mean()),
        "macro_recall": float(recall.mean()),
        "macro_f1": float(f1.mean()),
        "loss": float(losses.mean()),
        "per_class": [
            {
                "class": str(class_index),
                "precision": float(precision[class_index]),
                "recall": float(recall[class_index]),
                "f1": float(f1[class_index]),
                "support": int(support[class_index]),
            }
            for class_index in range(10)
        ],
        "confusion_matrix_counts": matrix.tolist(),
        "y_true": true.tolist(),
        "y_pred": predicted.astype(np.int64).tolist(),
    }


def _evaluate_model(model: tf.keras.Model, images: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    logits = model.predict(images, batch_size=BATCH_SIZE, verbose=0)
    return _metrics_from_logits(labels, logits)


def _cross_entropy(model: tf.keras.Model, images: np.ndarray, labels: np.ndarray) -> float:
    logits = np.asarray(model.predict(images, batch_size=BATCH_SIZE, verbose=0), dtype=np.float64)
    shifted = logits - logits.max(axis=1, keepdims=True)
    return float(np.mean(np.log(np.exp(shifted).sum(axis=1)) - shifted[np.arange(len(labels)), labels]))


def _weighted_average(weighted_models: list[tuple[list[np.ndarray], int]]) -> list[np.ndarray]:
    total_samples = sum(sample_count for _, sample_count in weighted_models)
    if total_samples <= 0:
        raise ValueError("Weighted averaging needs at least one sample")
    reference = weighted_models[0][0]
    return [
        (
            sum(weights[index].astype(np.float64) * sample_count for weights, sample_count in weighted_models)
            / total_samples
        ).astype(reference[index].dtype)
        for index in range(len(reference))
    ]


def _profiles() -> tuple[dict[str, Any], dict[str, float], dict[str, float]]:
    config = _read_json(PROFILE_PATH)
    reference = config["reference_measurements"]
    base_time = float(reference["base_round_time"]["value_seconds"])
    bandwidth = float(reference["effective_bandwidth"]["numerator_bytes"]) / float(
        reference["effective_bandwidth"]["denominator_exchange_seconds"]
    )
    profiles = config["nodes"]
    train_times = {
        node_id: base_time / float(profiles[node_id]["compute_multiplier"])
        for node_id in NODE_IDS
    }
    bandwidths = {
        node_id: bandwidth * float(profiles[node_id]["bandwidth_percent"]) / 100.0
        for node_id in NODE_IDS
    }
    return config, train_times, bandwidths


def _wire_id(node_id: str) -> str:
    return EMULATED_WORKER_IDS.get(node_id, node_id)


SIMULATED_PROFILE: dict[str, Any] = {}


def _transport_update(
    simulation_mode: str,
    run_id: str,
    round_index: int,
    receiver: str,
    source: str,
    source_weights: list[np.ndarray],
    sample_count: int,
) -> tuple[list[np.ndarray], int, dict[str, str] | None]:
    payload = serialize_weights(source_weights)
    if simulation_mode == "all_simulated":
        return [np.asarray(value).copy() for value in source_weights], len(payload), None
    # Hybrid-shaped mode keeps all traffic local but uses the worker HTTP weight envelope.
    headers = {
        "X-Run-ID": run_id,
        "X-Round": str(round_index),
        "X-Sender-Node": _wire_id(source),
        "X-Sample-Count": str(sample_count),
    }
    if headers["X-Run-ID"] != run_id or headers["X-Round"] != str(round_index):
        raise ValueError("Simulated worker message envelope failed run/round validation")
    if headers["X-Sender-Node"] != _wire_id(source) or int(headers["X-Sample-Count"]) != sample_count:
        raise ValueError("Simulated worker message envelope failed sender/sample validation")
    if receiver not in NODE_IDS or source not in NODE_IDS:
        raise ValueError("Simulated worker message referenced an unknown node")
    return deserialize_weights(payload), len(payload), headers


def _edge_names(adjacency: np.ndarray) -> list[list[str]]:
    return [[NODE_IDS[receiver], NODE_IDS[source]] for receiver, source in adjacency_edges(adjacency)]


def _matrix_with_null_diagonal(matrix: np.ndarray, name: str) -> list[list[float | None]]:
    values = np.asarray(matrix, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ValueError(f"{name} must be a square matrix")
    off_diagonal = ~np.eye(values.shape[0], dtype=bool)
    if not np.isfinite(values[off_diagonal]).all():
        raise ValueError(f"{name} has non-finite off-diagonal values")
    output: list[list[float | None]] = values.tolist()
    for index in range(values.shape[0]):
        output[index][index] = None
    return output


def _check_run_deadline(
    deadline: float | None,
    run_id: str,
    should_stop: Callable[[], bool] | None = None,
) -> None:
    if should_stop is not None and should_stop():
        raise InterruptedError(f"Global STOP requested while running {run_id}")
    if deadline is not None and time.perf_counter() >= deadline:
        raise TimeoutError(
            f"3-hour full-matrix time cap reached while running {run_id}"
        )


def _run_one(
    *,
    simulation_mode: str,
    data_setting: str,
    topology: str,
    seed: int,
    nodes: list[dict[str, Any]],
    architecture: dict[str, Any],
    initial_weights: list[np.ndarray],
    apg_parameters: APGParameters,
    base_time_seconds: float,
    bandwidths: dict[str, float],
    round_count: int = ROUNDS,
    local_epochs: int = LOCAL_EPOCHS,
    deadline: float | None = None,
    should_stop: Callable[[], bool] | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
    run_id_override: str | None = None,
) -> dict[str, Any]:
    run_suffix = f"-smoke{round_count}" if round_count != ROUNDS else ""
    run_id = run_id_override or (
        f"apgb-{simulation_mode}-{data_setting}-{topology}-s{seed}{run_suffix}".replace("_", "-")
    )
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(seed)
    models: dict[str, tf.keras.Model] = {}
    for node_id in NODE_IDS:
        _check_run_deadline(deadline, run_id, should_stop)
        model = _build_model(architecture)
        model.set_weights([np.array(weight, copy=True) for weight in initial_weights])
        _compile_training_model(model)
        models[node_id] = model

    local = {node["node_id"]: node for node in nodes}
    if topology == "mesh":
        graph = mesh_adjacency(8)
    elif topology == "ring":
        graph = ring_adjacency(8)
    elif topology == "apg":
        graph = mesh_adjacency(8)
    else:
        raise ValueError(f"Unknown topology: {topology}")
    prior_graph = graph.copy()
    weights = uniform_mesh_weights(8)
    loss_latest = np.full((8, 8), np.nan, dtype=np.float64)
    wait_latest = np.full((8, 8), np.nan, dtype=np.float64)
    age_rounds = np.full((8, 8), -1, dtype=np.int64)
    round_records: list[dict[str, Any]] = []
    convergence_records: list[dict[str, Any]] = []
    start = time.perf_counter()

    for round_index in range(round_count):
        _check_run_deadline(deadline, run_id, should_stop)
        graph_used = graph.copy()
        edges_used = adjacency_edges(graph_used)
        edges_changed = None if round_index == 0 else count_edge_changes(prior_graph, graph_used)
        local_histories: dict[str, dict[str, float]] = {}
        local_weights: dict[str, list[np.ndarray]] = {}
        for node_id in NODE_IDS:
            _check_run_deadline(deadline, run_id, should_stop)
            item = local[node_id]
            history = models[node_id].fit(
                item["train_images"],
                item["train_labels"],
                epochs=local_epochs,
                batch_size=BATCH_SIZE,
                shuffle=True,
                verbose=0,
            )
            _check_run_deadline(deadline, run_id, should_stop)
            local_histories[node_id] = {
                "loss": float(history.history["loss"][-1]),
                "accuracy": float(history.history["accuracy"][-1]),
            }
            local_weights[node_id] = [np.asarray(value).copy() for value in models[node_id].get_weights()]

        payloads = {node_id: serialize_weights(values) for node_id, values in local_weights.items()}
        bytes_sent = {node_id: 0 for node_id in NODE_IDS}
        bytes_received = {node_id: 0 for node_id in NODE_IDS}
        received: dict[str, list[tuple[list[np.ndarray], int]]] = {node_id: [] for node_id in NODE_IDS}
        message_records: list[dict[str, Any]] = []
        for receiver_index, source_index in edges_used:
            receiver = NODE_IDS[receiver_index]
            source = NODE_IDS[source_index]
            updated_weights, payload_bytes, headers = _transport_update(
                simulation_mode,
                run_id,
                round_index,
                receiver,
                source,
                local_weights[source],
                local[source]["train_rows"],
            )
            received[receiver].append((updated_weights, local[source]["train_rows"]))
            bytes_sent[source] += payload_bytes
            bytes_received[receiver] += payload_bytes
            message_records.append(
                {
                    "receiver": receiver,
                    "source": source,
                    "bytes": payload_bytes,
                    "source_label": "SIMULATED",
                    "transport": "IN_PROCESS_NUMPY" if simulation_mode == "all_simulated" else "WORKER_NPZ_HTTP_ENVELOPE_EMULATED_IN_PROCESS",
                    "headers": headers,
                }
            )

        # Metrics are measured before aggregation for APG L_ij, after aggregation for node validation.
        apg_inputs: dict[str, Any] | None = None
        apg_result = None
        if topology == "apg":
            measured_edges = set(edges_used)
            for receiver_index in range(8):
                for source_index in range(8):
                    if receiver_index == source_index:
                        age_rounds[receiver_index, source_index] = -1
                    elif (receiver_index, source_index) in measured_edges:
                        receiver = NODE_IDS[receiver_index]
                        source = NODE_IDS[source_index]
                        loss_latest[receiver_index, source_index] = _cross_entropy(
                            models[source],
                            local[receiver]["validation_images"],
                            local[receiver]["validation_labels"],
                        )
                        payload_bytes = len(payloads[source])
                        receive_wait = max(
                            0.0,
                            (base_time_seconds / float(SIMULATED_PROFILE["nodes"][source]["compute_multiplier"]))
                            - (base_time_seconds / float(SIMULATED_PROFILE["nodes"][receiver]["compute_multiplier"])),
                        )
                        link_wait = payload_bytes / min(bandwidths[receiver], bandwidths[source])
                        wait_latest[receiver_index, source_index] = receive_wait + link_wait
                        age_rounds[receiver_index, source_index] = 0
                    elif age_rounds[receiver_index, source_index] >= 0:
                        age_rounds[receiver_index, source_index] += 1
            if not np.isfinite(loss_latest[~np.eye(8, dtype=bool)]).all():
                raise RuntimeError("APG inputs are incomplete; round 1 Mesh must measure every pair")
            if not np.isfinite(wait_latest[~np.eye(8, dtype=bool)]).all():
                raise RuntimeError("APG wait-time inputs are incomplete")
            apg_result = optimize_topology(
                loss_latest,
                wait_latest,
                np.asarray([local[node_id]["theta"] for node_id in NODE_IDS]),
                weights,
                graph_used,
                apg_parameters,
            )
            apg_inputs = {
                "source_label": "SIMULATED",
                "loss_matrix_raw": _matrix_with_null_diagonal(loss_latest, "loss_matrix_raw"),
                "wait_matrix_simulated_seconds": _matrix_with_null_diagonal(
                    wait_latest, "wait_matrix_simulated_seconds"
                ),
                "loss_matrix_normalized": _matrix_with_null_diagonal(
                    normalize_off_diagonal(loss_latest), "loss_matrix_normalized"
                ),
                "wait_matrix_normalized": _matrix_with_null_diagonal(
                    normalize_off_diagonal(wait_latest), "wait_matrix_normalized"
                ),
                "measurement_age_rounds": age_rounds.tolist(),
                "theta_train_only": [local[node_id]["theta"].tolist() for node_id in NODE_IDS],
            }

        for receiver in NODE_IDS:
            sources = [source for source, _ in edges_used if NODE_IDS[source] == receiver]
            contributors = [(local_weights[receiver], local[receiver]["train_rows"])]
            contributors.extend(received[receiver])
            models[receiver].set_weights(_weighted_average(contributors))

        node_metrics: dict[str, Any] = {}
        for node_id in NODE_IDS:
            _check_run_deadline(deadline, run_id, should_stop)
            item = local[node_id]
            metrics = _evaluate_model(
                models[node_id], item["validation_images"], item["validation_labels"]
            )
            node_metrics[node_id] = {
                "node_role": SIMULATED_PROFILE["nodes"][node_id]["role"],
                "emulated_worker_id": SIMULATED_PROFILE["nodes"][node_id].get("worker_id"),
                "source_label": "SIMULATED",
                "shard_rows": item["shard_rows"],
                "train_rows": item["train_rows"],
                "validation_rows": item["validation_rows"],
                "shard_class_counts": item["shard_class_counts"],
                "train_class_counts": item["train_class_counts"],
                "validation_class_counts": item["validation_class_counts"],
                "train_accuracy": local_histories[node_id]["accuracy"],
                "train_loss": local_histories[node_id]["loss"],
                "validation_accuracy": metrics["accuracy"],
                "validation_macro_f1": metrics["macro_f1"],
                "validation_loss": metrics["loss"],
                "bytes_sent_simulated": int(bytes_sent[node_id]),
                "bytes_received_simulated": int(bytes_received[node_id]),
            }

        next_graph = apg_result.adjacency if apg_result is not None else graph_used
        selected_edges_next = _edge_names(next_graph)
        apg_delta = count_edge_changes(graph_used, next_graph) if apg_result is not None else 0
        objective_change = (
            abs(apg_result.objective_history[-1] - apg_result.objective_history[-2])
            if apg_result is not None and len(apg_result.objective_history) > 1
            else None
        )
        stop_criterion_met = (
            objective_change < apg_parameters.psi if objective_change is not None else None
        )
        round_record = {
            "round": round_index + 1,
            "split": "VALIDATION",
            "source_label": "SIMULATED",
            "topology": topology,
            "edge_semantics": "receiver selects source; payload source -> receiver",
            "edges_used": _edge_names(graph_used),
            "edge_count_used": int(graph_used.sum()),
            "edges_changed_vs_previous_round": edges_changed,
            "apg_selected_edges_for_next_round": selected_edges_next if apg_result is not None else None,
            "apg_edges_changed_for_next_round": apg_delta if apg_result is not None else None,
            "apg_edge_count_for_next_round": int(next_graph.sum()) if apg_result is not None else None,
            "g": apg_result.objective if apg_result is not None else None,
            "iterations_used": apg_result.iterations if apg_result is not None else 0,
            "safeguard_hits": apg_result.safeguard_hits if apg_result is not None else 0,
            "stop_tolerance_psi": apg_parameters.psi if apg_result is not None else None,
            "stop_abs_objective_change": objective_change,
            "stopped_by_objective_tolerance": stop_criterion_met,
            "mean_validation_accuracy": float(
                np.mean([item["validation_accuracy"] for item in node_metrics.values()])
            ),
            "mean_validation_macro_f1": float(
                np.mean([item["validation_macro_f1"] for item in node_metrics.values()])
            ),
            "objective_history": list(apg_result.objective_history) if apg_result is not None else None,
            "apg_weights": apg_result.weights.tolist() if apg_result is not None else None,
            "apg_inputs": apg_inputs,
            "bytes_sent_simulated": int(sum(bytes_sent.values())),
            "bytes_received_simulated": int(sum(bytes_received.values())),
            "per_node": node_metrics,
            "messages": message_records,
        }
        round_records.append(round_record)
        if progress_callback is not None:
            progress_callback(round_index + 1, round_count)
        for node_id in NODE_IDS:
            item = node_metrics[node_id]
            for split, accuracy, macro_f1, loss in (
                ("TRAIN", item["train_accuracy"], None, item["train_loss"]),
                ("VALIDATION", item["validation_accuracy"], item["validation_macro_f1"], item["validation_loss"]),
            ):
                convergence_records.append(
                    {
                        "run_id": run_id,
                        "simulation_mode": simulation_mode,
                        "data_setting": data_setting,
                        "topology": topology,
                        "seed": seed,
                        "round": round_index + 1,
                        "split": split,
                        "source": "SIMULATED",
                        "node_id": node_id,
                        "accuracy": accuracy,
                        "macro_f1": macro_f1,
                        "loss": loss,
                        "train_rows": item["train_rows"],
                        "validation_rows": item["validation_rows"],
                        "bytes_sent_simulated": item["bytes_sent_simulated"],
                        "bytes_received_simulated": item["bytes_received_simulated"],
                        "edge_list_json": json.dumps(round_record["edges_used"], separators=(",", ":")),
                        "edge_count": round_record["edge_count_used"],
                        "edges_changed_vs_previous_round": edges_changed,
                        "g": round_record["g"],
                        "iterations_used": round_record["iterations_used"],
                        "safeguard_hits": round_record["safeguard_hits"],
                        "stop_abs_objective_change": round_record["stop_abs_objective_change"],
                    }
                )
        for split, accuracy_key, f1_key, loss_key in (
            ("TRAIN", "train_accuracy", None, "train_loss"),
            ("VALIDATION", "validation_accuracy", "validation_macro_f1", "validation_loss"),
        ):
            convergence_records.append(
                {
                    "run_id": run_id,
                    "simulation_mode": simulation_mode,
                    "data_setting": data_setting,
                    "topology": topology,
                    "seed": seed,
                    "round": round_index + 1,
                    "split": split,
                    "source": "SIMULATED",
                    "node_id": "mean",
                    "accuracy": float(np.mean([item[accuracy_key] for item in node_metrics.values()])),
                    "macro_f1": float(np.mean([item[f1_key] for item in node_metrics.values()])) if f1_key else None,
                    "loss": float(np.mean([item[loss_key] for item in node_metrics.values()])),
                    "train_rows": TRAINING_ROWS_PER_NODE,
                    "validation_rows": VALIDATION_ROWS_PER_NODE,
                    "bytes_sent_simulated": round_record["bytes_sent_simulated"],
                    "bytes_received_simulated": round_record["bytes_received_simulated"],
                    "edge_list_json": json.dumps(round_record["edges_used"], separators=(",", ":")),
                    "edge_count": round_record["edge_count_used"],
                    "edges_changed_vs_previous_round": edges_changed,
                    "g": round_record["g"],
                    "iterations_used": round_record["iterations_used"],
                    "safeguard_hits": round_record["safeguard_hits"],
                    "stop_abs_objective_change": round_record["stop_abs_objective_change"],
                }
            )

        if apg_result is not None:
            weights = apg_result.weights.copy()
            prior_graph = graph_used.copy()
            graph = next_graph.copy()
        if (round_index + 1) % 5 == 0:
            mean_val = float(np.mean([item["validation_accuracy"] for item in node_metrics.values()]))
            print(
                f"mode={simulation_mode}; data={data_setting}; topology={topology}; seed={seed}; "
                f"round={round_index + 1}/{round_count}; validation_mean_accuracy={mean_val:.6f}; "
                f"source=SIMULATED",
                flush=True,
            )

    model_weights = {
        node_id: [np.asarray(value).copy() for value in models[node_id].get_weights()]
        for node_id in NODE_IDS
    }
    _check_run_deadline(deadline, run_id, should_stop)
    pc_elapsed = time.perf_counter() - start
    print(f"pc_simulation_wall_seconds={pc_elapsed:.3f}", flush=True)
    del models
    tf.keras.backend.clear_session()
    return {
        "run_id": run_id,
        "simulation_mode": simulation_mode,
        "source_label": "SIMULATED",
        "data_setting": data_setting,
        "topology": topology,
        "seed": seed,
        "split": "VALIDATION",
        "rounds": round_records,
        "convergence_records": convergence_records,
        "final_model_weights": model_weights,
        "pc_simulation_wall_seconds": pc_elapsed,
        "pc_wall_time_label": "PC simulation wall time; not hardware performance",
        "nodes": [
            {
                key: (value.tolist() if isinstance(value, np.ndarray) else value)
                for key, value in node.items()
                if key not in {"images", "labels", "train_images", "train_labels", "validation_images", "validation_labels"}
            }
            for node in nodes
        ],
    }


def _evaluate_test_once(
    runs: list[dict[str, Any]],
    architecture: dict[str, Any],
    test_path: Path,
    deadline: float | None = None,
) -> dict[str, Any]:
    # This is the only read of test.npz in the Phase B driver.
    test_bytes = test_path.read_bytes()
    test_hash = hashlib.sha256(test_bytes).hexdigest()
    with np.load(io.BytesIO(test_bytes), allow_pickle=False) as data:
        test_images = np.asarray(data["images"], dtype=np.float32)
        test_labels = np.asarray(data["labels"], dtype=np.int64).reshape(-1)
    if test_images.shape != (300, 28, 28, 1) or test_labels.shape != (300,):
        raise ValueError(f"Unexpected MNIST TEST shape: {test_images.shape}, {test_labels.shape}")

    for run in runs:
        _check_run_deadline(deadline, f"TEST evaluation for {run['run_id']}")
        node_predictions: dict[str, Any] = {}
        total_confusion = np.zeros((10, 10), dtype=np.int64)
        for node_id in NODE_IDS:
            _check_run_deadline(deadline, f"TEST evaluation for {run['run_id']}")
            model = _build_model(architecture)
            model.set_weights(run["final_model_weights"][node_id])
            logits = np.asarray(model.predict(test_images, batch_size=BATCH_SIZE, verbose=0))
            metrics = _metrics_from_logits(test_labels, logits)
            total_confusion += np.asarray(metrics["confusion_matrix_counts"], dtype=np.int64)
            node_predictions[node_id] = {
                "source": "SIMULATED",
                "split": "TEST",
                "metrics": {key: value for key, value in metrics.items() if key not in {"y_true", "y_pred", "confusion_matrix_counts"}},
                "y_true": test_labels.tolist(),
                "y_pred": metrics["y_pred"],
            }
            del model
        pooled = _metrics_from_confusion(total_confusion)
        run["final_test"] = {
            "split": "TEST",
            "source": "SIMULATED",
            "scope": "pooled predictions across eight simulated node models; same 300 TEST images are evaluated by each node model",
            "test_rows_per_node_model": int(test_labels.size),
            "pooled_prediction_count": int(test_labels.size * len(NODE_IDS)),
            "test_npz_sha256": test_hash,
            "per_node": node_predictions,
            "summary": pooled,
            "confusion_matrix_counts_pooled": total_confusion.tolist(),
            "confusion_matrix_row_normalized_percent": _normalize_confusion(total_confusion).tolist(),
            "y_true": test_labels.tolist(),
            "y_pred_by_node": {node_id: node_predictions[node_id]["y_pred"] for node_id in NODE_IDS},
        }
        saved_prediction_confusion = _confusion_from_saved_predictions(run)
        if not np.array_equal(saved_prediction_confusion, total_confusion):
            raise RuntimeError(f"Saved TEST predictions disagree with confusion counts for {run['run_id']}")
        run.pop("final_model_weights")
    return {
        "test_rows": int(test_labels.size),
        "class_support_per_test_split": [int(value) for value in np.bincount(test_labels, minlength=10)],
        "test_npz_sha256": test_hash,
        "loaded_once_after_all_training": True,
    }


def _metrics_from_confusion(confusion: np.ndarray) -> dict[str, Any]:
    matrix = np.asarray(confusion, dtype=np.int64)
    support = matrix.sum(axis=1)
    predicted = matrix.sum(axis=0)
    true_positive = np.diag(matrix).astype(np.float64)
    precision = np.divide(true_positive, predicted, out=np.zeros(10), where=predicted > 0)
    recall = np.divide(true_positive, support, out=np.zeros(10), where=support > 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros(10), where=(precision + recall) > 0)
    total = int(matrix.sum())
    return {
        "accuracy": float(true_positive.sum() / total) if total else 0.0,
        "macro_precision": float(precision.mean()),
        "macro_recall": float(recall.mean()),
        "macro_f1": float(f1.mean()),
        "per_class": [
            {
                "class": str(index),
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(support[index]),
            }
            for index in range(10)
        ],
    }


def _confusion_from_saved_predictions(run: dict[str, Any]) -> np.ndarray:
    test_result = run["final_test"]
    true = np.asarray(test_result["y_true"], dtype=np.int64)
    confusion = np.zeros((10, 10), dtype=np.int64)
    for node_id in NODE_IDS:
        predicted = np.asarray(test_result["y_pred_by_node"][node_id], dtype=np.int64)
        if predicted.shape != true.shape:
            raise ValueError(f"Saved TEST predictions have an unexpected shape for {run['run_id']}")
        np.add.at(confusion, (true, predicted), 1)
    return confusion


def _write_test_confusion_csv(runs: list[dict[str, Any]], path: Path) -> None:
    fields = (
        "run_id", "simulation_mode", "data_setting", "topology", "seed",
        "true_class", "predicted_class", "count",
    )
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        for run in runs:
            confusion = _confusion_from_saved_predictions(run)
            for true_class in range(10):
                for predicted_class in range(10):
                    writer.writerow(
                        {
                            "run_id": run["run_id"],
                            "simulation_mode": run["simulation_mode"],
                            "data_setting": run["data_setting"],
                            "topology": run["topology"],
                            "seed": run["seed"],
                            "true_class": true_class,
                            "predicted_class": predicted_class,
                            "count": int(confusion[true_class, predicted_class]),
                        }
                    )


def _normalize_confusion(confusion: np.ndarray) -> np.ndarray:
    matrix = np.asarray(confusion, dtype=np.float64)
    row_sums = matrix.sum(axis=1, keepdims=True)
    return np.divide(matrix * 100.0, row_sums, out=np.zeros_like(matrix), where=row_sums > 0)


def _mean_sd(values: list[float]) -> dict[str, float]:
    return {"mean": float(np.mean(values)), "sample_sd_ddof_1": float(np.std(values, ddof=1))}


def _build_summary(runs: list[dict[str, Any]], test_info: dict[str, Any]) -> dict[str, Any]:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for run in runs:
        groups.setdefault((run["simulation_mode"], run["data_setting"], run["topology"]), []).append(run)
    summaries = []
    for (mode, setting, topology), group in sorted(groups.items()):
        group.sort(key=lambda item: int(item["seed"]))
        metrics = {
            name: _mean_sd([float(run["final_test"]["summary"][name]) for run in group])
            for name in ("accuracy", "macro_precision", "macro_recall", "macro_f1")
        }
        summaries.append(
            {
                "simulation_mode": mode,
                "source_label": "SIMULATED",
                "data_setting": setting,
                "topology": topology,
                "split": "TEST",
                "seed_count": len(group),
                "metrics": metrics,
                "per_seed": [
                    {
                        "seed": run["seed"],
                        **run["final_test"]["summary"],
                    }
                    for run in group
                ],
            }
        )

    verdicts = []
    for mode in SIMULATION_MODES:
        for setting in DATA_SETTINGS:
            by_topology = {
                topology: {
                    int(run["seed"]): run
                    for run in runs
                    if run["simulation_mode"] == mode
                    and run["data_setting"] == setting
                    and run["topology"] == topology
                }
                for topology in TOPOLOGIES
            }
            apg_runs = by_topology["apg"]
            apg_adapts = any(
                round_info.get("apg_edges_changed_for_next_round", 0) > 0
                for run in apg_runs.values()
                for round_info in run["rounds"]
            )
            comparator_results = {}
            demonstrated = apg_adapts
            for comparator in ("ring", "mesh"):
                deltas = [
                    float(apg_runs[seed]["final_test"]["summary"]["accuracy"])
                    - float(by_topology[comparator][seed]["final_test"]["summary"]["accuracy"])
                    for seed in TRAINING_SEEDS
                ]
                delta_mean = float(np.mean(deltas))
                delta_sd = float(np.std(deltas, ddof=1))
                comparator_results[comparator] = {
                    "paired_accuracy_deltas": deltas,
                    "mean_delta": delta_mean,
                    "sample_sd_delta_ddof_1": delta_sd,
                    "benefit_beyond_seed_noise": delta_mean > delta_sd,
                }
                demonstrated = demonstrated and (delta_mean > delta_sd)
            verdicts.append(
                {
                    "simulation_mode": mode,
                    "source_label": "SIMULATED",
                    "data_setting": setting,
                    "split": "TEST",
                    "apg_topology_adapts": apg_adapts,
                    "comparisons": comparator_results,
                    "verdict": "benefit demonstrated beyond paired seed SD" if demonstrated else "no demonstrated benefit",
                    "test_split": test_info,
                }
            )
    return {"summaries": summaries, "apg_benefit_checks": verdicts}


def _validation_convergence_summary(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, int], list[tuple[int, float, float]]] = {}
    for run in runs:
        for round_info in run["rounds"]:
            per_node = list(round_info["per_node"].values())
            accuracy = float(np.mean([item["validation_accuracy"] for item in per_node]))
            macro_f1 = float(np.mean([item["validation_macro_f1"] for item in per_node]))
            key = (
                run["simulation_mode"],
                run["data_setting"],
                run["topology"],
                int(round_info["round"]),
            )
            grouped.setdefault(key, []).append((int(run["seed"]), accuracy, macro_f1))

    summary = []
    for (mode, setting, topology, round_number), seed_metrics in sorted(grouped.items()):
        seed_metrics.sort(key=lambda item: item[0])
        summary.append(
            {
                "simulation_mode": mode,
                "data_setting": setting,
                "topology": topology,
                "round": round_number,
                "seed_count": len(seed_metrics),
                "accuracy_mean": _mean_sd([item[1] for item in seed_metrics])["mean"],
                "accuracy_sample_sd_ddof_1": _mean_sd([item[1] for item in seed_metrics])[
                    "sample_sd_ddof_1"
                ],
                "macro_f1_mean": _mean_sd([item[2] for item in seed_metrics])["mean"],
                "macro_f1_sample_sd_ddof_1": _mean_sd([item[2] for item in seed_metrics])[
                    "sample_sd_ddof_1"
                ],
            }
        )
    return summary


def _write_validation_convergence_csv(
    runs: list[dict[str, Any]],
    path: Path,
) -> None:
    fields = (
        "simulation_mode", "data_setting", "topology", "round", "seed_count",
        "accuracy_mean", "accuracy_sample_sd_ddof_1",
        "macro_f1_mean", "macro_f1_sample_sd_ddof_1",
    )
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows(_validation_convergence_summary(runs))


def _write_apg_round_diagnostics(runs: list[dict[str, Any]], path: Path) -> None:
    fields = (
        "run_id", "simulation_mode", "data_setting", "seed", "round", "node_id",
        "edges_changed_vs_previous_round", "edge_count_used", "in_degree",
        "out_degree", "g", "iterations_used", "safeguard_hits",
        "stop_tolerance_psi", "stop_abs_objective_change",
        "stopped_by_objective_tolerance", "bytes_sent_simulated",
        "bytes_received_simulated",
    )
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        for run in runs:
            if run["topology"] != "apg":
                continue
            for round_info in run["rounds"]:
                in_degree = {node_id: 0 for node_id in NODE_IDS}
                out_degree = {node_id: 0 for node_id in NODE_IDS}
                for receiver, source in round_info["edges_used"]:
                    in_degree[receiver] += 1
                    out_degree[source] += 1
                for node_id in NODE_IDS:
                    writer.writerow(
                        {
                            "run_id": run["run_id"],
                            "simulation_mode": run["simulation_mode"],
                            "data_setting": run["data_setting"],
                            "seed": run["seed"],
                            "round": round_info["round"],
                            "node_id": node_id,
                            "edges_changed_vs_previous_round": round_info[
                                "edges_changed_vs_previous_round"
                            ],
                            "edge_count_used": round_info["edge_count_used"],
                            "in_degree": in_degree[node_id],
                            "out_degree": out_degree[node_id],
                            "g": round_info["g"],
                            "iterations_used": round_info["iterations_used"],
                            "safeguard_hits": round_info["safeguard_hits"],
                            "stop_tolerance_psi": round_info["stop_tolerance_psi"],
                            "stop_abs_objective_change": round_info[
                                "stop_abs_objective_change"
                            ],
                            "stopped_by_objective_tolerance": round_info[
                                "stopped_by_objective_tolerance"
                            ],
                            "bytes_sent_simulated": round_info["bytes_sent_simulated"],
                            "bytes_received_simulated": round_info["bytes_received_simulated"],
                        }
                    )


def _write_convergence_csv(
    records: list[dict[str, Any]],
    path: Path = CONVERGENCE_CSV,
) -> None:
    fields = (
        "run_id", "simulation_mode", "source", "data_setting", "topology", "seed",
        "round", "split", "node_id", "accuracy", "macro_f1", "loss",
        "train_rows", "validation_rows", "bytes_sent_simulated", "bytes_received_simulated",
        "edge_list_json", "edge_count", "edges_changed_vs_previous_round",
        "g", "iterations_used", "safeguard_hits", "stop_abs_objective_change",
    )
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        for row in records:
            writer.writerow(row)


def _write_smoke_json(run: dict[str, Any], path: Path) -> None:
    path.write_text(
        json.dumps(run, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _plot_convergence(runs: list[dict[str, Any]]) -> list[str]:
    created = []
    for mode in SIMULATION_MODES:
        for setting in DATA_SETTINGS:
            for topology in TOPOLOGIES:
                matching = [run for run in runs if run["simulation_mode"] == mode and run["data_setting"] == setting and run["topology"] == topology]
                validation_by_seed = {}
                for run in matching:
                    validation_by_seed[int(run["seed"])] = [
                        float(round_info["mean_validation_accuracy"])
                        for round_info in run["rounds"]
                    ]
                matrix = np.asarray([validation_by_seed[seed] for seed in TRAINING_SEEDS], dtype=np.float64)
                mean = matrix.mean(axis=0)
                lower = matrix.min(axis=0)
                upper = matrix.max(axis=0)
                figure, axis = plt.subplots(figsize=(8, 4.5), facecolor="white")
                rounds = np.arange(1, ROUNDS + 1)
                axis.plot(rounds, mean * 100, marker="o", color="#176b87", label="Mean across 3 seeds")
                axis.fill_between(rounds, lower * 100, upper * 100, color="#176b87", alpha=0.18, label="Seed min-max")
                axis.set(title=f"{setting} · {topology} · {mode} · VALIDATION · SIMULATED", xlabel="Round", ylabel="Validation accuracy (%)")
                axis.set_xticks(range(1, ROUNDS + 1, 3))
                axis.set_ylim(0, 100)
                axis.grid(alpha=0.2)
                axis.legend(frameon=False)
                figure.tight_layout()
                name = f"convergence_{mode}_{setting}_{topology}.png"
                figure.savefig(EXPORT_ROOT / name, dpi=150, facecolor="white")
                plt.close(figure)
                created.append(name)
    return created


def _plot_test_bars(summaries: dict[str, Any]) -> list[str]:
    created = []
    for mode in SIMULATION_MODES:
        for setting in DATA_SETTINGS:
            subset = [item for item in summaries if item["simulation_mode"] == mode and item["data_setting"] == setting]
            means = [item["metrics"]["accuracy"]["mean"] * 100 for item in subset]
            deviations = [item["metrics"]["accuracy"]["sample_sd_ddof_1"] * 100 for item in subset]
            figure, axis = plt.subplots(figsize=(7, 4.5), facecolor="white")
            positions = np.arange(len(subset))
            axis.bar(positions, means, yerr=deviations, capsize=5, color=["#4986a8", "#62a37c", "#bb8056"])
            axis.set_xticks(positions, [item["topology"] for item in subset])
            axis.set_ylim(0, 100)
            axis.set_ylabel("TEST accuracy (%)")
            axis.set_title(f"{setting} · {mode} · TEST · SIMULATED (mean +/- sample SD, n=3)")
            axis.grid(axis="y", alpha=0.2)
            figure.tight_layout()
            name = f"final_test_{mode}_{setting}.png"
            figure.savefig(EXPORT_ROOT / name, dpi=150, facecolor="white")
            plt.close(figure)
            created.append(name)
    return created


def _plot_confusions(runs: list[dict[str, Any]]) -> list[str]:
    created = []
    for mode in SIMULATION_MODES:
        for setting in DATA_SETTINGS:
            for topology in TOPOLOGIES:
                matching = [run for run in runs if run["simulation_mode"] == mode and run["data_setting"] == setting and run["topology"] == topology]
                summed = np.sum(
                    [_confusion_from_saved_predictions(run) for run in matching],
                    axis=0,
                )
                normalized = _normalize_confusion(summed)
                figure, axes = plt.subplots(1, 2, figsize=(12, 5), facecolor="white")
                for axis, matrix, title, fmt in (
                    (axes[0], summed, "Counts (seed-summed)", "d"),
                    (axes[1], normalized, "Row-normalized (%)", ".1f"),
                ):
                    image = axis.imshow(matrix, cmap="Blues")
                    axis.set(title=title, xlabel="Predicted class", ylabel="True class")
                    axis.set_xticks(range(10), CLASS_NAMES, rotation=45, ha="right")
                    axis.set_yticks(range(10), CLASS_NAMES)
                    threshold = float(np.max(matrix)) / 2.0 if matrix.size else 0.0
                    for row_index in range(10):
                        for col_index in range(10):
                            value = matrix[row_index, col_index]
                            axis.text(col_index, row_index, format(value, fmt), ha="center", va="center", color="white" if value > threshold else "black", fontsize=7)
                    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
                figure.suptitle(f"MNIST {setting} {topology} · {mode} · TEST · SIMULATED · seed-summed")
                figure.tight_layout()
                name = f"cm_MNIST_{mode}_{setting}_{topology}_TEST.png"
                figure.savefig(EXPORT_ROOT / name, dpi=150, facecolor="white")
                plt.close(figure)
                created.append(name)
    return created


def _write_phase_report(
    runs: list[dict[str, Any]],
    summaries: dict[str, Any],
    test_info: dict[str, Any],
    data_summary: dict[str, Any],
    timing_config: dict[str, Any],
) -> None:
    run_lines = []
    for item in summaries["summaries"]:
        run_lines.append(
            f"| {item['simulation_mode']} | {item['data_setting']} | {item['topology']} | TEST | "
            + " | ".join(
                f"{item['metrics'][name]['mean']:.6f} +/- "
                f"{item['metrics'][name]['sample_sd_ddof_1']:.6f}"
                for name in ("accuracy", "macro_f1", "macro_precision", "macro_recall")
            )
            + " | SIMULATED |"
        )
    validation_lines = []
    for row in _validation_convergence_summary(runs):
        validation_lines.append(
            f"| {row['simulation_mode']} | {row['data_setting']} | {row['topology']} | "
            f"{row['round']} | {row['accuracy_mean']:.6f} +/- "
            f"{row['accuracy_sample_sd_ddof_1']:.6f} | {row['macro_f1_mean']:.6f} +/- "
            f"{row['macro_f1_sample_sd_ddof_1']:.6f} |"
        )
    report = f"""# APG Phase B PC Simulation Report

Status: completed. No Jetson or master endpoint was contacted. No hardware training, deployment, mode change, or commit was performed. Every run and metric below is SIMULATED.

## Protocol

- Eight nodes; five simulated clients plus three PC-emulated worker-format nodes.
- Config: MNIST initial model with {len(TRAINING_SEEDS)} seeds {list(TRAINING_SEEDS)}, 30 rounds x 3 local epochs, Adam lr 0.01, batch 32.
- Two settings, reported separately: IID stratified and constrained Dirichlet alpha=0.1 on TRAIN only. Sharding seed 2026; exactly 300 rows/node, then 240 TRAIN and 60 local VALIDATION rows.
- TEST was loaded once after all 36 training runs. Each run saved per-node y_true/y_pred; confusion matrices pool eight node-model evaluations per seed. The same 300 TEST images are evaluated by each node model.
- Simulated profile source: `results_apg/SYNTHETIC_PROFILES.json`; all time and bytes fields are SYNTHETIC, including the hybrid-shaped simulation.

## Realized shards and Theta

```json
{json.dumps(data_summary, indent=2, sort_keys=True)}
```

## VALIDATION convergence by round

Per-round metrics are the mean across eight node models for each seed; displayed means and sample SDs (ddof=1) are across the three seeds. This table includes all topologies, and separates each data setting and simulation mode.

| Simulation mode | Data setting | Topology | Round | Accuracy mean +/- sample SD | Macro-F1 mean +/- sample SD |
|---|---|---|---:|---:|---:|
{chr(10).join(validation_lines)}

The full per-node TRAIN/VALIDATION rows are in `convergence.csv`; the compact three-seed per-round table is in `validation_convergence_summary.csv`.

## TEST summary

| Simulation mode | Data setting | Topology | Split | Accuracy mean +/- sample SD | Macro-F1 mean +/- sample SD | Macro precision mean +/- sample SD | Macro recall mean +/- sample SD | Source |
|---|---|---|---|---:|---:|---:|---:|---|
{chr(10).join(run_lines)}

Confusion matrices are reconstructed from the saved TEST predictions and checked against the run metrics. Per-run predictions and confusion counts are in `final_metrics.json`; the class-by-class counts are in `test_confusion_matrices.csv`. Accuracy is not used as a substitute for macro precision/recall/F1.

## APG round diagnostics

`apg_round_diagnostics.csv` reports edge changes, edge counts, receiver-centric in/out degrees for every node, g, iterations, safeguards, the absolute final objective change and psi stop result, and simulated communication bytes for each APG seed and round. `edges_changed_vs_previous_round` is blank for round 1 because it has no prior round.

The specified APG tolerance remains psi=0.01. The absolute objective-change stop criterion is `abs(g_k - g_(k-1)) < psi`; psi=0.01 stops APG after very few iterations in the smoke and full-matrix runs. No APG default was changed. APG's T timing input is synthetic and is not hardware performance.

## APG input and timing provenance

- Per-round L_ij is the neighbor model's cross-entropy loss on receiver i's local VALIDATION shard. Theta is computed from local TRAIN labels only.
- T_ij is SIMULATED using `max(0, train_time_j - train_time_i) + payload_bytes / min(bw_i, bw_j)`.
- `base_time_seconds={timing_config['base_time_seconds']}` is a SYNTHETIC reference derived from 41.04 seconds / 5 rounds of the previous hardware 5x20 run. The old run's local epoch count differs from this simulation; this is not a prediction of current PC or Jetson runtime.
- Effective bandwidth uses {timing_config['bandwidth_numerator_bytes']} aggregate bytes / {timing_config['bandwidth_denominator_exchange_seconds']} aggregate exchange seconds from Phase 5 Mesh alpha=0.5. It is an aggregate-derived SYNTHETIC proxy, not per-edge throughput.
- The receiver-centric edge mapping is `A[i,j]=1` means receiver i selects/receives source j; W is not an aggregation weight. PSO-GNN is NOT AVAILABLE/not implemented.

## Result decision

APG-vs-fixed comparisons use paired seed TEST accuracy differences. A benefit is called demonstrated only if APG's mean paired delta exceeds the paired sample SD versus both Ring and Mesh. Otherwise the result is reported as "no demonstrated benefit". These are three-seed descriptive results, not significance tests.

```json
{json.dumps(summaries['apg_benefit_checks'], indent=2, sort_keys=True)}
```

## Limitations

- Nodes 0-7 were all simulated on the PC. The hybrid-shaped run only emulates the worker NPZ weight payload and `X-Run-ID`, `X-Round`, `X-Sender-Node`, `X-Sample-Count` headers in process. No numbers are hardware results.
- The 300-image MNIST TEST set is small (one image is about 0.333333 percentage points). There are three seeds only.
- Real per-edge T_ij, real hardware RAM/swap, and hybrid physical results are NOT AVAILABLE in Phase B.
- Thesis training is 100 x 3; this experiment uses 30 x 3 as specified, a documented deviation.
- Source TEST split SHA-256: `{test_info['test_npz_sha256']}`; TEST rows: {test_info['test_rows']}.
"""
    REPORT_PATH.write_text(report, encoding="utf-8")


def _run_smoke(psi: float = 0.01) -> None:
    if psi == 0.01:
        output_files = SMOKE_OUTPUT_FILES
    elif psi == 0.001:
        output_tag = "_psi0001"
        output_files = tuple(
            name.replace(".json", f"{output_tag}.json").replace(
                "_convergence.csv", f"{output_tag}_convergence.csv"
            )
            for name in SMOKE_OUTPUT_FILES
        )
    elif psi == 1e-6:
        output_tag = "_psi000001"
        output_files = tuple(
            name.replace(".json", f"{output_tag}.json").replace(
                "_convergence.csv", f"{output_tag}_convergence.csv"
            )
            for name in SMOKE_OUTPUT_FILES
        )
    else:
        raise ValueError(f"Unsupported smoke-only APG psi sensitivity: {psi}")
    existing = [name for name in output_files if (EXPORT_ROOT / name).exists()]
    if existing:
        raise FileExistsError(f"Refusing to overwrite existing smoke outputs: {existing}")
    global SIMULATED_PROFILE
    SIMULATED_PROFILE = _read_json(PROFILE_PATH)
    report_path, _ = _latest_training_run()
    train_images, train_labels, train_hash = _load_train_only(report_path)
    architecture, initial_weights, model_hashes = _load_initial_model()
    seed_splits, shard_records = _make_data_setting(train_images, train_labels, "iid_stratified")
    selected_seed = next(item for item in seed_splits if item["seed"] == 17)
    profile, _, bandwidths = _profiles()
    base_time = float(profile["reference_measurements"]["base_round_time"]["value_seconds"])
    apg_parameters = APGParameters(psi=psi)
    started = time.perf_counter()
    smoke_run = _run_one(
        simulation_mode="all_simulated",
        data_setting="iid_stratified",
        topology="apg",
        seed=17,
        nodes=selected_seed["nodes"],
        architecture=architecture,
        initial_weights=initial_weights,
        apg_parameters=apg_parameters,
        base_time_seconds=base_time,
        bandwidths=bandwidths,
        round_count=3,
    )
    measured_seconds = time.perf_counter() - started
    print(f"smoke_wall_seconds={measured_seconds:.3f}", flush=True)
    smoke_run["run_wall_seconds"] = measured_seconds
    convergence_records = smoke_run.pop("convergence_records")
    smoke_run.pop("final_model_weights")
    smoke_run["rounds_requested"] = 3
    smoke_run["test_split_opened"] = False
    smoke_run["source_label"] = "SIMULATED"
    smoke_run["train_npz_sha256"] = train_hash
    smoke_run["initial_model"] = model_hashes
    smoke_run["shard_class_counts"] = {
        node["node_id"]: node["shard_class_counts"] for node in shard_records
    }
    full_matrix_runs = (
        len(SIMULATION_MODES) * len(DATA_SETTINGS) * len(TOPOLOGIES) * len(TRAINING_SEEDS)
    )
    estimated_full_seconds = measured_seconds * full_matrix_runs * ROUNDS / 3.0
    smoke_run["full_matrix_estimate"] = {
        "seconds": estimated_full_seconds,
        "hours": estimated_full_seconds / 3600.0,
        "formula": "smoke wall seconds x (36 runs x 30 rounds) / (1 run x 3 rounds)",
        "label": "ESTIMATE from one smoke run; assumes linear scaling and excludes final TEST/plot overhead",
    }
    smoke_run["split"] = "VALIDATION"
    smoke_run["test_note"] = "TEST was not read in smoke mode."

    _write_smoke_json(smoke_run, EXPORT_ROOT / output_files[0])
    _write_convergence_csv(convergence_records, EXPORT_ROOT / output_files[1])
    print(f"smoke_psi={psi}")
    print(f"smoke_run_id={smoke_run['run_id']}; source=SIMULATED; split=VALIDATION; rounds=3")
    print("full_matrix_time_estimate=" + json.dumps(smoke_run["full_matrix_estimate"], sort_keys=True))
    print(f"test_split_opened={smoke_run['test_split_opened']}")
    for name in output_files:
        print(f"created=results_apg/{name}")


def _run_full_matrix() -> None:
    matrix_started = time.perf_counter()
    deadline = matrix_started + MATRIX_TIME_CAP_SECONDS
    EXPORT_ROOT.mkdir(exist_ok=True)
    expected_outputs = set(OUTPUT_FILES)
    existing = sorted(name for name in expected_outputs if (EXPORT_ROOT / name).exists())
    if existing:
        raise FileExistsError(f"Refusing to overwrite existing Phase B outputs: {existing}")
    if MATRIX_RUN_DIR.exists():
        raise FileExistsError(f"Refusing to overwrite existing per-run output directory: {MATRIX_RUN_DIR}")
    MATRIX_RUN_DIR.mkdir()

    global SIMULATED_PROFILE
    SIMULATED_PROFILE = _read_json(PROFILE_PATH)
    report_path, preprocessing_report = _latest_training_run()
    train_images, train_labels, train_hash = _load_train_only(report_path)
    architecture, initial_weights, model_hashes = _load_initial_model()
    if tuple(NODE_IDS) != tuple(f"node_{index}" for index in range(8)):
        raise AssertionError("Node numbering is not deterministic")
    base_time = float(SIMULATED_PROFILE["reference_measurements"]["base_round_time"]["value_seconds"])
    bandwidth_ref = (
        float(SIMULATED_PROFILE["reference_measurements"]["effective_bandwidth"]["numerator_bytes"])
        / float(SIMULATED_PROFILE["reference_measurements"]["effective_bandwidth"]["denominator_exchange_seconds"])
    )
    bandwidths = {
        node_id: bandwidth_ref * float(SIMULATED_PROFILE["nodes"][node_id]["bandwidth_percent"]) / 100.0
        for node_id in NODE_IDS
    }
    apg_parameters = APGParameters()
    all_runs: list[dict[str, Any]] = []
    all_convergence: list[dict[str, Any]] = []
    shard_summaries: dict[str, Any] = {}
    for setting in DATA_SETTINGS:
        seed_splits, shard_records = _make_data_setting(train_images, train_labels, setting)
        shard_summaries[setting] = [
            {
                "node_id": record["node_id"],
                "shard_rows": record["shard_rows"],
                "class_counts": record["shard_class_counts"],
            }
            for record in shard_records
        ]
        for simulation_mode in SIMULATION_MODES:
            for topology in TOPOLOGIES:
                for seed_split in seed_splits:
                    run_id = (
                        f"apgb-{simulation_mode}-{setting}-{topology}-s{seed_split['seed']}"
                    ).replace("_", "-")
                    run_started = time.perf_counter()
                    print(
                        f"starting_run={run_id}; completed_runs={len(all_runs)}/36; "
                        f"elapsed_total_seconds={run_started - matrix_started:.1f}",
                        flush=True,
                    )
                    try:
                        run = _run_one(
                            simulation_mode=simulation_mode,
                            data_setting=setting,
                            topology=topology,
                            seed=seed_split["seed"],
                            nodes=seed_split["nodes"],
                            architecture=architecture,
                            initial_weights=initial_weights,
                            apg_parameters=apg_parameters,
                            base_time_seconds=base_time,
                            bandwidths=bandwidths,
                            deadline=deadline,
                        )
                    except Exception as error:
                        print(
                            f"failed_run={run_id}; error={type(error).__name__}: {error}; "
                            f"completed_runs={len(all_runs)}/36; "
                            f"elapsed_total_seconds={time.perf_counter() - matrix_started:.1f}",
                            flush=True,
                        )
                        raise
                    try:
                        convergence_records = run.pop("convergence_records")
                        run_output = {
                            key: value
                            for key, value in run.items()
                            if key != "final_model_weights"
                        }
                        run_output["convergence_records_file"] = f"{run_id}_convergence.csv"
                        _write_smoke_json(run_output, MATRIX_RUN_DIR / f"{run_id}.json")
                        _write_convergence_csv(
                            convergence_records,
                            MATRIX_RUN_DIR / f"{run_id}_convergence.csv",
                        )
                    except Exception as error:
                        print(
                            f"failed_run_output={run_id}; error={type(error).__name__}: {error}; "
                            f"completed_runs={len(all_runs)}/36; "
                            f"elapsed_total_seconds={time.perf_counter() - matrix_started:.1f}",
                            flush=True,
                        )
                        raise
                    all_convergence.extend(convergence_records)
                    all_runs.append(run)
                    elapsed_run = time.perf_counter() - run_started
                    elapsed_total = time.perf_counter() - matrix_started
                    print(
                        f"completed_run={run['run_id']}; rounds={len(run['rounds'])}; "
                        f"source=SIMULATED; run_elapsed_seconds={elapsed_run:.2f}; "
                        f"elapsed_total_seconds={elapsed_total:.2f}; "
                        f"saved=results_apg/matrix_runs/{run_id}.json,"
                        f"results_apg/matrix_runs/{run_id}_convergence.csv",
                        flush=True,
                    )
                    if time.perf_counter() >= deadline:
                        raise TimeoutError(
                            f"3-hour full-matrix time cap reached after {len(all_runs)}/36 "
                            f"completed runs; next TEST evaluation was not started"
                        )

    print(
        f"training_runs_completed={len(all_runs)}/36; "
        f"elapsed_total_seconds={time.perf_counter() - matrix_started:.2f}; "
        "TEST evaluation starts only after all training has finished",
        flush=True,
    )
    test_path = report_path.parent / "test.npz"
    test_info = _evaluate_test_once(all_runs, architecture, test_path, deadline=deadline)
    summary = _build_summary(all_runs, test_info)
    _write_convergence_csv(all_convergence)
    _write_validation_convergence_csv(
        all_runs,
        EXPORT_ROOT / "validation_convergence_summary.csv",
    )
    _write_apg_round_diagnostics(all_runs, EXPORT_ROOT / "apg_round_diagnostics.csv")
    _write_test_confusion_csv(all_runs, EXPORT_ROOT / "test_confusion_matrices.csv")
    chart_names = _plot_convergence(all_runs)
    chart_names.extend(_plot_test_bars(summary["summaries"]))
    chart_names.extend(_plot_confusions(all_runs))

    config_output = {
        "dataset": "mnist_demo",
        "simulation_modes": list(SIMULATION_MODES),
        "data_settings": list(DATA_SETTINGS),
        "topologies": list(TOPOLOGIES),
        "training_seeds": list(TRAINING_SEEDS),
        "sharding_seed": 2026,
        "rounds": ROUNDS,
        "local_epochs_per_round": LOCAL_EPOCHS,
        "learning_rate": LEARNING_RATE,
        "batch_size": BATCH_SIZE,
        "nodes": list(NODE_IDS),
        "source_label": "SIMULATED",
        "initial_model": model_hashes,
        "training_npz_sha256": train_hash,
        "test_data": test_info,
        "preprocessing_report": str(report_path.relative_to(ROOT)),
        "split_sizes": preprocessing_report["split_sizes"],
        "shard_summaries": shard_summaries,
        "synthetic_profiles": SIMULATED_PROFILE,
        "effective_bandwidth_bytes_per_second_derived": bandwidth_ref,
        "phase_b_runs": len(all_runs),
        "test_loaded_once_after_all_training": True,
    }
    (EXPORT_ROOT / "simulation_config.json").write_text(
        json.dumps(config_output, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    final_output = {
        "status": "COMPLETED",
        "source_label": "SIMULATED",
        "split_labels": {"convergence": "VALIDATION", "final_metrics": "TEST"},
        "test_data": test_info,
        "shard_summaries": shard_summaries,
        "runs": all_runs,
        **summary,
        "charts": chart_names,
    }
    (FINAL_JSON).write_text(
        json.dumps(final_output, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    _write_phase_report(all_runs, summary, test_info, shard_summaries, {
        "base_time_seconds": base_time,
        "bandwidth_numerator_bytes": SIMULATED_PROFILE["reference_measurements"]["effective_bandwidth"]["numerator_bytes"],
        "bandwidth_denominator_exchange_seconds": SIMULATED_PROFILE["reference_measurements"]["effective_bandwidth"]["denominator_exchange_seconds"],
    })

    print(f"runs_completed={len(all_runs)}; source=SIMULATED")
    print(f"test_rows={test_info['test_rows']}; test_loaded_once_after_training=True")
    for item in summary["summaries"]:
        metrics = item["metrics"]
        print(
            f"summary mode={item['simulation_mode']} data={item['data_setting']} "
            f"topology={item['topology']} split=TEST source=SIMULATED "
            f"accuracy={metrics['accuracy']['mean']:.6f}+/-{metrics['accuracy']['sample_sd_ddof_1']:.6f} "
            f"macro_f1={metrics['macro_f1']['mean']:.6f}+/-{metrics['macro_f1']['sample_sd_ddof_1']:.6f}"
        )
    for name in OUTPUT_FILES:
        print(f"created=results_apg/{name}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run only 3 rounds, one seed, IID, APG, all-simulated; TEST remains closed.",
    )
    parser.add_argument(
        "--smoke-psi-sensitivity",
        action="store_true",
        help="For --smoke only, use psi=0.001 and write separately named smoke outputs.",
    )
    parser.add_argument(
        "--smoke-psi-1e-6",
        action="store_true",
        help="For --smoke only, use psi=1e-6 and write separately named smoke outputs.",
    )
    args = parser.parse_args()
    if args.smoke_psi_sensitivity and args.smoke_psi_1e_6:
        parser.error("Choose at most one smoke psi sensitivity option")
    if (args.smoke_psi_sensitivity or args.smoke_psi_1e_6) and not args.smoke:
        parser.error("Smoke psi sensitivity options require --smoke")
    if args.smoke:
        psi = 0.001 if args.smoke_psi_sensitivity else 1e-6 if args.smoke_psi_1e_6 else 0.01
        _run_smoke(psi=psi)
    else:
        _run_full_matrix()


if __name__ == "__main__":
    main()