import csv
import json
from types import SimpleNamespace

import numpy as np
import pytest

from topology_opt.apg import APGParameters
from scripts import run_apg_phase_b as driver
from scripts.run_apg_phase_b import (
    DATASET_ROOT,
    _build_model,
    _compile_training_model,
    _load_initial_model,
    _transport_update,
)


def test_hybrid_shaped_simulation_uses_worker_weight_envelope_in_process() -> None:
    weights = [np.arange(12, dtype=np.float32).reshape(3, 4), np.array([1.0, 2.0], dtype=np.float32)]

    received, payload_bytes, headers = _transport_update(
        "hybrid_shaped_simulation",
        "apgb-hybrid-iid-ring-s17",
        4,
        "node_6",
        "node_5",
        weights,
        240,
    )

    assert headers == {
        "X-Run-ID": "apgb-hybrid-iid-ring-s17",
        "X-Round": "4",
        "X-Sender-Node": "worker1",
        "X-Sample-Count": "240",
    }
    assert payload_bytes > 0
    assert len(received) == len(weights)
    for actual, expected in zip(received, weights):
        np.testing.assert_array_equal(actual, expected)


def test_all_simulated_transport_is_in_process_and_has_no_worker_headers() -> None:
    weights = [np.array([0.25, 0.75], dtype=np.float32)]

    received, payload_bytes, headers = _transport_update(
        "all_simulated",
        "apgb-all-iid-ring-s17",
        0,
        "node_0",
        "node_7",
        weights,
        240,
    )

    assert headers is None
    assert payload_bytes > 0
    np.testing.assert_array_equal(received[0], weights[0])


class _TinyModel:
    def __init__(self) -> None:
        self.weights = [np.zeros(10, dtype=np.float32)]

    def set_weights(self, weights: list[np.ndarray]) -> None:
        self.weights = [np.asarray(weight).copy() for weight in weights]

    def get_weights(self) -> list[np.ndarray]:
        return [weight.copy() for weight in self.weights]

    def fit(self, images, labels, **kwargs):
        self.weights[0][0] += 0.01
        return SimpleNamespace(history={"loss": [0.5], "accuracy": [0.5]})

    def predict(self, images, **kwargs):
        logits = np.zeros((len(images), 10), dtype=np.float32)
        logits[:, 0] = 1.0 + self.weights[0][0]
        return logits


def test_two_round_train_only_run_serializes_to_json_and_csv(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(driver, "_build_model", lambda architecture: _TinyModel())
    monkeypatch.setattr(driver, "_compile_training_model", lambda model: model)
    monkeypatch.setattr(
        driver,
        "SIMULATED_PROFILE",
        {
            "nodes": {
                node_id: {
                    "role": "simulated",
                    "worker_id": None,
                    "compute_multiplier": 1.0,
                }
                for node_id in driver.NODE_IDS
            }
        },
    )
    nodes = []
    for node_index, node_id in enumerate(driver.NODE_IDS):
        labels = np.array([node_index % 10, (node_index + 1) % 10], dtype=np.int64)
        counts = np.bincount(labels, minlength=10).tolist()
        nodes.append(
            {
                "node_id": node_id,
                "train_images": np.zeros((2, 1), dtype=np.float32),
                "train_labels": labels,
                "validation_images": np.zeros((2, 1), dtype=np.float32),
                "validation_labels": labels,
                "shard_rows": 4,
                "train_rows": 2,
                "validation_rows": 2,
                "shard_class_counts": counts,
                "train_class_counts": counts,
                "validation_class_counts": counts,
                "theta": np.asarray(counts, dtype=np.float64) / 2.0,
            }
        )

    run = driver._run_one(
        simulation_mode="all_simulated",
        data_setting="iid_stratified",
        topology="apg",
        seed=17,
        nodes=nodes,
        architecture={},
        initial_weights=[np.zeros(10, dtype=np.float32)],
        apg_parameters=APGParameters(max_iterations=2),
        base_time_seconds=0.1,
        bandwidths={node_id: 1.0 for node_id in driver.NODE_IDS},
        round_count=2,
        local_epochs=1,
    )
    assert len(run["rounds"]) == 2
    assert run["pc_simulation_wall_seconds"] >= 0.0
    assert run["split"] == "VALIDATION"

    convergence_records = run.pop("convergence_records")
    run.pop("final_model_weights")
    json.dumps(run, allow_nan=False)
    json_path = tmp_path / "tiny_run.json"
    csv_path = tmp_path / "tiny_convergence.csv"
    driver._write_smoke_json(run, json_path)
    driver._write_convergence_csv(convergence_records, csv_path)

    restored = json.loads(json_path.read_text(encoding="utf-8"))
    assert len(restored["rounds"]) == 2
    for round_record in restored["rounds"]:
        for matrix_name in (
            "loss_matrix_raw",
            "wait_matrix_simulated_seconds",
            "loss_matrix_normalized",
            "wait_matrix_normalized",
        ):
            matrix = round_record["apg_inputs"][matrix_name]
            assert all(matrix[index][index] is None for index in range(len(matrix)))
    with csv_path.open(encoding="utf-8", newline="") as source:
        csv_records = list(csv.DictReader(source))
    assert len(csv_records) == len(convergence_records) == 36
    assert {record["split"] for record in csv_records} == {"TRAIN", "VALIDATION"}


def test_matrix_serializer_rejects_nonfinite_off_diagonal_value() -> None:
    matrix = np.array([[np.nan, 1.0], [np.nan, np.nan]], dtype=np.float64)
    with pytest.raises(ValueError, match="non-finite off-diagonal values"):
        driver._matrix_with_null_diagonal(matrix, "test_matrix")


def test_local_mnist_training_step_records_loss_and_accuracy() -> None:
    reports = sorted(
        (DATASET_ROOT / "processed").glob("run-*/preprocessing_report.json"),
        key=lambda path: path.stat().st_mtime_ns,
    )
    assert reports, "Saved MNIST preprocessing report is required for this local TRAIN test"
    shard_path = reports[-1].parent / "workers" / "worker1.npz"
    with np.load(shard_path, allow_pickle=False) as shard:
        images = np.asarray(shard["images"][:32], dtype=np.float32)
        labels = np.asarray(shard["labels"][:32], dtype=np.int64)

    architecture, initial_weights, _ = _load_initial_model()
    model = _build_model(architecture)
    model.set_weights(initial_weights)
    _compile_training_model(model)
    history = model.fit(images, labels, epochs=1, batch_size=32, shuffle=False, verbose=0)

    assert len(history.history["loss"]) == 1
    assert len(history.history["accuracy"]) == 1
    assert np.isfinite(history.history["loss"][0])
    assert np.isfinite(history.history["accuracy"][0])