"""Run the final PC-only MNIST test evaluation and save its metrics as JSON."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf

from mnist_test_split_check import (
    BATCH_SIZE,
    DATASET_ROOT,
    LEARNING_RATE,
    LOCAL_EPOCHS,
    ROUNDS,
    TRAINING_SEEDS,
    WORKER_IDS,
    WORKER_ROWS,
    _evaluate,
    _installed_initial_weights,
    _load_worker_training_data,
    _summary,
    _train_centralized_seed,
    _train_federated_seed,
)


ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = DATASET_ROOT / "models" / "initial_model"
OUTPUT_PATH = ROOT / "data" / "mnist_final_test_eval.json"
EXPECTED_ARCHITECTURE_SHA256 = "b66616d37317a211cd9dc2cb8d02b6213cfc292d049a2258dc7711d1e7f73bb2"
EXPECTED_WEIGHTS_SHA256 = "d23e3d771bb065c347655ac932ab90cc3a8ce12ea01345c533ffca004be78d75"
EXPECTED_SPLIT_SIZES = {"train": 2400, "validation": 300, "test": 300}


def _latest_preprocessing_report() -> tuple[Path, dict[str, Any]]:
    reports = sorted(
        (DATASET_ROOT / "processed").glob("run-*/preprocessing_report.json"),
        key=lambda path: path.stat().st_mtime_ns,
    )
    if not reports:
        raise FileNotFoundError("Persisted MNIST preprocessing report is missing")
    report_path = reports[-1]
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("dataset_name") != "mnist_demo":
        raise ValueError("Latest preprocessing report is not mnist_demo")
    if report.get("split_sizes") != EXPECTED_SPLIT_SIZES:
        raise ValueError("MNIST split sizes do not match the expected 2400/300/300 split")
    if report.get("split_indices_disjoint") is not True:
        raise ValueError("Persisted MNIST split indices are not confirmed disjoint")
    if report.get("worker_shards_disjoint") is not True:
        raise ValueError("Persisted MNIST worker shards are not confirmed disjoint")
    return report_path, report


def _verify_installed_model() -> tuple[dict[str, Any], list[np.ndarray], dict[str, str]]:
    architecture_path = MODEL_ROOT / "architecture.json"
    weights_path = MODEL_ROOT / "initial_weights.npz"
    architecture_hash = hashlib.sha256(architecture_path.read_bytes()).hexdigest()
    weights_hash = hashlib.sha256(weights_path.read_bytes()).hexdigest()
    if architecture_hash != EXPECTED_ARCHITECTURE_SHA256:
        raise ValueError("Installed MNIST architecture hash differs from the verified bundle")
    if weights_hash != EXPECTED_WEIGHTS_SHA256:
        raise ValueError("Installed MNIST weights hash differ from the verified bundle")
    architecture, weights = _installed_initial_weights()
    return architecture, weights, {
        "architecture_sha256": architecture_hash,
        "weights_sha256": weights_hash,
    }


def _verify_worker_shards(
    report_path: Path,
    report: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    verified: dict[str, dict[str, Any]] = {}
    uploaded_workers = report.get("uploaded_workers", {})
    for worker_id in WORKER_IDS:
        shard_path = report_path.parent / "workers" / f"{worker_id}.npz"
        actual_hash = hashlib.sha256(shard_path.read_bytes()).hexdigest()
        worker_report = uploaded_workers.get(worker_id, {})
        expected_hash = worker_report.get("shard_sha256")
        if actual_hash != expected_hash:
            raise ValueError(f"Persisted real training shard hash mismatch for {worker_id}")
        if worker_report.get("rows") != 800:
            raise ValueError(f"Expected 800 persisted shard rows for {worker_id}")
        verified[worker_id] = {
            "shard_sha256": actual_hash,
            "shard_rows": int(worker_report["rows"]),
            "training_rows_per_seed": WORKER_ROWS,
        }
    return verified


def _load_untouched_test_split(test_path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(test_path, allow_pickle=False) as test_split:
        images = np.asarray(test_split["images"], dtype=np.float32)
        labels = np.asarray(test_split["labels"], dtype=np.int64).reshape(-1)
    if images.shape != (300, 28, 28, 1) or labels.shape != (300,):
        raise ValueError(f"Unexpected untouched MNIST test split shape: {images.shape}, {labels.shape}")
    return images, labels


def _metric_summary(results: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    summaries: dict[str, dict[str, float]] = {}
    for metric in ("accuracy", "macro_f1"):
        mean, sample_std = _summary([row[metric] for row in results])
        summaries[metric] = {"mean": mean, "sample_std_ddof_1": sample_std}
    return summaries


def main() -> None:
    if OUTPUT_PATH.exists():
        raise FileExistsError(f"Refusing to overwrite existing result: {OUTPUT_PATH}")

    report_path, report = _latest_preprocessing_report()
    architecture, initial_weights, model_hashes = _verify_installed_model()
    worker_shards = _verify_worker_shards(report_path, report)
    worker_data = _load_worker_training_data()
    if len(worker_data) != len(WORKER_IDS):
        raise ValueError("Loaded worker training shards do not match the expected worker count")

    federated_models: list[tf.keras.Model] = []
    centralized_models: list[tf.keras.Model] = []
    for seed in TRAINING_SEEDS:
        federated_models.append(
            _train_federated_seed(architecture, initial_weights, worker_data, seed)
        )
        centralized_models.append(
            _train_centralized_seed(architecture, initial_weights, worker_data, seed)
        )

    # The test split is opened only after all six training runs have completed.
    test_path = report_path.parent / "test.npz"
    test_hash = hashlib.sha256(test_path.read_bytes()).hexdigest()
    test_images, test_labels = _load_untouched_test_split(test_path)
    federated_results = [_evaluate(model, test_images, test_labels) for model in federated_models]
    centralized_results = [_evaluate(model, test_images, test_labels) for model in centralized_models]

    federated_by_seed = [
        {"seed": seed, **result}
        for seed, result in zip(TRAINING_SEEDS, federated_results)
    ]
    centralized_by_seed = [
        {"seed": seed, **result}
        for seed, result in zip(TRAINING_SEEDS, centralized_results)
    ]
    train_rows_per_seed = WORKER_ROWS * len(WORKER_IDS)
    example_presentations = train_rows_per_seed * ROUNDS * LOCAL_EPOCHS
    result = {
        "dataset_name": "mnist_demo",
        "evaluation_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "evaluation_scope": "PC-only; no cluster requests or hardware training",
        "data_integrity": {
            "preprocessing_report": str(report_path.relative_to(ROOT)),
            "split_sizes": report["split_sizes"],
            "split_indices_disjoint": report["split_indices_disjoint"],
            "worker_shards_disjoint": report["worker_shards_disjoint"],
            "test_rows": int(test_labels.size),
            "test_npz_sha256": test_hash,
            "worker_shards": worker_shards,
        },
        "protocol": {
            "initial_model": model_hashes,
            "training_seeds": list(TRAINING_SEEDS),
            "rounds": ROUNDS,
            "local_epochs_per_round": LOCAL_EPOCHS,
            "learning_rate": LEARNING_RATE,
            "batch_size": BATCH_SIZE,
            "optimizer": "Adam",
            "federated_workers": list(WORKER_IDS),
            "persisted_rows_per_worker": 800,
            "training_rows_per_worker_per_seed": WORKER_ROWS,
            "federated_aggregation": "sample-count-weighted model-weight averaging each round",
            "centralized_rows_per_epoch": train_rows_per_seed,
            "federated_example_presentations_per_seed": example_presentations,
            "centralized_example_presentations_per_seed": example_presentations,
            "same_total_training_example_presentations": True,
            "summary_standard_deviation": "sample standard deviation (ddof=1) across seeds",
        },
        "evaluation_disclosure": {
            "validation_was_used_for_tuning": True,
            "test_split_used_for_training_or_tuning": False,
            "test_split_evaluated_after_all_training": True,
            "test_split_note": "The 300-image test split is used only for the final reported metrics.",
        },
        "federated_test": {
            "per_seed": federated_by_seed,
            "summary": _metric_summary(federated_results),
        },
        "centralized_test": {
            "per_seed": centralized_by_seed,
            "summary": _metric_summary(centralized_results),
        },
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print("federated_test_per_seed=" + json.dumps(federated_by_seed, sort_keys=True))
    print("federated_test_summary=" + json.dumps(result["federated_test"]["summary"], sort_keys=True))
    print("centralized_test_per_seed=" + json.dumps(centralized_by_seed, sort_keys=True))
    print("centralized_test_summary=" + json.dumps(result["centralized_test"]["summary"], sort_keys=True))
    print(f"test_rows={test_labels.size}; test_npz_sha256={test_hash}")
    print(f"result_json={OUTPUT_PATH.relative_to(ROOT)}")
    print("evaluation_note=validation used for tuning; test split used only for final metrics")


if __name__ == "__main__":
    main()