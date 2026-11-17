"""Run the Phase 6-equivalent centralized CIFAR-10 NAS baseline on the PC."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import tensorflow as tf

from common.cifar_model import build_cifar10_model, classification_metrics, compile_cifar10_model
from common.nas import architecture_parameters, architecture_spec


TRAIN_PATH = ROOT / "data" / "raw" / "cifar10_train.npz"
TEST_PATH = ROOT / "data" / "raw" / "cifar10_test.npz"
OUTPUT_PATH = ROOT / "data" / "m8_centralized_nas_baseline.json"
ARCHITECTURE_ID = "b1_c16_k5"
TRAINING_SEEDS = (17, 42, 2026)
ROUNDS = 10
EPOCHS_PER_ROUND = 5
BATCH_SIZE = 128
LEARNING_RATE = 0.001
TRAIN_ROWS = 50_000
TEST_ROWS = 10_000
EXPECTED_CLASS_ROWS = 5_000
PHASE6_NAS_MEAN_ACCURACY = 0.6163
PHASE6_NAS_MEAN_MACRO_F1 = 0.6136


def _load_training_split() -> tuple[np.ndarray, np.ndarray]:
    with np.load(TRAIN_PATH, allow_pickle=False) as dataset:
        images = np.asarray(dataset["images"], dtype=np.uint8)
        labels = np.asarray(dataset["labels"], dtype=np.int64).reshape(-1)
    if images.shape != (TRAIN_ROWS, 32, 32, 3) or labels.shape != (TRAIN_ROWS,):
        raise ValueError(f"Unexpected CIFAR-10 training split shape: {images.shape}, {labels.shape}")
    class_counts = np.bincount(labels, minlength=10)
    if labels.min() != 0 or labels.max() != 9 or not np.all(class_counts == EXPECTED_CLASS_ROWS):
        raise ValueError("CIFAR-10 training split does not contain 5,000 rows per class")
    return images, labels


def _load_test_split() -> tuple[np.ndarray, np.ndarray]:
    with np.load(TEST_PATH, allow_pickle=False) as dataset:
        images = np.asarray(dataset["images"], dtype=np.uint8)
        labels = np.asarray(dataset["labels"], dtype=np.int64).reshape(-1)
    if images.shape != (TEST_ROWS, 32, 32, 3) or labels.shape != (TEST_ROWS,):
        raise ValueError(f"Unexpected CIFAR-10 test split shape: {images.shape}, {labels.shape}")
    class_counts = np.bincount(labels, minlength=10)
    if labels.min() != 0 or labels.max() != 9 or not np.all(class_counts == 1_000):
        raise ValueError("CIFAR-10 test split does not contain 1,000 rows per class")
    return images, labels


def _summary(values: list[float]) -> dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "sample_std_ddof_1": float(np.std(values, ddof=1)),
    }


def _build_model(seed: int) -> tf.keras.Model:
    model = build_cifar10_model(seed=seed, architecture=ARCHITECTURE_ID)
    compile_cifar10_model(model)
    actual_lr = float(tf.keras.backend.get_value(model.optimizer.learning_rate))
    if not math.isclose(actual_lr, LEARNING_RATE, rel_tol=1e-6):
        raise ValueError(f"Unexpected Adam learning rate: {actual_lr}")
    if model.count_params() != 11_466:
        raise ValueError(f"Unexpected {ARCHITECTURE_ID} model parameter count")
    flips = [layer for layer in model.layers if isinstance(layer, tf.keras.layers.RandomFlip)]
    if len(flips) != 1 or flips[0].mode != "horizontal":
        raise ValueError("CIFAR model must use only the Phase 6 horizontal RandomFlip augmentation")
    return model


def _epoch_progress(seed: int, epoch_offset: int) -> tf.keras.callbacks.Callback:
    class EpochProgress(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch: int, logs: dict[str, Any] | None = None) -> None:
            values = logs or {}
            print(
                f"seed={seed}; epoch={epoch_offset + epoch + 1}/"
                f"{ROUNDS * EPOCHS_PER_ROUND}; "
                f"loss={float(values.get('loss', float('nan'))):.5f}; "
                f"accuracy={float(values.get('accuracy', float('nan'))):.5f}",
                flush=True,
            )

    return EpochProgress()


def _save_json(payload: dict[str, Any]) -> None:
    temporary_path = OUTPUT_PATH.with_suffix(".json.tmp")
    temporary_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary_path.replace(OUTPUT_PATH)


def _estimate_one_epoch() -> None:
    if not TRAIN_PATH.is_file():
        raise FileNotFoundError("Local prepared CIFAR-10 training file is required")
    train_images, train_labels = _load_training_split()
    model = _build_model(TRAINING_SEEDS[0])
    started = time.perf_counter()
    model.fit(
        train_images,
        train_labels,
        epochs=1,
        batch_size=BATCH_SIZE,
        shuffle=True,
        verbose=0,
        callbacks=[_epoch_progress(TRAINING_SEEDS[0], 0)],
    )
    epoch_seconds = time.perf_counter() - started
    projected_seconds = epoch_seconds * len(TRAINING_SEEDS) * ROUNDS * EPOCHS_PER_ROUND
    print(f"measured_one_epoch_seconds={epoch_seconds:.2f}")
    print(
        f"projected_full_training_seconds={projected_seconds:.1f}; "
        f"projected_minutes={projected_seconds / 60:.1f}; "
        "projection scales one measured epoch across 150 epochs and excludes final evaluation/overhead"
    )
    print("estimate_only=True; test_split_opened=False; result_json_not_written=True")


def _run_full_evaluation() -> None:
    if OUTPUT_PATH.exists():
        raise FileExistsError(f"Refusing to overwrite existing result: {OUTPUT_PATH}")
    if not TRAIN_PATH.is_file() or not TEST_PATH.is_file():
        raise FileNotFoundError("Local prepared CIFAR-10 train/test files are required")

    train_sha256 = hashlib.sha256(TRAIN_PATH.read_bytes()).hexdigest()
    train_images, train_labels = _load_training_split()
    architecture = architecture_spec(ARCHITECTURE_ID)
    if architecture_parameters(ARCHITECTURE_ID) != 11_466:
        raise ValueError("b1_c16_k5 does not match the recorded 11,466-parameter architecture")

    models: list[tf.keras.Model] = []
    seed_training: list[dict[str, Any]] = []
    for seed in TRAINING_SEEDS:
        model = _build_model(seed)

        round_rows: list[dict[str, Any]] = []
        for round_index in range(ROUNDS):
            started = time.perf_counter()
            history = model.fit(
                train_images,
                train_labels,
                epochs=EPOCHS_PER_ROUND,
                batch_size=BATCH_SIZE,
                shuffle=True,
                verbose=0,
                callbacks=[_epoch_progress(seed, round_index * EPOCHS_PER_ROUND)],
            )
            row = {
                "round": round_index + 1,
                "epochs": EPOCHS_PER_ROUND,
                "training_loss": float(history.history["loss"][-1]),
                "training_accuracy": float(history.history["accuracy"][-1]),
                "duration_seconds": round(time.perf_counter() - started, 3),
            }
            round_rows.append(row)
            print(
                f"seed={seed}; round={round_index + 1}/{ROUNDS}; "
                f"epochs={EPOCHS_PER_ROUND}; rows={TRAIN_ROWS}; "
                f"duration_seconds={row['duration_seconds']:.2f}"
            )
        models.append(model)
        seed_training.append({"seed": seed, "rounds": round_rows})
        _save_json(
            {
                "status": "TRAINING",
                "experiment": "M8 centralized NAS-architecture baseline",
                "completed_seeds": [row["seed"] for row in seed_training],
                "phase6_configuration": {
                    "architecture": ARCHITECTURE_ID,
                    "parameter_count": 11_466,
                    "optimizer": "Adam",
                    "learning_rate": LEARNING_RATE,
                    "batch_size": BATCH_SIZE,
                    "augmentation": "RandomFlip(mode='horizontal') in the model; no other augmentation",
                    "train_rows": TRAIN_ROWS,
                    "test_rows": TEST_ROWS,
                    "rounds": ROUNDS,
                    "epochs_per_round": EPOCHS_PER_ROUND,
                    "training_seeds": list(TRAINING_SEEDS),
                },
                "train_npz_sha256": train_sha256,
                "training_rounds": seed_training,
                "test_split_loaded": False,
            }
        )
        print(f"partial_result_saved={OUTPUT_PATH.relative_to(ROOT)}; seed={seed}", flush=True)

    # Test examples are first loaded after all three seeds have completed training.
    test_sha256 = hashlib.sha256(TEST_PATH.read_bytes()).hexdigest()
    test_images, test_labels = _load_test_split()
    per_seed = []
    for seed, model in zip(TRAINING_SEEDS, models):
        logits = model.predict(test_images, batch_size=256, verbose=0)
        per_seed.append({"seed": seed, **classification_metrics(test_labels, logits)})
        row = per_seed[-1]
        print(
            f"centralized_test seed={seed}; accuracy={row['accuracy']:.6f}; "
            f"macro_f1={row['macro_f1']:.6f}"
        )

    training_examples_per_seed = TRAIN_ROWS * ROUNDS * EPOCHS_PER_ROUND
    result = {
        "status": "COMPLETED",
        "experiment": "M8 centralized NAS-architecture baseline",
        "dataset": "CIFAR-10 official Python train/test split",
        "evaluation_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "execution_scope": "PC-only; no cluster endpoints or Jetson access",
        "phase6_configuration": {
            "architecture": ARCHITECTURE_ID,
            "architecture_spec": architecture,
            "parameter_count": 11_466,
            "optimizer": "Adam",
            "learning_rate": LEARNING_RATE,
            "batch_size": BATCH_SIZE,
            "augmentation": "RandomFlip(mode='horizontal') in the model; no other augmentation",
            "training_rows": TRAIN_ROWS,
            "test_rows": TEST_ROWS,
            "worker_shard_rows_phase6": [16_670, 16_670, 16_660],
            "nas_search_validation_rows_per_worker": 3_330,
            "nas_search_train_rows_per_worker": [13_340, 13_340, 13_330],
            "rounds": ROUNDS,
            "epochs_per_round": EPOCHS_PER_ROUND,
            "training_seeds": list(TRAINING_SEEDS),
            "shuffle": True,
        },
        "centralized_training_equivalence": {
            "training_rows_per_epoch": TRAIN_ROWS,
            "total_epochs_per_seed": ROUNDS * EPOCHS_PER_ROUND,
            "example_presentations_per_seed": training_examples_per_seed,
            "dfl_equivalent_example_presentations_per_seed": training_examples_per_seed,
        },
        "data_integrity": {
            "train_npz_sha256": train_sha256,
            "test_npz_sha256": test_sha256,
            "train_class_rows": EXPECTED_CLASS_ROWS,
            "test_class_rows": 1_000,
        },
        "test_split_disclosure": {
            "test_used_for_training_or_tuning": False,
            "test_loaded_after_all_seed_training": True,
            "validation_was_used_for_phase6_nas_tuning": True,
        },
        "phase6_dfl_nas_reference": {
            "architecture": ARCHITECTURE_ID,
            "mean_accuracy": PHASE6_NAS_MEAN_ACCURACY,
            "mean_macro_f1": PHASE6_NAS_MEAN_MACRO_F1,
            "training_seeds": list(TRAINING_SEEDS),
            "rounds": ROUNDS,
            "epochs_per_round": EPOCHS_PER_ROUND,
            "source": "docs/phase6-neural-architecture-search.md; reported values",
        },
        "centralized_test": {
            "per_seed": per_seed,
            "summary": {
                "accuracy": _summary([row["accuracy"] for row in per_seed]),
                "macro_f1": _summary([row["macro_f1"] for row in per_seed]),
            },
        },
        "training_rounds": seed_training,
    }
    _save_json(result)
    print("centralized_test_summary=" + json.dumps(result["centralized_test"]["summary"], sort_keys=True))
    print(f"test_rows={test_labels.size}; test_npz_sha256={test_sha256}")
    print(f"result_json={OUTPUT_PATH.relative_to(ROOT)}")
    print("test_split_used_only_after_training=True")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--estimate-epoch",
        action="store_true",
        help="Measure one training epoch and estimate full runtime without opening the test split.",
    )
    arguments = parser.parse_args()
    if arguments.estimate_epoch:
        _estimate_one_epoch()
    else:
        _run_full_evaluation()


if __name__ == "__main__":
    main()