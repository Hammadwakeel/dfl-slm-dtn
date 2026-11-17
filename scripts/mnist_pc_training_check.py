"""Train PC-only MNIST controls using the actual normalized worker shards."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import tensorflow as tf

from worker.nas_search import _build_dataset_candidate, _split_indices

DATASET_ROOT = ROOT / "data" / "datasets" / "mnist_demo"
MODEL_ROOT = DATASET_ROOT / "models" / "initial_model"
OUTPUT_PATH = ROOT / "data" / "mnist_pc_training_check.json"
WORKER_IDS = ("worker1", "worker2", "worker3")
WORKER_ARCHITECTURE = {"id": "b1_c8_k3", "blocks": 1, "channels": 8, "kernel_size": 3}
SPLIT_SEED = 2026
TRAINING_SEEDS = (17, 42, 2026)
EPOCHS = 20
LEARNING_RATE = 0.001
BATCH_SIZE = 128


def _dataset() -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
]:
    report_paths = sorted(
        (DATASET_ROOT / "processed").glob("run-*/preprocessing_report.json"),
        key=lambda path: path.stat().st_mtime_ns,
    )
    if not report_paths:
        raise FileNotFoundError("MNIST preprocessing report is missing")
    images_train: list[np.ndarray] = []
    labels_train: list[np.ndarray] = []
    images_validation: list[np.ndarray] = []
    labels_validation: list[np.ndarray] = []
    per_worker: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    for worker_id in WORKER_IDS:
        path = report_paths[-1].parent / "workers" / f"{worker_id}.npz"
        with np.load(path, allow_pickle=False) as shard:
            images = np.asarray(shard["images"], dtype=np.float32)
            labels = np.asarray(shard["labels"], dtype=np.int64).reshape(-1)
            classes = np.asarray(shard["class_names"]).tolist()
        if images.shape != (800, 28, 28, 1) or labels.shape != (800,):
            raise RuntimeError(f"Unexpected shard shape on {worker_id}: {images.shape}, {labels.shape}")
        if len(classes) != 10 or not np.array_equal(np.unique(labels), np.arange(10)):
            raise RuntimeError(f"Unexpected MNIST classes or labels on {worker_id}")
        train_indices, validation_indices = _split_indices(labels, SPLIT_SEED)
        worker_train_images = images[train_indices]
        worker_train_labels = labels[train_indices]
        worker_validation_images = images[validation_indices]
        worker_validation_labels = labels[validation_indices]
        per_worker.append(
            (
                worker_train_images,
                worker_train_labels,
                worker_validation_images,
                worker_validation_labels,
            )
        )
        images_train.append(worker_train_images)
        labels_train.append(worker_train_labels)
        images_validation.append(worker_validation_images)
        labels_validation.append(worker_validation_labels)
        print(
            f"{worker_id}: train={train_indices.size}; validation={validation_indices.size}; "
            f"shape={images.shape}; dtype={images.dtype}; labels_dtype={labels.dtype}"
        )
    return (
        np.concatenate(images_train),
        np.concatenate(labels_train),
        np.concatenate(images_validation),
        np.concatenate(labels_validation),
        per_worker,
    )


def _build_from_initial_architecture(architecture: dict[str, Any]) -> tf.keras.Model:
    layers: list[tf.keras.layers.Layer] = [
        tf.keras.layers.Input(shape=tuple(architecture["input_shape"]), dtype=tf.float32)
    ]
    for layer in architecture["layers"]:
        if layer["type"] == "conv2d":
            layers.append(
                tf.keras.layers.Conv2D(
                    filters=layer["filters"],
                    kernel_size=tuple(layer["kernel_size"]),
                    padding=layer["padding"],
                    activation=layer["activation"],
                )
            )
        elif layer["type"] == "global_average_pooling2d":
            layers.append(tf.keras.layers.GlobalAveragePooling2D())
        elif layer["type"] == "dense":
            layers.append(
                tf.keras.layers.Dense(
                    units=layer["units"],
                    activation=layer["activation"],
                )
            )
        else:
            raise RuntimeError(f"Unsupported serialized initial-model layer: {layer['type']}")
    return tf.keras.Sequential(layers, name="mnist_initial_model_json")


def _compile(model: tf.keras.Model, learning_rate: float = LEARNING_RATE) -> None:
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )


def _train(
    name: str,
    seed: int,
    model: tf.keras.Model,
    train_images: np.ndarray,
    train_labels: np.ndarray,
    validation_images: np.ndarray,
    validation_labels: np.ndarray,
    *,
    epochs: int = EPOCHS,
    learning_rate: float = LEARNING_RATE,
    batch_size: int = BATCH_SIZE,
    report_epochs: tuple[int, ...] | None = None,
) -> dict[str, Any]:
    _compile(model, learning_rate=learning_rate)
    actual_learning_rate = float(tf.keras.backend.get_value(model.optimizer.learning_rate))
    if not math.isclose(actual_learning_rate, learning_rate, rel_tol=1e-5):
        raise RuntimeError(f"Unexpected optimizer learning rate: {actual_learning_rate}")
    history = model.fit(
        train_images,
        train_labels,
        validation_data=(validation_images, validation_labels),
        epochs=epochs,
        batch_size=batch_size,
        shuffle=True,
        verbose=0,
    )
    rows = [
        {
            "epoch": index + 1,
            "train_accuracy": float(history.history["accuracy"][index]),
            "validation_accuracy": float(history.history["val_accuracy"][index]),
            "train_loss": float(history.history["loss"][index]),
            "validation_loss": float(history.history["val_loss"][index]),
        }
        for index in range(epochs)
    ]
    if any(
        not math.isfinite(value)
        for row in rows
        for value in (
            row["train_accuracy"],
            row["validation_accuracy"],
            row["train_loss"],
            row["validation_loss"],
        )
    ):
        raise RuntimeError(f"Non-finite training metrics for {name}, seed={seed}")
    logits = model.predict(
        validation_images,
        batch_size=batch_size,
        verbose=0,
    )
    predictions = np.asarray(logits).argmax(axis=1)
    confusion = np.zeros((10, 10), dtype=np.int64)
    np.add.at(confusion, (validation_labels, predictions), 1)
    true_positive = np.diag(confusion).astype(np.float64)
    denominator = confusion.sum(axis=0) + confusion.sum(axis=1)
    per_class_f1 = np.divide(
        2.0 * true_positive,
        denominator,
        out=np.zeros_like(true_positive),
        where=denominator > 0,
    )
    final_macro_f1 = float(per_class_f1.mean())
    output = {
        "name": name,
        "seed": seed,
        "optimizer": "Adam",
        "learning_rate": actual_learning_rate,
        "loss": "SparseCategoricalCrossentropy(from_logits=True)",
        "batch_size": batch_size,
        "train_samples": int(train_labels.size),
        "validation_samples": int(validation_labels.size),
        "final_macro_f1": final_macro_f1,
        "epochs": rows,
    }
    print(f"\n{name}; seed={seed}; final_macro_f1={final_macro_f1:.4f}")
    print("epoch | train_acc | val_acc | train_loss | val_loss")
    selected_epochs = set(report_epochs or tuple(range(1, epochs + 1)))
    for row in rows:
        if row["epoch"] not in selected_epochs:
            continue
        print(
            f"{row['epoch']:>5} | {row['train_accuracy']:.4f} | "
            f"{row['validation_accuracy']:.4f} | {row['train_loss']:.4f} | "
            f"{row['validation_loss']:.4f}"
        )
    return output


def _weighted_average(weights_by_worker: list[tuple[list[np.ndarray], int]]) -> list[np.ndarray]:
    if not weights_by_worker:
        raise ValueError("Cannot average an empty worker list")
    total_samples = sum(sample_count for _, sample_count in weights_by_worker)
    reference = weights_by_worker[0][0]
    return [
        (
            sum(
                weights[index].astype(np.float64) * sample_count
                for weights, sample_count in weights_by_worker
            )
            / total_samples
        ).astype(reference[index].dtype)
        for index in range(len(reference))
    ]


def _simulate_worker_schedule(
    per_worker: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    seed: int,
    *,
    rounds: int = 5,
    local_epochs: int = 2,
    learning_rate: float = LEARNING_RATE,
    batch_size: int = BATCH_SIZE,
    model_kind: str = "worker_builder",
    initial_architecture: dict[str, Any] | None = None,
    initial_weights: list[np.ndarray] | None = None,
) -> dict[str, Any]:
    models: list[tf.keras.Model] = []
    for _worker_index in range(3):
        tf.keras.utils.set_random_seed(seed)
        if model_kind == "initial_model_json":
            if initial_architecture is None or initial_weights is None:
                raise ValueError("Initial-model schedule requires JSON architecture and weights")
            model = _build_from_initial_architecture(initial_architecture)
            model(np.zeros((1, 28, 28, 1), dtype=np.float32), training=False)
            model.set_weights([np.array(weight, copy=True) for weight in initial_weights])
        elif model_kind == "worker_builder":
            model = _build_dataset_candidate(
                input_shape=(28, 28, 1),
                num_classes=10,
                architecture=WORKER_ARCHITECTURE,
                seed=seed,
            )
        else:
            raise ValueError(f"Unsupported PC schedule model_kind: {model_kind}")
        _compile(model, learning_rate=learning_rate)
        models.append(model)

    round_results: list[dict[str, Any]] = []
    for round_index in range(1, rounds + 1):
        local_train_accuracies: dict[str, float] = {}
        worker_weights: list[tuple[list[np.ndarray], int]] = []
        for worker_index, (train_images, train_labels, _, _) in enumerate(per_worker):
            history = models[worker_index].fit(
                train_images,
                train_labels,
                epochs=local_epochs,
                batch_size=batch_size,
                shuffle=True,
                verbose=0,
            )
            worker_id = WORKER_IDS[worker_index]
            local_train_accuracies[worker_id] = float(history.history["accuracy"][-1])
            worker_weights.append(
                ([np.asarray(weight) for weight in models[worker_index].get_weights()], int(train_labels.size))
            )
        averaged = _weighted_average(worker_weights)
        for model in models:
            model.set_weights(averaged)
        validation_accuracies = {}
        for worker_index, (_, _, validation_images, validation_labels) in enumerate(per_worker):
            logits = models[worker_index].predict(
                validation_images,
                batch_size=min(batch_size, int(validation_labels.size)),
                verbose=0,
            )
            validation_accuracies[WORKER_IDS[worker_index]] = float(
                np.mean(np.asarray(logits).argmax(axis=1) == validation_labels)
            )
        row = {
            "round": round_index,
            "local_train_accuracy": local_train_accuracies,
            "validation_accuracy_after_average": validation_accuracies,
            "mean_validation_accuracy": float(np.mean(list(validation_accuracies.values()))),
        }
        round_results.append(row)
        print(
            f"federated round={round_index}; mean_val_acc={row['mean_validation_accuracy']:.4f}; "
            f"workers={json.dumps(validation_accuracies, sort_keys=True)}"
        )

    final_worker_metrics: dict[str, dict[str, float]] = {}
    for worker_index, (_, _, validation_images, validation_labels) in enumerate(per_worker):
        logits = models[worker_index].predict(
            validation_images,
            batch_size=min(batch_size, int(validation_labels.size)),
            verbose=0,
        )
        predictions = np.asarray(logits).argmax(axis=1)
        confusion = np.zeros((10, 10), dtype=np.int64)
        np.add.at(confusion, (validation_labels, predictions), 1)
        true_positive = np.diag(confusion).astype(np.float64)
        denominator = confusion.sum(axis=0) + confusion.sum(axis=1)
        per_class_f1 = np.divide(
            2.0 * true_positive,
            denominator,
            out=np.zeros_like(true_positive),
            where=denominator > 0,
        )
        final_worker_metrics[WORKER_IDS[worker_index]] = {
            "accuracy": float(np.mean(predictions == validation_labels)),
            "macro_f1": float(per_class_f1.mean()),
        }
    return {
        "seed": seed,
        "model_kind": model_kind,
        "architecture": (
            initial_architecture if model_kind == "initial_model_json" else WORKER_ARCHITECTURE
        ),
        "workers": 3,
        "train_rows_per_worker": [int(item[1].size) for item in per_worker],
        "rounds": rounds,
        "local_epochs_per_round": local_epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "weight_averaging": "sample-count-weighted, after each round",
        "rounds_detail": round_results,
        "final_worker_metrics": final_worker_metrics,
        "mean_final_accuracy": float(np.mean([item["accuracy"] for item in final_worker_metrics.values()])),
        "mean_final_macro_f1": float(np.mean([item["macro_f1"] for item in final_worker_metrics.values()])),
    }


def main() -> None:
    train_images, train_labels, validation_images, validation_labels, per_worker = _dataset()
    architecture_path = MODEL_ROOT / "architecture.json"
    weights_path = MODEL_ROOT / "initial_weights.npz"
    if not architecture_path.is_file() or not weights_path.is_file():
        raise FileNotFoundError("PC initial-model architecture or weights are missing")
    architecture = json.loads(architecture_path.read_text(encoding="utf-8"))
    if architecture.get("input_shape") != [28, 28, 1] or architecture.get("num_classes") != 10:
        raise RuntimeError("Initial architecture JSON does not describe 1x28x28 MNIST")
    with np.load(weights_path, allow_pickle=False) as weights:
        initial_weights = [
            np.asarray(weights["conv1_kernel"]),
            np.asarray(weights["conv1_bias"]),
            np.asarray(weights["conv2_kernel"]),
            np.asarray(weights["conv2_bias"]),
            np.asarray(weights["classifier_kernel"]),
            np.asarray(weights["classifier_bias"]),
        ]
    print(
        "worker_training_config: optimizer=Adam; learning_rate=0.001; "
        "loss=SparseCategoricalCrossentropy(from_logits=True); batch_size=128; "
        "weight_averaging=sample-count-weighted across workers; "
        "worker_dataset_training_loads_installed_initial_weights=False"
    )
    results = {
        "dataset_name": "mnist_demo",
        "split_seed": SPLIT_SEED,
        "training_seeds": list(TRAINING_SEEDS),
        "train_images_shape": list(train_images.shape),
        "validation_images_shape": list(validation_images.shape),
        "train_labels_dtype": str(train_labels.dtype),
        "validation_labels_dtype": str(validation_labels.dtype),
        "initial_architecture_json": architecture,
        "worker_training_architecture": WORKER_ARCHITECTURE,
        "models": {
            "initial_architecture_json_with_installed_weights": [],
            "worker_builder_b1_c8_k3_fresh_weights": [],
        },
    }

    for seed in TRAINING_SEEDS:
        tf.keras.utils.set_random_seed(seed)
        json_model = _build_from_initial_architecture(architecture)
        json_model(np.zeros((1, 28, 28, 1), dtype=np.float32), training=False)
        json_model.set_weights([np.array(weight, copy=True) for weight in initial_weights])
        results["models"]["initial_architecture_json_with_installed_weights"].append(
            _train(
                "initial_architecture_json_with_installed_weights",
                seed,
                json_model,
                train_images,
                train_labels,
                validation_images,
                validation_labels,
            )
        )

        tf.keras.utils.set_random_seed(seed)
        worker_model = _build_dataset_candidate(
            input_shape=(28, 28, 1),
            num_classes=10,
            architecture=WORKER_ARCHITECTURE,
            seed=seed,
        )
        results["models"]["worker_builder_b1_c8_k3_fresh_weights"].append(
            _train(
                "worker_builder_b1_c8_k3_fresh_weights",
                seed,
                worker_model,
                train_images,
                train_labels,
                validation_images,
                validation_labels,
            )
        )

    tf.keras.utils.set_random_seed(TRAINING_SEEDS[-1])
    results["federated_worker_schedule"] = _simulate_worker_schedule(
        per_worker,
        seed=TRAINING_SEEDS[-1],
    )
    results["initial_weights_loaded_by_worker_path"] = False
    OUTPUT_PATH.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"pc_training_json={OUTPUT_PATH}")


if __name__ == "__main__":
    main()
