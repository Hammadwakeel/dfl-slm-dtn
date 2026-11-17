"""Evaluate federated and centralized MNIST training on the untouched test split."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf


ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = ROOT / "data" / "datasets" / "mnist_demo"
MODEL_ROOT = DATASET_ROOT / "models" / "initial_model"
WORKER_IDS = ("worker1", "worker2", "worker3")
WORKER_ROWS = 640
ROUNDS = 5
LOCAL_EPOCHS = 20
LEARNING_RATE = 0.01
BATCH_SIZE = 32
TRAINING_SEEDS = (17, 42, 2026)
TRAIN_SELECTION_SEED = 2026
REFERENCE_HARDWARE_SECONDS = 10.811
REFERENCE_TRAIN_ROWS_PER_WORKER = 720


def _build_model(architecture: dict[str, Any]) -> tf.keras.Model:
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
            raise ValueError(f"Unsupported serialized layer type: {layer['type']}")
    model = tf.keras.Sequential(layers, name="mnist_installed_initial_model")
    model(np.zeros((1, *architecture["input_shape"]), dtype=np.float32), training=False)
    return model


def _compile(model: tf.keras.Model) -> None:
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    )


def _installed_initial_weights() -> tuple[dict[str, Any], list[np.ndarray]]:
    architecture_path = MODEL_ROOT / "architecture.json"
    weights_path = MODEL_ROOT / "initial_weights.npz"
    metadata_path = MODEL_ROOT / "metadata.json"
    architecture_bytes = architecture_path.read_bytes()
    weights_bytes = weights_path.read_bytes()
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if hashlib.sha256(architecture_bytes).hexdigest() != metadata["architecture_sha256"]:
        raise ValueError("Installed MNIST architecture hash does not match metadata")
    if hashlib.sha256(weights_bytes).hexdigest() != metadata["weights_sha256"]:
        raise ValueError("Installed MNIST weights hash does not match metadata")
    architecture = json.loads(architecture_bytes.decode("utf-8"))
    if architecture.get("input_shape") != [28, 28, 1] or architecture.get("num_classes") != 10:
        raise ValueError("Installed initial model is not a 28x28 grayscale, ten-class model")
    weight_names = (
        "conv1_kernel",
        "conv1_bias",
        "conv2_kernel",
        "conv2_bias",
        "classifier_kernel",
        "classifier_bias",
    )
    with np.load(weights_path, allow_pickle=False) as archive:
        if set(archive.files) != set(weight_names):
            raise ValueError("Installed MNIST initial-model weight names are unexpected")
        weights = [np.asarray(archive[name]).copy() for name in weight_names]
    return architecture, weights


def _load_worker_training_data() -> list[tuple[np.ndarray, np.ndarray]]:
    report_paths = sorted(
        (DATASET_ROOT / "processed").glob("run-*/preprocessing_report.json"),
        key=lambda path: path.stat().st_mtime_ns,
    )
    if not report_paths:
        raise FileNotFoundError("Persisted MNIST preprocessing report is missing")
    report_path = report_paths[-1]
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("dataset_name") != "mnist_demo" or report.get("split_sizes", {}).get("test") != 300:
        raise ValueError("Persisted MNIST split is not the expected 300-row test split")

    worker_data: list[tuple[np.ndarray, np.ndarray]] = []
    for worker_id in WORKER_IDS:
        shard_path = report_path.parent / "workers" / f"{worker_id}.npz"
        with np.load(shard_path, allow_pickle=False) as shard:
            images = np.asarray(shard["images"], dtype=np.float32)
            labels = np.asarray(shard["labels"], dtype=np.int64).reshape(-1)
        if images.shape != (800, 28, 28, 1) or labels.shape != (800,):
            raise ValueError(f"Unexpected persisted training shard shape for {worker_id}")
        worker_data.append((images, labels))
    return worker_data


def _stratified_subset(
    images: np.ndarray,
    labels: np.ndarray,
    *,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    classes = np.unique(labels)
    if classes.size != 10 or WORKER_ROWS % classes.size:
        raise ValueError("Worker shard must contain ten classes for a balanced 640-row subset")
    rows_per_class = WORKER_ROWS // classes.size
    rng = np.random.default_rng(seed)
    selected: list[int] = []
    for label in classes:
        candidates = np.flatnonzero(labels == label)
        if candidates.size < rows_per_class:
            raise ValueError(f"Worker shard has too few training rows for class {label}")
        selected.extend(rng.choice(candidates, size=rows_per_class, replace=False).tolist())
    rng.shuffle(selected)
    indices = np.asarray(selected, dtype=np.int64)
    return images[indices], labels[indices]


def _evaluate(model: tf.keras.Model, images: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    logits = model.predict(images, batch_size=BATCH_SIZE, verbose=0)
    predictions = np.asarray(logits).argmax(axis=1)
    confusion = np.zeros((10, 10), dtype=np.int64)
    np.add.at(confusion, (labels, predictions), 1)
    true_positive = np.diag(confusion).astype(np.float64)
    denominator = confusion.sum(axis=0) + confusion.sum(axis=1)
    per_class_f1 = np.divide(
        2.0 * true_positive,
        denominator,
        out=np.zeros_like(true_positive),
        where=denominator > 0,
    )
    return {
        "accuracy": float(np.mean(predictions == labels)),
        "macro_f1": float(per_class_f1.mean()),
    }


def _weighted_average(worker_weights: list[list[np.ndarray]]) -> list[np.ndarray]:
    reference = worker_weights[0]
    return [
        (
            sum(weights[index].astype(np.float64) * WORKER_ROWS for weights in worker_weights)
            / (len(worker_weights) * WORKER_ROWS)
        ).astype(reference[index].dtype)
        for index in range(len(reference))
    ]


def _train_federated_seed(
    architecture: dict[str, Any],
    initial_weights: list[np.ndarray],
    worker_data: list[tuple[np.ndarray, np.ndarray]],
    seed: int,
) -> tf.keras.Model:
    tf.keras.utils.set_random_seed(seed)
    selected_data = [
        _stratified_subset(images, labels, seed=TRAIN_SELECTION_SEED + index)
        for index, (images, labels) in enumerate(worker_data)
    ]
    models: list[tf.keras.Model] = []
    for _ in WORKER_IDS:
        model = _build_model(architecture)
        model.set_weights([weight.copy() for weight in initial_weights])
        _compile(model)
        models.append(model)

    for round_index in range(ROUNDS):
        for model, (images, labels) in zip(models, selected_data):
            model.fit(
                images,
                labels,
                epochs=LOCAL_EPOCHS,
                batch_size=BATCH_SIZE,
                shuffle=True,
                verbose=0,
            )
        averaged_weights = _weighted_average([model.get_weights() for model in models])
        for model in models:
            model.set_weights(averaged_weights)
        print(f"seed={seed}; federated_round={round_index + 1}/{ROUNDS}; workers=3; rows_per_worker=640")
    return models[0]


def _train_centralized_seed(
    architecture: dict[str, Any],
    initial_weights: list[np.ndarray],
    worker_data: list[tuple[np.ndarray, np.ndarray]],
    seed: int,
) -> tf.keras.Model:
    images = np.concatenate(
        [_stratified_subset(x, y, seed=TRAIN_SELECTION_SEED + index)[0] for index, (x, y) in enumerate(worker_data)]
    )
    labels = np.concatenate(
        [_stratified_subset(x, y, seed=TRAIN_SELECTION_SEED + index)[1] for index, (x, y) in enumerate(worker_data)]
    )
    tf.keras.utils.set_random_seed(seed)
    model = _build_model(architecture)
    model.set_weights([weight.copy() for weight in initial_weights])
    _compile(model)
    for round_index in range(ROUNDS):
        model.fit(
            images,
            labels,
            epochs=LOCAL_EPOCHS,
            batch_size=BATCH_SIZE,
            shuffle=True,
            verbose=0,
        )
        print(f"seed={seed}; centralized_round={round_index + 1}/{ROUNDS}; rows=1920")
    return model


def _summary(values: list[float]) -> tuple[float, float]:
    return float(np.mean(values)), float(np.std(values, ddof=1))


def main() -> None:
    architecture, initial_weights = _installed_initial_weights()
    worker_data = _load_worker_training_data()
    federated_models: list[tf.keras.Model] = []
    centralized_models: list[tf.keras.Model] = []
    for seed in TRAINING_SEEDS:
        federated_models.append(
            _train_federated_seed(architecture, initial_weights, worker_data, seed)
        )
        centralized_models.append(
            _train_centralized_seed(architecture, initial_weights, worker_data, seed)
        )

    report_paths = sorted(
        (DATASET_ROOT / "processed").glob("run-*/preprocessing_report.json"),
        key=lambda path: path.stat().st_mtime_ns,
    )
    test_path = report_paths[-1].parent / "test.npz"
    with np.load(test_path, allow_pickle=False) as test_split:
        test_images = np.asarray(test_split["images"], dtype=np.float32)
        test_labels = np.asarray(test_split["labels"], dtype=np.int64).reshape(-1)
    if test_images.shape != (300, 28, 28, 1) or test_labels.shape != (300,):
        raise ValueError(f"Unexpected untouched MNIST test split shape: {test_images.shape}, {test_labels.shape}")

    federated_results = [_evaluate(model, test_images, test_labels) for model in federated_models]
    centralized_results = [_evaluate(model, test_images, test_labels) for model in centralized_models]
    for seed, result in zip(TRAINING_SEEDS, federated_results):
        print(
            f"federated_test seed={seed}; accuracy={result['accuracy']:.4f}; "
            f"macro_f1={result['macro_f1']:.4f}"
        )
    federated_accuracy = _summary([row["accuracy"] for row in federated_results])
    federated_macro_f1 = _summary([row["macro_f1"] for row in federated_results])
    print(
        "federated_test_summary (mean, sample_std; n=3): "
        f"accuracy={federated_accuracy[0]:.4f}, {federated_accuracy[1]:.4f}; "
        f"macro_f1={federated_macro_f1[0]:.4f}, {federated_macro_f1[1]:.4f}"
    )
    for seed, result in zip(TRAINING_SEEDS, centralized_results):
        print(
            f"centralized_test seed={seed}; accuracy={result['accuracy']:.4f}; "
            f"macro_f1={result['macro_f1']:.4f}"
        )
    centralized_accuracy = _summary([row["accuracy"] for row in centralized_results])
    centralized_macro_f1 = _summary([row["macro_f1"] for row in centralized_results])
    print(
        "centralized_test_summary (mean, sample_std; n=3): "
        f"accuracy={centralized_accuracy[0]:.4f}, {centralized_accuracy[1]:.4f}; "
        f"macro_f1={centralized_macro_f1[0]:.4f}, {centralized_macro_f1[1]:.4f}"
    )
    baseline_steps = ROUNDS * 2 * math.ceil(REFERENCE_TRAIN_ROWS_PER_WORKER / 128)
    target_steps = ROUNDS * LOCAL_EPOCHS * math.ceil(WORKER_ROWS / BATCH_SIZE)
    estimated_seconds = REFERENCE_HARDWARE_SECONDS * target_steps / baseline_steps
    print(
        "Jetson time estimate (not measured): "
        f"{estimated_seconds:.1f} s ({estimated_seconds / 60:.1f} min) for 5x20 at batch 32; "
        "assumes 720 training rows per worker in the earlier 5x2, batch-128 run and 640 "
        "rows per worker here, with runtime scaling linearly by total minibatch steps "
        "(60 baseline steps vs 2,000 target steps); fixed overhead is ignored."
    )
    print(
        "Evaluation note: validation was used for tuning; the untouched 300-image test split "
        "was used only for these reported numbers."
    )


if __name__ == "__main__":
    main()