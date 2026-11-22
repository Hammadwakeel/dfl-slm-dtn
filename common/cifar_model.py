"""Shared small CIFAR-10 CNN definition and classification metrics."""

from __future__ import annotations

from typing import Any

import numpy as np

from common.nas import FIXED_ARCHITECTURE, MAX_PARAMETERS, architecture_spec


def build_cifar10_model(
    seed: int = 0,
    architecture: str | dict[str, Any] = FIXED_ARCHITECTURE,
) -> Any:
    import tensorflow as tf

    spec = architecture_spec(architecture)
    tf.keras.utils.set_random_seed(seed)
    for gpu in tf.config.list_physical_devices("GPU"):
        if not tf.config.experimental.get_memory_growth(gpu):
            tf.config.experimental.set_memory_growth(gpu, True)
    layers = [
        tf.keras.layers.Input(shape=(32, 32, 3), dtype=tf.uint8),
        tf.keras.layers.RandomFlip("horizontal", seed=seed),
        tf.keras.layers.Rescaling(1.0 / 255.0),
    ]
    for block in range(spec["blocks"]):
        layers.append(
            tf.keras.layers.Conv2D(
                spec["channels"],
                kernel_size=spec["kernel_size"],
                strides=2 if block == 0 else 1,
                padding="same",
                activation="relu",
            )
        )
        layers.append(tf.keras.layers.MaxPooling2D(pool_size=2))
    layers.extend([tf.keras.layers.Flatten(), tf.keras.layers.Dense(10)])
    model = tf.keras.Sequential(layers, name=f"cifar10_{spec['id']}")
    if model.count_params() > MAX_PARAMETERS:
        raise ValueError("Architecture exceeds the CIFAR-10 parameter cap")
    return model


def compile_cifar10_model(model: Any) -> None:
    import tensorflow as tf

    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )


def model_size_bytes(model: Any) -> int:
    return int(sum(np.asarray(weight).nbytes for weight in model.get_weights()))


def classification_metrics(labels: np.ndarray, logits: np.ndarray) -> dict[str, Any]:
    truth = np.asarray(labels, dtype=np.int64).reshape(-1)
    predictions = np.asarray(logits)
    if predictions.ndim != 2 or predictions.shape[0] != truth.shape[0]:
        raise ValueError("logits must have shape (number of labels, number of classes)")
    num_classes = predictions.shape[1]
    if truth.size == 0 or num_classes < 2:
        raise ValueError("classification metrics require non-empty data with at least two classes")
    if truth.min() < 0 or truth.max() >= predictions.shape[1]:
        raise ValueError("labels must be within the model's class range")
    predicted = predictions.argmax(axis=1)
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(confusion, (truth, predicted), 1)
    true_positive = np.diag(confusion).astype(np.float64)
    predicted_count = confusion.sum(axis=0)
    actual_count = confusion.sum(axis=1)
    per_class_precision = np.divide(
        true_positive,
        predicted_count,
        out=np.zeros_like(true_positive),
        where=predicted_count > 0,
    )
    per_class_recall = np.divide(
        true_positive,
        actual_count,
        out=np.zeros_like(true_positive),
        where=actual_count > 0,
    )
    denominator = predicted_count + actual_count
    per_class_f1 = np.divide(
        2.0 * true_positive,
        denominator,
        out=np.zeros_like(true_positive),
        where=denominator > 0,
    )
    row_normalized = np.divide(
        confusion * 100.0,
        actual_count[:, np.newaxis],
        out=np.zeros_like(confusion, dtype=np.float64),
        where=actual_count[:, np.newaxis] > 0,
    )
    shifted = predictions - predictions.max(axis=1, keepdims=True)
    log_partition = np.log(np.exp(shifted).sum(axis=1))
    cross_entropy = log_partition - shifted[np.arange(truth.size), truth]
    return {
        "accuracy": float(np.mean(predicted == truth)),
        "loss": float(np.mean(cross_entropy)),
        "macro_precision": float(np.mean(per_class_precision)),
        "macro_recall": float(np.mean(per_class_recall)),
        "macro_f1": float(np.mean(per_class_f1)),
        "per_class_precision": per_class_precision.tolist(),
        "per_class_recall": per_class_recall.tolist(),
        "per_class_f1": per_class_f1.tolist(),
        "confusion_matrix_counts": confusion.tolist(),
        "confusion_matrix_row_normalized_percent": row_normalized.tolist(),
    }
