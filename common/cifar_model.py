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


def classification_metrics(labels: np.ndarray, logits: np.ndarray) -> dict[str, float]:
    truth = np.asarray(labels, dtype=np.int64).reshape(-1)
    predictions = np.asarray(logits)
    if predictions.ndim != 2 or predictions.shape[0] != truth.shape[0]:
        raise ValueError("logits must have shape (number of labels, number of classes)")
    if truth.size == 0 or predictions.shape[1] != 10:
        raise ValueError("CIFAR-10 metrics require non-empty data with ten classes")
    predicted = predictions.argmax(axis=1)
    confusion = np.zeros((10, 10), dtype=np.int64)
    np.add.at(confusion, (truth, predicted), 1)
    true_positive = np.diag(confusion).astype(np.float64)
    denominator = confusion.sum(axis=0) + confusion.sum(axis=1)
    per_class_f1 = np.divide(
        2.0 * true_positive,
        denominator,
        out=np.zeros_like(true_positive),
        where=denominator > 0,
    )
    shifted = predictions - predictions.max(axis=1, keepdims=True)
    log_partition = np.log(np.exp(shifted).sum(axis=1))
    cross_entropy = log_partition - shifted[np.arange(truth.size), truth]
    return {
        "accuracy": float(np.mean(predicted == truth)),
        "loss": float(np.mean(cross_entropy)),
        "macro_f1": float(np.mean(per_class_f1)),
    }
