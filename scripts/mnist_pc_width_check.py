"""PC-only width test for the MNIST global-average CNN under the 12k cap."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import tensorflow as tf

from mnist_pc_training_check import _dataset

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = ROOT / "data" / "mnist_pc_width_check.json"
FILTER_COUNTS = (16, 32)
SEED = 2026
EPOCHS = 50
LEARNING_RATE = 0.01
BATCH_SIZE = 32


def _parameter_count(filters: int) -> int:
    conv1 = 3 * 3 * 1 * filters + filters
    conv2 = 3 * 3 * filters * filters + filters
    classifier = filters * 10 + 10
    return conv1 + conv2 + classifier


def _model(filters: int) -> tf.keras.Model:
    return tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(28, 28, 1), dtype=tf.float32),
            tf.keras.layers.Conv2D(filters, 3, padding="same", activation="relu"),
            tf.keras.layers.Conv2D(filters, 3, padding="same", activation="relu"),
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(10),
        ],
        name=f"mnist_cnn_c{filters}",
    )


def _macro_f1(labels: np.ndarray, logits: np.ndarray) -> float:
    predictions = np.asarray(logits).argmax(axis=1)
    confusion = np.zeros((10, 10), dtype=np.int64)
    np.add.at(confusion, (labels, predictions), 1)
    true_positive = np.diag(confusion).astype(np.float64)
    denominator = confusion.sum(axis=0) + confusion.sum(axis=1)
    values = np.divide(
        2.0 * true_positive,
        denominator,
        out=np.zeros_like(true_positive),
        where=denominator > 0,
    )
    return float(values.mean())


def main() -> None:
    train_images, train_labels, validation_images, validation_labels, _ = _dataset()
    results = {"seed": SEED, "epochs": EPOCHS, "learning_rate": LEARNING_RATE, "batch_size": BATCH_SIZE, "models": []}
    for filters in FILTER_COUNTS:
        parameters = _parameter_count(filters)
        if parameters > 12_000:
            raise RuntimeError(f"Width {filters} exceeds the 12,000-parameter cap: {parameters}")
        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(SEED)
        model = _model(filters)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=["accuracy"],
        )
        history = model.fit(
            train_images,
            train_labels,
            validation_data=(validation_images, validation_labels),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            shuffle=True,
            verbose=0,
        )
        logits = model.predict(validation_images, batch_size=BATCH_SIZE, verbose=0)
        record = {
            "filters": filters,
            "parameter_count": parameters,
            "train_accuracy_epoch_50": float(history.history["accuracy"][-1]),
            "validation_accuracy_epoch_50": float(history.history["val_accuracy"][-1]),
            "final_macro_f1": _macro_f1(validation_labels, logits),
            "accuracy_by_epoch": [float(value) for value in history.history["val_accuracy"]],
        }
        results["models"].append(record)
        print(
            f"filters={filters}; parameters={parameters}; "
            f"train_accuracy={record['train_accuracy_epoch_50']:.4f}; "
            f"validation_accuracy={record['validation_accuracy_epoch_50']:.4f}; "
            f"macro_f1={record['final_macro_f1']:.4f}"
        )
    OUTPUT_PATH.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"width_check_json={OUTPUT_PATH}")


if __name__ == "__main__":
    main()
