"""Evaluate every worker's trained model on a shared, class-balanced split.

Worker-reported accuracy is measured on each worker's own (non-IID) validation
slice, so it overstates how well a model handles all classes. The coordinator
therefore also evaluates each worker's newest model on the dataset's global
validation split (used for decisions) and on the official test split (reported
only, never used to choose anything).

This module runs in a separate TensorFlow CPU environment:

    python -m common.global_evaluation <request.json>

The request is ``{"splits": {"validation": "...npz", "test": "...npz"}, "models":
[{"node_id", "architecture", "weights_b64"}, ...]}``; the result JSON is printed
to stdout.
"""

from __future__ import annotations

import base64
import io
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

EVALUATION_BATCH_SIZE = 256


def classification_metrics(labels: np.ndarray, predictions: np.ndarray, num_classes: int) -> dict[str, Any]:
    """Accuracy, macro-F1 and per-class recall from integer labels and predictions."""
    labels = np.asarray(labels, dtype=np.int64)
    predictions = np.asarray(predictions, dtype=np.int64)
    if labels.shape != predictions.shape or labels.size == 0:
        raise ValueError("Labels and predictions must be non-empty and the same length")
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(confusion, (labels, predictions), 1)
    true_positive = np.diag(confusion).astype(np.float64)
    support = confusion.sum(axis=1).astype(np.float64)
    predicted = confusion.sum(axis=0).astype(np.float64)
    recall = np.divide(true_positive, support, out=np.zeros(num_classes), where=support > 0)
    precision = np.divide(true_positive, predicted, out=np.zeros(num_classes), where=predicted > 0)
    denominator = precision + recall
    f1 = np.divide(2 * precision * recall, denominator, out=np.zeros(num_classes), where=denominator > 0)
    present = support > 0
    return {
        "accuracy": round(float(true_positive.sum() / labels.size), 6),
        "macro_f1": round(float(f1[present].mean()), 6),
        "per_class_recall": [round(float(value), 6) for value in recall],
        "samples": int(labels.size),
    }


def summarize(per_worker: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Mean, standard deviation and range of worker metrics."""
    rows = [row for row in per_worker.values() if isinstance(row, dict) and "accuracy" in row]
    if not rows:
        return {"workers_evaluated": 0}
    accuracy = np.asarray([row["accuracy"] for row in rows], dtype=np.float64)
    macro_f1 = np.asarray([row["macro_f1"] for row in rows], dtype=np.float64)
    return {
        "workers_evaluated": len(rows),
        "mean_accuracy": round(float(accuracy.mean()), 6),
        "std_accuracy": round(float(accuracy.std()), 6),
        "min_accuracy": round(float(accuracy.min()), 6),
        "max_accuracy": round(float(accuracy.max()), 6),
        "mean_macro_f1": round(float(macro_f1.mean()), 6),
    }


def _decode_weights(encoded: str) -> list[np.ndarray]:
    with np.load(io.BytesIO(base64.b64decode(encoded, validate=True)), allow_pickle=False) as archive:
        keys = sorted(archive.files, key=lambda key: int(key.removeprefix("weight_")))
        return [np.asarray(archive[key]) for key in keys]


def _load_split(path: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    with np.load(path, allow_pickle=False) as split:
        return (
            np.asarray(split["images"], dtype=np.float32),
            np.asarray(split["labels"], dtype=np.int64),
            [str(value) for value in split["class_names"].tolist()],
        )


def evaluate_models(split_paths: dict[str, str], models: list[dict[str, Any]]) -> dict[str, Any]:
    """Evaluate each model once per split; returns {split: {per_worker, summary...}}."""
    import tensorflow as tf

    from worker.model_factory import build_dataset_model

    splits = {name: _load_split(path) for name, path in split_paths.items()}
    class_names = next(iter(splits.values()))[2]
    if any(item[2] != class_names for item in splits.values()):
        raise ValueError("Evaluation splits declare different classes")
    input_shape = tuple(int(value) for value in next(iter(splits.values()))[0].shape[1:])
    per_split: dict[str, dict[str, dict[str, Any]]] = {name: {} for name in splits}
    for item in models:
        node_id = str(item["node_id"])
        try:
            model = build_dataset_model(
                input_shape=input_shape,
                num_classes=len(class_names),
                architecture=item["architecture"],
                seed=0,
            )
            model.set_weights(_decode_weights(str(item["weights_b64"])))
            for name, (images, labels, _) in splits.items():
                logits = model.predict(images, batch_size=EVALUATION_BATCH_SIZE, verbose=0)
                per_split[name][node_id] = classification_metrics(labels, np.argmax(logits, axis=1), len(class_names))
        except Exception as error:  # noqa: BLE001 - one bad model must not hide the others
            for name in splits:
                per_split[name][node_id] = {"error": f"{type(error).__name__}: {error}"[:300]}
        finally:
            tf.keras.backend.clear_session()
    return {
        "classes": class_names,
        **{name: {"per_worker": rows, **summarize(rows)} for name, rows in per_split.items()},
    }


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: python -m common.global_evaluation <request.json>", file=sys.stderr)
        return 2
    request = json.loads(Path(argv[1]).read_text(encoding="utf-8"))
    result = evaluate_models(dict(request["splits"]), list(request["models"]))
    sys.stdout.write(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
