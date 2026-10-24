"""Persistent system-Python evaluator for held-out CIFAR-10 inference metrics."""

from __future__ import annotations

import base64
import io
import json
import resource
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.cifar_model import (
    build_cifar10_model,
    classification_metrics,
    compile_cifar10_model,
)


def _load_weights(value: str) -> list[np.ndarray]:
    payload = base64.b64decode(value, validate=True)
    with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
        keys = sorted(archive.files, key=lambda key: int(key.removeprefix("weight_")))
        if keys != [f"weight_{index}" for index in range(4)]:
            raise ValueError("Model weights have an unexpected structure")
        return [np.asarray(archive[key]) for key in keys]


def evaluate(test_path: Path, train_path: Path) -> None:
    with np.load(test_path, allow_pickle=False) as dataset:
        images = np.asarray(dataset["images"], dtype=np.uint8)
        labels = np.asarray(dataset["labels"], dtype=np.int64)
    if images.shape != (10_000, 32, 32, 3) or labels.shape != (10_000,):
        raise ValueError("Held-out data must be the 10,000 CIFAR-10 test images")
    with np.load(train_path, allow_pickle=False) as dataset:
        train_images = np.asarray(dataset["images"], dtype=np.uint8)
        train_labels = np.asarray(dataset["labels"], dtype=np.int64)
    if train_images.shape != (50_000, 32, 32, 3) or train_labels.shape != (50_000,):
        raise ValueError("Centralized baseline data must contain all 50,000 training images")
    model = build_cifar10_model(seed=0)
    print(json.dumps({"ready": True, "test_samples": int(labels.size)}), flush=True)
    for line in sys.stdin:
        try:
            request = json.loads(line)
            if request.get("type") == "baseline":
                baseline_model = build_cifar10_model(seed=request["seed"])
                compile_cifar10_model(baseline_model)
                rows = []
                for round_index in range(request["rounds"]):
                    started = time.perf_counter()
                    history = baseline_model.fit(
                        train_images,
                        train_labels,
                        epochs=request["epochs"],
                        batch_size=128,
                        shuffle=True,
                        verbose=0,
                    )
                    logits = baseline_model.predict(images, batch_size=256, verbose=0)
                    rows.append(
                        {
                            "round": round_index + 1,
                            "training_loss": float(history.history["loss"][-1]),
                            "training_accuracy": float(history.history["accuracy"][-1]),
                            **classification_metrics(labels, logits),
                            "runtime_seconds": round(time.perf_counter() - started, 3),
                            "peak_process_rss_mb": round(
                                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
                                1,
                            ),
                        }
                    )
                print(json.dumps({"baseline_rounds": rows}, allow_nan=False), flush=True)
            else:
                models = request["models"]
                if not isinstance(models, dict) or not models:
                    raise ValueError("models must be a non-empty mapping")
                results: dict[str, Any] = {}
                for node_id, encoded_weights in models.items():
                    started = time.perf_counter()
                    model.set_weights(_load_weights(encoded_weights))
                    logits = model.predict(images, batch_size=256, verbose=0)
                    results[node_id] = {
                        **classification_metrics(labels, logits),
                        "evaluation_seconds": round(time.perf_counter() - started, 3),
                    }
                print(json.dumps({"results": results}, allow_nan=False), flush=True)
        except Exception as error:
            print(
                json.dumps({"error": f"{type(error).__name__}: {error}"}),
                flush=True,
            )


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("Usage: cifar_evaluator.py <held-out-test-npz> <baseline-train-npz>")
    evaluate(Path(sys.argv[1]), Path(sys.argv[2]))


if __name__ == "__main__":
    main()
