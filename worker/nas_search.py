"""Worker-local, GPU-backed CNN candidate evaluation using shard-only validation data."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from common.cifar_model import build_cifar10_model, classification_metrics, compile_cifar10_model
from common.nas import ARCHITECTURE_SPACE, architecture_parameters


MAX_SEARCH_SECONDS = 300
MAX_SEARCH_RSS_MB = 2048
VALIDATION_FRACTION = 0.2


def _rss_mb() -> float:
    with open("/proc/self/status", encoding="ascii") as stream:
        for line in stream:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024
    raise RuntimeError("VmRSS is missing from /proc/self/status")


def _split_indices(labels: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    training: list[int] = []
    validation: list[int] = []
    for label in np.unique(labels):
        indices = np.flatnonzero(labels == label)
        rng.shuffle(indices)
        validation_count = max(1, int(round(indices.size * VALIDATION_FRACTION)))
        validation.extend(indices[:validation_count].tolist())
        training.extend(indices[validation_count:].tolist())
    return np.asarray(training, dtype=np.int64), np.asarray(validation, dtype=np.int64)


def search_cifar10_architectures(
    images: np.ndarray,
    labels: np.ndarray,
    *,
    seed: int,
    time_limit_seconds: int,
    node_id: str,
    logger: Any,
) -> dict[str, Any]:
    if not 1 <= time_limit_seconds <= MAX_SEARCH_SECONDS:
        raise ValueError(f"Search time limit must be between 1 and {MAX_SEARCH_SECONDS} seconds")
    import tensorflow as tf

    devices = tf.config.list_physical_devices("GPU")
    if not devices:
        raise RuntimeError("NAS search requires an enumerated TensorFlow GPU")
    train_indices, validation_indices = _split_indices(labels, seed)
    started = time.monotonic()
    deadline = started + time_limit_seconds
    candidates = []
    for spec in ARCHITECTURE_SPACE:
        if time.monotonic() >= deadline:
            raise TimeoutError("NAS search time budget expired before all candidates were evaluated")
        parameter_count = architecture_parameters(spec)
        if parameter_count > 12_000:
            raise ValueError(f"Candidate {spec['id']} exceeds the architecture parameter cap")
        candidate_started = time.perf_counter()
        model = build_cifar10_model(seed=seed, architecture=spec)
        compile_cifar10_model(model)
        exceeded: list[str] = []
        peak_rss_mb = [_rss_mb()]

        class SearchBudget(tf.keras.callbacks.Callback):
            def on_train_batch_end(self, batch: int, logs: Any = None) -> None:
                rss_mb = _rss_mb()
                peak_rss_mb[0] = max(peak_rss_mb[0], rss_mb)
                if time.monotonic() >= deadline:
                    exceeded.append("time")
                    self.model.stop_training = True
                elif rss_mb > MAX_SEARCH_RSS_MB:
                    exceeded.append("memory")
                    self.model.stop_training = True

        model.fit(
            images[train_indices],
            labels[train_indices],
            epochs=1,
            batch_size=128,
            shuffle=True,
            verbose=0,
            callbacks=[SearchBudget()],
        )
        peak_rss_mb[0] = max(peak_rss_mb[0], _rss_mb())
        if exceeded:
            reason = exceeded[-1]
            raise TimeoutError(
                f"NAS candidate {spec['id']} exceeded the {reason} budget "
                f"(RSS {peak_rss_mb[0]:.1f} MiB)"
            )
        if peak_rss_mb[0] > MAX_SEARCH_RSS_MB:
            raise MemoryError(
                f"NAS candidate {spec['id']} exceeded the {MAX_SEARCH_RSS_MB} MiB RSS cap"
            )
        logits = model.predict(images[validation_indices], batch_size=256, verbose=0)
        metrics = classification_metrics(labels[validation_indices], logits)
        peak_rss_mb[0] = max(peak_rss_mb[0], _rss_mb())
        candidate_peak_rss_mb = round(peak_rss_mb[0], 1)
        elapsed_seconds = round(time.perf_counter() - candidate_started, 3)
        if time.monotonic() >= deadline:
            raise TimeoutError(f"NAS candidate {spec['id']} exceeded the time budget")
        if candidate_peak_rss_mb > MAX_SEARCH_RSS_MB:
            raise MemoryError(
                f"NAS candidate {spec['id']} exceeded the {MAX_SEARCH_RSS_MB} MiB RSS cap"
            )
        result = {
            "architecture": dict(spec),
            "parameter_count": parameter_count,
            "validation_samples": int(validation_indices.size),
            "validation_accuracy": metrics["accuracy"],
            "elapsed_seconds": elapsed_seconds,
            "peak_rss_mb": candidate_peak_rss_mb,
        }
        candidates.append(result)
        logger(
            "nas_candidate_evaluated",
            architecture=spec,
            parameter_count=parameter_count,
            validation_accuracy=metrics["accuracy"],
            elapsed_seconds=elapsed_seconds,
            peak_rss_mb=candidate_peak_rss_mb,
            validation_samples=int(validation_indices.size),
        )
        del model
        tf.keras.backend.clear_session()
    return {
        "node_id": node_id,
        "seed": seed,
        "gpu": [device.name for device in devices],
        "validation_samples": int(validation_indices.size),
        "training_samples": int(train_indices.size),
        "time_limit_seconds": time_limit_seconds,
        "memory_limit_mb": MAX_SEARCH_RSS_MB,
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "peak_rss_mb": round(max(row["peak_rss_mb"] for row in candidates), 1),
        "candidates": candidates,
    }


def search_cifar10_architectures_isolated(
    data_path: str,
    *,
    seed: int,
    time_limit_seconds: int,
    node_id: str,
    logger: Any,
) -> dict[str, Any]:
    if not 1 <= time_limit_seconds <= MAX_SEARCH_SECONDS:
        raise ValueError(f"Search time limit must be between 1 and {MAX_SEARCH_SECONDS} seconds")
    environment = os.environ.copy()
    for name in (
        "DFL_API_TOKEN",
        "DFL_PEERS_JSON_B64",
        "DFL_TRAINING_SHARDS_JSON_B64",
    ):
        environment.pop(name, None)
    environment["PYTHONUNBUFFERED"] = "1"
    command = [
        sys.executable,
        "-u",
        "-m",
        "worker.nas_search",
        data_path,
        node_id,
        str(seed),
        str(time_limit_seconds),
    ]
    try:
        completed = subprocess.run(
            command,
            cwd=Path(__file__).resolve().parents[1],
            env=environment,
            capture_output=True,
            text=True,
            timeout=time_limit_seconds + 30,
            check=False,
            encoding="utf-8",
        )
    except subprocess.TimeoutExpired as error:
        _forward_search_events(error.stdout, logger)
        raise TimeoutError("Isolated NAS search exceeded its process deadline") from error
    result = _forward_search_events(completed.stdout, logger)
    if completed.returncode != 0:
        detail = completed.stderr.strip()[-1000:]
        raise RuntimeError(
            f"Isolated NAS search exited with code {completed.returncode}: {detail}"
        )
    if result is None or not isinstance(result, dict):
        raise RuntimeError("Isolated NAS search returned no result object")
    return result


def _forward_search_events(output: str | bytes | None, logger: Any) -> dict[str, Any] | None:
    if output is None:
        return None
    if isinstance(output, bytes):
        output = output.decode("utf-8", errors="replace")
    result = None
    for line in output.splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, dict):
            continue
        if event.get("type") == "nas_candidate_evaluated":
            details = dict(event)
            details.pop("type", None)
            logger("nas_candidate_evaluated", **details)
        elif event.get("type") == "nas_search_result":
            result = event.get("data")
    return result


def main() -> None:
    if len(sys.argv) != 5:
        raise SystemExit(
            "Usage: python -m worker.nas_search <shard.npz> <node_id> <seed> <time_limit>"
        )
    path, node_id, seed_value, limit_value = sys.argv[1:]
    with np.load(path, allow_pickle=False) as shard:
        images = np.asarray(shard["images"], dtype=np.uint8)
        labels = np.asarray(shard["labels"], dtype=np.int64).reshape(-1)
    result = search_cifar10_architectures(
        images,
        labels,
        seed=int(seed_value),
        time_limit_seconds=int(limit_value),
        node_id=node_id,
        logger=lambda event, **details: print(
            json.dumps({"type": event, **details}, allow_nan=False),
            flush=True,
        ),
    )
    print(json.dumps({"type": "nas_search_result", "data": result}, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
