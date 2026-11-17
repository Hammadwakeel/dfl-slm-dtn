"""Worker-local, GPU-backed CNN candidate evaluation using shard-only validation data."""

from __future__ import annotations

import gc
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

from common.cifar_model import build_cifar10_model, classification_metrics, compile_cifar10_model
from common.dataset_nas import MAX_DATASET_NAS_PARAMETERS
from common.model_design import validate_model_design
from common.nas import (
    ARCHITECTURE_SPACE,
    architecture_parameters,
)
from worker.model_factory import build_dataset_model


MAX_SEARCH_SECONDS = 300
MAX_SEARCH_RSS_MB = 3072
MIN_SEARCH_AVAILABLE_MEMORY_MB = 1024
DATASET_NAS_TRAIN_BATCH_SIZE = 8
DATASET_NAS_SHUFFLE_BUFFER_SIZE = 32
DATASET_NAS_VALIDATION_BATCH_SIZE = 16
VALIDATION_FRACTION = 0.2


def dataset_nas_input_pipeline_limits(
    training_samples: int,
    validation_samples: int,
) -> tuple[int, int, int]:
    if training_samples < 1 or validation_samples < 1:
        raise ValueError("Dataset NAS requires non-empty training and validation splits")
    return (
        min(DATASET_NAS_TRAIN_BATCH_SIZE, training_samples),
        min(DATASET_NAS_SHUFFLE_BUFFER_SIZE, training_samples),
        min(DATASET_NAS_VALIDATION_BATCH_SIZE, validation_samples),
    )


def _rss_mb() -> float:
    with open("/proc/self/status", encoding="ascii") as stream:
        for line in stream:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024
    raise RuntimeError("VmRSS is missing from /proc/self/status")


def _available_system_memory_mb() -> float:
    with open("/proc/meminfo", encoding="ascii") as stream:
        for line in stream:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 1024
    raise RuntimeError("MemAvailable is missing from /proc/meminfo")


def _enable_gpu_memory_growth(tf: Any, devices: list[Any]) -> None:
    for device in devices:
        if not tf.config.experimental.get_memory_growth(device):
            tf.config.experimental.set_memory_growth(device, True)


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


def load_dataset_shard(path: str | Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    try:
        with np.load(path, allow_pickle=False) as shard:
            images = np.asarray(shard["images"])
            labels = np.asarray(shard["labels"], dtype=np.int64).reshape(-1)
            raw_classes = np.asarray(shard["class_names"])
    except (OSError, ValueError, KeyError) as error:
        raise ValueError(f"Dataset NAS shard is invalid: {error}") from error
    if (
        images.ndim != 4
        or images.shape[0] == 0
        or images.shape[1] < 1
        or images.shape[1] > 2048
        or images.shape[2] < 1
        or images.shape[2] > 2048
        or images.shape[-1] not in {1, 3}
        or images.dtype.kind not in {"u", "i", "f"}
        or not np.isfinite(images).all()
    ):
        raise ValueError("Dataset NAS images must be finite N x H x W x C numeric data, C=1 or C=3")
    if (
        labels.ndim != 1
        or labels.size != images.shape[0]
        or labels.size == 0
        or labels.min() < 0
        or raw_classes.ndim != 1
        or raw_classes.dtype.kind not in {"U", "S"}
    ):
        raise ValueError("Dataset NAS labels or class names are invalid")
    classes = (
        [value.decode("utf-8") for value in raw_classes.tolist()]
        if raw_classes.dtype.kind == "S"
        else [str(value) for value in raw_classes.tolist()]
    )
    if (
        len(classes) < 2
        or any(not name for name in classes)
        or len(set(classes)) != len(classes)
        or labels.max() >= len(classes)
    ):
        raise ValueError("Dataset NAS class names do not match the label values")
    return images, labels, classes


def _build_dataset_candidate(
    *,
    input_shape: tuple[int, int, int],
    num_classes: int,
    architecture: dict[str, Any],
    seed: int,
) -> Any:
    import tensorflow as tf

    tf.keras.utils.set_random_seed(seed)
    layers: list[Any] = [tf.keras.layers.Input(shape=input_shape, dtype=tf.float32)]
    for block in range(architecture["blocks"]):
        layers.append(
            tf.keras.layers.Conv2D(
                architecture["channels"],
                kernel_size=architecture["kernel_size"],
                strides=2 if block == 0 else 1,
                padding="same",
                activation="relu",
            )
        )
        layers.append(tf.keras.layers.MaxPooling2D(pool_size=2, padding="same"))
    layers.extend(
        [
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(num_classes),
        ]
    )
    return tf.keras.Sequential(layers, name=f"dataset_{architecture['id']}")


def search_dataset_architectures(
    images: np.ndarray,
    labels: np.ndarray,
    class_names: list[str],
    *,
    seed: int,
    time_limit_seconds: int,
    node_id: str,
    logger: Any,
    architectures: list[dict[str, Any]] | None = None,
    epochs: int = 1,
    cancel_event: threading.Event | None = None,
    reuse_tensorflow_runtime: bool = False,
    memory_constrained: bool = False,
) -> dict[str, Any]:
    if not 30 <= time_limit_seconds <= MAX_SEARCH_SECONDS:
        raise ValueError(f"Search time limit must be between 30 and {MAX_SEARCH_SECONDS} seconds")
    if not isinstance(seed, int) or not 0 <= seed <= 2**32 - 1:
        raise ValueError("Dataset NAS seed is out of range")
    if (
        images.ndim != 4
        or images.shape[0] != labels.size
        or images.shape[-1] not in {1, 3}
        or not np.isfinite(images).all()
        or labels.ndim != 1
        or labels.size == 0
        or labels.min() < 0
        or labels.max() >= len(class_names)
    ):
        raise ValueError("Dataset NAS inputs do not match the selected dataset")
    if not isinstance(epochs, int) or isinstance(epochs, bool) or not 1 <= epochs <= 10:
        raise ValueError("Dataset NAS fidelity must be between 1 and 10 epochs")
    if architectures is None:
        from common.dataset_nas import (
            CIFAR_INITIAL_POPULATION_SIZE,
            INITIAL_POPULATION_SIZE,
            initial_population,
        )

        candidates_to_run = initial_population(
            seed=seed,
            input_shape=(
                int(images.shape[1]),
                int(images.shape[2]),
                int(images.shape[3]),
            ),
            num_classes=len(class_names),
            size=(
                CIFAR_INITIAL_POPULATION_SIZE
                if memory_constrained
                else INITIAL_POPULATION_SIZE
            ),
            memory_constrained=memory_constrained,
        )
    else:
        if not isinstance(architectures, list) or not 1 <= len(architectures) <= 8:
            raise ValueError("Dataset NAS requires between one and eight architectures per stage")
        candidates_to_run = architectures
    if any(
        not isinstance(candidate, dict)
        or not isinstance(candidate.get("id"), str)
        or not candidate["id"]
        for candidate in candidates_to_run
    ):
        raise ValueError("Dataset NAS architectures must have a non-empty string ID")
    candidate_ids = [candidate.get("id") for candidate in candidates_to_run]
    if len(set(candidate_ids)) != len(candidate_ids):
        raise ValueError("Dataset NAS candidate IDs must be unique within a stage")
    available_memory_mb = _available_system_memory_mb()
    if available_memory_mb < MIN_SEARCH_AVAILABLE_MEMORY_MB:
        raise MemoryError(
            "Dataset NAS requires at least "
            f"{MIN_SEARCH_AVAILABLE_MEMORY_MB} MiB available system memory; "
            f"only {available_memory_mb:.1f} MiB is available"
        )
    import tensorflow as tf

    devices = tf.config.list_physical_devices("GPU")
    if not devices:
        raise RuntimeError("NAS search requires an enumerated TensorFlow GPU")
    if not reuse_tensorflow_runtime:
        _enable_gpu_memory_growth(tf, devices)
        tf.config.threading.set_intra_op_parallelism_threads(1)
        tf.config.threading.set_inter_op_parallelism_threads(1)
    train_indices, validation_indices = _split_indices(labels, seed)
    started = time.monotonic()
    deadline = started + time_limit_seconds
    candidates = []
    train_batch_size, shuffle_buffer_size, validation_batch_size = (
        dataset_nas_input_pipeline_limits(
            int(train_indices.size),
            int(validation_indices.size),
        )
    )
    training_data = tf.data.Dataset.from_tensor_slices(
        (images[train_indices], labels[train_indices])
    )
    training_data = training_data.shuffle(
        buffer_size=shuffle_buffer_size,
        seed=seed,
        reshuffle_each_iteration=True,
    ).batch(train_batch_size)
    training_data = training_data.map(
        lambda batch_images, batch_labels: (
            tf.cast(batch_images, tf.float32),
            batch_labels,
        ),
        num_parallel_calls=1,
    )
    dataset_options = tf.data.Options()
    dataset_options.threading.private_threadpool_size = 1
    dataset_options.threading.max_intra_op_parallelism = 1
    training_data = training_data.with_options(dataset_options)
    for candidate in candidates_to_run:
        if cancel_event is not None and cancel_event.is_set():
            raise RuntimeError("Dataset NAS search was cancelled")
        if time.monotonic() >= deadline:
            raise TimeoutError("NAS search time budget expired before all candidates were evaluated")
        candidate_id = candidate.get("id", "<unknown>")
        candidate_started = time.perf_counter()
        try:
            normalized_architecture, parameter_count = validate_model_design(
                candidate,
                tuple(int(value) for value in images.shape[1:]),
                len(class_names),
            )
        except ValueError as error:
            raise ValueError(f"Dataset NAS architecture is invalid: {error}") from error
        if (
            candidate.get("model_type") != "custom"
            or candidate.get("parameter_count") != parameter_count
            or parameter_count > MAX_DATASET_NAS_PARAMETERS
        ):
            raise ValueError(
                f"Dataset NAS candidate {candidate.get('id', '<unknown>')} has an invalid "
                f"parameter count or exceeds the {MAX_DATASET_NAS_PARAMETERS:,}-parameter cap"
            )
        model = build_dataset_model(
            input_shape=tuple(int(value) for value in images.shape[1:]),
            num_classes=len(class_names),
            architecture=normalized_architecture,
            seed=seed,
        )
        model_summary_lines: list[str] = []
        model.summary(print_fn=model_summary_lines.append)
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=normalized_architecture["training"]["learning_rate"]
        )
        model.compile(
            optimizer=optimizer,
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=["accuracy"],
        )
        if _available_system_memory_mb() < MIN_SEARCH_AVAILABLE_MEMORY_MB:
            raise MemoryError(
                f"Dataset NAS candidate {candidate_id} stopped before training to preserve "
                f"{MIN_SEARCH_AVAILABLE_MEMORY_MB} MiB of available system memory"
            )
        if _rss_mb() > MAX_SEARCH_RSS_MB:
            raise MemoryError(
                f"Dataset NAS candidate {candidate_id} exceeded "
                f"the {MAX_SEARCH_RSS_MB} MiB RSS cap during model initialization"
            )
        exceeded: list[str] = []
        peak_rss_mb = [_rss_mb()]

        class SearchBudget(tf.keras.callbacks.Callback):
            def on_train_batch_end(self, batch: int, logs: Any = None) -> None:
                rss_mb = _rss_mb()
                peak_rss_mb[0] = max(peak_rss_mb[0], rss_mb)
                if time.monotonic() >= deadline:
                    exceeded.append("time")
                    self.model.stop_training = True
                elif cancel_event is not None and cancel_event.is_set():
                    exceeded.append("cancelled")
                    self.model.stop_training = True
                elif rss_mb > MAX_SEARCH_RSS_MB:
                    exceeded.append("process-memory")
                    self.model.stop_training = True
                elif _available_system_memory_mb() < MIN_SEARCH_AVAILABLE_MEMORY_MB:
                    exceeded.append("system-memory")
                    self.model.stop_training = True

        model.fit(
            training_data,
            epochs=epochs,
            verbose=0,
            callbacks=[SearchBudget()],
        )
        peak_rss_mb[0] = max(peak_rss_mb[0], _rss_mb())
        if exceeded:
            reason = exceeded[-1]
            if reason == "cancelled":
                raise RuntimeError("Dataset NAS search was cancelled")
            if reason == "time":
                raise TimeoutError(
                    f"Dataset NAS candidate {candidate_id} exceeded the time budget "
                    f"(RSS {peak_rss_mb[0]:.1f} MiB)"
                )
            if reason == "system-memory":
                raise MemoryError(
                    f"Dataset NAS candidate {candidate_id} stopped to preserve at least "
                    f"{MIN_SEARCH_AVAILABLE_MEMORY_MB} MiB of available system memory"
                )
            raise MemoryError(
                f"Dataset NAS candidate {candidate_id} exceeded the "
                f"{MAX_SEARCH_RSS_MB} MiB process RSS cap"
            )
        if cancel_event is not None and cancel_event.is_set():
            raise RuntimeError("Dataset NAS search was cancelled")
        validation_data = tf.data.Dataset.from_tensor_slices(
            images[validation_indices]
        ).batch(validation_batch_size)
        validation_data = validation_data.map(
            lambda batch_images: tf.cast(batch_images, tf.float32),
            num_parallel_calls=1,
        ).with_options(dataset_options)
        logits = model.predict(validation_data, verbose=0)
        predictions = logits.argmax(axis=1)
        truth = labels[validation_indices]
        confusion = np.zeros((len(class_names), len(class_names)), dtype=np.int64)
        np.add.at(confusion, (truth, predictions), 1)
        true_positive = np.diag(confusion).astype(np.float64)
        predicted_count = confusion.sum(axis=0)
        actual_count = confusion.sum(axis=1)
        precision = np.divide(
            true_positive,
            predicted_count,
            out=np.zeros_like(true_positive),
            where=predicted_count > 0,
        )
        recall = np.divide(
            true_positive,
            actual_count,
            out=np.zeros_like(true_positive),
            where=actual_count > 0,
        )
        f1 = np.divide(
            2 * precision * recall,
            precision + recall,
            out=np.zeros_like(precision),
            where=(precision + recall) > 0,
        )
        peak_rss_mb[0] = max(peak_rss_mb[0], _rss_mb())
        if _available_system_memory_mb() < MIN_SEARCH_AVAILABLE_MEMORY_MB:
            raise MemoryError(
                f"Dataset NAS candidate {candidate_id} completed with less than "
                f"{MIN_SEARCH_AVAILABLE_MEMORY_MB} MiB of available system memory"
            )
        if peak_rss_mb[0] > MAX_SEARCH_RSS_MB:
            raise MemoryError(
                f"Dataset NAS candidate {candidate_id} exceeded "
                f"the {MAX_SEARCH_RSS_MB} MiB RSS cap"
            )
        accuracy = float(np.mean(logits.argmax(axis=1) == labels[validation_indices]))
        elapsed_seconds = round(time.perf_counter() - candidate_started, 3)
        candidate_result = {
            "node_id": node_id,
            "architecture": dict(candidate),
            "parameter_count": parameter_count,
            "validation_samples": int(validation_indices.size),
            "validation_accuracy": accuracy,
            "validation_macro_precision": float(np.mean(precision)),
            "validation_macro_recall": float(np.mean(recall)),
            "validation_macro_f1": float(np.mean(f1)),
            "validation_loss": float(
                tf.keras.losses.sparse_categorical_crossentropy(
                    truth, logits, from_logits=True
                ).numpy().mean()
            ),
            "validation_confusion_matrix": confusion.tolist(),
            "model_summary": "\n".join(model_summary_lines),
            "fidelity_epochs": epochs,
            "elapsed_seconds": elapsed_seconds,
            "peak_rss_mb": round(max(peak_rss_mb[0], _rss_mb()), 1),
        }
        candidates.append(candidate_result)
        logger("nas_candidate_evaluated", **candidate_result)
        del model
        tf.keras.backend.clear_session()
        gc.collect()
        if time.monotonic() >= deadline:
            raise TimeoutError(f"Dataset NAS candidate {candidate_id} exceeded the time budget")
    return {
        "node_id": node_id,
        "dataset_classes": class_names,
        "seed": seed,
        "gpu": [device.name for device in devices],
        "validation_samples": int(validation_indices.size),
        "training_samples": int(train_indices.size),
        "time_limit_seconds": time_limit_seconds,
        "fidelity_epochs": epochs,
        "search_method": "pareto_evolutionary_multifidelity",
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


def search_dataset_architectures_isolated(
    shard_path: str,
    *,
    seed: int,
    time_limit_seconds: int,
    node_id: str,
    logger: Any,
    architectures: list[dict[str, Any]] | None = None,
    epochs: int = 1,
    cancel_event: threading.Event | None = None,
) -> dict[str, Any]:
    if not 30 <= time_limit_seconds <= MAX_SEARCH_SECONDS:
        raise ValueError(f"Search time limit must be between 30 and {MAX_SEARCH_SECONDS} seconds")
    environment = os.environ.copy()
    for name in (
        "DFL_API_TOKEN",
        "DFL_PEERS_JSON_B64",
        "DFL_TRAINING_SHARDS_JSON_B64",
    ):
        environment.pop(name, None)
    environment["PYTHONUNBUFFERED"] = "1"
    for name in (
        "TF_NUM_INTRAOP_THREADS",
        "TF_NUM_INTEROP_THREADS",
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        environment[name] = "1"
    command = [
        sys.executable,
        "-u",
        "-m",
        "worker.nas_search",
        "--dataset-shard",
        shard_path,
        node_id,
        str(seed),
        str(time_limit_seconds),
    ]
    # Always pass the fidelity: the first generation has no architectures but must
    # still train for the requested epochs (it silently used 1 before).
    command.extend((str(epochs), json.dumps(architectures, separators=(",", ":"))))
    if cancel_event is None:
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
            raise TimeoutError(
                "Isolated dataset NAS search exceeded its process deadline"
            ) from error
        stdout, stderr, returncode = (
            completed.stdout,
            completed.stderr,
            completed.returncode,
        )
    else:
        process = subprocess.Popen(
            command,
            cwd=Path(__file__).resolve().parents[1],
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
        )
        deadline = time.monotonic() + time_limit_seconds + 30
        while True:
            try:
                stdout, stderr = process.communicate(
                    timeout=min(0.5, max(deadline - time.monotonic(), 0.01))
                )
                break
            except subprocess.TimeoutExpired:
                if cancel_event.is_set():
                    process.terminate()
                    try:
                        stdout, stderr = process.communicate(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        stdout, stderr = process.communicate()
                    _forward_search_events(stdout, logger)
                    raise RuntimeError("Isolated dataset NAS search was cancelled")
                if time.monotonic() >= deadline:
                    process.kill()
                    stdout, stderr = process.communicate()
                    _forward_search_events(stdout, logger)
                    raise TimeoutError(
                        "Isolated dataset NAS search exceeded its process deadline"
                    )
        returncode = process.returncode
    result = _forward_search_events(stdout, logger)
    if returncode != 0:
        stderr_lines = stderr.strip().splitlines()
        detail = stderr_lines[-1] if stderr_lines else "No error details were returned."
        if stderr.strip():
            logger(
                "nas_search_failed",
                node_id=node_id,
                returncode=returncode,
                stderr_tail="\n".join(stderr_lines[-20:])[-4000:],
            )
        raise RuntimeError(
            f"Isolated dataset NAS search exited with code {returncode}: "
            f"{detail[-500:]}"
        )
    if result is None or not isinstance(result, dict):
        raise RuntimeError("Isolated dataset NAS search returned no result object")
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
    if sys.argv[1:2] == ["--dataset-shard"] and len(sys.argv) in {6, 8}:
        _, _, path, node_id, seed_value, limit_value, *search_options = sys.argv
        epochs = int(search_options[0]) if search_options else 1
        architectures = json.loads(search_options[1]) if len(search_options) == 2 else None
        images, labels, class_names = load_dataset_shard(path)
        result = search_dataset_architectures(
            images,
            labels,
            class_names,
            seed=int(seed_value),
            time_limit_seconds=int(limit_value),
            node_id=node_id,
            architectures=architectures,
            epochs=epochs,
            logger=lambda event, **details: print(
                json.dumps({"type": event, **details}, allow_nan=False),
                flush=True,
            ),
        )
        print(
            json.dumps({"type": "nas_search_result", "data": result}, allow_nan=False),
            flush=True,
        )
        return
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
