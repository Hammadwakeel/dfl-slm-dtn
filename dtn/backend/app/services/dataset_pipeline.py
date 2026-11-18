"""Runs local image preprocessing and installs the resulting worker shards."""

import hashlib
import io
import json
import shutil
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from app.core.config import DeviceConfig
from app.services.datasets import DatasetStore, validate_dataset_name
from app.services.preprocessing import PreprocessingError, preprocess_image_dataset
from app.services.worker_shards import (
    WorkerShardError,
    get_worker_dataset_metadata,
    get_worker_training_status,
    upload_worker_global_validation,
    upload_worker_shard,
)

# Each preprocessing run stores a full normalized copy of the dataset (about 450 MB
# for CIFAR-10), so only the newest few runs are kept on disk.
KEEP_PROCESSED_RUNS = 3
# Shared validation subset every worker scores each round on: class-balanced, drawn
# from the run's validation split (disjoint from every worker's training shard).
GLOBAL_VALIDATION_PER_CLASS = 200
GLOBAL_VALIDATION_SEED = 2026


def prune_processed_runs(dataset_path: Path, *, keep: int = KEEP_PROCESSED_RUNS) -> list[str]:
    """Delete all but the newest ``keep`` preprocessing runs; return the removed names."""
    processed = dataset_path / "processed"
    if keep < 1 or not processed.is_dir():
        return []
    runs = sorted(
        (path for path in processed.glob("run-*") if path.is_dir() and not path.is_symlink()),
        key=lambda path: path.stat().st_mtime_ns,
        reverse=True,
    )
    removed = []
    for path in runs[keep:]:
        shutil.rmtree(path, ignore_errors=True)
        removed.append(path.name)
    return removed


def preprocess_and_ship(
    dataset_store: DatasetStore,
    workers: Sequence[DeviceConfig],
    dataset_name: str,
    *,
    normalization: str = "standard",
    split_strategy: str = "stratified",
    target_size: tuple[int, int] = (32, 32),
    seed: int = 42,
    replace_existing: bool = False,
) -> dict[str, Any]:
    validate_dataset_name(dataset_name)
    selected_workers = sorted(
        (worker for worker in workers if worker.role == "worker"),
        key=lambda worker: worker.node_id,
    )
    if (
        len(selected_workers) < 3
        or len({worker.node_id for worker in selected_workers}) != len(selected_workers)
    ):
        raise PreprocessingError("At least three uniquely configured worker nodes are required")
    for worker in selected_workers:
        get_worker_training_status(worker)

    dataset_path = (dataset_store.root / dataset_name).resolve()
    if dataset_path.parent != dataset_store.root.resolve() or not dataset_path.is_dir():
        raise PreprocessingError(f"Dataset not found: {dataset_name}")
    preprocessing_started = time.perf_counter()
    report = preprocess_image_dataset(
        dataset_path,
        normalization=normalization,
        split_strategy=split_strategy,
        target_size=target_size,
        worker_ids=[worker.node_id for worker in selected_workers],
        seed=seed,
    )
    report["preprocessing_duration_seconds"] = round(
        time.perf_counter() - preprocessing_started, 3
    )
    shipping_started = time.perf_counter()
    uploaded: dict[str, dict[str, Any]] = {}
    try:
        for worker in selected_workers:
            expected = report["workers"][worker.node_id]
            response = upload_worker_shard(
                worker,
                dataset_name,
                Path(expected["shard_path"]),
                replace_existing=replace_existing,
            )
            verified = get_worker_dataset_metadata(worker, dataset_name)
            expected_counts = {
                name: count
                for name, count in expected["class_counts"].items()
                if count > 0
            }
            if (
                response.get("rows") != expected["rows"]
                or response.get("class_counts") != expected_counts
                or verified.get("rows") != expected["rows"]
                or verified.get("class_counts") != expected_counts
                or verified.get("shard_sha256") != response.get("shard_sha256")
            ):
                raise WorkerShardError(
                    f"{worker.node_id} installed counts do not match the preprocessed shard"
                )
            uploaded[worker.node_id] = {
                "rows": verified["rows"],
                "class_counts": verified["class_counts"],
                "shard_sha256": verified["shard_sha256"],
            }
    finally:
        report["uploaded_workers"] = uploaded
        report["shipping_duration_seconds"] = round(
            time.perf_counter() - shipping_started, 3
        )
        (Path(report["output_path"]) / "preprocessing_report.json").write_text(
            json.dumps(report, indent=2),
            encoding="utf-8",
        )
    try:
        report["global_validation"] = ship_global_validation(
            dataset_store,
            selected_workers,
            dataset_name,
            run_path=Path(report["output_path"]),
        )
    except (PreprocessingError, WorkerShardError, OSError, ValueError) as error:
        # Older workers without the endpoint still train; rounds then lack global scores.
        report["global_validation"] = {"error": str(error)[:500]}
    # Only after every shard was installed and verified: older runs are now unused.
    report["removed_processed_runs"] = prune_processed_runs(dataset_path)
    return report

def balanced_validation_subset(
    validation_path: Path,
    *,
    per_class: int = GLOBAL_VALIDATION_PER_CLASS,
    seed: int = GLOBAL_VALIDATION_SEED,
) -> bytes:
    """Deterministic class-balanced subset of a validation split, as NPZ bytes."""
    with np.load(validation_path, allow_pickle=False) as split:
        images = np.asarray(split["images"])
        labels = np.asarray(split["labels"], dtype=np.int64)
        class_names = np.asarray(split["class_names"])
    rng = np.random.default_rng(seed)
    counts = np.bincount(labels, minlength=len(class_names))
    take = int(min(per_class, counts.min()))
    if take < 1:
        raise PreprocessingError("The validation split is missing at least one class")
    chosen = np.sort(
        np.concatenate(
            [rng.permutation(np.flatnonzero(labels == index))[:take] for index in range(len(class_names))]
        )
    )
    buffer = io.BytesIO()
    np.savez_compressed(buffer, images=images[chosen], labels=labels[chosen], class_names=class_names)
    return buffer.getvalue()


def installed_preprocessing_run(
    dataset_store: DatasetStore,
    workers: Sequence[DeviceConfig],
    dataset_name: str,
) -> Path:
    """The preprocessing run whose shards are the ones installed on every worker."""
    processed = (dataset_store.root / dataset_name).resolve() / "processed"
    installed = {
        worker.node_id: get_worker_dataset_metadata(worker, dataset_name).get("shard_sha256")
        for worker in workers
    }
    reports = sorted(
        processed.glob("run-*/preprocessing_report.json"),
        key=lambda path: path.stat().st_mtime_ns,
        reverse=True,
    )
    for report_path in reports:
        uploaded = json.loads(report_path.read_text(encoding="utf-8")).get("uploaded_workers", {})
        if all(uploaded.get(node, {}).get("shard_sha256") == sha for node, sha in installed.items()):
            return report_path.parent
    raise PreprocessingError(
        f"No preprocessing run of {dataset_name} matches the shards installed on the workers"
    )


def ship_global_validation(
    dataset_store: DatasetStore,
    workers: Sequence[DeviceConfig],
    dataset_name: str,
    *,
    run_path: Path | None = None,
) -> dict[str, Any]:
    """Install the same balanced validation subset on every worker."""
    validate_dataset_name(dataset_name)
    selected = sorted((worker for worker in workers if worker.role == "worker"), key=lambda worker: worker.node_id)
    if run_path is None:
        run_path = installed_preprocessing_run(dataset_store, selected, dataset_name)
    payload = balanced_validation_subset(run_path / "validation.npz")
    sha256 = hashlib.sha256(payload).hexdigest()
    installed = {}
    for worker in selected:
        response = upload_worker_global_validation(worker, dataset_name, payload)
        if response.get("sha256") != sha256:
            raise WorkerShardError(f"{worker.node_id} stored a different global validation split")
        installed[worker.node_id] = {"rows": response.get("rows"), "class_counts": response.get("class_counts")}
    return {
        "dataset_name": dataset_name,
        "preprocessing_run": run_path.name,
        "rows": next(iter(installed.values()))["rows"] if installed else 0,
        "per_class": GLOBAL_VALIDATION_PER_CLASS,
        "sha256": sha256,
        "workers": installed,
    }
