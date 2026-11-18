"""Runs local image preprocessing and installs the resulting worker shards."""

import json
import shutil
import time
from pathlib import Path
from typing import Any, Sequence

from app.core.config import DeviceConfig
from app.services.datasets import DatasetStore, validate_dataset_name
from app.services.preprocessing import PreprocessingError, preprocess_image_dataset
from app.services.worker_shards import (
    WorkerShardError,
    get_worker_dataset_metadata,
    get_worker_training_status,
    upload_worker_shard,
)

# Each preprocessing run stores a full normalized copy of the dataset (about 450 MB
# for CIFAR-10), so only the newest few runs are kept on disk.
KEEP_PROCESSED_RUNS = 3


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
    # Only after every shard was installed and verified: older runs are now unused.
    report["removed_processed_runs"] = prune_processed_runs(dataset_path)
    return report