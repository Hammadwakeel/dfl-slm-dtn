"""The supervisor run's best trained model, kept so it can be put back on the workers.

Workers keep only the model of their newest training. Whenever an experiment
becomes the run's best (or an accepted prune replaces it), the coordinator saves
that worker model with its trained weights. Continuation training starts from
this checkpoint, and the finished run installs it on every worker so the cluster
ends with the run's best model rather than whatever was trained last.
"""

from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Sequence

from app.core.config import DeviceConfig
from app.services.datasets import DatasetStore
from app.services.initial_models import InitialModelError, create_and_distribute_initial_model
from app.services.pruning import _decode_weights, select_source_model
from app.services.worker_shards import get_worker_latest_model

BEST_CHECKPOINT_FILE = "best_checkpoint.json"
CHECKPOINT_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


def _checkpoint_path(dataset_store: DatasetStore, dataset_name: str) -> Path:
    return (dataset_store.root / dataset_name).resolve() / "models" / BEST_CHECKPOINT_FILE


def _require_id(checkpoint_id: Any) -> str:
    if not isinstance(checkpoint_id, str) or CHECKPOINT_ID.fullmatch(checkpoint_id) is None:
        raise InitialModelError("A valid checkpoint_id is required")
    return checkpoint_id


def save_best_checkpoint(
    dataset_store: DatasetStore,
    workers: Sequence[DeviceConfig],
    dataset_name: str,
    *,
    checkpoint_id: str,
    source_worker: str | None = None,
) -> dict[str, Any]:
    """Save the newest run's model of ``source_worker`` (else the best-validated worker)."""
    checkpoint_id = _require_id(checkpoint_id)
    selected = sorted((worker for worker in workers if worker.role == "worker"), key=lambda worker: worker.node_id)
    with ThreadPoolExecutor(max_workers=max(1, len(selected))) as pool:
        fetched = list(pool.map(lambda worker: get_worker_latest_model(worker, dataset_name), selected))
    node_id, source = select_source_model(
        {worker.node_id: model for worker, model in zip(selected, fetched)},
        preferred_worker=source_worker,
    )
    architecture = source.get("architecture")
    if not isinstance(architecture, dict):
        raise InitialModelError(f"{node_id} returned a trained model without its architecture")
    weights_b64 = str(source.get("weights_b64", ""))
    try:
        _decode_weights(weights_b64)  # never store a checkpoint that could not be installed
    except InitialModelError:
        raise
    except Exception as error:  # noqa: BLE001 - corrupt base64/npz from a worker
        raise InitialModelError(f"{node_id} returned unreadable trained weights") from error
    metadata = {
        "checkpoint_id": checkpoint_id,
        "dataset_name": dataset_name,
        "source_worker": node_id,
        "source_run_id": source.get("run_id"),
        "source_round": source.get("round"),
        "weights_sha256": source.get("weights_sha256"),
    }
    path = _checkpoint_path(dataset_store, dataset_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps({**metadata, "architecture": architecture, "weights_b64": weights_b64}), encoding="utf-8")
    temporary.replace(path)
    return {**metadata, "saved": True}


def install_best_checkpoint(
    dataset_store: DatasetStore,
    workers: Sequence[DeviceConfig],
    dataset_name: str,
    *,
    checkpoint_id: str,
    seed: int = 0,
) -> dict[str, Any]:
    """Install the saved checkpoint (weights included) as every worker's model."""
    checkpoint_id = _require_id(checkpoint_id)
    path = _checkpoint_path(dataset_store, dataset_name)
    if not path.is_file():
        raise InitialModelError("No best checkpoint is stored for this dataset")
    try:
        stored = json.loads(path.read_text(encoding="utf-8"))
        architecture = stored["architecture"]
        weights = _decode_weights(str(stored["weights_b64"]))
    except (OSError, KeyError, TypeError, ValueError) as error:
        raise InitialModelError("The stored best checkpoint is unreadable") from error
    if stored.get("checkpoint_id") != checkpoint_id:
        # Another supervisor run saved a newer checkpoint; never install a foreign model.
        raise InitialModelError("The stored best checkpoint belongs to another run")
    deployed = create_and_distribute_initial_model(
        dataset_store,
        workers,
        dataset_name,
        seed=seed,
        architecture=architecture,
        replace_existing=True,
        initial_weights=weights,
    )
    return {
        "dataset_name": dataset_name,
        "installed": True,
        "checkpoint_id": checkpoint_id,
        "source_worker": stored.get("source_worker"),
        "source_run_id": stored.get("source_run_id"),
        "architecture_sha256": deployed.get("architecture_sha256"),
        "weights_sha256": deployed.get("weights_sha256"),
    }
