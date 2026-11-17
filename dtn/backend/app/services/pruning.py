"""Lightweight structured pruning of the newest trained DFL model.

The coordinator never averages weights. It *selects* the trained model of the
worker with the best validation accuracy in the newest run, removes the least
important channels/units (``common.pruning``), and redistributes that smaller
model — with its trained weights — as every worker's starting point. A short
decentralized fine-tune then recovers accuracy on all workers.
"""

from __future__ import annotations

import base64
import io
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Sequence

import numpy as np

from common.pruning import PruningError, prune_model
from app.core.config import DeviceConfig
from app.services.datasets import DatasetStore
from app.services.initial_models import InitialModelError, create_and_distribute_initial_model
from app.services.worker_shards import get_worker_latest_model


def _decode_weights(encoded: str) -> list[np.ndarray]:
    with np.load(io.BytesIO(base64.b64decode(encoded, validate=True)), allow_pickle=False) as archive:
        keys = sorted(archive.files, key=lambda key: int(key.removeprefix("weight_")))
        if keys != [f"weight_{index}" for index in range(len(keys))]:
            raise InitialModelError("A worker returned trained weights with an unexpected layout")
        return [np.asarray(archive[key]) for key in keys]


def select_source_model(models: dict[str, dict[str, Any] | None]) -> tuple[str, dict[str, Any]]:
    """Pick the best-validated worker model from the newest run every worker finished."""
    available = {node: model for node, model in models.items() if isinstance(model, dict)}
    if not available:
        raise InitialModelError("No worker has a trained model for this dataset yet; train first")
    newest = max(available.values(), key=lambda model: float(model.get("saved_at") or 0))
    run_id = newest.get("run_id")
    same_run = {node: model for node, model in available.items() if model.get("run_id") == run_id}
    accuracy = lambda item: float(item[1].get("validation_accuracy") or 0)  # noqa: E731
    return max(sorted(same_run.items()), key=accuracy)


def prune_and_distribute(
    dataset_store: DatasetStore,
    workers: Sequence[DeviceConfig],
    dataset_name: str,
    *,
    ratio: float,
    seed: int = 0,
) -> dict[str, Any]:
    selected = sorted((worker for worker in workers if worker.role == "worker"), key=lambda worker: worker.node_id)
    with ThreadPoolExecutor(max_workers=max(1, len(selected))) as pool:
        fetched = list(pool.map(lambda worker: get_worker_latest_model(worker, dataset_name), selected))
    node_id, source = select_source_model({worker.node_id: model for worker, model in zip(selected, fetched)})
    architecture = source.get("architecture")
    if not isinstance(architecture, dict):
        raise InitialModelError(f"{node_id} returned a trained model without its architecture")
    input_shape = tuple(int(value) for value in architecture.get("input_shape") or ())
    classes = architecture.get("classes") or []
    if len(input_shape) != 3 or len(classes) < 2:
        raise InitialModelError("The trained model does not record its input shape and classes")
    try:
        pruned, weights, report = prune_model(
            architecture,
            _decode_weights(str(source.get("weights_b64", ""))),
            input_shape=input_shape,  # type: ignore[arg-type]
            num_classes=len(classes),
            ratio=ratio,
        )
    except PruningError as error:
        raise InitialModelError(f"Pruning failed: {error}") from error
    deployed = create_and_distribute_initial_model(
        dataset_store,
        workers,
        dataset_name,
        seed=seed,
        architecture=pruned,
        replace_existing=True,
        initial_weights=weights,
    )
    return {
        **report,
        "dataset_name": dataset_name,
        "architecture_id": pruned["id"],
        "source_worker": node_id,
        "source_run_id": source.get("run_id"),
        "source_round": source.get("round"),
        "source_validation_accuracy": source.get("validation_accuracy"),
        "architecture_sha256": deployed.get("architecture_sha256"),
        "weights_sha256": deployed.get("weights_sha256"),
    }
