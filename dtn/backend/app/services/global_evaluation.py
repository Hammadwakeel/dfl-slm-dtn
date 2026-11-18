"""Global, class-balanced evaluation of every worker's newest trained model.

Workers report accuracy on their own non-IID validation slices. After each
decentralized experiment the coordinator fetches every worker's newest model and
evaluates it on the dataset's shared validation split (used for decisions) and
the official test split (reported only). TensorFlow runs in a separate CPU
environment (``DTN_EVAL_PYTHON``) so the backend itself stays TensorFlow-free.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Sequence

from app.core.config import DeviceConfig
from app.services.datasets import DatasetStore
from app.services.worker_shards import WorkerShardError, get_worker_latest_model

REPO_ROOT = Path(__file__).resolve().parents[4]
EVALUATION_SPLITS = ("validation", "test")
EVALUATION_TIMEOUT_SECONDS = 600


class GlobalEvaluationError(RuntimeError):
    """Global evaluation could not run; training results are unaffected."""


def evaluation_python() -> Path | None:
    configured = os.environ.get("DTN_EVAL_PYTHON")
    candidates = (
        [Path(configured)]
        if configured
        else [
            REPO_ROOT / "dtn" / "backend" / ".venv-eval" / "Scripts" / "python.exe",
            REPO_ROOT / "dtn" / "backend" / ".venv-eval" / "bin" / "python",
        ]
    )
    return next((path for path in candidates if path.is_file()), None)


def latest_split_paths(dataset_store: DatasetStore, dataset_name: str) -> dict[str, Path]:
    """Global validation/test splits of the newest preprocessing run."""
    processed = (dataset_store.root / dataset_name).resolve() / "processed"
    reports = sorted(
        processed.glob("run-*/preprocessing_report.json"),
        key=lambda path: path.stat().st_mtime_ns,
    )
    if not reports:
        raise GlobalEvaluationError(f"{dataset_name} has not been preprocessed")
    run_path = reports[-1].parent
    paths = {name: run_path / f"{name}.npz" for name in EVALUATION_SPLITS}
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise GlobalEvaluationError(f"The preprocessing run has no {', '.join(missing)} split")
    return paths


def fetch_worker_models(
    workers: Sequence[DeviceConfig],
    dataset_name: str,
    *,
    run_id: str | None = None,
) -> list[dict[str, Any]]:
    """Newest custom model of every worker (optionally only those from ``run_id``)."""
    selected = sorted((worker for worker in workers if worker.role == "worker"), key=lambda worker: worker.node_id)

    def fetch(worker: DeviceConfig) -> dict[str, Any] | None:
        try:
            return get_worker_latest_model(worker, dataset_name)
        except WorkerShardError:
            return None

    with ThreadPoolExecutor(max_workers=max(1, len(selected))) as pool:
        fetched = list(pool.map(fetch, selected))
    models = []
    for worker, model in zip(selected, fetched):
        if not isinstance(model, dict) or not model.get("weights_b64") or not isinstance(model.get("architecture"), dict):
            continue
        if run_id is not None and model.get("run_id") != run_id:
            continue
        models.append(
            {
                "node_id": worker.node_id,
                "architecture": model["architecture"],
                "weights_b64": model["weights_b64"],
                "run_id": model.get("run_id"),
                "round": model.get("round"),
            }
        )
    return models


def run_evaluator(python: Path, split_paths: dict[str, Path], models: list[dict[str, Any]]) -> dict[str, Any]:
    request = {
        "splits": {name: str(path) for name, path in split_paths.items()},
        "models": [{key: item[key] for key in ("node_id", "architecture", "weights_b64")} for item in models],
    }
    with tempfile.TemporaryDirectory(prefix="dtn-eval-") as directory:
        request_path = Path(directory) / "request.json"
        request_path.write_text(json.dumps(request), encoding="utf-8")
        environment = {**os.environ, "PYTHONPATH": str(REPO_ROOT), "TF_CPP_MIN_LOG_LEVEL": "2"}
        try:
            completed = subprocess.run(
                [str(python), "-m", "common.global_evaluation", str(request_path)],
                cwd=REPO_ROOT,
                env=environment,
                capture_output=True,
                text=True,
                encoding="utf-8",
                timeout=EVALUATION_TIMEOUT_SECONDS,
                check=False,
            )
        except subprocess.TimeoutExpired as error:
            raise GlobalEvaluationError("Global evaluation timed out") from error
    if completed.returncode != 0:
        raise GlobalEvaluationError(f"Global evaluation failed: {completed.stderr.strip()[-400:]}")
    try:
        return json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise GlobalEvaluationError("Global evaluation returned invalid JSON") from error


def evaluate_newest_models(
    dataset_store: DatasetStore,
    workers: Sequence[DeviceConfig],
    dataset_name: str,
    *,
    run_id: str | None = None,
) -> dict[str, Any]:
    python = evaluation_python()
    if python is None:
        raise GlobalEvaluationError("No TensorFlow evaluation environment is configured (DTN_EVAL_PYTHON)")
    split_paths = latest_split_paths(dataset_store, dataset_name)
    models = fetch_worker_models(workers, dataset_name, run_id=run_id)
    if not models:
        raise GlobalEvaluationError("No worker has a trained model from this experiment")
    started = time.perf_counter()
    result = run_evaluator(python, split_paths, models)
    result["dataset_name"] = dataset_name
    result["run_id"] = run_id
    result["elapsed_seconds"] = round(time.perf_counter() - started, 2)
    return result
