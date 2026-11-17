"""Authenticated transfer of preprocessed image shards to worker Jetsons."""

import json
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

from app.core.config import DeviceConfig


class WorkerShardError(RuntimeError):
    pass


def get_worker_training_status(
    worker: DeviceConfig,
    *,
    timeout_seconds: float = 10.0,
) -> dict[str, Any]:
    if worker.role != "worker":
        raise ValueError(f"Cannot query training status on non-worker node {worker.node_id}")
    request = Request(
        f"{worker.training_url}/v1/status",
        headers={
            "Authorization": f"Bearer {worker.token}",
            "Accept": "application/json",
        },
    )
    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            result = json.loads(response.read().decode("utf-8"))
    except HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace").strip()[:500]
        raise WorkerShardError(
            f"{worker.node_id} returned HTTP {error.code} while checking training status: {detail}"
        ) from error
    except (URLError, TimeoutError, OSError) as error:
        raise WorkerShardError(
            f"Could not reach {worker.node_id} to check training status: {error}"
        ) from error
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise WorkerShardError(
            f"{worker.node_id} returned invalid JSON for training status"
        ) from error
    if not isinstance(result, dict) or result.get("node_id") != worker.node_id:
        raise WorkerShardError(f"{worker.node_id} returned an invalid training status")
    if result.get("active_run_id") is not None or result.get("scheduled_run_id") is not None:
        raise WorkerShardError(f"{worker.node_id} has a training job running or scheduled")
    return result


def get_worker_dataset_metadata(
    worker: DeviceConfig,
    dataset_name: str,
    *,
    timeout_seconds: float = 10.0,
) -> dict[str, Any]:
    if worker.role != "worker":
        raise ValueError(f"Cannot query datasets on non-worker node {worker.node_id}")
    request = Request(
        f"{worker.training_url}/v1/datasets/{quote(dataset_name, safe='')}",
        headers={
            "Authorization": f"Bearer {worker.token}",
            "Accept": "application/json",
        },
    )
    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            result = json.loads(response.read().decode("utf-8"))
    except HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace").strip()[:500]
        raise WorkerShardError(
            f"{worker.node_id} returned HTTP {error.code} while verifying dataset shard: {detail}"
        ) from error
    except (URLError, TimeoutError, OSError) as error:
        raise WorkerShardError(
            f"Could not reach {worker.node_id} to verify dataset shard: {error}"
        ) from error
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise WorkerShardError(
            f"{worker.node_id} returned invalid JSON while verifying dataset shard"
        ) from error
    if (
        not isinstance(result, dict)
        or result.get("node_id") != worker.node_id
        or result.get("dataset_name") != dataset_name
    ):
        raise WorkerShardError(f"{worker.node_id} returned mismatched dataset metadata")
    return result


def install_worker_initial_model(
    worker: DeviceConfig,
    dataset_name: str,
    bundle: bytes,
    *,
    timeout_seconds: float = 60.0,
) -> dict[str, Any]:
    if worker.role != "worker":
        raise ValueError(f"Cannot install a model on non-worker node {worker.node_id}")
    request = Request(
        f"{worker.training_url}/v1/datasets/{quote(dataset_name, safe='')}/initial-model",
        data=bundle,
        headers={
            "Authorization": f"Bearer {worker.token}",
            "Content-Type": "application/octet-stream",
            "Accept": "application/json",
        },
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            result = json.loads(response.read().decode("utf-8"))
    except HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace").strip()[:500]
        raise WorkerShardError(
            f"{worker.node_id} returned HTTP {error.code} while installing initial model: {detail}"
        ) from error
    except (URLError, TimeoutError, OSError) as error:
        raise WorkerShardError(
            f"Could not reach {worker.node_id} to install initial model: {error}"
        ) from error
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise WorkerShardError(
            f"{worker.node_id} returned invalid JSON after initial model installation"
        ) from error
    if (
        not isinstance(result, dict)
        or result.get("node_id") != worker.node_id
        or result.get("dataset_name") != dataset_name
    ):
        raise WorkerShardError(f"{worker.node_id} returned mismatched initial model metadata")
    return result


def get_worker_dataset_initial_model(
    worker: DeviceConfig,
    dataset_name: str,
    *,
    timeout_seconds: float = 10.0,
) -> dict[str, Any]:
    result = _read_worker_initial_model(
        worker,
        dataset_name,
        timeout_seconds=timeout_seconds,
        allow_missing=False,
    )
    if result is None:
        raise WorkerShardError(f"{worker.node_id} has no initial model for {dataset_name}")
    return result


def get_worker_dataset_initial_model_if_present(
    worker: DeviceConfig,
    dataset_name: str,
    *,
    timeout_seconds: float = 10.0,
) -> dict[str, Any] | None:
    return _read_worker_initial_model(
        worker,
        dataset_name,
        timeout_seconds=timeout_seconds,
        allow_missing=True,
    )


def _read_worker_initial_model(
    worker: DeviceConfig,
    dataset_name: str,
    *,
    timeout_seconds: float,
    allow_missing: bool,
) -> dict[str, Any] | None:
    if worker.role != "worker":
        raise ValueError(f"Cannot query a model on non-worker node {worker.node_id}")
    request = Request(
        f"{worker.training_url}/v1/datasets/{quote(dataset_name, safe='')}/initial-model",
        headers={
            "Authorization": f"Bearer {worker.token}",
            "Accept": "application/json",
        },
    )
    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            result = json.loads(response.read().decode("utf-8"))
    except HTTPError as error:
        if allow_missing and error.code == 404:
            return None
        detail = error.read().decode("utf-8", errors="replace").strip()[:500]
        raise WorkerShardError(
            f"{worker.node_id} returned HTTP {error.code} while verifying initial model: {detail}"
        ) from error
    except (URLError, TimeoutError, OSError) as error:
        raise WorkerShardError(
            f"Could not reach {worker.node_id} to verify initial model: {error}"
        ) from error
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise WorkerShardError(
            f"{worker.node_id} returned invalid JSON while verifying initial model"
        ) from error
    if (
        not isinstance(result, dict)
        or result.get("node_id") != worker.node_id
        or result.get("dataset_name") != dataset_name
    ):
        raise WorkerShardError(f"{worker.node_id} returned mismatched initial model metadata")
    return result


def delete_worker_initial_model(
    worker: DeviceConfig,
    dataset_name: str,
    *,
    timeout_seconds: float = 30.0,
) -> None:
    if worker.role != "worker":
        raise ValueError(f"Cannot remove a model from non-worker node {worker.node_id}")
    request = Request(
        f"{worker.training_url}/v1/datasets/{quote(dataset_name, safe='')}/initial-model",
        headers={
            "Authorization": f"Bearer {worker.token}",
            "Accept": "application/json",
        },
        method="DELETE",
    )
    try:
        with urlopen(request, timeout=timeout_seconds):
            return
    except HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace").strip()[:500]
        raise WorkerShardError(
            f"{worker.node_id} returned HTTP {error.code} while removing initial model: {detail}"
        ) from error
    except (URLError, TimeoutError, OSError) as error:
        raise WorkerShardError(
            f"Could not reach {worker.node_id} to remove initial model: {error}"
        ) from error


def upload_worker_shard(
    worker: DeviceConfig,
    dataset_name: str,
    shard_path: Path,
    *,
    replace_existing: bool = False,
    timeout_seconds: float = 60.0,
) -> dict[str, Any]:
    if worker.role != "worker":
        raise ValueError(f"Cannot upload a training shard to non-worker node {worker.node_id}")
    url = (
        f"{worker.training_url}/v1/datasets/{quote(dataset_name, safe='')}/shard"
    )
    payload = shard_path.read_bytes()
    request = Request(
        url,
        data=payload,
        headers={
            "Authorization": f"Bearer {worker.token}",
            "Content-Type": "application/octet-stream",
            "Accept": "application/json",
        },
        method="PUT" if replace_existing else "POST",
    )
    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            result = json.loads(response.read().decode("utf-8"))
    except HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace").strip()[:500]
        raise WorkerShardError(
            f"{worker.node_id} returned HTTP {error.code} while "
            f"{'replacing' if replace_existing else 'installing'} dataset shard: {detail}"
        ) from error
    except (URLError, TimeoutError, OSError) as error:
        raise WorkerShardError(
            f"Could not reach {worker.node_id} to install dataset shard: {error}"
        ) from error
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise WorkerShardError(
            f"{worker.node_id} returned invalid JSON after shard upload"
        ) from error
    if not isinstance(result, dict):
        raise WorkerShardError(f"{worker.node_id} returned a non-object shard response")
    if result.get("node_id") != worker.node_id:
        raise WorkerShardError(f"{worker.node_id} returned a mismatched shard identity")
    return result
