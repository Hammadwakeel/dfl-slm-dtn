"""Authenticated worker process inspection and controls."""

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from app.core.config import DeviceConfig


class WorkerOperationError(RuntimeError):
    def __init__(self, message: str, status_code: int = 502) -> None:
        super().__init__(message)
        self.status_code = status_code


def _request_worker(
    worker: DeviceConfig,
    path: str,
    *,
    method: str = "GET",
    timeout_seconds: float = 10.0,
) -> dict[str, Any]:
    try:
        worker_url = worker.training_url
    except ValueError as error:
        raise WorkerOperationError(str(error), 409) from error
    request = Request(
        f"{worker_url}{path}",
        data=b"" if method == "POST" else None,
        headers={
            "Authorization": f"Bearer {worker.token}",
            "Accept": "application/json",
        },
        method=method,
    )
    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            result = json.loads(response.read().decode("utf-8"))
    except HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace").strip()[:500]
        status_code = 409 if error.code == 409 else 502
        raise WorkerOperationError(
            f"{worker.node_id} returned HTTP {error.code}: {detail}",
            status_code,
        ) from error
    except (URLError, TimeoutError, OSError) as error:
        raise WorkerOperationError(
            f"Could not reach {worker.node_id}: {error}"
        ) from error
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise WorkerOperationError(
            f"{worker.node_id} returned invalid JSON"
        ) from error
    if not isinstance(result, dict):
        raise WorkerOperationError(f"{worker.node_id} returned an invalid response")
    return result


def get_worker_processes(
    workers: tuple[DeviceConfig, ...],
    *,
    timeout_seconds: float = 10.0,
) -> dict[str, Any]:
    results: dict[str, dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=max(1, len(workers))) as executor:
        requests = {
            executor.submit(
                _request_worker,
                worker,
                "/v1/processes",
                timeout_seconds=timeout_seconds,
            ): worker
            for worker in workers
        }
        for future in as_completed(requests):
            worker = requests[future]
            try:
                report = future.result()
                if (
                    report.get("node_id") != worker.node_id
                    or not isinstance(report.get("processes"), list)
                    or not isinstance(report.get("memory_available_mb"), (int, float))
                    or not isinstance(report.get("training"), dict)
                    or not isinstance(report.get("nas_active"), bool)
                ):
                    raise WorkerOperationError(
                        f"{worker.node_id} returned an invalid process snapshot"
                    )
                results[worker.node_id] = {"node_id": worker.node_id, **report}
            except WorkerOperationError as error:
                results[worker.node_id] = {
                    "node_id": worker.node_id,
                    "error": str(error),
                }

    return {
        "workers": [results[worker.node_id] for worker in workers],
    }


def control_worker(
    worker: DeviceConfig,
    action: str,
    *,
    timeout_seconds: float = 10.0,
) -> dict[str, Any]:
    if action == "stop_training":
        path = "/v1/cancel"
    elif action == "stop_nas":
        path = "/v1/nas/cancel"
    elif action == "reboot":
        path = "/v1/system/reboot"
    else:
        raise ValueError(f"Unsupported worker action: {action}")
    return _request_worker(
        worker,
        path,
        method="POST",
        timeout_seconds=timeout_seconds,
    )
