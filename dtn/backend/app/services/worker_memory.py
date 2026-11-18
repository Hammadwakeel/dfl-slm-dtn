"""Coordinated cleanup of idle worker training memory and Linux page caches."""

import json
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from app.core.config import DeviceConfig
from app.services.worker_shards import WorkerShardError, get_worker_training_status


class WorkerMemoryCleanupError(RuntimeError):
    pass


def _cleanup_worker(worker: DeviceConfig, timeout_seconds: float, recycle: bool = False) -> dict[str, Any]:
    request = Request(
        f"{worker.training_url}/v1/memory/cleanup{'?recycle=1' if recycle else ''}",
        data=b"",
        headers={
            "Authorization": f"Bearer {worker.token}",
            "Accept": "application/json",
        },
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            result = json.loads(response.read().decode("utf-8"))
    except HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace").strip()[:500]
        raise WorkerMemoryCleanupError(
            f"{worker.node_id} returned HTTP {error.code} while cleaning memory: {detail}"
        ) from error
    except (URLError, TimeoutError, OSError) as error:
        raise WorkerMemoryCleanupError(
            f"Could not reach {worker.node_id} to clean memory: {error}"
        ) from error
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise WorkerMemoryCleanupError(
            f"{worker.node_id} returned invalid JSON while cleaning memory"
        ) from error
    metric_names = (
        "memory_available_before_mb",
        "memory_available_after_mb",
        "process_rss_before_mb",
        "process_rss_after_mb",
    )
    if not isinstance(result, dict) or result.get("node_id") != worker.node_id:
        raise WorkerMemoryCleanupError(
            f"{worker.node_id} returned an invalid memory-cleanup result"
        )
    cleaned_runs = result.get("cleaned_runs")
    if (
        result.get("process_memory_cleaned") is not True
        or not isinstance(result.get("os_page_cache_dropped"), bool)
        or not isinstance(cleaned_runs, int)
        or isinstance(cleaned_runs, bool)
        or cleaned_runs < 0
        or any(
            not isinstance(result.get(key), (int, float))
            or isinstance(result.get(key), bool)
            or not math.isfinite(result[key])
            or result[key] < 0
            for key in metric_names
        )
    ):
        raise WorkerMemoryCleanupError(
            f"{worker.node_id} returned an invalid memory-cleanup result"
        )
    cache_dropped = result["os_page_cache_dropped"]
    response = {
        "node_id": worker.node_id,
        "success": cache_dropped,
        "cleaned_runs": cleaned_runs,
        "process_memory_cleaned": True,
        "os_page_cache_dropped": cache_dropped,
        "recycle_scheduled": result.get("recycle_scheduled") is True,
        **{key: result[key] for key in metric_names},
    }
    if not cache_dropped:
        detail = result.get("error")
        response["error"] = (
            detail
            if isinstance(detail, str) and detail
            else "Worker model memory was cleared, but OS page-cache cleanup failed."
        )
    return response


def cleanup_worker_memory_cluster(
    workers: tuple[DeviceConfig, ...],
    *,
    timeout_seconds: float = 10.0,
    recycle: bool = False,
) -> dict[str, Any]:
    """Clean idle memory on every worker; ``recycle`` restarts bloated idle services."""
    if not workers:
        raise ValueError("No configured worker nodes are available for memory cleanup")

    preflight_failures: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=len(workers)) as executor:
        checks = {
            executor.submit(
                get_worker_training_status,
                worker,
                timeout_seconds=timeout_seconds,
            ): worker
            for worker in workers
        }
        for future in as_completed(checks):
            worker = checks[future]
            try:
                future.result()
            except (WorkerShardError, ValueError) as error:
                preflight_failures[worker.node_id] = str(error)

    if preflight_failures:
        return {
            "complete": False,
            "phase": "preflight",
            "message": "No cleanup was started because at least one worker is busy or unreachable.",
            "workers": [
                {
                    "node_id": worker.node_id,
                    "success": False,
                    "error": preflight_failures.get(
                        worker.node_id,
                        "Skipped because another worker failed preflight; no cleanup was started.",
                    ),
                }
                for worker in workers
            ],
        }

    results: dict[str, dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=len(workers)) as executor:
        cleanups = {
            executor.submit(_cleanup_worker, worker, timeout_seconds, recycle): worker
            for worker in workers
        }
        for future in as_completed(cleanups):
            worker = cleanups[future]
            try:
                results[worker.node_id] = future.result()
            except WorkerMemoryCleanupError as error:
                results[worker.node_id] = {
                    "node_id": worker.node_id,
                    "success": False,
                    "error": str(error),
                }

    ordered_results = [results[worker.node_id] for worker in workers]
    complete = all(result["success"] for result in ordered_results)
    return {
        "complete": complete,
        "phase": "cleanup",
        "message": (
            "Memory cleanup completed on all workers."
            if complete
            else "Memory cleanup did not complete on every worker."
        ),
        "workers": ordered_results,
        "recycled": [result["node_id"] for result in ordered_results if result.get("recycle_scheduled")],
    }
