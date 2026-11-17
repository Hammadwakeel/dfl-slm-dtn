"""Read-only status check for the most recent dataset NAS group."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "dtn" / "backend"))

from app.core.config import Settings  # noqa: E402
from app.services.worker_shards import get_worker_training_status  # noqa: E402


def main() -> None:
    settings = Settings.from_env()
    request = Request(
        settings.master_slm_url.rstrip("/") + "/results",
        headers={
            "Authorization": f"Bearer {settings.master_slm_token}",
            "Accept": "application/json",
        },
    )
    with urlopen(request, timeout=15) as response:
        results: dict[str, Any] = json.loads(response.read().decode("utf-8"))
    groups = [
        group
        for group in results.get("groups", [])
        if isinstance(group, dict) and group.get("kind") == "dataset_architecture_search"
    ]
    if not groups:
        raise RuntimeError("No dataset NAS group is recorded")
    group = groups[-1]
    print(
        "latest_dataset_nas="
        + json.dumps(
            {
                key: group.get(key)
                for key in (
                    "group_id",
                    "dataset_name",
                    "status",
                    "error",
                    "selected_architecture",
                    "selected_parameter_count",
                    "search_elapsed_seconds",
                    "search_peak_rss_mb",
                )
            },
            sort_keys=True,
        )
    )
    for worker_search in group.get("worker_search", []):
        selected_id = (group.get("selected_architecture") or {}).get("id")
        candidate = next(
            (
                item
                for item in worker_search.get("candidates", [])
                if item.get("architecture", {}).get("id") == selected_id
            ),
            None,
        )
        print(
            f"{worker_search.get('node_id')}: selected_validation_accuracy="
            f"{None if candidate is None else candidate.get('validation_accuracy')}; "
            f"peak_rss_mb={worker_search.get('peak_rss_mb')}"
        )
    workers = sorted(
        (node for node in settings.nodes if node.role == "worker"),
        key=lambda node: node.node_id,
    )
    for worker in workers:
        status = get_worker_training_status(worker)
        print(
            f"{worker.node_id}: active_run_id={status.get('active_run_id')}; "
            f"scheduled_run_id={status.get('scheduled_run_id')}"
        )
    print(f"master_active_group_id={results.get('active_group_id')}")


if __name__ == "__main__":
    main()
