"""Read-only analysis of a completed dataset-specific NAS group."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "dtn" / "backend"))
sys.path.insert(0, str(ROOT))

from app.core.config import Settings  # noqa: E402
from worker.nas_search import _split_indices  # noqa: E402


GROUP_ID = "dataset-nas-1793171089-3bcbb3"


def _get_json(url: str, token: str) -> Any:
    request = Request(
        url,
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/json",
        },
    )
    with urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _analyze_worker(
    shard_path: Path,
    *,
    worker_id: str,
    seed: int,
    class_names: list[str],
) -> tuple[dict[str, Any], np.ndarray]:
    with np.load(shard_path, allow_pickle=False) as shard:
        images = np.asarray(shard["images"])
        labels = np.asarray(shard["labels"], dtype=np.int64).reshape(-1)
        actual_classes = np.asarray(shard["class_names"]).tolist()
    if actual_classes and isinstance(actual_classes[0], bytes):
        actual_classes = [value.decode("utf-8") for value in actual_classes]
    actual_classes = [str(value) for value in actual_classes]
    if actual_classes != class_names:
        raise RuntimeError(f"{worker_id} shard class order differs from NAS search")
    training_indices, validation_indices = _split_indices(labels, seed)
    class_counts = {
        class_name: int(np.count_nonzero(labels[validation_indices] == index))
        for index, class_name in enumerate(class_names)
    }
    data_digest = hashlib.sha256()
    data_digest.update(str(images[validation_indices].dtype).encode("ascii"))
    data_digest.update(np.asarray(images[validation_indices].shape, dtype="<i8").tobytes())
    data_digest.update(np.ascontiguousarray(images[validation_indices]).tobytes())
    data_digest.update(np.ascontiguousarray(labels[validation_indices], dtype="<i8").tobytes())
    analysis = {
        "worker_id": worker_id,
        "total_samples": int(labels.size),
        "training_samples": int(training_indices.size),
        "validation_samples": int(validation_indices.size),
        "training_batches_per_epoch": int(
            np.ceil(training_indices.size / min(128, int(training_indices.size)))
        ),
        "validation_class_counts": class_counts,
        "validation_indices_sha256": _sha256(
            np.ascontiguousarray(validation_indices, dtype="<i8").tobytes()
        ),
        "validation_data_sha256": data_digest.hexdigest(),
    }
    return analysis, validation_indices


def main() -> None:
    settings = Settings.from_env()
    response = _get_json(
        "http://127.0.0.1:8000/api/autonomy/history",
        settings.api_token,
    )
    matching = [
        row
        for row in response
        if isinstance(row, dict) and row.get("decision_id") == "c53a1ddb-2777-4872-8cb7-7a10cf710382"
    ]
    if not matching:
        raise RuntimeError("The actual NAS approval history record is unavailable")
    print(
        "real_history_record_key_names="
        + json.dumps(sorted(matching[0].keys()))
    )
    print(
        "real_history_details_key_names="
        + json.dumps(sorted((matching[0].get("details") or {}).keys()))
    )

    results = _get_json(
        settings.master_slm_url.rstrip("/") + "/results",
        settings.master_slm_token,
    )
    group = next(
        (
            item
            for item in results.get("groups", [])
            if isinstance(item, dict) and item.get("group_id") == GROUP_ID
        ),
        None,
    )
    if group is None:
        raise RuntimeError(f"Recorded NAS group {GROUP_ID} is missing")
    print("recorded_nas_group_key_names=" + json.dumps(sorted(group.keys())))
    seed = int(group["seed"])
    class_names = [str(value) for value in group["dataset_classes"]]
    analyses: dict[str, dict[str, Any]] = {}
    root = settings.dataset_root / "cifar_subset_demo" / "processed"
    reports = sorted(root.glob("run-*/preprocessing_report.json"), key=lambda path: path.stat().st_mtime_ns)
    if not reports:
        raise RuntimeError("No processed dataset report is available")
    run_root = reports[-1].parent

    worker_search = {
        str(item["node_id"]): item
        for item in group["worker_search"]
    }
    for worker_id in ("worker1", "worker2", "worker3"):
        analysis, _ = _analyze_worker(
            run_root / "workers" / f"{worker_id}.npz",
            worker_id=worker_id,
            seed=seed,
            class_names=class_names,
        )
        analyses[worker_id] = analysis
        search = worker_search[worker_id]
        print(
            f"{worker_id}: training_samples={analysis['training_samples']}; "
            f"validation_samples={analysis['validation_samples']}; "
            f"training_steps_per_candidate={analysis['training_batches_per_epoch']}; "
            f"epochs_per_candidate=1"
        )
        print(f"{worker_id}: validation_class_counts={json.dumps(analysis['validation_class_counts'],sort_keys=True)}")
        print(
            f"{worker_id}: validation_indices_sha256={analysis['validation_indices_sha256']}; "
            f"validation_data_sha256={analysis['validation_data_sha256']}"
        )
        candidates = sorted(
            search["candidates"],
            key=lambda row: row["architecture"]["id"],
        )
        for candidate in candidates:
            print(
                f"{worker_id} candidate={candidate['architecture']['id']}; "
                f"validation_accuracy={candidate['validation_accuracy']:.10f}; "
                f"validation_samples={candidate['validation_samples']}; "
                f"elapsed_seconds={candidate['elapsed_seconds']}; "
                f"peak_rss_mb={candidate['peak_rss_mb']}"
            )

    identical_indices = len(
        {item["validation_indices_sha256"] for item in analyses.values()}
    ) == 1
    identical_validation_data = len(
        {item["validation_data_sha256"] for item in analyses.values()}
    ) == 1
    print(
        f"validation_indices_identical={identical_indices}; "
        f"validation_data_identical={identical_validation_data}"
    )
    print(
        "group_search_elapsed_seconds="
        + str(group.get("search_elapsed_seconds"))
        + "; selected_architecture="
        + json.dumps(group.get("selected_architecture"), sort_keys=True)
    )


if __name__ == "__main__":
    main()
