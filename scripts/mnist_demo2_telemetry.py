"""Capture real PC RAM and shard-shipping telemetry through the backend."""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "dtn" / "backend"))
sys.path.insert(0, str(ROOT / "scripts"))

from app.core.config import DeviceConfig, Settings  # noqa: E402
from app.services.worker_shards import (  # noqa: E402
    get_worker_dataset_metadata,
    WorkerShardError,
)
from mnist_dataset_acceptance import (  # noqa: E402
    DATASET_NAME as ORIGINAL_DATASET_NAME,
    SEED,
    SAMPLE_PER_CLASS,
    AcceptanceError,
    _MemorySampler,
    _assert_worker_absent,
    _build_upload_archive,
    _check_preflight,
    _multipart_upload,
    _read_mnist,
    _sha256,
)
from m5_feedback_loop import BackendClient, _sample_tegrastats  # noqa: E402

DATASET_NAME = "mnist_demo2"
OUTPUT_PATH = ROOT / "data" / "mnist_demo2_telemetry.json"


def _dataset_report(envelope: Any) -> dict[str, Any]:
    if not isinstance(envelope, dict) or envelope.get("status") != "approved":
        raise AcceptanceError("MNIST demo2 preprocessing was not approved")
    result = envelope.get("result")
    report = result.get("data") if isinstance(result, dict) else None
    if not isinstance(report, dict) or report.get("dataset_name") != DATASET_NAME:
        raise AcceptanceError("MNIST demo2 approval omitted its preprocessing report")
    return report


def _worker_rows(
    report: dict[str, Any],
    workers: list[DeviceConfig],
) -> dict[str, dict[str, Any]]:
    expected_rows = report.get("workers")
    if not isinstance(expected_rows, dict) or set(expected_rows) != {
        worker.node_id for worker in workers
    }:
        raise AcceptanceError("MNIST demo2 report omitted configured worker shards")
    rows: dict[str, dict[str, Any]] = {}
    for worker in workers:
        expected = expected_rows[worker.node_id]
        installed = get_worker_dataset_metadata(worker, DATASET_NAME)
        shard_path = Path(expected["shard_path"])
        if not shard_path.is_file():
            raise AcceptanceError(f"PC shard missing for {worker.node_id}")
        pc_hash = _sha256(shard_path)
        if (
            installed.get("rows") != expected.get("rows")
            or installed.get("channels") != 1
            or installed.get("shape", [])[1:] != [28, 28, 1]
            or installed.get("shard_sha256") != pc_hash
        ):
            raise WorkerShardError(f"{worker.node_id} MNIST demo2 shard verification failed")
        rows[worker.node_id] = {
            "rows": installed["rows"],
            "pc_shard_sha256": pc_hash,
            "worker_shard_sha256": installed["shard_sha256"],
        }
    if report.get("split_sizes") != {"train": 2400, "validation": 300, "test": 300}:
        raise AcceptanceError("MNIST demo2 split sizes differ from the expected seeded split")
    return rows


def main() -> None:
    settings = Settings.from_env()
    client = BackendClient(f"http://127.0.0.1:{settings.port}", settings.api_token)
    workers = sorted(
        (node for node in settings.nodes if node.role == "worker"),
        key=lambda node: node.node_id,
    )
    result: dict[str, Any] = {
        "dataset_name": DATASET_NAME,
        "status": "RUNNING",
        "seed": SEED,
        "images_per_class": SAMPLE_PER_CLASS,
        "pc_peak_system_used_mb": None,
        "pc_memory_samples": 0,
        "preprocessing_duration_seconds": None,
        "shipping_duration_seconds": None,
        "worker_rows_and_hashes": {},
    }
    sampler: _MemorySampler | None = None
    try:
        _check_preflight(client, settings, workers)
        dataset_root = settings.dataset_root
        dataset_root.mkdir(parents=True, exist_ok=True)
        if (dataset_root / DATASET_NAME).exists():
            raise AcceptanceError("mnist_demo2 already exists; refusing to overwrite it")
        for worker in workers:
            _assert_worker_absent(worker, DATASET_NAME)

        original_zip = dataset_root / "mnist_demo_upload.zip"
        if not original_zip.is_file():
            raise AcceptanceError("Original MNIST ZIP is missing; cannot verify same fixed-seed sample")
        images, labels = _read_mnist()
        original_counts = _build_upload_archive(images, labels, original_zip)
        archive_path = dataset_root / "mnist_demo2_upload.zip"
        second_counts = _build_upload_archive(images, labels, archive_path)
        if original_counts != second_counts:
            raise AcceptanceError("MNIST demo2 fixed-seed class counts differ from mnist_demo")
        archive = archive_path.read_bytes()
        result["upload_sha256"] = hashlib.sha256(archive).hexdigest()

        upload_status, metadata = _multipart_upload(
            client,
            DATASET_NAME,
            archive_path.name,
            archive,
        )
        if upload_status != 201:
            raise AcceptanceError(f"MNIST demo2 upload returned HTTP {upload_status}")
        if (
            metadata.get("image_count") != 3000
            or metadata.get("class_counts") != original_counts
            or any(
                metadata.get("image_channels", {}).get(str(label)) != {"1": 300}
                for label in range(10)
            )
        ):
            raise AcceptanceError("MNIST demo2 upload metadata differs from the original sample")
        result["upload_status"] = upload_status
        result["image_count"] = metadata["image_count"]

        action = {
            "action": "preprocess_dataset",
            "parameters": {
                "dataset_name": DATASET_NAME,
                "normalization": "standard",
                "dataset_split": "stratified",
                "target_size": {"height": 28, "width": 28},
                "seed": SEED,
            },
        }
        reason = "Measure real PC preprocessing and shard shipping for a second copy of the fixed-seed MNIST sample."
        evidence = {
            "dataset_name": DATASET_NAME,
            "image_count": 3000,
            "channels": 1,
            "target_size": {"height": 28, "width": 28},
            "seed": SEED,
            "preprocessing_on_pc": True,
            "telemetry_capture": True,
        }
        dispatched = client.request(
            "POST",
            "/api/autonomy/dispatch",
            {"action": action, "reason": reason, "evidence": evidence},
        )
        decision = dispatched.get("decision")
        if (
            dispatched.get("status") != "pending"
            or not isinstance(decision, dict)
            or decision.get("status") != "PENDING"
            or decision.get("action") != "preprocess_dataset"
            or decision.get("parameters") != action["parameters"]
            or not isinstance(decision.get("id"), str)
        ):
            raise AcceptanceError("MNIST demo2 preprocessing did not create its expected ASK_HUMAN decision")
        result["decision_id"] = decision["id"]

        sampler = _MemorySampler()
        with _sample_tegrastats(settings, "mnist-demo2-preprocess"):
            with sampler:
                action_started = time.perf_counter()
                approved = client.request(
                    "POST",
                    f"/api/autonomy/decisions/{decision['id']}/approve",
                    timeout=1200,
                )
                result["approval_elapsed_seconds"] = round(
                    time.perf_counter() - action_started, 3
                )
        report = _dataset_report(approved)
        result["pc_peak_system_used_mb"] = sampler.result["peak_system_used_mb"]
        result["pc_memory_samples"] = sampler.result["samples"]
        result["preprocessing_duration_seconds"] = report.get(
            "preprocessing_duration_seconds"
        )
        result["shipping_duration_seconds"] = report.get("shipping_duration_seconds")
        if result["shipping_duration_seconds"] is None:
            raise AcceptanceError("Backend report omitted shipping_duration_seconds")
        result["channels"] = report.get("channels")
        result["image_shape"] = report.get("image_shape")
        result["split_sizes"] = report.get("split_sizes")
        result["worker_rows_and_hashes"] = _worker_rows(report, workers)
        result["status"] = "PASS"
    except Exception as error:
        result["status"] = "FAIL"
        result["error_type"] = type(error).__name__
        raise
    finally:
        if sampler is not None:
            result["pc_peak_system_used_mb"] = sampler.result["peak_system_used_mb"]
            result["pc_memory_samples"] = sampler.result["samples"]
        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        OUTPUT_PATH.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"telemetry_json={OUTPUT_PATH}")
        print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
