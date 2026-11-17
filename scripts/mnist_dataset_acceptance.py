"""Upload real MNIST and approve PC preprocessing/sharding through the backend API."""

from __future__ import annotations

import ctypes
import hashlib
import io
import json
import os
import threading
import time
import zipfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator
from urllib.error import HTTPError
from urllib.parse import quote
from urllib.request import Request, urlopen

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(ROOT / "dtn" / "backend"))
sys.path.insert(0, str(ROOT / "scripts"))

from app.core.config import DeviceConfig, Settings  # noqa: E402
from app.services.worker_shards import (  # noqa: E402
    get_worker_dataset_metadata,
    get_worker_training_status,
    WorkerShardError,
)
from m5_feedback_loop import BackendClient, _sample_tegrastats  # noqa: E402

MNIST_ARCHIVE = Path.home() / ".keras" / "datasets" / "mnist.npz"
MNIST_SHA256 = "731c5ac602752760c8e48fbffcf8c3b850d9dc2a2aedcf2cc48468fc17b673d1"
DATASET_NAME = "mnist_demo"
UPLOAD_FILENAME = "mnist_demo_upload.zip"
SAMPLE_PER_CLASS = 300
SEED = 42


class AcceptanceError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class _MemoryStatus(ctypes.Structure):
    _fields_ = [
        ("dwLength", ctypes.c_ulong),
        ("dwMemoryLoad", ctypes.c_ulong),
        ("ullTotalPhys", ctypes.c_ulonglong),
        ("ullAvailPhys", ctypes.c_ulonglong),
        ("ullTotalPageFile", ctypes.c_ulonglong),
        ("ullAvailPageFile", ctypes.c_ulonglong),
        ("ullTotalVirtual", ctypes.c_ulonglong),
        ("ullAvailVirtual", ctypes.c_ulonglong),
        ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
    ]


def _system_memory_used_mb() -> float:
    if os.name != "nt":
        raise AcceptanceError("PC peak RAM sampling requires Windows GlobalMemoryStatusEx")
    status = _MemoryStatus()
    status.dwLength = ctypes.sizeof(status)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        raise OSError("GlobalMemoryStatusEx failed")
    return (status.ullTotalPhys - status.ullAvailPhys) / (1024 * 1024)


class _MemorySampler:
    def __init__(self) -> None:
        self._stop = threading.Event()
        self._peak_mb = 0.0
        self._samples = 0
        self._error: Exception | None = None
        self._thread = threading.Thread(target=self._sample, daemon=True)

    def _sample(self) -> None:
        while not self._stop.is_set():
            try:
                self._peak_mb = max(self._peak_mb, _system_memory_used_mb())
                self._samples += 1
            except Exception as error:
                self._error = error
                self._stop.set()
                return
            self._stop.wait(0.5)

    def __enter__(self) -> _MemorySampler:
        self._thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self._stop.set()
        self._thread.join(timeout=3)
        if self._thread.is_alive():
            raise AcceptanceError("PC memory sampler did not stop")
        if self._error is not None:
            raise AcceptanceError(f"PC memory sampling failed: {self._error}") from self._error
        if self._samples == 0:
            raise AcceptanceError("PC memory sampler collected no samples")

    @property
    def result(self) -> dict[str, int | float]:
        return {"peak_system_used_mb": round(self._peak_mb, 1), "samples": self._samples}


def _multipart_upload(
    client: BackendClient,
    dataset_name: str,
    filename: str,
    archive: bytes,
) -> tuple[int, dict[str, Any]]:
    boundary = f"----mnist-{hashlib.sha256(os.urandom(16)).hexdigest()}"
    parts = [
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"dataset_name\"\r\n\r\n{dataset_name}\r\n".encode(),
        (
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; "
            f"filename=\"{filename}\"\r\nContent-Type: application/zip\r\n\r\n"
        ).encode()
        + archive
        + b"\r\n",
        f"--{boundary}--\r\n".encode(),
    ]
    request = Request(
        client.base_url + "/api/datasets/upload",
        data=b"".join(parts),
        headers={
            **client.headers,
            "Content-Type": f"multipart/form-data; boundary={boundary}",
        },
        method="POST",
    )
    try:
        with urlopen(request, timeout=120) as response:
            return response.status, json.loads(response.read().decode("utf-8"))
    except HTTPError as error:
        body = error.read().decode("utf-8", errors="replace")
        try:
            detail = json.loads(body).get("detail", body)
        except json.JSONDecodeError:
            detail = body
        return error.code, {"detail": str(detail)}


def _read_mnist() -> tuple[np.ndarray, np.ndarray]:
    if not MNIST_ARCHIVE.is_file():
        raise AcceptanceError(f"Official MNIST cache is missing: {MNIST_ARCHIVE}")
    actual_hash = _sha256(MNIST_ARCHIVE)
    if actual_hash != MNIST_SHA256:
        raise AcceptanceError("Cached MNIST archive SHA-256 does not match the official archive")
    with np.load(MNIST_ARCHIVE, allow_pickle=False) as archive:
        images = np.asarray(archive["x_train"])
        labels = np.asarray(archive["y_train"])
    if images.shape != (60_000, 28, 28) or labels.shape != (60_000,):
        raise AcceptanceError("Official MNIST training archive has unexpected dimensions")
    if images.dtype != np.uint8 or not np.array_equal(np.unique(labels), np.arange(10)):
        raise AcceptanceError("Official MNIST arrays have unexpected types or labels")
    return images, labels


def _build_upload_archive(
    images: np.ndarray,
    labels: np.ndarray,
    output_path: Path,
) -> dict[str, int]:
    rng = np.random.default_rng(SEED)
    selected_by_label = {
        label: sorted(
            int(value)
            for value in rng.choice(
                np.flatnonzero(labels == label),
                size=SAMPLE_PER_CLASS,
                replace=False,
            )
        )
        for label in range(10)
    }
    counts: dict[str, int] = {}
    if output_path.exists():
        with zipfile.ZipFile(output_path, "r") as zipped:
            infos = [item for item in zipped.infolist() if not item.is_dir()]
            if len(infos) != 3_000:
                raise AcceptanceError("Existing MNIST upload ZIP does not contain 3,000 images")
            observed: dict[int, set[int]] = {label: set() for label in range(10)}
            for info in infos:
                parts = info.filename.split("/")
                if (
                    len(parts) != 2
                    or not parts[0].isdigit()
                    or not parts[1].endswith(".png")
                    or not parts[1][:-4].isdigit()
                ):
                    raise AcceptanceError("Existing MNIST ZIP has an unexpected image path")
                label = int(parts[0])
                source_index = int(parts[1][:-4])
                if (
                    label not in selected_by_label
                    or source_index not in selected_by_label[label]
                    or source_index in observed[label]
                    or int(labels[source_index]) != label
                ):
                    raise AcceptanceError("Existing MNIST ZIP does not match the fixed-seed sample")
                with Image.open(io.BytesIO(zipped.read(info))) as image:
                    if image.mode != "L" or not np.array_equal(
                        np.asarray(image),
                        images[source_index],
                    ):
                        raise AcceptanceError(
                            "Existing MNIST ZIP contains pixels not matching official MNIST"
                        )
                observed[label].add(source_index)
            if any(
                observed[label] != set(selected_by_label[label])
                for label in range(10)
            ):
                raise AcceptanceError("Existing MNIST ZIP has incomplete class samples")
        return {str(label): SAMPLE_PER_CLASS for label in range(10)}

    with zipfile.ZipFile(output_path, "x", compression=zipfile.ZIP_STORED) as zipped:
        for label in range(10):
            for source_index in selected_by_label[label]:
                image_buffer = io.BytesIO()
                Image.fromarray(images[source_index]).save(
                    image_buffer,
                    format="PNG",
                )
                zipped.writestr(
                    f"{label}/{source_index:05d}.png",
                    image_buffer.getvalue(),
                )
            counts[str(label)] = len(selected_by_label[label])
    if sum(counts.values()) != 3_000:
        raise AcceptanceError("MNIST upload archive does not contain exactly 3,000 images")
    return counts


def _check_preflight(
    client: BackendClient,
    settings: Settings,
    workers: list[DeviceConfig],
) -> None:
    state = client.request("GET", "/api/autonomy/state")
    if (
        state.get("mode") != "ASK_HUMAN"
        or state.get("stop_requested") is not False
        or state.get("pending_decisions") != 0
        or any(loop.get("status") == "RUNNING" for loop in state.get("loops", []))
    ):
        raise AcceptanceError("MNIST action requires ASK_HUMAN, STOP off, and no pending/running work")
    results = client.request("GET", "/api/slm/results")
    if results.get("active_group_id") is not None:
        raise AcceptanceError("Master training/NAS group is active")
    if [worker.node_id for worker in workers] != ["worker1", "worker2", "worker3"]:
        raise AcceptanceError("Expected worker1, worker2, and worker3")
    for worker in workers:
        status = get_worker_training_status(worker)
        if status.get("active_run_id") is not None or status.get("scheduled_run_id") is not None:
            raise AcceptanceError(f"{worker.node_id} is not idle")
        print(f"{worker.node_id}: authenticated and idle")
    print("backend: ASK_HUMAN; STOP inactive; no active master group")


def _assert_worker_absent(worker: DeviceConfig, dataset_name: str) -> None:
    try:
        get_worker_dataset_metadata(worker, dataset_name)
    except WorkerShardError as error:
        if "HTTP 404" not in str(error):
            raise AcceptanceError(
                f"{worker.node_id} absence check failed: {error}"
            ) from error
        return
    raise AcceptanceError(
        f"{worker.node_id} unexpectedly already contains {dataset_name}"
    )


def _expect_rejection(
    client: BackendClient,
    root: Path,
    dataset_name: str,
    filename: str,
    archive: bytes,
    *,
    expected_status: int,
    expected_text: str,
) -> None:
    status, response = _multipart_upload(client, dataset_name, filename, archive)
    detail = str(response.get("detail", ""))
    if status != expected_status or expected_text.lower() not in detail.lower():
        raise AcceptanceError(
            f"Negative upload {dataset_name} returned HTTP {status} with unexpected detail"
        )
    candidate = (root / dataset_name).resolve()
    if candidate.exists():
        raise AcceptanceError(f"Rejected upload left a dataset directory: {dataset_name}")
    if any(candidate.parent.glob(f".{candidate.name}-*")):
        raise AcceptanceError(f"Rejected upload left a staging directory: {dataset_name}")
    print(f"negative upload {dataset_name}: rejected HTTP {status}; {detail}")


def _zip_with_entry(path: str, payload: bytes) -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as zipped:
        zipped.writestr(path, payload)
    return output.getvalue()


def _approved_report(envelope: Any) -> dict[str, Any]:
    if not isinstance(envelope, dict) or envelope.get("status") != "approved":
        raise AcceptanceError("Preprocessing decision did not return approved status")
    result = envelope.get("result")
    report = result.get("data") if isinstance(result, dict) else None
    if (
        not isinstance(report, dict)
        or report.get("dataset_name") != DATASET_NAME
        or not isinstance(report.get("workers"), dict)
    ):
        raise AcceptanceError("Approved action response omitted the preprocessing report")
    return report


def _verify_report(
    report: dict[str, Any],
    dataset_root: Path,
    workers: list[DeviceConfig],
) -> dict[str, Any]:
    if (
        report.get("image_count") != 3_000
        or report.get("source_class_counts") != {str(label): 300 for label in range(10)}
        or report.get("target_size") != {"height": 28, "width": 28}
        or report.get("channels") != 1
        or report.get("image_shape") != [28, 28, 1]
        or report.get("normalization", {}).get("method") != "standard"
        or report.get("split_strategy") != "stratified"
        or report.get("seed") != SEED
        or report.get("split_indices_disjoint") is not True
        or report.get("worker_shards_disjoint") is not True
    ):
        raise AcceptanceError("Preprocessing report does not match MNIST acceptance parameters")
    expected_split_counts = {
        "train": {str(label): 240 for label in range(10)},
        "validation": {str(label): 30 for label in range(10)},
        "test": {str(label): 30 for label in range(10)},
    }
    if report.get("split_class_counts") != expected_split_counts:
        raise AcceptanceError("MNIST split class counts do not match the expected stratified split")
    if report.get("split_sizes") != {"train": 2_400, "validation": 300, "test": 300}:
        raise AcceptanceError("MNIST split sizes do not match 80/10/10")

    worker_reports = report.get("workers", {})
    uploaded_workers = report.get("uploaded_workers", {})
    if set(worker_reports) != {worker.node_id for worker in workers}:
        raise AcceptanceError("Preprocessing report omitted a configured worker")
    if set(uploaded_workers) != {worker.node_id for worker in workers}:
        raise AcceptanceError("Not all worker shards were successfully uploaded")
    if sum(int(item["rows"]) for item in worker_reports.values()) != 2_400:
        raise AcceptanceError("Worker sample counts do not add up to all MNIST training data")

    pc_hashes: dict[str, str] = {}
    total_rows = 0
    for worker in workers:
        expected = worker_reports[worker.node_id]
        shard_path = Path(expected["shard_path"])
        if not shard_path.is_file():
            raise AcceptanceError(f"PC shard file is missing for {worker.node_id}")
        local_hash = _sha256(shard_path)
        installed = get_worker_dataset_metadata(worker, DATASET_NAME)
        uploaded = uploaded_workers[worker.node_id]
        if (
            installed.get("rows") != expected["rows"]
            or installed.get("class_counts") != expected["class_counts"]
            or installed.get("shape", [])[1:] != [28, 28, 1]
            or installed.get("channels") != 1
            or installed.get("image_size") != [28, 28]
            or installed.get("shard_sha256") != local_hash
            or uploaded.get("shard_sha256") != local_hash
        ):
            raise AcceptanceError(f"{worker.node_id} installed shard does not match the PC shard")
        if expected["class_counts"] != {str(label): 80 for label in range(10)}:
            raise AcceptanceError(f"{worker.node_id} does not have 80 train images per class")
        pc_hashes[worker.node_id] = local_hash
        total_rows += int(installed["rows"])
        print(
            f"{worker.node_id}: rows={installed['rows']}; "
            f"class_counts={json.dumps(installed['class_counts'], sort_keys=True)}; "
            f"PC_sha256={local_hash}; Jetson_sha256={installed['shard_sha256']}"
        )
    if total_rows != report["split_sizes"]["train"]:
        raise AcceptanceError("Installed worker samples do not sum to the training split")
    return pc_hashes


def main() -> None:
    settings = Settings.from_env()
    client = BackendClient(f"http://127.0.0.1:{settings.port}", settings.api_token)
    workers = sorted(
        (node for node in settings.nodes if node.role == "worker"),
        key=lambda node: node.node_id,
    )
    dataset_root = settings.dataset_root
    dataset_root.mkdir(parents=True, exist_ok=True)
    archive_path = dataset_root / UPLOAD_FILENAME
    if (dataset_root / DATASET_NAME).exists():
        raise AcceptanceError(f"Dataset {DATASET_NAME} already exists on the PC")
    images, labels = _read_mnist()
    class_counts = _build_upload_archive(images, labels, archive_path)
    archive = archive_path.read_bytes()
    print(
        f"official_mnist_path={MNIST_ARCHIVE}; bytes={MNIST_ARCHIVE.stat().st_size}; "
        f"sha256={MNIST_SHA256}; full_train_images={len(labels)}; full_test_images=10000; "
        f"sample_counts={json.dumps(class_counts, sort_keys=True)}"
    )
    print(
        f"upload_zip={archive_path}; bytes={len(archive)}; "
        f"sha256={hashlib.sha256(archive).hexdigest()}"
    )

    _check_preflight(client, settings, workers)
    for worker in workers:
        _assert_worker_absent(worker, DATASET_NAME)

    traversal_name = "mnist_path_traversal_negative"
    corrupt_name = "mnist_corrupt_negative"
    traversal = _zip_with_entry(
        "0/../escape.png",
        b"valid-path-payload-is-not-reached",
    )
    _expect_rejection(
        client,
        dataset_root,
        traversal_name,
        "traversal.zip",
        traversal,
        expected_status=422,
        expected_text="path traversal",
    )
    invalid_png = _zip_with_entry("0/corrupt.png", b"not a PNG image")
    _expect_rejection(
        client,
        dataset_root,
        corrupt_name,
        "corrupt.png.zip",
        invalid_png,
        expected_status=422,
        expected_text="invalid or corrupt image",
    )
    _expect_rejection(
        client,
        dataset_root,
        "../mnist-invalid-name",
        "invalid-name.zip",
        archive,
        expected_status=422,
        expected_text="dataset_name must match",
    )

    upload_status, metadata = _multipart_upload(
        client,
        DATASET_NAME,
        UPLOAD_FILENAME,
        archive,
    )
    if upload_status != 201:
        raise AcceptanceError(f"MNIST upload returned unexpected HTTP {upload_status}")
    listed = client.request("GET", "/api/datasets")
    if not any(item.get("dataset_name") == DATASET_NAME for item in listed):
        raise AcceptanceError("Dataset list did not show mnist_demo after upload")
    if (
        metadata.get("image_count") != 3_000
        or metadata.get("class_counts") != class_counts
        or any(
            metadata.get("image_channels", {}).get(str(label)) != {"1": 300}
            for label in range(10)
        )
    ):
        raise AcceptanceError("MNIST upload metadata has incorrect image/channel counts")
    metadata_path = dataset_root / DATASET_NAME / "metadata.json"
    metadata_hash_before_duplicate = _sha256(metadata_path)
    duplicate_status, duplicate = _multipart_upload(
        client,
        DATASET_NAME,
        UPLOAD_FILENAME,
        archive,
    )
    if duplicate_status != 409 or "already exists" not in str(duplicate.get("detail", "")).lower():
        raise AcceptanceError("Duplicate mnist_demo upload was not rejected clearly")
    if _sha256(metadata_path) != metadata_hash_before_duplicate:
        raise AcceptanceError("Duplicate upload changed the existing PC dataset metadata")
    print("negative upload mnist_demo duplicate: rejected HTTP 409; existing upload unchanged")
    for worker in workers:
        _assert_worker_absent(worker, traversal_name)
        _assert_worker_absent(worker, corrupt_name)
        _assert_worker_absent(worker, DATASET_NAME)
    if (dataset_root / traversal_name).exists() or (dataset_root / corrupt_name).exists():
        raise AcceptanceError("A rejected negative upload left PC dataset contents")
    if any(dataset_root.glob(f".{traversal_name}-*")) or any(
        dataset_root.glob(f".{corrupt_name}-*")
    ):
        raise AcceptanceError("A rejected negative upload left PC staging contents")
    print("negative upload cleanup: no rejected test dataset exists on PC or workers")

    _check_preflight(client, settings, workers)
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
    reason = (
        "Preprocess the uploaded real MNIST training sample on the PC and ship "
        "stratified worker shards (28x28, standard normalization, seed 42)."
    )
    evidence = {
        "dataset_name": DATASET_NAME,
        "image_count": 3_000,
        "class_counts": class_counts,
        "target_size": {"height": 28, "width": 28},
        "channels": 1,
        "split": {"train": 0.8, "validation": 0.1, "test": 0.1},
        "seed": SEED,
        "preprocessing_on_pc": True,
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
        or decision.get("reason") != reason
        or decision.get("evidence") != evidence
        or not isinstance(decision.get("id"), str)
    ):
        raise AcceptanceError("MNIST preprocess action did not create the expected pending decision")
    print(
        f"pending decision_id={decision['id']}; reason={decision['reason']}; "
        f"evidence={json.dumps(decision['evidence'], sort_keys=True)}"
    )
    with _sample_tegrastats(settings, "mnist-shard-shipping"):
        with _MemorySampler() as pc_memory:
            started = time.perf_counter()
            approved = client.request(
                "POST",
                f"/api/autonomy/decisions/{decision['id']}/approve",
                timeout=1200,
            )
            action_elapsed = round(time.perf_counter() - started, 3)
    report = _approved_report(approved)
    history = client.request("GET", "/api/autonomy/history")
    events = [
        item.get("event")
        for item in history
        if item.get("decision_id") == decision["id"]
    ]
    if not {"created", "approved"}.issubset(events):
        raise AcceptanceError("MNIST preprocess decision history omitted approval")

    shard_hashes = _verify_report(report, dataset_root, workers)
    after_upload = client.request("GET", f"/api/datasets/{DATASET_NAME}")
    if after_upload.get("image_channels") != metadata.get("image_channels"):
        raise AcceptanceError("PC dataset metadata changed unexpectedly during preprocessing")
    print(
        "MNIST preprocessing result="
        + json.dumps(
            {
                "decision_id": decision["id"],
                "history_events": events,
                "image_count": report["image_count"],
                "source_class_counts": report["source_class_counts"],
                "source_image_channels": metadata["image_channels"],
                "target_size": report["target_size"],
                "channels": report["channels"],
                "image_shape": report["image_shape"],
                "normalization": report["normalization"]["method"],
                "split_sizes": report["split_sizes"],
                "split_class_counts": report["split_class_counts"],
                "split_indices_disjoint": report["split_indices_disjoint"],
                "worker_shards_disjoint": report["worker_shards_disjoint"],
                "worker_rows_total": sum(
                    item["rows"] for item in report["workers"].values()
                ),
                "preprocessing_duration_seconds": report[
                    "preprocessing_duration_seconds"
                ],
                "shipping_duration_seconds": report.get("shipping_duration_seconds"),
                "action_elapsed_seconds": action_elapsed,
                "pc_peak_memory": pc_memory.result,
                "shard_hashes": shard_hashes,
            },
            sort_keys=True,
        )
    )
    print("DATASET STEP (MNIST) PASS")


if __name__ == "__main__":
    main()
