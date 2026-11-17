"""Upload and preprocess an explicit 1,000-per-class CIFAR-10 image dataset."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
import tempfile
import uuid
import zipfile
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "dtn" / "backend"))
sys.path.insert(0, str(ROOT))

from app.core.config import Settings  # noqa: E402
from app.services.worker_shards import (  # noqa: E402
    get_worker_dataset_metadata,
    get_worker_training_status,
)
from prepare_cifar10 import CLASS_NAMES, load_cifar10  # noqa: E402
from m4_hardware_check import BackendClient, HardwareCheckError, _approved_action_data, _dispatch_and_approve  # noqa: E402


class UploadFlowError(RuntimeError):
    pass


def _build_zip(archive_path: Path, output_path: Path, per_class: int) -> None:
    images, labels, _, _ = load_cifar10(archive_path)
    if images.shape != (50_000, 32, 32, 3) or labels.shape != (50_000,):
        raise UploadFlowError("Local CIFAR training archive has unexpected dimensions")
    if per_class < 1:
        raise UploadFlowError("per_class must be positive")
    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED) as zipped:
        for label, class_name in enumerate(CLASS_NAMES):
            indices = np.flatnonzero(labels == label)[:per_class]
            if indices.size != per_class:
                raise UploadFlowError(
                    f"Local archive has only {indices.size} images for {class_name}"
                )
            for ordinal, index in enumerate(indices):
                buffer = io.BytesIO()
                Image.fromarray(images[index]).save(buffer, format="PNG", optimize=True)
                zipped.writestr(
                    f"{class_name}/{ordinal:04d}.png",
                    buffer.getvalue(),
                )


def _upload_zip(
    backend_url: str,
    token: str,
    dataset_name: str,
    zip_path: Path,
) -> dict[str, Any]:
    boundary = "----dfl-dataset-" + uuid.uuid4().hex
    archive = zip_path.read_bytes()
    chunks = [
        f"--{boundary}\r\n".encode("ascii"),
        b'Content-Disposition: form-data; name="dataset_name"\r\n\r\n',
        dataset_name.encode("ascii"),
        b"\r\n",
        f"--{boundary}\r\n".encode("ascii"),
        (
            'Content-Disposition: form-data; name="file"; '
            f'filename="{dataset_name}.zip"\r\n'
        ).encode("ascii"),
        b"Content-Type: application/zip\r\n\r\n",
        archive,
        b"\r\n",
        f"--{boundary}--\r\n".encode("ascii"),
    ]
    request = Request(
        backend_url.rstrip("/") + "/api/datasets/upload",
        data=b"".join(chunks),
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/json",
            "Content-Type": f"multipart/form-data; boundary={boundary}",
        },
        method="POST",
    )
    try:
        with urlopen(request, timeout=300) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as error:
        raise UploadFlowError(
            f"Real dataset upload returned HTTP {error.code}"
        ) from error
    except (URLError, TimeoutError, OSError) as error:
        raise UploadFlowError(
            f"Real dataset upload failed: {type(error).__name__}: {error}"
        ) from error
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise UploadFlowError("Dataset upload returned invalid JSON") from error


def _verify_shards(
    dataset_root: Path,
    report: dict[str, Any],
    workers: list[Any],
) -> None:
    run_root = Path(report["output_path"])
    split_sizes = report["split_sizes"]
    if split_sizes != {"train": 8000, "validation": 1000, "test": 1000}:
        raise UploadFlowError(f"Unexpected stratified split sizes: {split_sizes}")
    worker_rows: set[str] = set()
    row_hashes: dict[str, set[str]] = {}
    total_rows = 0
    for worker in workers:
        worker_id = worker.node_id
        worker_report = report["workers"][worker_id]
        shard_path = Path(worker_report["shard_path"])
        expected = report["uploaded_workers"][worker_id]
        pc_hash = hashlib.sha256(shard_path.read_bytes()).hexdigest()
        if expected["shard_sha256"] != pc_hash:
            raise UploadFlowError(f"{worker_id} PC report hash differs from local shard")
        installed = get_worker_dataset_metadata(worker, report["dataset_name"])
        if (
            installed.get("rows") != worker_report["rows"]
            or installed.get("class_counts") != worker_report["class_counts"]
            or installed.get("shard_sha256") != pc_hash
        ):
            raise UploadFlowError(f"{worker_id} authenticated shard metadata/hash mismatch")
        counts = worker_report["class_counts"]
        rows = int(worker_report["rows"])
        total_rows += rows
        row_hashes[worker_id] = set()
        with np.load(shard_path, allow_pickle=False) as shard:
            images = np.asarray(shard["images"])
            labels = np.asarray(shard["labels"], dtype=np.int64)
        if len(images) != rows or len(labels) != rows:
            raise UploadFlowError(f"{worker_id} shard arrays do not match reported rows")
        for image, label in zip(images, labels, strict=True):
            row_hashes[worker_id].add(
                hashlib.sha256(
                    np.ascontiguousarray(image).tobytes()
                    + int(label).to_bytes(4, "little", signed=False)
                ).hexdigest()
            )
        print(
            f"{worker_id}: rows={rows}; class_counts={json.dumps(counts, sort_keys=True)}; "
            f"PC_shard_sha256={pc_hash}; Jetson_shard_sha256={installed['shard_sha256']}"
        )
    if total_rows != split_sizes["train"]:
        raise UploadFlowError(
            f"Worker total {total_rows} does not equal training rows {split_sizes['train']}"
        )
    worker_ids = [worker.node_id for worker in workers]
    for left_index, left in enumerate(worker_ids):
        for right in worker_ids[left_index + 1 :]:
            if row_hashes[left] & row_hashes[right]:
                raise UploadFlowError(f"Worker shards overlap: {left} and {right}")
    print(
        f"split_sizes={json.dumps(split_sizes, sort_keys=True)}; "
        f"worker_rows_sum={total_rows}; pairwise_worker_overlap=none"
    )
    print("per-class source counts=" + json.dumps(report["source_class_counts"], sort_keys=True))


def run(args: argparse.Namespace) -> None:
    settings = Settings.from_env()
    backend_url = args.backend_url.rstrip("/")
    backend = BackendClient(backend_url, settings.api_token)
    workers = sorted(
        (node for node in settings.nodes if node.role == "worker"),
        key=lambda node: node.node_id,
    )
    if [node.node_id for node in workers] != ["worker1", "worker2", "worker3"]:
        raise UploadFlowError("Expected exactly worker1, worker2, and worker3")
    state = backend.request("GET", "/api/autonomy/state")
    if state.get("mode") != "ASK_HUMAN" or state.get("stop_requested") is not False:
        raise UploadFlowError("Autonomy must be ASK_HUMAN with global STOP inactive")
    if state.get("pending_decisions"):
        raise UploadFlowError("Review pending autonomy decisions before dataset upload")
    existing = backend.request("GET", "/api/datasets")
    uploaded_metadata = next(
        (
            item
            for item in existing
            if item.get("dataset_name") == args.dataset_name
        ),
        None,
    )
    if args.per_class != 1000:
        raise UploadFlowError("This acceptance requires exactly 1,000 images per class")
    for worker in workers:
        status = get_worker_training_status(worker)
        print(
            f"{worker.node_id}: idle; active_run_id={status.get('active_run_id')}; "
            f"scheduled_run_id={status.get('scheduled_run_id')}"
        )

    expected_counts = {name: 1000 for name in CLASS_NAMES}
    if uploaded_metadata is None:
        with tempfile.TemporaryDirectory(prefix="cifar-10k-upload-") as directory:
            zip_path = Path(directory) / f"{args.dataset_name}.zip"
            _build_zip(args.cifar_archive, zip_path, args.per_class)
            print(f"generated upload archive bytes={zip_path.stat().st_size}")
            uploaded_metadata = _upload_zip(
                backend_url,
                settings.api_token,
                args.dataset_name,
                zip_path,
            )
        if (
            uploaded_metadata.get("image_count") != 10_000
            or uploaded_metadata.get("class_counts") != expected_counts
        ):
            raise UploadFlowError(
                "Upload metadata does not contain exactly 10,000 images balanced across classes"
            )
        print(f"real upload complete: dataset_name={args.dataset_name}; image_count=10000")
    elif (
        uploaded_metadata.get("image_count") == 10_000
        and uploaded_metadata.get("class_counts") == expected_counts
        and uploaded_metadata.get("classes") == list(CLASS_NAMES)
    ):
        print(
            "existing upload confirmed read-only; "
            f"dataset_name={args.dataset_name}; image_count=10000"
        )
    else:
        raise UploadFlowError(
            "An existing dataset with this name does not match the expected 10k balanced upload; "
            "refusing to overwrite or preprocess it"
        )

    _, upload_result = _dispatch_and_approve(
        backend,
        action={
            "action": "upload_dataset",
            "parameters": {"dataset_name": args.dataset_name},
        },
        reason="Confirm the explicitly uploaded 1,000-per-class CIFAR image dataset is available.",
        evidence={"dataset_name": args.dataset_name, "total_images": 10_000},
    )
    if (
        upload_result.get("image_count") != 10_000
        or upload_result.get("class_counts") != expected_counts
    ):
        raise UploadFlowError("Approved upload_dataset action returned unexpected metadata")
    print("upload_dataset ASK_HUMAN approval: PASS")

    for worker in workers:
        status = get_worker_training_status(worker)
        if status.get("active_run_id") is not None or status.get("scheduled_run_id") is not None:
            raise UploadFlowError(f"{worker.node_id} became busy before preprocessing")
    _, preprocess_result = _dispatch_and_approve(
        backend,
        action={
            "action": "preprocess_dataset",
            "parameters": {
                "dataset_name": args.dataset_name,
                "normalization": "standard",
                "dataset_split": "stratified",
            },
        },
        reason="Preprocess and shard the explicitly named 10,000-image dataset.",
        evidence={
            "dataset_name": args.dataset_name,
            "normalization": "standard",
            "dataset_split": "stratified",
            "workers_idle_checked": True,
        },
    )
    if preprocess_result.get("dataset_name") != args.dataset_name:
        raise UploadFlowError("Approved preprocessing result used a different dataset")
    _verify_shards(settings.dataset_root, preprocess_result, workers)
    print("real upload -> ASK_HUMAN preprocess -> worker shard flow: PASS")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--per-class", type=int, default=1000)
    parser.add_argument(
        "--cifar-archive",
        type=Path,
        default=ROOT / "data" / "raw" / "cifar-10-python.tar.gz",
    )
    parser.add_argument("--backend-url", default="http://127.0.0.1:8000")
    args = parser.parse_args()
    if not args.cifar_archive.is_file():
        raise SystemExit(f"CIFAR source archive not found: {args.cifar_archive}")
    try:
        run(args)
    except (UploadFlowError, HardwareCheckError) as error:
        raise SystemExit(f"cifar_10k dataset flow FAILED: {error}") from error


if __name__ == "__main__":
    main()
