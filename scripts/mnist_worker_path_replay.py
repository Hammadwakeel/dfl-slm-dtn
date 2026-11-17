"""PC-only federated replay through WorkerTrainer.run_round; never reads test.npz."""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np

from common.nas import FIXED_ARCHITECTURE
from worker import training_service
from worker.training_service import Peer, WorkerConfig, WorkerTrainer


DATASET_ROOT = ROOT / "data" / "datasets" / "mnist_demo"
WORKER_IDS = ("worker1", "worker2", "worker3")
SEED = 2026
ROUNDS = 5
LOCAL_EPOCHS = 20
TOPOLOGY = "bidirectional_ring"
EXPECTED_ARCHITECTURE_SHA256 = "b66616d37317a211cd9dc2cb8d02b6213cfc292d049a2258dc7711d1e7f73bb2"
EXPECTED_WEIGHTS_SHA256 = "d23e3d771bb065c347655ac932ab90cc3a8ce12ea01345c533ffca004be78d75"


class _PeerResponse:
    status = 202

    def __enter__(self) -> "_PeerResponse":
        return self

    def __exit__(self, *_: Any) -> None:
        return None


def _latest_preprocessing_report() -> tuple[Path, dict[str, Any]]:
    reports = sorted(
        (DATASET_ROOT / "processed").glob("run-*/preprocessing_report.json"),
        key=lambda path: path.stat().st_mtime_ns,
    )
    if not reports:
        raise FileNotFoundError("Persisted mnist_demo preprocessing report is missing")
    report_path = reports[-1]
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("dataset_name") != "mnist_demo":
        raise ValueError("Latest preprocessing report is not mnist_demo")
    return report_path, report


def _verify_local_inputs(report_path: Path, report: dict[str, Any]) -> dict[str, Path]:
    model_dir = DATASET_ROOT / "models" / "initial_model"
    architecture_path = model_dir / "architecture.json"
    weights_path = model_dir / "initial_weights.npz"
    metadata_path = model_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    architecture_hash = hashlib.sha256(architecture_path.read_bytes()).hexdigest()
    weights_hash = hashlib.sha256(weights_path.read_bytes()).hexdigest()
    if (
        architecture_hash != EXPECTED_ARCHITECTURE_SHA256
        or weights_hash != EXPECTED_WEIGHTS_SHA256
        or metadata.get("architecture_sha256") != architecture_hash
        or metadata.get("weights_sha256") != weights_hash
    ):
        raise ValueError("Local installed initial-model hashes do not match expected metadata")

    worker_paths: dict[str, Path] = {}
    for worker_id in WORKER_IDS:
        shard_path = report_path.parent / "workers" / f"{worker_id}.npz"
        expected_hash = report.get("uploaded_workers", {}).get(worker_id, {}).get("shard_sha256")
        actual_hash = hashlib.sha256(shard_path.read_bytes()).hexdigest()
        if actual_hash != expected_hash:
            raise ValueError(f"Local shard hash mismatch for {worker_id}")
        with np.load(shard_path, allow_pickle=False) as shard:
            images = np.asarray(shard["images"])
            labels = np.asarray(shard["labels"]).reshape(-1)
        if images.shape != (800, 28, 28, 1) or labels.shape != (800,):
            raise ValueError(f"Unexpected shard shape for {worker_id}: {images.shape}, {labels.shape}")
        worker_paths[worker_id] = shard_path
        print(f"input {worker_id}: rows=800; local_shard_sha256={actual_hash}")
    print(
        "input initial_model: architecture_sha256="
        f"{architecture_hash}; weights_sha256={weights_hash}"
    )
    return worker_paths


def _prepare_local_trainers(
    temporary_root: Path,
    worker_paths: dict[str, Path],
) -> dict[str, WorkerTrainer]:
    source_model_dir = DATASET_ROOT / "models" / "initial_model"

    dummy_cifar_path = temporary_root / "unused_cifar_shard.npz"
    np.savez(
        dummy_cifar_path,
        images=np.zeros((1, 32, 32, 3), dtype=np.uint8),
        labels=np.array([0], dtype=np.int64),
    )
    workers: dict[str, WorkerTrainer] = {}
    for worker_id in WORKER_IDS:
        dataset_root = temporary_root / worker_id / "datasets"
        dataset_dir = dataset_root / "mnist_demo"
        dataset_dir.mkdir(parents=True)
        shutil.copyfile(worker_paths[worker_id], dataset_dir / "train.npz")
        model_dir = dataset_dir / "models" / "initial_model"
        model_dir.mkdir(parents=True)
        for name in ("architecture.json", "initial_weights.npz", "metadata.json"):
            shutil.copyfile(source_model_dir / name, model_dir / name)
        peers = tuple(
            Peer(peer_id, f"http://{peer_id}", "p" * 48)
            for peer_id in WORKER_IDS
            if peer_id != worker_id
        )
        config = WorkerConfig(
            node_id=worker_id,
            api_token="a" * 48,
            port=5004,
            data_path=str(dummy_cifar_path),
            dataset_root=str(dataset_root),
            peers=peers,
            log_path=str(temporary_root / f"{worker_id}.jsonl"),
            peer_timeout_seconds=120.0,
            dataset_learning_rate=0.01,
            dataset_batch_size=32,
        )
        trainer = WorkerTrainer(config)
        trainer.configure(
            rounds=ROUNDS,
            epochs=LOCAL_EPOCHS,
            topology=TOPOLOGY,
            architecture=FIXED_ARCHITECTURE,
            dataset_name="mnist_demo",
        )
        workers[worker_id] = trainer
    return workers


def _local_peer_urlopen(workers: dict[str, WorkerTrainer]):
    def local_urlopen(request, timeout=60):
        parsed = urlparse(request.full_url)
        receiver = workers.get(parsed.hostname or "")
        if receiver is None or parsed.path != "/v1/weights" or request.get_method() != "POST":
            raise RuntimeError("Unexpected URL in local-only peer transport")
        headers = {key.lower(): value for key, value in request.header_items()}
        receiver.accept_peer_weights(
            run_id=headers["x-run-id"],
            round_index=int(headers["x-round"]),
            sender=headers["x-sender-node"],
            sample_count=int(headers["x-sample-count"]),
            payload=request.data,
        )
        return _PeerResponse()

    return local_urlopen


def main() -> None:
    report_path, report = _latest_preprocessing_report()
    worker_paths = _verify_local_inputs(report_path, report)
    pc_results = json.loads(
        (ROOT / "data" / "mnist_pc_learnability_sweep.json").read_text(encoding="utf-8")
    )
    pc_reference = next(
        row
        for row in pc_results["federated_schedule_comparison"]
        if row.get("model_kind") == "initial_model_json"
        and row.get("rounds") == ROUNDS
        and row.get("local_epochs_per_round") == LOCAL_EPOCHS
        and row.get("seed") == SEED
    )

    with tempfile.TemporaryDirectory(prefix="mnist-worker-run-round-") as temporary:
        workers = _prepare_local_trainers(Path(temporary), worker_paths)
        run_id = "pc-worker-run-round-2026"
        for trainer in workers.values():
            if run_id in trainer._models:
                raise AssertionError("Replay must enter run_round with no pre-created model")
            if run_id in trainer._run_model_audits:
                raise AssertionError("Replay must enter run_round with no pre-created audit record")
            print(f"worker_model {trainer.config.node_id}: initialized_by_first_run_round=True")

        original_urlopen = training_service.urlopen
        original_memory = (
            training_service._memory_available_kb,
            training_service._resident_memory_mb,
            training_service._peak_process_rss_mb,
        )
        training_service.urlopen = _local_peer_urlopen(workers)
        training_service._memory_available_kb = lambda: 1024 * 1024
        training_service._resident_memory_mb = lambda: 256.0
        training_service._peak_process_rss_mb = lambda: 256.0
        try:
            for round_index in range(ROUNDS):
                with ThreadPoolExecutor(max_workers=3) as pool:
                    futures = {
                        worker_id: pool.submit(
                            trainer.run_round,
                            run_id,
                            round_index,
                            LOCAL_EPOCHS,
                            SEED,
                            TOPOLOGY,
                        )
                        for worker_id, trainer in workers.items()
                    }
                    rows = {worker_id: future.result() for worker_id, future in futures.items()}
                values = {worker_id: row["validation_accuracy"] for worker_id, row in rows.items()}
                mean_accuracy = float(np.mean(list(values.values())))
                if round_index == 0:
                    for worker_id, row in rows.items():
                        if (
                            not row["initial_model_loaded"]
                            or not math.isclose(row["learning_rate"], 0.01, rel_tol=1e-6)
                            or row["batch_size"] != 32
                            or row["optimizer"] != "Adam"
                            or row["architecture_sha256"] != EXPECTED_ARCHITECTURE_SHA256
                            or row["weights_sha256"] != EXPECTED_WEIGHTS_SHA256
                        ):
                            raise AssertionError(
                                f"Unexpected first-round audit fields for {worker_id}: "
                                f"{json.dumps({key: row.get(key) for key in ('learning_rate', 'batch_size', 'optimizer', 'initial_model_loaded', 'architecture_sha256', 'weights_sha256')}, sort_keys=True)}"
                            )
                print(
                    f"round={round_index + 1}; validation_accuracy="
                    f"{json.dumps(values, sort_keys=True)}; mean={mean_accuracy:.6f}; "
                    f"local_epochs={LOCAL_EPOCHS}; sample_count_per_worker=640"
                )
        finally:
            training_service.urlopen = original_urlopen
            (
                training_service._memory_available_kb,
                training_service._resident_memory_mb,
                training_service._peak_process_rss_mb,
            ) = original_memory

    expected = float(pc_reference["mean_final_accuracy"])
    actual_rows = [trainer.results(run_id)["runs"][run_id] for trainer in workers.values()]
    final_values = [rows[-1]["validation_accuracy"] for rows in actual_rows]
    actual = float(np.mean(final_values))
    print(
        "pc_worker_run_round_summary: validation_split_metric (160 imgs/worker)="
        f"{actual:.6f}; saved_pc_initial_model_reference={expected:.6f}; "
        f"absolute_gap={actual - expected:+.6f}"
    )
    if abs(actual - expected) <= 0.03:
        print("replay_assessment=consistent with saved PC initial-model 5x20 reference")
    else:
        print("replay_assessment=below saved PC reference; stop for diagnosis, no test-split tuning")


if __name__ == "__main__":
    main()
