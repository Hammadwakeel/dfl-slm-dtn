"""Authenticated worker API for local CIFAR-10 training and direct peer averaging."""

from __future__ import annotations

import io
import base64
import gc
import hashlib
import json
import math
import logging
import logging.handlers
import os
import re
import stat
import shutil
import sys
import subprocess
import threading
import tempfile
import time
import zipfile
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, urlparse
from urllib.request import Request, urlopen

import numpy as np

from worker.aggregation import (
    AGGREGATION_RULES,
    SYNC_MODES,
    Contribution,
    aggregate,
    model_signature,
)
from common.auth import is_authorized
from common.cifar_model import build_cifar10_model, compile_cifar10_model, model_size_bytes
from common.dataset_nas import smoke_test_architecture
from common.model_design import (
    AUGMENTATIONS,
    MAX_MODEL_PARAMETERS,
    MAX_MODEL_WEIGHT_BYTES,
    ModelDesignError,
    validate_model_design,
    validate_normalization,
)
from common.nas import FIXED_ARCHITECTURE, architecture_spec
from worker.model_factory import build_dataset_model
from worker.nas_search import (
    MAX_SEARCH_SECONDS,
    load_dataset_shard,
    search_dataset_architectures,
    search_dataset_architectures_isolated,
)

LOGGER = logging.getLogger("dfl_slm.worker_training")
# A long-lived service that still holds more than this after cleanup is restarted
# (on request) so TensorFlow/allocator growth cannot accumulate across runs.
WORKER_RECYCLE_RSS_MB = 1536
RECYCLE_EXIT_CODE = 75
RECYCLE_DELAY_SECONDS = 1.0
ROUND_WEIGHTS_RETAINED = 2
# Newest aggregated model of a dataset run, kept on disk for pruning.
LATEST_MODEL_WEIGHTS = "latest_model.npz"
# Shared, class-balanced validation subset (same for every worker), used to score
# each round on all classes instead of only this worker's non-IID slice.
GLOBAL_VALIDATION_FILE = "global_validation.npz"
MAX_GLOBAL_VALIDATION_ROWS = 5000
# "learned" uses an explicit receiver -> senders map chosen by APG topology learning.
TRAINING_TOPOLOGIES = frozenset({"ring", "directed_ring", "bidirectional_ring", "mesh", "chain", "star", "learned"})
LATEST_MODEL_METADATA = "latest_model.json"
RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
MAX_WEIGHT_BYTES = MAX_MODEL_WEIGHT_BYTES
MAX_DATASET_SHARD_BYTES = 256 * 1024 * 1024
MAX_DATASET_SHARD_EXPANDED_BYTES = 512 * 1024 * 1024
MAX_INITIAL_MODEL_PARAMETERS = MAX_MODEL_PARAMETERS
DATASET_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
DEFAULT_DATASET_ROOT = "/var/lib/dfl-slm-dtn/datasets"
DEFAULT_DATASET_LEARNING_RATE = 0.01
DEFAULT_DATASET_BATCH_SIZE = 32


@dataclass(frozen=True)
class Peer:
    node_id: str
    base_url: str
    token: str


@dataclass(frozen=True)
class WorkerConfig:
    node_id: str
    api_token: str
    port: int
    data_path: str
    peers: tuple[Peer, ...]
    log_path: str
    shard_paths: dict[str, str] = field(default_factory=dict)
    dataset_root: str = DEFAULT_DATASET_ROOT
    bind_host: str = "0.0.0.0"
    peer_timeout_seconds: float = 600.0
    dataset_learning_rate: float = DEFAULT_DATASET_LEARNING_RATE
    dataset_batch_size: int = DEFAULT_DATASET_BATCH_SIZE


    @classmethod
    def from_env(cls) -> "WorkerConfig":
        node_id = os.environ.get("DFL_NODE_ID", "").strip()
        token = os.environ.get("DFL_API_TOKEN", "")
        peers_encoded = os.environ.get("DFL_PEERS_JSON_B64", "")
        peers_value = json.loads(base64.b64decode(peers_encoded, validate=True)) if peers_encoded else []
        shards_encoded = os.environ.get("DFL_TRAINING_SHARDS_JSON_B64", "")
        shard_paths = (
            json.loads(base64.b64decode(shards_encoded, validate=True))
            if shards_encoded
            else {}
        )
        if re.fullmatch(r"worker[1-9][0-9]*", node_id) is None:
            raise ValueError("DFL_NODE_ID must identify a numbered worker")
        if len(token) < 32:
            raise ValueError("DFL_API_TOKEN must contain at least 32 characters")
        peers = tuple(
            Peer(
                node_id=str(item["node_id"]),
                base_url=str(item["base_url"]).rstrip("/"),
                token=str(item["token"]),
            )
            for item in peers_value
        )
        peer_ids = [peer.node_id for peer in peers]
        worker_ids = sorted((node_id, *peer_ids))
        expected_worker_ids = [f"worker{index}" for index in range(1, len(worker_ids) + 1)]
        if (
            len(peers) < 2
            or len(set(peer_ids)) != len(peer_ids)
            or node_id in peer_ids
            or worker_ids != expected_worker_ids
        ):
            raise ValueError(
                "Each worker must be configured with every other worker peer"
            )
        if any(len(peer.token) < 32 or not peer.base_url.startswith("http://") for peer in peers):
            raise ValueError("Peer URLs and tokens must be configured")
        port = int(os.environ.get("DFL_TRAINING_PORT", "5004"))
        timeout = float(os.environ.get("DFL_PEER_TIMEOUT_SECONDS", "600"))
        dataset_learning_rate = float(
            os.environ.get("DFL_DATASET_LEARNING_RATE", str(DEFAULT_DATASET_LEARNING_RATE))
        )
        dataset_batch_size = int(
            os.environ.get("DFL_DATASET_BATCH_SIZE", str(DEFAULT_DATASET_BATCH_SIZE))
        )
        if (
            not 1 <= port <= 65535
            or timeout <= 0
            or not math.isfinite(dataset_learning_rate)
            or dataset_learning_rate <= 0
            or not 1 <= dataset_batch_size <= 4096
        ):
            raise ValueError("Training port and peer timeout are out of range")
        return cls(
            node_id=node_id,
            api_token=token,
            port=port,
            data_path=os.environ.get(
                "DFL_TRAINING_DATA_PATH",
                "/opt/dfl-slm-dtn/worker/data/train_shard.npz",
            ),
            peers=peers,
            log_path=os.environ.get(
                "DFL_TRAINING_LOG_PATH",
                "/var/log/dfl-slm-worker-training/events.jsonl",
            ),
            shard_paths={str(key): str(value) for key, value in shard_paths.items()},
            dataset_root=os.environ.get(
                "DFL_TRAINING_DATASET_ROOT",
                DEFAULT_DATASET_ROOT,
            ),
            bind_host=os.environ.get("DFL_BIND_HOST", "0.0.0.0"),
            peer_timeout_seconds=timeout,
            dataset_learning_rate=dataset_learning_rate,
            dataset_batch_size=dataset_batch_size,
        )


class WorkerMemoryBusyError(RuntimeError):
    pass


def ring_neighbours(node_id: str, worker_ids: tuple[str, ...]) -> tuple[str, ...]:
    if node_id not in worker_ids or len(worker_ids) < 3:
        raise ValueError("Ring topology requires a configured member of a worker ring")
    index = worker_ids.index(node_id)
    return (worker_ids[(index + 1) % len(worker_ids)],)


def mesh_neighbours(node_id: str, worker_ids: tuple[str, ...]) -> tuple[str, ...]:
    if node_id not in worker_ids:
        raise ValueError("Mesh topology requires a configured worker")
    return tuple(peer for peer in worker_ids if peer != node_id)


def topology_neighbours(
    node_id: str,
    worker_ids: tuple[str, ...],
    topology: str,
) -> tuple[str, ...]:
    if (
        node_id not in worker_ids
        or len(worker_ids) < 3
        or len(set(worker_ids)) != len(worker_ids)
    ):
        raise ValueError("Supported topologies require at least three unique configured workers")
    ordered = tuple(sorted(worker_ids))
    index = ordered.index(node_id)
    if topology in {"ring", "directed_ring"}:
        return (ordered[(index + 1) % len(ordered)],)
    if topology == "bidirectional_ring":
        return tuple(
            sorted(
                {
                    ordered[(index - 1) % len(ordered)],
                    ordered[(index + 1) % len(ordered)],
                }
            )
        )
    if topology == "mesh":
        return mesh_neighbours(node_id, ordered)
    if topology == "chain":
        return tuple(
            ordered[peer_index]
            for peer_index in (index - 1, index + 1)
            if 0 <= peer_index < len(ordered)
        )
    if topology == "star":
        hub = ordered[0]
        return tuple(peer for peer in ordered if peer != node_id) if node_id == hub else (hub,)
    raise ValueError(f"Unsupported topology: {topology}")


def _cross_entropy(labels: Any, logits: Any) -> float:
    """Mean sparse categorical cross-entropy of linear logits (numerically stable)."""
    values = np.asarray(logits, dtype=np.float64)
    shifted = values - values.max(axis=1, keepdims=True)
    log_probabilities = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
    targets = np.asarray(labels, dtype=np.int64).reshape(-1)
    return float(-log_probabilities[np.arange(targets.size), targets].mean())


def _validate_learned_senders(raw: Any) -> dict[str, tuple[str, ...]]:
    if not isinstance(raw, dict) or not raw:
        raise ValueError("The learned topology needs a receiver -> senders map")
    senders_map: dict[str, tuple[str, ...]] = {}
    for receiver, senders in raw.items():
        if (
            not isinstance(receiver, str)
            or not isinstance(senders, list)
            or not senders
            or any(not isinstance(sender, str) or sender == receiver for sender in senders)
            or len(set(senders)) != len(senders)
        ):
            raise ValueError("Every learned-topology receiver needs one or more distinct other senders")
        senders_map[receiver] = tuple(sorted(senders))
    return senders_map


def topology_senders(
    node_id: str,
    worker_ids: tuple[str, ...],
    topology: str,
) -> tuple[str, ...]:
    if topology in {"ring", "directed_ring"}:
        ordered = tuple(sorted(worker_ids))
        return (ordered[(ordered.index(node_id) - 1) % len(ordered)],)
    return topology_neighbours(node_id, worker_ids, topology)


def _read_dataset_arrays(
    payload: bytes,
) -> tuple[np.ndarray, np.ndarray, list[str], dict[str, int]]:
    """Validate an uploaded images/labels/class_names NPZ and return its arrays."""
    if not payload or len(payload) > MAX_DATASET_SHARD_BYTES:
        raise ValueError("Dataset shard payload size is out of range")
    try:
        archive = zipfile.ZipFile(io.BytesIO(payload))
    except zipfile.BadZipFile as error:
        raise ValueError("Dataset shard is not a valid NPZ archive") from error
    with archive:
        names = [entry.filename for entry in archive.infolist()]
        if set(names) != {"images.npy", "labels.npy", "class_names.npy"} or len(names) != 3:
            raise ValueError("Dataset shard must contain images, labels, and class_names arrays")
        if any(stat.S_ISLNK(entry.external_attr >> 16) for entry in archive.infolist()):
            raise ValueError("Dataset shard cannot contain symbolic links")
        if sum(entry.file_size for entry in archive.infolist()) > MAX_DATASET_SHARD_EXPANDED_BYTES:
            raise ValueError("Dataset shard exceeds the expanded size limit")
    try:
        with np.load(io.BytesIO(payload), allow_pickle=False) as shard:
            images = np.asarray(shard["images"])
            labels = np.asarray(shard["labels"])
            class_values = np.asarray(shard["class_names"])
    except (OSError, ValueError, KeyError, zipfile.BadZipFile) as error:
        raise ValueError("Dataset shard contains invalid NumPy arrays") from error
    if (
        images.ndim != 4
        or images.shape[0] == 0
        or images.shape[1] < 1
        or images.shape[1] > 2048
        or images.shape[2] < 1
        or images.shape[2] > 2048
        or images.shape[3] not in {1, 3}
        or images.dtype.kind not in {"u", "i", "f"}
    ):
        raise ValueError("Dataset images must have shape (N, H, W, C), with C=1 or C=3")
    if images.dtype.kind == "f" and not np.isfinite(images).all():
        raise ValueError("Dataset images contain non-finite values")
    if labels.ndim != 1 or labels.size != images.shape[0] or labels.dtype.kind not in {"u", "i"}:
        raise ValueError("Dataset labels must be an integer vector matching the image count")
    if class_values.ndim != 1 or class_values.size < 2 or class_values.dtype.kind not in {"U", "S"}:
        raise ValueError("Dataset class names must be a string vector with at least two classes")
    class_names = [str(value) for value in class_values.tolist()]
    if class_values.dtype.kind == "S":
        class_names = [value.decode("utf-8") for value in class_values.tolist()]
    if len(set(class_names)) != len(class_names) or min(labels) < 0 or max(labels) >= len(class_names):
        raise ValueError("Dataset labels do not match the declared class names")
    class_counts = {
        class_names[int(label)]: int(count)
        for label, count in zip(*np.unique(labels, return_counts=True))
    }

    return images, labels, class_names, class_counts


def install_dataset_shard(
    config: WorkerConfig,
    dataset_name: str,
    payload: bytes,
    *,
    replace_existing: bool = False,
) -> dict[str, Any]:
    if DATASET_NAME_PATTERN.fullmatch(dataset_name) is None:
        raise ValueError("Invalid dataset name")
    images, labels, class_names, class_counts = _read_dataset_arrays(payload)

    root = Path(config.dataset_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    target = root / dataset_name
    if target.is_symlink() or (target.exists() and not target.is_dir()):
        raise ValueError("Dataset path is not a safe directory")
    if target.exists():
        if target.resolve().parent != root:
            raise ValueError("Dataset path escapes the configured root")
        if any((target / name).is_symlink() for name in ("train.npz", "metadata.json")):
            raise ValueError("Dataset shard files cannot be symbolic links")
        existing_metadata_path = target / "metadata.json"
        if existing_metadata_path.is_file():
            existing = json.loads(existing_metadata_path.read_text(encoding="utf-8"))
            if existing.get("shard_sha256") == hashlib.sha256(payload).hexdigest():
                return existing
        if not replace_existing:
            raise FileExistsError(f"Dataset shard already exists with different contents: {dataset_name}")

    with tempfile.TemporaryDirectory(prefix=f".{dataset_name}-", dir=root) as temporary:
        staging = Path(temporary) / "stage"
        staging.mkdir()
        shard_path = staging / "train.npz"
        shard_path.write_bytes(payload)
        metadata = {
            "node_id": config.node_id,
            "dataset_name": dataset_name,
            "rows": int(labels.size),
            "shape": list(images.shape),
            "image_size": [int(images.shape[1]), int(images.shape[2])],
            "channels": int(images.shape[3]),
            "dtype": str(images.dtype),
            "class_names": class_names,
            "class_counts": class_counts,
            "shard_bytes": len(payload),
            "shard_sha256": hashlib.sha256(payload).hexdigest(),
            "installed_at": datetime.now(timezone.utc).isoformat(),
        }
        staged_metadata = staging / "metadata.json"
        staged_metadata.write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        if target.exists() and replace_existing:
            backup = Path(temporary) / "backup"
            backup.mkdir()
            existing_files = tuple(
                name
                for name in ("train.npz", "metadata.json")
                if (target / name).exists()
            )
            try:
                for name in existing_files:
                    os.replace(target / name, backup / name)
                os.replace(shard_path, target / "train.npz")
                os.replace(staged_metadata, target / "metadata.json")
            except OSError:
                for name in ("train.npz", "metadata.json"):
                    (target / name).unlink(missing_ok=True)
                for name in existing_files:
                    if (backup / name).exists():
                        os.replace(backup / name, target / name)
                raise
        else:
            try:
                os.rename(staging, target)
            except FileExistsError as error:
                raise FileExistsError(
                    f"Dataset shard already exists with different contents: {dataset_name}"
                ) from error
    return metadata


def install_global_validation(
    config: WorkerConfig,
    dataset_name: str,
    payload: bytes,
) -> dict[str, Any]:
    """Store the shared validation subset next to an installed shard of the same dataset."""
    if DATASET_NAME_PATTERN.fullmatch(dataset_name) is None:
        raise ValueError("Invalid dataset name")
    root = Path(config.dataset_root).resolve()
    target = root / dataset_name
    metadata_path = target / "metadata.json"
    if target.is_symlink() or not metadata_path.is_file() or target.resolve().parent != root:
        raise ValueError(f"Install the {dataset_name} training shard before its global validation split")
    images, labels, class_names, class_counts = _read_dataset_arrays(payload)
    shard = json.loads(metadata_path.read_text(encoding="utf-8"))
    if class_names != shard.get("class_names"):
        raise ValueError("Global validation classes do not match the installed shard")
    if list(images.shape[1:]) != list(shard.get("shape", [])[1:]):
        raise ValueError("Global validation image shape does not match the installed shard")
    if labels.size > MAX_GLOBAL_VALIDATION_ROWS:
        raise ValueError(f"Global validation split exceeds {MAX_GLOBAL_VALIDATION_ROWS} rows")
    destination = target / GLOBAL_VALIDATION_FILE
    if destination.is_symlink():
        raise ValueError("Global validation file cannot be a symbolic link")
    staged = target / f".{GLOBAL_VALIDATION_FILE}.tmp"
    staged.write_bytes(payload)
    os.replace(staged, destination)
    return {
        "node_id": config.node_id,
        "dataset_name": dataset_name,
        "rows": int(labels.size),
        "class_counts": class_counts,
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def load_global_validation(dataset_path: Path) -> tuple[np.ndarray, np.ndarray] | None:
    """Arrays of the installed global validation split, or None when absent."""
    path = dataset_path / GLOBAL_VALIDATION_FILE
    if not path.is_file() or path.is_symlink():
        return None
    with np.load(path, allow_pickle=False) as split:
        return (
            np.asarray(split["images"], dtype=np.float32),
            np.asarray(split["labels"], dtype=np.int64),
        )


def install_initial_model(
    config: WorkerConfig,
    dataset_name: str,
    payload: bytes,
) -> dict[str, Any]:
    if DATASET_NAME_PATTERN.fullmatch(dataset_name) is None:
        raise ValueError("Invalid dataset name")
    if not payload or len(payload) > MAX_DATASET_SHARD_BYTES:
        raise ValueError("Initial model payload size is out of range")
    try:
        archive = zipfile.ZipFile(io.BytesIO(payload))
    except zipfile.BadZipFile as error:
        raise ValueError("Initial model is not a valid ZIP archive") from error
    with archive:
        entries = archive.infolist()
        entry_names = {entry.filename for entry in entries}
        if (
            len(entries) not in {1, 2}
            or "architecture.json" not in entry_names
            or entry_names - {"architecture.json", "initial_weights.npz"}
        ):
            raise ValueError("Initial model bundle has an unexpected file layout")
        if any(stat.S_ISLNK(entry.external_attr >> 16) for entry in entries):
            raise ValueError("Initial model cannot contain symbolic links")
        if sum(entry.file_size for entry in entries) > MAX_DATASET_SHARD_EXPANDED_BYTES:
            raise ValueError("Initial model exceeds the expanded size limit")
        architecture_bytes = archive.read("architecture.json")
        weights_bytes = archive.read("initial_weights.npz") if "initial_weights.npz" in entry_names else b""

    try:
        architecture = json.loads(architecture_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("Initial model architecture is invalid JSON") from error
    dataset_path = Path(config.dataset_root).resolve() / dataset_name
    if not dataset_path.is_dir() or dataset_path.is_symlink():
        raise FileNotFoundError(f"Dataset shard is not installed for {dataset_name}")

    model_type = architecture.get("model_type") if isinstance(architecture, dict) else None
    if model_type in {"custom", "pretrained_deit"}:
        if architecture.get("dataset_name") != dataset_name:
            raise ValueError("Initial model architecture does not match the dataset")
        shard_images, _, shard_classes = load_dataset_shard(dataset_path / "train.npz")
        input_shape = tuple(int(value) for value in shard_images.shape[1:])
        if (
            architecture.get("input_shape") != list(input_shape)
            or architecture.get("classes") != shard_classes
            or architecture.get("num_classes") != len(shard_classes)
        ):
            raise ValueError("Initial model architecture does not match the installed dataset shard")
        try:
            normalized_design, estimated_parameters = validate_model_design(
                architecture,
                input_shape,
                len(shard_classes),
            )
            if normalized_design["model_type"] == "pretrained_deit":
                normalized_normalization = validate_normalization(
                    architecture.get("normalization"),
                    input_shape[2],
                )
                if architecture.get("normalization") != normalized_normalization:
                    raise ModelDesignError("Initial model normalization metadata is invalid")
        except ModelDesignError as error:
            raise ValueError(str(error)) from error
        if architecture.get("parameter_count") != estimated_parameters:
            raise ValueError("Initial model parameter count does not match its architecture")
        model = build_dataset_model(
            input_shape=input_shape,
            num_classes=len(shard_classes),
            architecture=architecture,
            seed=int(architecture.get("initialization_seed", 0)),
        )
        actual_parameters = int(model.count_params())
        if actual_parameters > MAX_INITIAL_MODEL_PARAMETERS:
            raise ValueError(
                f"Initialized model has {actual_parameters:,} parameters; cap is "
                f"{MAX_INITIAL_MODEL_PARAMETERS:,}"
            )
        if model_type == "custom" and actual_parameters != estimated_parameters:
            raise ValueError("Model layer parameter count differs from its validated estimate")
        if model_type == "custom" and weights_bytes:
            # A supplied bundle (e.g. a pruned, already-trained model) starts from those
            # weights instead of a fresh seeded initialization.
            supplied = deserialize_weights(weights_bytes)
            expected_shapes = [tuple(value.shape) for value in model.get_weights()]
            if [tuple(value.shape) for value in supplied] != expected_shapes:
                raise ValueError("Supplied initial weights do not match the model architecture")
            model.set_weights([value.astype(np.float32) for value in supplied])
        weights = [np.asarray(value) for value in model.get_weights()]
        weights_bytes = serialize_weights(weights)
        parameter_count = actual_parameters
        weight_shapes: dict[str, Any] | None = None
    else:
        if model_type != "small_cnn" or architecture.get("dataset_name") != dataset_name:
            raise ValueError("Initial model architecture does not match the dataset")
        classes = architecture.get("classes")
        weight_shapes = architecture.get("weight_shapes")
        parameter_count = architecture.get("parameter_count")
        if (
            not isinstance(classes, list)
            or len(classes) < 2
            or any(not isinstance(name, str) or not name for name in classes)
            or len(set(classes)) != len(classes)
            or architecture.get("num_classes") != len(classes)
            or not isinstance(weight_shapes, dict)
            or not isinstance(parameter_count, int)
            or not 1 <= parameter_count <= MAX_INITIAL_MODEL_PARAMETERS
        ):
            raise ValueError("Initial model architecture fields are invalid")
        try:
            with np.load(io.BytesIO(weights_bytes), allow_pickle=False) as archive:
                weights = {name: np.asarray(archive[name]) for name in archive.files}
        except (OSError, ValueError, KeyError, zipfile.BadZipFile) as error:
            raise ValueError("Initial model weights are invalid") from error
        if set(weights) != set(weight_shapes):
            raise ValueError("Initial model weight names do not match the architecture")
        actual_parameters = 0
        for name, array in weights.items():
            expected_shape = weight_shapes[name]
            if (
                not isinstance(expected_shape, list)
                or list(array.shape) != expected_shape
                or array.dtype.kind not in {"f", "i", "u"}
                or not np.isfinite(array).all()
            ):
                raise ValueError(f"Initial model weight {name} has invalid shape or values")
            actual_parameters += int(array.size)
        if actual_parameters != parameter_count:
            raise ValueError("Initial model parameter count does not match its weight arrays")

    model_path = dataset_path / "models" / "initial_model"
    architecture_sha256 = hashlib.sha256(architecture_bytes).hexdigest()
    weights_sha256 = hashlib.sha256(weights_bytes).hexdigest()
    metadata = {
        "node_id": config.node_id,
        "dataset_name": dataset_name,
        "parameter_count": parameter_count,
        "architecture_sha256": architecture_sha256,
        "weights_sha256": weights_sha256,
    }
    if model_path.exists():
        existing_metadata = model_path / "metadata.json"
        if existing_metadata.is_file():
            existing = json.loads(existing_metadata.read_text(encoding="utf-8"))
            if all(existing.get(key) == value for key, value in metadata.items() if key != "node_id"):
                return metadata
        raise FileExistsError(f"A different initial model is already installed for {dataset_name}")

    with tempfile.TemporaryDirectory(prefix=".initial-model-", dir=dataset_path) as temporary:
        staging = Path(temporary) / "initial_model"
        staging.mkdir()
        (staging / "architecture.json").write_bytes(architecture_bytes)
        (staging / "initial_weights.npz").write_bytes(weights_bytes)
        (staging / "metadata.json").write_text(
            json.dumps(metadata, indent=2) + "\n",
            encoding="utf-8",
        )
        model_path.parent.mkdir(parents=True, exist_ok=True)
        os.rename(staging, model_path)
    return metadata


PEER_SEND_ATTEMPTS = 3
PEER_SEND_BACKOFF_SECONDS = (1.0, 3.0)


def _send_with_retry(
    request: Request,
    *,
    timeout: float,
    peer_id: str,
    cancel_event: threading.Event | None = None,
) -> int:
    """POST peer weights, retrying transient failures; returns the attempts used.

    A dropped connection (reset, timeout) or a 5xx answer is retried with a short
    backoff; the receiver acknowledges an identical re-upload, so a retry after a
    lost response is safe. 4xx answers (auth, signature, architecture) fail at once.
    """
    for attempt in range(1, PEER_SEND_ATTEMPTS + 1):
        try:
            with urlopen(request, timeout=timeout) as response:
                if response.status != HTTPStatus.ACCEPTED:
                    raise RuntimeError(f"Peer {peer_id} returned HTTP {response.status}")
            return attempt
        except HTTPError as error:
            if error.code < 500 or attempt == PEER_SEND_ATTEMPTS:
                raise
            failure: Exception = error
        except (URLError, TimeoutError, OSError) as error:
            if attempt == PEER_SEND_ATTEMPTS:
                raise
            failure = error
        _log_event(
            "peer_send_retry",
            peer_id=peer_id,
            attempt=attempt,
            error=f"{type(failure).__name__}: {failure}"[:300],
        )
        if cancel_event is not None and cancel_event.is_set():
            raise RuntimeError("Training was cancelled during peer exchange")
        time.sleep(PEER_SEND_BACKOFF_SECONDS[min(attempt - 1, len(PEER_SEND_BACKOFF_SECONDS) - 1)])
    raise RuntimeError("unreachable")


def _augmented_dataset(
    images: np.ndarray,
    labels: np.ndarray,
    *,
    batch_size: int,
    seed: int,
    max_shift: int = 4,
) -> Any:
    """Shuffled batches with a random horizontal flip and a random shift of up to
    ``max_shift`` pixels (reflect-pad, then random crop back to the input size).

    Augmentation only changes the training inputs; the model and its exchanged weights
    are unchanged, so peers can still average them.
    """
    import tensorflow as tf

    height, width, channels = (int(value) for value in images.shape[1:])

    def augment(image: Any, label: Any) -> tuple[Any, Any]:
        image = tf.image.random_flip_left_right(image)
        padded = tf.pad(
            image,
            [[max_shift, max_shift], [max_shift, max_shift], [0, 0]],
            mode="REFLECT",
        )
        image = tf.image.random_crop(padded, [height, width, channels])
        return image, label

    return (
        tf.data.Dataset.from_tensor_slices((images, labels))
        .shuffle(int(labels.size), seed=seed, reshuffle_each_iteration=True)
        .map(augment, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )


def _load_installed_dataset_initial_model(
    config: WorkerConfig,
    dataset_name: str,
    *,
    input_shape: tuple[int, int, int],
    class_names: list[str],
) -> tuple[Any, dict[str, Any]] | None:
    dataset_root = Path(config.dataset_root).resolve()
    model_path = dataset_root / dataset_name / "models" / "initial_model"
    if not model_path.exists():
        return None
    if model_path.is_symlink() or not model_path.is_dir():
        raise ValueError("Installed initial-model bundle path is not a regular directory")

    architecture_path = model_path / "architecture.json"
    weights_path = model_path / "initial_weights.npz"
    metadata_path = model_path / "metadata.json"
    if not architecture_path.is_file() or not weights_path.is_file() or not metadata_path.is_file():
        raise ValueError("Installed initial-model bundle is incomplete")
    if any(path.is_symlink() for path in (architecture_path, weights_path, metadata_path)):
        raise ValueError("Installed initial-model bundle cannot contain symbolic links")

    architecture_bytes = architecture_path.read_bytes()
    weights_bytes = weights_path.read_bytes()
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("Installed initial-model metadata is invalid") from error
    architecture_hash = hashlib.sha256(architecture_bytes).hexdigest()
    weights_hash = hashlib.sha256(weights_bytes).hexdigest()
    if (
        not isinstance(metadata, dict)
        or metadata.get("dataset_name") != dataset_name
        or metadata.get("architecture_sha256") != architecture_hash
        or metadata.get("weights_sha256") != weights_hash
    ):
        raise ValueError("Installed initial-model bundle hash mismatch")

    bundle = io.BytesIO()
    with zipfile.ZipFile(bundle, "w", compression=zipfile.ZIP_DEFLATED) as zipped:
        zipped.writestr("architecture.json", architecture_bytes)
        zipped.writestr("initial_weights.npz", weights_bytes)
    try:
        install_initial_model(config, dataset_name, bundle.getvalue())
    except (OSError, ValueError, FileExistsError, zipfile.BadZipFile) as error:
        raise ValueError(f"Installed initial-model bundle is malformed: {error}") from error

    try:
        architecture = json.loads(architecture_bytes.decode("utf-8"))
        with np.load(io.BytesIO(weights_bytes), allow_pickle=False) as archive:
            weights = {name: np.asarray(archive[name]) for name in archive.files}
    except (UnicodeDecodeError, json.JSONDecodeError, OSError, ValueError, KeyError) as error:
        raise ValueError("Installed initial-model architecture or weights are unreadable") from error
    if (
        architecture.get("input_shape") != list(input_shape)
        or architecture.get("classes") != class_names
        or architecture.get("num_classes") != len(class_names)
    ):
        raise ValueError("Installed initial-model architecture does not match the dataset shard")

    import tensorflow as tf

    if architecture.get("model_type") in {"custom", "pretrained_deit"}:
        model = build_dataset_model(
            input_shape=input_shape,
            num_classes=len(class_names),
            architecture=architecture,
            seed=int(architecture.get("initialization_seed", 0)),
        )
        weights = deserialize_weights(weights_bytes)
        if len(model.get_weights()) != len(weights):
            raise ValueError("Installed model weight count does not match the architecture")
        for expected, actual in zip(model.get_weights(), weights):
            if expected.shape != actual.shape or not np.isfinite(actual).all():
                raise ValueError("Installed model weights do not match the architecture")
        model.set_weights(weights)
        if model.count_params() != architecture.get("parameter_count"):
            raise ValueError("Installed model parameter count does not match its architecture")
        return model, architecture

    layer_specs = architecture.get("layers")
    if not isinstance(layer_specs, list) or not layer_specs:
        raise ValueError("Installed initial-model architecture has no layers")
    layers: list[Any] = [tf.keras.layers.Input(shape=input_shape, dtype=tf.float32)]
    weight_order: list[str] = []
    convolution_index = 0
    dense_count = 0
    for spec in layer_specs:
        if not isinstance(spec, dict):
            raise ValueError("Installed initial-model layer specification is malformed")
        layer_type = spec.get("type")
        if layer_type == "conv2d":
            convolution_index += 1
            kernel_size = spec.get("kernel_size")
            if (
                not isinstance(spec.get("filters"), int)
                or not isinstance(kernel_size, list)
                or len(kernel_size) != 2
                or any(not isinstance(value, int) or value < 1 for value in kernel_size)
                or spec.get("padding") not in {"same", "valid"}
                or not isinstance(spec.get("activation"), str)
            ):
                raise ValueError("Installed initial-model Conv2D layer is malformed")
            layers.append(
                tf.keras.layers.Conv2D(
                    filters=spec["filters"],
                    kernel_size=tuple(kernel_size),
                    padding=spec["padding"],
                    activation=spec["activation"],
                )
            )
            weight_order.extend(
                [f"conv{convolution_index}_kernel", f"conv{convolution_index}_bias"]
            )
        elif layer_type == "global_average_pooling2d":
            layers.append(tf.keras.layers.GlobalAveragePooling2D())
        elif layer_type == "dense":
            dense_count += 1
            if dense_count != 1 or spec.get("units") != len(class_names):
                raise ValueError("Installed initial-model Dense layer is unsupported")
            if not isinstance(spec.get("activation"), str):
                raise ValueError("Installed initial-model Dense activation is malformed")
            layers.append(
                tf.keras.layers.Dense(
                    units=spec["units"],
                    activation=spec["activation"],
                )
            )
            weight_order.extend(["classifier_kernel", "classifier_bias"])
        else:
            raise ValueError(f"Installed initial-model layer type is unsupported: {layer_type}")
    if not convolution_index or dense_count != 1 or set(weight_order) != set(weights):
        raise ValueError("Installed initial-model layer/weight mapping is incomplete")

    model = tf.keras.Sequential(layers, name=f"dataset_{dataset_name}_initial")
    model(np.zeros((1, *input_shape), dtype=np.float32), training=False)
    model.set_weights([weights[name] for name in weight_order])
    return model, architecture


def serialize_weights(weights: list[np.ndarray]) -> bytes:
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **{f"weight_{index}": value for index, value in enumerate(weights)})
    payload = buffer.getvalue()
    if len(payload) > MAX_WEIGHT_BYTES:
        raise ValueError("Serialized model weights exceed the configured transfer limit")
    return payload


def deserialize_weights(payload: bytes) -> list[np.ndarray]:
    if not payload or len(payload) > MAX_WEIGHT_BYTES:
        raise ValueError("Model weight payload has an invalid size")
    with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
        keys = sorted(archive.files, key=lambda key: int(key.removeprefix("weight_")))
        if not keys or keys != [f"weight_{index}" for index in range(len(keys))]:
            raise ValueError("Model weight payload has an unexpected structure")
        weights = [np.asarray(archive[key]) for key in keys]
    if any(array.dtype.kind not in "fiu" or array.nbytes > MAX_WEIGHT_BYTES for array in weights):
        raise ValueError("Model weight payload contains invalid arrays")
    return weights


def _memory_available_kb() -> int:
    with open("/proc/meminfo", encoding="ascii") as stream:
        for line in stream:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1])
    raise RuntimeError("MemAvailable is missing from /proc/meminfo")


def _process_snapshot() -> dict[str, Any]:
    result = subprocess.run(
        [
            "/usr/bin/ps",
            "-eo",
            "pid=,ppid=,user=,pcpu=,pmem=,rss=,comm=",
            "--sort=-rss",
        ],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env={"LC_ALL": "C"},
        timeout=3,
    )
    processes = []
    for line in result.stdout.splitlines():
        fields = line.split(maxsplit=6)
        if len(fields) != 7:
            raise ValueError("Process table returned a malformed row")
        pid, ppid, user, cpu, memory, rss_kb, name = fields
        processes.append(
            {
                "pid": int(pid),
                "ppid": int(ppid),
                "user": user,
                "cpu_percent": float(cpu),
                "memory_percent": float(memory),
                "rss_mb": round(int(rss_kb) / 1024, 1),
                "name": name,
            }
        )
    return {
        "sampled_at": datetime.now(timezone.utc).isoformat(),
        "processes": processes[:30],
        "memory_available_mb": _memory_available_mb(),
    }


def _memory_available_mb() -> float:
    return round(_memory_available_kb() / 1024, 1)


def _resident_memory_mb() -> float:
    with open("/proc/self/status", encoding="ascii") as stream:
        for line in stream:
            if line.startswith("VmRSS:"):
                return round(int(line.split()[1]) / 1024, 1)
    raise RuntimeError("VmRSS is missing from /proc/self/status")


def _peak_process_rss_mb() -> float:
    import resource

    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)


def _trim_process_heap() -> None:
    if not sys.platform.startswith("linux"):
        return
    import ctypes

    try:
        trim = ctypes.CDLL(None).malloc_trim
    except (AttributeError, OSError) as error:
        LOGGER.warning("Could not release freed training memory to the OS: %s", error)
        return
    trim.argtypes = [ctypes.c_size_t]
    trim.restype = ctypes.c_int
    trim(0)


def _log_event(event: str, **details: Any) -> None:
    LOGGER.info(
        "%s",
        json.dumps(
            {"timestamp": time.time(), "node_id": os.environ.get("DFL_NODE_ID"), "event": event, **details},
            sort_keys=True,
            separators=(",", ":"),
        ),
    )


class WorkerTrainer:
    def __init__(self, config: WorkerConfig) -> None:
        self.config = config
        self._lock = threading.RLock()
        self._condition = threading.Condition(self._lock)
        self._incoming: dict[tuple[str, int], dict[str, tuple[list[np.ndarray], int, int]]] = {}
        self._models: dict[str, Any] = {}
        self._run_architectures: dict[str, Any] = {}
        self._run_model_audits: dict[str, dict[str, Any]] = {}
        self._completed_rounds: dict[str, int] = {}
        self._round_weights: dict[tuple[str, int], list[np.ndarray]] = {}
        self._round_sample_counts: dict[tuple[str, int], int] = {}
        # One train/validation split per run instead of fresh fancy-index copies every round.
        self._run_splits: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]] = {}
        self._results: dict[str, list[dict[str, Any]]] = {}
        self._run_errors: dict[str, str] = {}
        self._active_run: str | None = None
        self._active_round: int | None = None
        self._active_epoch: int | None = None
        self._total_epochs: int | None = None
        self._active_phase: str | None = None
        self._scheduled_run: str | None = None
        self._reboot_requested = False
        self._run_thread: threading.Thread | None = None
        self._cancel = threading.Event()
        self._peer_map = {peer.node_id: peer for peer in config.peers}
        self.dataset_split = "stratified"
        self.dataset_name: str | None = None
        self._dataset_class_names: list[str] | None = None
        self._global_validation: tuple[np.ndarray, np.ndarray] | None = None
        self.images, self.labels = self._load_shard(config.data_path)
        self._rounds = 10
        self._epochs = 5
        self._topology = "ring"
        self._architecture = architecture_spec(FIXED_ARCHITECTURE)
        self._aggregation = "fedavg"
        self._sync_mode = "sync"
        self._sync_deadline_seconds: float | None = None
        self._min_peers = 1
        self._learned_senders: dict[str, tuple[str, ...]] = {}
        self._measure_peers = False
        # Run-level augmentation override (None = use the architecture's setting).
        self._augmentation: str | None = None
        # Build the configured architecture from its seed instead of the installed
        # initial model (NAS v2 federated candidate check).
        self._fresh_model = False
        self._arrivals: dict[tuple[str, int], dict[str, float]] = {}

    def configure_peers(self, peers: tuple[Peer, ...]) -> dict[str, Any]:
        peer_ids = [peer.node_id for peer in peers]
        worker_ids = sorted((self.config.node_id, *peer_ids))
        expected_ids = [f"worker{index}" for index in range(1, len(worker_ids) + 1)]
        if (
            len(peers) < 2
            or len(set(peer_ids)) != len(peer_ids)
            or self.config.node_id in peer_ids
            or worker_ids != expected_ids
            or any(len(peer.token) < 32 or not peer.base_url.startswith("http://") for peer in peers)
        ):
            raise ValueError("Each worker must be configured with every other worker peer")
        with self._condition:
            if self._active_run is not None or self._scheduled_run is not None:
                raise RuntimeError("Cannot change worker peers while training is active")
            self.config = replace(self.config, peers=peers)
            self._peer_map = {peer.node_id: peer for peer in peers}
        return {"configured_peers": peer_ids}

    @staticmethod
    def _exit_for_recycle() -> None:
        _log_event("worker_recycling", exit_code=RECYCLE_EXIT_CODE)
        logging.shutdown()
        os._exit(RECYCLE_EXIT_CODE)

    def cleanup_memory(self, nas_lock: threading.Lock, *, recycle: bool = False) -> dict[str, Any]:
        """Free idle model memory; with ``recycle``, restart a bloated idle service.

        TensorFlow and the allocator keep memory that ``clear_session`` cannot return,
        so a long-lived service grows across runs. When asked to recycle and the
        process still holds more than ``WORKER_RECYCLE_RSS_MB`` after cleanup, the
        service exits with ``RECYCLE_EXIT_CODE`` and systemd (Restart=on-failure)
        starts a fresh one; shards and models live on disk, so nothing is lost.
        """
        if not nas_lock.acquire(blocking=False):
            raise WorkerMemoryBusyError("Cannot clean worker memory while NAS search is active")
        try:
            with self._condition:
                if (
                    self._active_run is not None
                    or self._scheduled_run is not None
                    or self._reboot_requested
                ):
                    raise WorkerMemoryBusyError(
                        "Cannot clean worker memory while training is active or reboot is requested"
                    )

                memory_before_mb = _memory_available_mb()
                process_rss_before_mb = _resident_memory_mb()
                cache_drop_error = None
                try:
                    self._drop_os_page_cache()
                except RuntimeError as error:
                    cache_drop_error = str(error)

                run_ids = (
                    set(self._models)
                    | set(self._run_architectures)
                    | set(self._run_model_audits)
                    | set(self._completed_rounds)
                    | set(self._results)
                    | set(self._run_errors)
                    | {key[0] for key in self._round_weights}
                    | {key[0] for key in self._round_sample_counts}
                    | {key[0] for key in self._incoming}
                )
                had_models = bool(self._models)
                self._models.clear()
                self._run_architectures.clear()
                self._run_model_audits.clear()
                self._completed_rounds.clear()
                self._round_weights.clear()
                self._round_sample_counts.clear()
                self._run_splits.clear()
                self._results.clear()
                self._run_errors.clear()
                self._incoming.clear()

                if "tensorflow" in sys.modules:
                    tensorflow = sys.modules["tensorflow"]
                    tensorflow.keras.backend.clear_session()
                elif had_models:
                    import tensorflow as tf

                    tf.keras.backend.clear_session()
                gc.collect()
                _trim_process_heap()
                result = {
                    "node_id": self.config.node_id,
                    "cleaned_runs": len(run_ids),
                    "memory_available_before_mb": memory_before_mb,
                    "memory_available_after_mb": _memory_available_mb(),
                    "process_rss_before_mb": process_rss_before_mb,
                    "process_rss_after_mb": _resident_memory_mb(),
                    "process_memory_cleaned": True,
                    "os_page_cache_dropped": cache_drop_error is None,
                }
                if cache_drop_error is not None:
                    result["error"] = cache_drop_error
                result["recycle_scheduled"] = bool(
                    recycle and result["process_rss_after_mb"] > WORKER_RECYCLE_RSS_MB
                )
                if result["recycle_scheduled"]:
                    # Reject new work until the process is replaced.
                    self._reboot_requested = True
                    timer = threading.Timer(RECYCLE_DELAY_SECONDS, self._exit_for_recycle)
                    timer.daemon = True
                    timer.start()
        finally:
            nas_lock.release()

        _log_event("worker_memory_cleaned", **result)
        return result

    def _drop_os_page_cache(self) -> None:
        request = Request(
            "http://127.0.0.1:5005/drop-caches",
            data=b"",
            headers={"Authorization": f"Bearer {self.config.api_token}"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=5) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace").strip()[:300]
            raise RuntimeError(
                f"Privileged cache cleanup helper returned HTTP {error.code}: {detail}"
            ) from error
        except (URLError, TimeoutError, OSError) as error:
            raise RuntimeError(
                f"Could not reach the privileged cache cleanup helper: {error}"
            ) from error
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise RuntimeError("Privileged cache cleanup helper returned invalid JSON") from error
        if not isinstance(payload, dict) or payload.get("status") != "ok":
            raise RuntimeError("Privileged cache cleanup helper returned an invalid response")

    def status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "node_id": self.config.node_id,
                "sample_count": int(self.labels.size),
                "dataset_split": self.dataset_split,
                "dataset_name": self.dataset_name,
                "active_run_id": self._active_run,
                "scheduled_run_id": self._scheduled_run,
                "active_round": self._active_round,
                "active_epoch": self._active_epoch,
                "total_epochs": self._total_epochs,
                "active_phase": self._active_phase,
                "reboot_requested": self._reboot_requested,
                "training_config": {
                    "rounds": self._rounds,
                    "epochs": self._epochs,
                    "topology": self._topology,
                    "architecture": self._architecture,
                    "dataset_learning_rate": self.config.dataset_learning_rate,
                    "dataset_batch_size": self.config.dataset_batch_size,
                },
                "last_error": self._run_errors.get(self._scheduled_run or self._active_run or ""),
                "runs": {
                    run_id: {
                        "rounds_completed": len(rows),
                        "last_round": rows[-1] if rows else None,
                    }
                    for run_id, rows in self._results.items()
                },
            }

    def begin_reboot(self) -> bool:
        with self._condition:
            if (
                self._active_run is not None
                or self._scheduled_run is not None
                or self._reboot_requested
            ):
                return False
            self._reboot_requested = True
            return True

    def cancel_reboot(self) -> None:
        with self._condition:
            self._reboot_requested = False
            self._condition.notify_all()

    def results(self, run_id: str | None = None) -> dict[str, Any]:
        with self._lock:
            selected = (
                {run_id: list(self._results.get(run_id, []))}
                if run_id is not None
                else {key: list(value) for key, value in self._results.items()}
            )
            errors = {
                key: value
                for key, value in self._run_errors.items()
                if run_id is None or key == run_id
            }
            progress = None
            if self._active_run is not None and (
                run_id is None or run_id == self._active_run
            ):
                progress = {
                    "run_id": self._active_run,
                    "round": (self._active_round + 1)
                    if self._active_round is not None
                    else None,
                    "epoch": self._active_epoch,
                    "epochs": self._total_epochs,
                    "phase": self._active_phase,
                }
        return {
            "node_id": self.config.node_id,
            "runs": selected,
            "errors": errors,
            "progress": progress,
        }

    def cleanup_run(self, run_id: str) -> dict[str, Any]:
        if RUN_ID_PATTERN.fullmatch(run_id) is None:
            raise ValueError("Invalid run ID")
        deadline = time.monotonic() + self.config.peer_timeout_seconds
        with self._condition:
            while self._active_run is not None or self._scheduled_run is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise RuntimeError("Cannot clean worker memory while training is active")
                self._condition.wait(timeout=remaining)

            had_model = run_id in self._models
            had_run_data = (
                had_model
                or run_id in self._run_architectures
                or run_id in self._run_model_audits
                or run_id in self._completed_rounds
                or run_id in self._results
                or run_id in self._run_errors
                or any(key[0] == run_id for key in self._round_weights)
                or any(key[0] == run_id for key in self._incoming)
            )
            self._models.pop(run_id, None)
            self._run_splits.pop(run_id, None)
            self._run_architectures.pop(run_id, None)
            self._run_model_audits.pop(run_id, None)
            self._completed_rounds.pop(run_id, None)
            self._results.pop(run_id, None)
            self._run_errors.pop(run_id, None)
            for key in [key for key in self._round_weights if key[0] == run_id]:
                self._round_weights.pop(key, None)
                self._round_sample_counts.pop(key, None)
            for key in [key for key in self._incoming if key[0] == run_id]:
                self._incoming.pop(key, None)

            if had_run_data:
                if had_model:
                    import tensorflow as tf

                    tf.keras.backend.clear_session()
                gc.collect()
                _trim_process_heap()

        _log_event("run_memory_cleaned", run_id=run_id)
        return {"run_id": run_id, "cleaned": True}

    @staticmethod
    def _load_shard(path: str) -> tuple[np.ndarray, np.ndarray]:
        with np.load(path, allow_pickle=False) as shard:
            images = np.asarray(shard["images"], dtype=np.uint8)
            labels = np.asarray(shard["labels"], dtype=np.int64).reshape(-1)
        if images.ndim != 4 or images.shape[1:] != (32, 32, 3):
            raise ValueError("Worker training shard must have shape (N, 32, 32, 3)")
        if images.shape[0] != labels.size or labels.size == 0:
            raise ValueError("Worker training shard has mismatched or empty image/label arrays")
        if labels.min() < 0 or labels.max() > 9:
            raise ValueError("Worker training shard contains labels outside CIFAR-10")
        return images, labels

    def configure(
        self,
        rounds: int,
        epochs: int,
        topology: str,
        dataset_split: str = "stratified",
        architecture: str | dict[str, Any] = FIXED_ARCHITECTURE,
        dataset_name: str | None = None,
        aggregation: str = "fedavg",
        sync_mode: str = "sync",
        sync_deadline_seconds: float | None = None,
        min_peers: int = 1,
        learned_senders: dict[str, list[str]] | None = None,
        measure_peers: bool = False,
        augmentation: str | None = None,
        fresh_model: bool = False,
    ) -> dict[str, Any]:
        if not isinstance(measure_peers, bool):
            raise ValueError("measure_peers must be a boolean")
        if not isinstance(fresh_model, bool):
            raise ValueError("fresh_model must be a boolean")
        if augmentation is not None and augmentation not in AUGMENTATIONS:
            raise ValueError(f"augmentation must be one of {', '.join(AUGMENTATIONS)}")
        senders_map = _validate_learned_senders(learned_senders) if topology == "learned" else {}
        if aggregation not in AGGREGATION_RULES:
            raise ValueError(f"aggregation must be one of {', '.join(AGGREGATION_RULES)}")
        if sync_mode not in SYNC_MODES:
            raise ValueError(f"sync_mode must be one of {', '.join(SYNC_MODES)}")
        if sync_deadline_seconds is not None and not 5 <= float(sync_deadline_seconds) <= 7200:
            raise ValueError("sync_deadline_seconds must be between 5 and 7200")
        if not isinstance(min_peers, int) or not 0 <= min_peers <= 64:
            raise ValueError("min_peers must be an integer between 0 and 64")
        if not 1 <= rounds <= 100 or not 1 <= epochs <= 50:
            raise ValueError("Rounds must be 1-100 and local epochs 1-50")
        if topology not in TRAINING_TOPOLOGIES:
            raise ValueError("Unsupported training topology")
        if dataset_split not in {
            "stratified",
            "dirichlet_a01",
            "dirichlet_a03",
            "dirichlet_a05",
            "dirichlet_a07",
        }:
            raise ValueError("Unsupported CIFAR-10 dataset split")
        if dataset_name is not None:
            if DATASET_NAME_PATTERN.fullmatch(dataset_name) is None:
                raise ValueError("Invalid dataset_name")


            shard_path = Path(self.config.dataset_root).resolve() / dataset_name / "train.npz"
            images, labels, class_names = load_dataset_shard(shard_path)
            global_validation = load_global_validation(shard_path.parent)
            if isinstance(architecture, dict) and architecture.get("model_type") in {
                "custom",
                "pretrained_deit",
            }:
                try:
                    selected_architecture, _ = validate_model_design(
                        architecture,
                        tuple(int(value) for value in images.shape[1:]),
                        len(class_names),
                    )
                except ModelDesignError as error:
                    raise ValueError(str(error)) from error
                selected_architecture = {
                    **selected_architecture,
                    **{
                        key: value
                        for key, value in architecture.items()
                        if key not in {"layers", "model_type"}
                    },
                    "model_type": architecture["model_type"],
                }
                if architecture.get("model_id") is not None:
                    selected_architecture["model_id"] = architecture["model_id"]
            else:
                selected_architecture = architecture_spec(architecture)
            selected_dataset_split = f"dataset:{dataset_name}"
        else:
            selected_architecture = architecture_spec(architecture)
            shard_path = (
                self.config.data_path
                if dataset_split == "stratified"
                else self.config.shard_paths.get(dataset_split)
            )
            if not shard_path:
                raise ValueError(f"Worker shard for {dataset_split} is not installed")
            images, labels = self._load_shard(shard_path)
            class_names = None
            global_validation = None
            selected_dataset_split = dataset_split
        with self._lock:
            if self._scheduled_run is not None or self._active_run is not None:
                raise RuntimeError("Cannot change worker configuration while training is active")
            self.images = images
            self.labels = labels
            self.dataset_split = selected_dataset_split
            self.dataset_name = dataset_name
            self._dataset_class_names = class_names
            self._global_validation = global_validation
            self._rounds = rounds
            self._epochs = epochs
            self._topology = topology
            self._architecture = selected_architecture
            self._aggregation = aggregation
            self._sync_mode = sync_mode
            self._sync_deadline_seconds = (
                float(sync_deadline_seconds) if sync_deadline_seconds is not None else None
            )
            self._min_peers = min_peers
            self._learned_senders = senders_map
            self._measure_peers = measure_peers
            self._augmentation = augmentation
            self._fresh_model = fresh_model
        return {
            "rounds": rounds,
            "epochs": epochs,
            "topology": topology,
            "dataset_split": selected_dataset_split,
            "dataset_name": dataset_name,
            "sample_count": int(labels.size),
            "architecture": selected_architecture,
            "aggregation": aggregation,
            "sync_mode": sync_mode,
            "sync_deadline_seconds": sync_deadline_seconds,
            "min_peers": min_peers,
            "learned_senders": {key: list(value) for key, value in senders_map.items()},
            "measure_peers": measure_peers,
            "augmentation": augmentation,
            "fresh_model": fresh_model,
        }

    def _measure_peer_models(
        self,
        model: Any,
        peer_contributions: list[Contribution],
        validation_images: Any,
        validation_labels: Any,
        arrivals: dict[str, float],
        exchange_started: float,
    ) -> dict[str, Any]:
        peer_losses: dict[str, float] = {}
        if validation_images is not None and validation_labels is not None and len(validation_labels):
            for contribution in peer_contributions:
                model.set_weights(contribution.weights)
                logits = model.predict(
                    validation_images,
                    batch_size=min(256, int(validation_labels.size)),
                    verbose=0,
                )
                peer_losses[contribution.node_id] = round(_cross_entropy(validation_labels, logits), 6)
        labels = np.asarray(self.labels).reshape(-1) if getattr(self, "labels", None) is not None else np.zeros(0, int)
        class_count = len(self._dataset_class_names or []) or int(labels.max(initial=-1)) + 1
        return {
            "peer_losses": peer_losses,
            # Seconds after this worker finished training until each peer model arrived
            # (0 when it was already waiting).
            "peer_arrival_seconds": {
                sender: round(max(0.0, arrived - exchange_started), 3) for sender, arrived in arrivals.items()
            },
            "class_histogram": np.bincount(labels.astype(np.int64), minlength=class_count).astype(int).tolist(),
        }

    def _round_senders(self, node_id: str, worker_ids: tuple[str, ...], topology: str) -> tuple[str, ...]:
        """Peers whose models ``node_id`` aggregates this round."""
        if topology == "learned":
            senders = self._learned_senders.get(node_id)
            if senders is None or not set(senders) <= set(worker_ids) - {node_id}:
                raise ValueError(f"The learned topology has no valid senders for {node_id}")
            return senders
        return topology_senders(node_id, worker_ids, topology)

    def _round_neighbours(self, node_id: str, worker_ids: tuple[str, ...], topology: str) -> tuple[str, ...]:
        """Peers ``node_id`` sends its model to this round."""
        if topology == "learned":
            return tuple(sorted(peer for peer in worker_ids if node_id in self._learned_senders.get(peer, ())))
        return topology_neighbours(node_id, worker_ids, topology)

    def start_run(self, run_id: str, seed: int, topology: str) -> dict[str, Any]:
        if not RUN_ID_PATTERN.fullmatch(run_id) or not 0 <= seed <= 2**32 - 1:
            raise ValueError("run_id or seed is invalid")
        if topology not in TRAINING_TOPOLOGIES:
            raise ValueError("Unsupported training topology")
        with self._lock:
            if self._reboot_requested:
                raise RuntimeError("The Jetson is preparing to reboot")
            if self._scheduled_run is not None or self._active_run is not None:
                raise RuntimeError(f"Training is already active for {self._scheduled_run}")
            rounds, epochs = self._rounds, self._epochs
            self._cancel.clear()
            self._scheduled_run = run_id
            self._run_errors.pop(run_id, None)
            self._run_thread = threading.Thread(
                target=self._run_configured_rounds,
                args=(run_id, rounds, epochs, seed, topology),
                name=f"training-{run_id}",
                daemon=True,
            )
            self._run_thread.start()
        _log_event(
            "training_started",
            run_id=run_id,
            rounds=rounds,
            epochs=epochs,
            seed=seed,
            topology=topology,
        )
        return {"accepted": True, "run_id": run_id, "rounds": rounds, "epochs": epochs}

    def _create_run_model(self, run_id: str, seed: int) -> Any:
        initial_model_loaded = False
        architecture_sha256: str | None = None
        weights_sha256: str | None = None
        if self.dataset_name is None:
            model = build_cifar10_model(seed, self._architecture)
            compile_cifar10_model(model)
            model_architecture: Any = self._architecture
            batch_size = self.config.dataset_batch_size
        else:
            if self._dataset_class_names is None:
                raise RuntimeError("Dataset class metadata is unavailable")
            model, installed_architecture = (
                None
                if self._fresh_model
                else _load_installed_dataset_initial_model(
                    self.config,
                    self.dataset_name,
                    input_shape=tuple(int(value) for value in self.images.shape[1:]),
                    class_names=self._dataset_class_names,
                )
            ) or (None, None)
            if model is None:
                if self._architecture.get("model_type") in {"custom", "pretrained_deit"}:
                    model = build_dataset_model(
                        input_shape=tuple(int(value) for value in self.images.shape[1:]),
                        num_classes=len(self._dataset_class_names),
                        architecture=self._architecture,
                        seed=seed,
                    )
                else:
                    from worker.nas_search import _build_dataset_candidate

                    model = _build_dataset_candidate(
                        input_shape=tuple(int(value) for value in self.images.shape[1:]),
                        num_classes=len(self._dataset_class_names),
                        architecture=self._architecture,
                        seed=seed,
                    )
                model_architecture = self._architecture
            else:
                model_architecture = installed_architecture
                initial_model_loaded = True
                metadata_path = (
                    Path(self.config.dataset_root).resolve()
                    / self.dataset_name
                    / "models"
                    / "initial_model"
                    / "metadata.json"
                )
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                architecture_sha256 = metadata["architecture_sha256"]
                weights_sha256 = metadata["weights_sha256"]
            import tensorflow as tf

            # An explicit request (e.g. an HPO result) sets the training settings;
            # otherwise the installed model trains with its own recorded settings
            # rather than the worker-wide defaults.
            training_settings = self._architecture.get("training") or (
                model_architecture.get("training")
                if initial_model_loaded and isinstance(model_architecture, dict)
                else None
            ) or {}
            optimizer_name = training_settings.get("optimizer", "adam")
            learning_rate = float(
                training_settings.get(
                    "learning_rate",
                    self.config.dataset_learning_rate,
                )
            )
            optimizer_types = {
                "adam": tf.keras.optimizers.Adam,
                "adamw": tf.keras.optimizers.AdamW,
                "sgd": tf.keras.optimizers.SGD,
                "rmsprop": tf.keras.optimizers.RMSprop,
            }
            if optimizer_name not in optimizer_types:
                raise ValueError(f"Unsupported optimizer: {optimizer_name}")
            model.compile(
                optimizer=optimizer_types[optimizer_name](learning_rate=learning_rate),
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=["accuracy"],
            )
            batch_size = int(
                training_settings.get("batch_size", self.config.dataset_batch_size)
            )
        import tensorflow as tf

        self._models[run_id] = model
        self._run_architectures[run_id] = model_architecture
        self._results[run_id] = []
        augmentation = "none"
        if self.dataset_name is not None and self._augmentation is not None:
            augmentation = self._augmentation
        elif self.dataset_name is not None:
            requested_training = self._architecture.get("training") or {}
            installed_training = (
                model_architecture.get("training") if isinstance(model_architecture, dict) else None
            ) or {}
            augmentation = str(
                requested_training.get("augmentation")
                or installed_training.get("augmentation")
                or "none"
            )
        self._run_model_audits[run_id] = {
            "learning_rate": float(tf.keras.backend.get_value(model.optimizer.learning_rate)),
            "batch_size": batch_size,
            "augmentation": augmentation,
            "optimizer": type(model.optimizer).__name__,
            "initial_model_loaded": initial_model_loaded,
            "architecture_sha256": architecture_sha256,
            "weights_sha256": weights_sha256,
        }
        return model

    def _run_configured_rounds(
        self,
        run_id: str,
        rounds: int,
        epochs: int,
        seed: int,
        topology: str,
    ) -> None:
        try:
            for round_index in range(rounds):
                if self._cancel.is_set():
                    break
                self.run_round(run_id, round_index, epochs, seed, topology)
        except Exception as error:
            with self._lock:
                self._run_errors[run_id] = f"{type(error).__name__}: {error}"
        finally:
            with self._condition:
                self._scheduled_run = None
                self._condition.notify_all()

    def cancel(self) -> dict[str, Any]:
        with self._lock:
            active_run = self._scheduled_run or self._active_run
            if active_run is None:
                return {"accepted": False, "message": "No training round is active."}
            self._cancel.set()
            self._condition.notify_all()
        _log_event("training_cancel_requested", run_id=active_run)
        return {"accepted": True, "run_id": active_run}

    def _persist_latest_model(
        self,
        run_id: str,
        round_index: int,
        weights: list[np.ndarray],
        architecture: Any,
        result: dict[str, Any],
    ) -> None:
        """Keep the newest aggregated model on disk so it survives memory cleanup.

        Only custom dataset models are kept; pruning reads this file after training.
        A failure here never fails the round.
        """
        dataset_name = getattr(self, "dataset_name", None)
        if not dataset_name or not isinstance(architecture, dict) or architecture.get("model_type") != "custom":
            return
        try:
            models_dir = Path(self.config.dataset_root).resolve() / dataset_name / "models"
            models_dir.mkdir(parents=True, exist_ok=True)
            payload = serialize_weights([np.asarray(item) for item in weights])
            metadata = {
                "dataset_name": dataset_name,
                "run_id": run_id,
                "round": round_index,
                "architecture": architecture,
                "validation_accuracy": result.get("validation_accuracy"),
                "validation_macro_f1": result.get("validation_macro_f1"),
                "weights_sha256": hashlib.sha256(payload).hexdigest(),
                "saved_at": time.time(),
            }
            for name, data in (
                (LATEST_MODEL_WEIGHTS, payload),
                (LATEST_MODEL_METADATA, (json.dumps(metadata) + "\n").encode("utf-8")),
            ):
                temporary = models_dir / f".{name}.tmp"
                temporary.write_bytes(data)
                os.replace(temporary, models_dir / name)
        except (OSError, ValueError) as error:
            _log_event("latest_model_save_failed", run_id=run_id, round=round_index, error=str(error))

    def latest_model(self, dataset_name: str) -> dict[str, Any]:
        """Return the newest aggregated model of a dataset run with base64 weights."""
        models_dir = Path(self.config.dataset_root).resolve() / dataset_name / "models"
        weights_path = models_dir / LATEST_MODEL_WEIGHTS
        metadata_path = models_dir / LATEST_MODEL_METADATA
        if not weights_path.is_file() or not metadata_path.is_file():
            raise FileNotFoundError(f"No trained model is stored for {dataset_name}")
        payload = weights_path.read_bytes()
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("weights_sha256") != hashlib.sha256(payload).hexdigest():
            raise ValueError("Stored trained model is incomplete")
        return {**metadata, "weights_b64": base64.b64encode(payload).decode("ascii")}

    def weights(self, run_id: str, round_index: int) -> tuple[bytes, int]:
        with self._lock:
            weights = self._round_weights.get((run_id, round_index))
            if weights is None:
                raise KeyError("Requested model is not available")
            sample_count = self._round_sample_counts[(run_id, round_index)]
        return serialize_weights(weights), sample_count

    def accept_peer_weights(
        self,
        run_id: str,
        round_index: int,
        sender: str,
        sample_count: int,
        payload: bytes,
        signature: str | None = None,
    ) -> bool:
        """Store a peer's weights; returns False when the round is already finished."""
        if sender not in self._peer_map or sample_count < 1:
            raise ValueError("Unknown peer or invalid sample count")
        weights = deserialize_weights(payload)
        received_signature = model_signature(weights)
        if signature is not None and signature != received_signature:
            raise ValueError("Peer model signature header does not match its weights")
        key = (run_id, round_index)
        with self._condition:
            if round_index <= self._completed_rounds.get(run_id, -1):
                # A straggler's update for a round this worker already aggregated
                # (semi-synchronous mode); keeping it would only leak memory.
                _log_event("stale_peer_weights_dropped", run_id=run_id, round=round_index, peer_id=sender)
                return False
            model = self._models.get(run_id)
            if model is not None:
                local_signature = model_signature([np.asarray(item) for item in model.get_weights()])
                if local_signature != received_signature:
                    raise ValueError(
                        f"Architecture mismatch: {sender} sent model {received_signature}, "
                        f"this worker trains {local_signature}; redistribute one initial model"
                    )
            senders = self._incoming.setdefault(key, {})
            if sender in senders:
                # A sender retries when the connection drops after the upload; the
                # identical model is acknowledged again, a different one is an error.
                stored = senders[sender][0]
                if len(stored) == len(weights) and all(
                    np.array_equal(left, right) for left, right in zip(stored, weights)
                ):
                    return True
                raise ValueError("Duplicate weights received from a peer for this round")
            senders[sender] = (weights, sample_count, len(payload))
            self._arrivals.setdefault(key, {})[sender] = time.monotonic()
            self._condition.notify_all()
        _log_event(
            "peer_weights_received",
            run_id=run_id,
            round=round_index,
            peer_id=sender,
            sample_count=sample_count,
            bytes_received=len(payload),
        )
        return True

    def _run_split(
        self,
        run_id: str,
        seed: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
        """Return this run's train/validation arrays, materialized once per run."""
        with self._lock:
            cached = self._run_splits.get(run_id)
        if cached is not None:
            return cached
        if self.dataset_name is None:
            split = (self.images, self.labels, None, None)
        else:
            from worker.nas_search import _split_indices

            training_indices, validation_indices = _split_indices(self.labels, seed)
            split = (
                self.images[training_indices],
                self.labels[training_indices],
                self.images[validation_indices],
                self.labels[validation_indices],
            )
        with self._lock:
            self._run_splits[run_id] = split
        return split

    def run_round(
        self,
        run_id: str,
        round_index: int,
        epochs: int,
        seed: int,
        topology: str,
    ) -> dict[str, Any]:
        if not RUN_ID_PATTERN.fullmatch(run_id):
            raise ValueError("run_id must be a safe identifier")
        if not 0 <= round_index < 100 or not 1 <= epochs <= 50 or seed < 0:
            raise ValueError("Round, epoch, or seed value is out of range")
        worker_ids = tuple(sorted((self.config.node_id, *self._peer_map.keys())))
        neighbors = self._round_neighbours(self.config.node_id, worker_ids, topology)
        expected_senders = self._round_senders(self.config.node_id, worker_ids, topology)
        with self._lock:
            if self._reboot_requested:
                raise RuntimeError("The Jetson is preparing to reboot")
            if self._active_run is not None:
                raise RuntimeError(f"Training is already active for {self._active_run}")
            current_round = self._completed_rounds.get(run_id, -1)
            if round_index != current_round + 1:
                raise ValueError(f"Expected round {current_round + 1}, received {round_index}")
            if run_id not in self._models:
                self._create_run_model(run_id, seed)
            model = self._models[run_id]
            self._active_run = run_id
            self._active_round = round_index
            self._active_epoch = 0
            self._total_epochs = epochs
            self._active_phase = "local_training"
        started = time.perf_counter()
        before_available_kb = _memory_available_kb()
        rss_before = _resident_memory_mb()
        initial_validation_accuracy: float | None = None
        try:
            import tensorflow as tf

            training_images, training_labels, validation_images, validation_labels = (
                self._run_split(run_id, seed)
            )
            train_sample_count = int(training_labels.size)

            if validation_images is not None and validation_labels is not None:
                from common.cifar_model import classification_metrics

                if round_index == 0:
                    initial_logits = model.predict(
                        validation_images,
                        batch_size=min(256, int(validation_labels.size)),
                        verbose=0,
                    )
                    initial_validation_accuracy = classification_metrics(
                        validation_labels,
                        initial_logits,
                    )["accuracy"]

            cancel_event = self._cancel
            trainer = self

            class StopOnRequest(tf.keras.callbacks.Callback):
                def on_epoch_begin(self, epoch: int, logs: Any = None) -> None:
                    with trainer._lock:
                        if trainer._active_run == run_id:
                            trainer._active_epoch = epoch + 1

                def on_train_batch_end(self, batch: int, logs: Any = None) -> None:
                    if cancel_event.is_set():
                        self.model.stop_training = True

            train_started = time.perf_counter()
            fit_batch_size = int(self._run_model_audits[run_id]["batch_size"])
            augmentation = self._run_model_audits[run_id].get("augmentation", "none")
            if augmentation == "flip_shift":
                training_input = _augmented_dataset(
                    training_images,
                    training_labels,
                    batch_size=fit_batch_size,
                    seed=int(seed) * 1000 + int(round_index),
                )
                history = model.fit(
                    training_input,
                    epochs=epochs,
                    verbose=0,
                    callbacks=[StopOnRequest()],
                )
            else:
                history = model.fit(
                    training_images,
                    training_labels,
                    epochs=epochs,
                    batch_size=fit_batch_size,
                    shuffle=True,
                    verbose=0,
                    callbacks=[StopOnRequest()],
                )
            train_seconds = time.perf_counter() - train_started
            if self._cancel.is_set():
                raise RuntimeError("Training was cancelled before peer exchange")
            with self._lock:
                self._active_phase = "peer_exchange"
            local_weights = [np.asarray(item) for item in model.get_weights()]
            payload = serialize_weights(local_weights)
            local_signature = model_signature(local_weights)
            sent_bytes = 0
            exchange_started = time.perf_counter()
            exchange_monotonic = time.monotonic()
            for peer_id in neighbors:
                peer = self._peer_map[peer_id]
                transfer_started = time.perf_counter()
                request = Request(
                    f"{peer.base_url}/v1/weights",
                    data=payload,
                    headers={
                        "Authorization": f"Bearer {self.config.api_token}",
                        "Content-Type": "application/octet-stream",
                        "X-Run-ID": run_id,
                        "X-Round": str(round_index),
                        "X-Sender-Node": self.config.node_id,
                        "X-Sample-Count": str(train_sample_count),
                        "X-Model-Signature": local_signature,
                    },
                    method="POST",
                )
                try:
                    attempts = _send_with_retry(
                        request,
                        timeout=self.config.peer_timeout_seconds,
                        peer_id=peer_id,
                        cancel_event=self._cancel,
                    )
                    sent_bytes += len(payload)
                    _log_event(
                        "peer_weights_sent",
                        run_id=run_id,
                        round=round_index,
                        peer_id=peer_id,
                        bytes_sent=len(payload),
                        transfer_seconds=round(time.perf_counter() - transfer_started, 3),
                        attempts=attempts,
                    )
                except (HTTPError, URLError, TimeoutError, OSError) as error:
                    raise RuntimeError(f"Peer exchange with {peer_id} failed: {error}") from error
            with self._lock:
                sync_mode = self._sync_mode
                aggregation_rule = self._aggregation
                wait_budget = (
                    self.config.peer_timeout_seconds
                    if sync_mode == "sync" or self._sync_deadline_seconds is None
                    else self._sync_deadline_seconds
                )
                required_peers = min(self._min_peers, len(expected_senders))
            deadline = time.monotonic() + wait_budget
            peer_wait_started = time.perf_counter()
            with self._condition:
                while not set(expected_senders).issubset(
                    self._incoming.get((run_id, round_index), {})
                ):
                    if self._cancel.is_set():
                        raise RuntimeError("Training was cancelled during peer exchange")
                    arrived = set(self._incoming.get((run_id, round_index), {}))
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        missing = sorted(set(expected_senders) - arrived)
                        if sync_mode == "semi_sync" and len(arrived & set(expected_senders)) >= required_peers:
                            # Semi-synchronous: aggregate what arrived instead of failing the
                            # whole round because one peer is slow.
                            break
                        raise TimeoutError(f"Timed out waiting for peer weights from {missing}")
                    self._condition.wait(timeout=min(remaining, 1.0))
                incoming = self._incoming.pop((run_id, round_index), {})
                arrivals = self._arrivals.pop((run_id, round_index), {})
            peer_wait_seconds = time.perf_counter() - peer_wait_started
            used_senders = [peer_id for peer_id in expected_senders if peer_id in incoming]
            missing_senders = sorted(set(expected_senders) - set(used_senders))
            contributions = [
                Contribution(self.config.node_id, local_weights, train_sample_count, len(expected_senders))
            ]
            bytes_received = 0
            for peer_id in used_senders:
                peer_weights, sample_count, received_bytes = incoming[peer_id]
                contributions.append(
                    Contribution(
                        peer_id,
                        peer_weights,
                        sample_count,
                        len(self._round_senders(peer_id, worker_ids, topology)),
                    )
                )
                bytes_received += received_bytes
            peer_measurements: dict[str, Any] = {}
            if self._measure_peers:
                # APG topology learning inputs: how well each sender's model fits this
                # worker's validation data, and how long this worker waited for it.
                peer_measurements = self._measure_peer_models(
                    model, contributions[1:], validation_images, validation_labels, arrivals, exchange_monotonic
                )
            averaged = aggregate(aggregation_rule, contributions)
            del contributions, incoming
            model.set_weights(averaged)
            validation_metrics: dict[str, Any] = {}
            if validation_images is not None and validation_labels is not None:
                logits = model.predict(
                    validation_images,
                    batch_size=min(256, int(validation_labels.size)),
                    verbose=0,
                )
                metrics = classification_metrics(validation_labels, logits)
                validation_metrics = {
                    "validation_accuracy": metrics["accuracy"],
                    "validation_macro_precision": metrics["macro_precision"],
                    "validation_macro_recall": metrics["macro_recall"],
                    "validation_macro_f1": metrics["macro_f1"],
                    "validation_sample_count": int(validation_labels.size),
                    "validation_per_class_precision": metrics["per_class_precision"],
                    "validation_per_class_recall": metrics["per_class_recall"],
                    "validation_per_class_f1": metrics["per_class_f1"],
                    "validation_confusion_matrix_counts": metrics["confusion_matrix_counts"],
                    "validation_confusion_matrix_row_normalized_percent": (
                        metrics["confusion_matrix_row_normalized_percent"]
                    ),
                }
            global_validation = self._global_validation
            if global_validation is not None:
                from common.cifar_model import classification_metrics as score_logits

                global_images, global_labels = global_validation
                global_logits = model.predict(
                    global_images,
                    batch_size=min(256, int(global_labels.size)),
                    verbose=0,
                )
                global_metrics = score_logits(global_labels, global_logits)
                del global_logits
                validation_metrics.update(
                    {
                        "global_validation_accuracy": global_metrics["accuracy"],
                        "global_validation_macro_f1": global_metrics["macro_f1"],
                        "global_validation_sample_count": int(global_labels.size),
                        "global_validation_per_class_recall": global_metrics["per_class_recall"],
                    }
                )
            exchange_seconds = time.perf_counter() - exchange_started
            rss_after = _resident_memory_mb()
            after_available_kb = _memory_available_kb()
            peak_rss_mb = _peak_process_rss_mb()
            history_values = {
                name: float(values[-1])
                for name, values in history.history.items()
                if values
            }
            result = {
                "node_id": self.config.node_id,
                "run_id": run_id,
                "round": round_index + 1,
                "topology": topology,
                # The model that actually trained: an installed initial model takes
                # precedence over the configured architecture, so report both.
                "architecture": self._run_architectures.get(run_id, self._architecture),
                "requested_architecture": self._architecture,
                "parameter_count": (
                    int(model.count_params()) if hasattr(model, "count_params") else None
                ),
                "dataset_split": self.dataset_split,
                "dataset_name": self.dataset_name,
                "augmentation": self._run_model_audits.get(run_id, {}).get("augmentation", "none"),
                "local_epochs": epochs,
                "sample_count": train_sample_count,
                **self._run_model_audits[run_id],
                "training_loss": history_values.get("loss"),
                "training_accuracy": history_values.get("accuracy"),
                **(
                    {"initial_validation_accuracy": initial_validation_accuracy}
                    if initial_validation_accuracy is not None
                    else {}
                ),
                **validation_metrics,
                "training_seconds": round(train_seconds, 3),
                "round_seconds": round(time.perf_counter() - started, 3),
                "rss_before_mb": rss_before,
                "rss_after_mb": rss_after,
                "peak_process_rss_mb": peak_rss_mb,
                "system_memory_available_delta_mb": round(
                    (after_available_kb - before_available_kb) / 1024,
                    1,
                ),
                "bytes_sent": sent_bytes,
                "bytes_received": bytes_received,
                "peer_messages_sent": len(neighbors),
                "peer_messages_received": len(used_senders),
                "aggregation": aggregation_rule,
                "sync_mode": sync_mode,
                "peers_expected": list(expected_senders),
                "peers_used": used_senders,
                "missing_peers": missing_senders,
                "model_signature": local_signature,
                **peer_measurements,
                "exchange_seconds": round(exchange_seconds, 3),
                "peer_wait_seconds": round(peer_wait_seconds, 3),
                "neighbours": list(neighbors),
                "weight_bytes": model_size_bytes(model),
            }
            with self._lock:
                self._completed_rounds[run_id] = round_index
                self._round_weights[(run_id, round_index)] = [
                    np.asarray(item).copy() for item in averaged
                ]
                self._round_sample_counts[(run_id, round_index)] = train_sample_count
                # The master reads a round's weights right after that round; older rounds
                # only grew worker memory for the whole run.
                for stale_key in [
                    key for key in self._incoming if key[0] == run_id and key[1] <= round_index
                ]:
                    self._incoming.pop(stale_key, None)
                for stale_round in range(0, round_index - ROUND_WEIGHTS_RETAINED + 1):
                    self._round_weights.pop((run_id, stale_round), None)
                    self._round_sample_counts.pop((run_id, stale_round), None)
                self._results[run_id].append(result)
                run_architecture = self._run_architectures.get(run_id)
            self._persist_latest_model(run_id, round_index, averaged, run_architecture, result)
            _log_event("round_completed", **result)
            return result
        except Exception as error:
            with self._condition:
                self._incoming.pop((run_id, round_index), None)
            _log_event(
                "round_failed",
                run_id=run_id,
                round=round_index,
                topology=topology,
                error_type=type(error).__name__,
                error=str(error),
            )
            raise
        finally:
            with self._condition:
                self._active_run = None
                self._active_round = None
                self._active_epoch = None
                self._total_epochs = None
                self._active_phase = None
                self._condition.notify_all()


def make_handler(config: WorkerConfig, trainer: WorkerTrainer) -> type[BaseHTTPRequestHandler]:
    nas_lock = threading.Lock()
    nas_control_lock = threading.Lock()
    nas_cancel_event = threading.Event()
    nas_search_active = False

    class WorkerHandler(BaseHTTPRequestHandler):
        server_version = "DFLWorkerTraining/1.0"
        sys_version = ""

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path == "/healthz":
                self._send_json(HTTPStatus.OK, {"status": "ok"})
                return
            if not is_authorized(self.headers.get("Authorization"), config.api_token):
                self._unauthorized()
                return
            if parsed.path == "/v1/status":
                self._send_json(HTTPStatus.OK, trainer.status())
            elif parsed.path == "/v1/processes":
                try:
                    report = _process_snapshot()
                    report["node_id"] = config.node_id
                    report["training"] = trainer.status()
                    with nas_control_lock:
                        report["nas_active"] = nas_search_active
                    self._send_json(HTTPStatus.OK, report)
                except (OSError, subprocess.SubprocessError, ValueError) as error:
                    self._send_json(
                        HTTPStatus.SERVICE_UNAVAILABLE,
                        {"error": f"Could not inspect worker processes: {error}"},
                    )
            elif parsed.path == "/v1/results":
                run_id = parse_qs(parsed.query).get("run_id", [None])[0]
                self._send_json(HTTPStatus.OK, trainer.results(run_id))
            elif re.fullmatch(r"/v1/datasets/[A-Za-z0-9][A-Za-z0-9._-]{0,63}/latest-model", parsed.path):
                try:
                    self._send_json(HTTPStatus.OK, trainer.latest_model(parsed.path.split("/")[-2]))
                except FileNotFoundError as error:
                    self._send_json(HTTPStatus.NOT_FOUND, {"error": str(error)})
                except (OSError, ValueError, json.JSONDecodeError) as error:
                    self._send_json(HTTPStatus.CONFLICT, {"error": str(error)})
            elif re.fullmatch(r"/v1/datasets/[A-Za-z0-9][A-Za-z0-9._-]{0,63}", parsed.path):
                dataset_name = parsed.path.rsplit("/", 1)[-1]
                metadata_path = (
                    Path(config.dataset_root).resolve()
                    / dataset_name
                    / "metadata.json"
                )
                with trainer._lock:
                    if not metadata_path.is_file():
                        self._send_json(HTTPStatus.NOT_FOUND, {"error": "dataset_not_found"})
                        return
                    self._send_json(
                        HTTPStatus.OK,
                        json.loads(metadata_path.read_text(encoding="utf-8")),
                    )
            elif re.fullmatch(
                r"/v1/datasets/[A-Za-z0-9][A-Za-z0-9._-]{0,63}/initial-model",
                parsed.path,
            ):
                dataset_name = parsed.path.split("/")[-2]
                metadata_path = (
                    Path(config.dataset_root).resolve()
                    / dataset_name
                    / "models"
                    / "initial_model"
                    / "metadata.json"
                )
                if not metadata_path.is_file():
                    self._send_json(HTTPStatus.NOT_FOUND, {"error": "initial_model_not_found"})
                    return
                self._send_json(
                    HTTPStatus.OK,
                    json.loads(metadata_path.read_text(encoding="utf-8")),
                )
            elif parsed.path == "/v1/weights":
                query = parse_qs(parsed.query)
                try:
                    run_id = query["run_id"][0]
                    round_index = int(query["round"][0])
                    payload, sample_count = trainer.weights(run_id, round_index)
                except (KeyError, ValueError) as error:
                    self._send_json(HTTPStatus.NOT_FOUND, {"error": str(error)})
                    return
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "application/octet-stream")
                self.send_header("Content-Length", str(len(payload)))
                self.send_header("X-Sample-Count", str(sample_count))
                self.end_headers()
                self.wfile.write(payload)
            else:
                self._send_json(HTTPStatus.NOT_FOUND, {"error": "not_found"})

        def do_POST(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path == "/v1/weights":
                self._receive_peer_weights()
                return
            if not is_authorized(self.headers.get("Authorization"), config.api_token):
                self._discard_request_body(MAX_DATASET_SHARD_BYTES)
                self._unauthorized()
                return
            if parsed.path == "/v1/config":
                self._configure()
            elif parsed.path == "/v1/peers":
                self._configure_peers()
            elif parsed.path == "/v1/nas/search":
                self._nas_search()
            elif re.fullmatch(
                r"/v1/datasets/[A-Za-z0-9][A-Za-z0-9._-]{0,63}/shard",
                parsed.path,
            ):
                dataset_name = parsed.path.split("/")[-2]
                self._install_dataset_shard(dataset_name)
            elif re.fullmatch(
                r"/v1/datasets/[A-Za-z0-9][A-Za-z0-9._-]{0,63}/initial-model",
                parsed.path,
            ):
                dataset_name = parsed.path.split("/")[-2]
                self._install_initial_model(dataset_name)
            elif parsed.path == "/v1/start":
                self._start_run()
            elif parsed.path == "/v1/train-round":
                self._train_round()
            elif parsed.path == "/v1/cancel":
                self._send_json(HTTPStatus.OK, trainer.cancel())
            elif parsed.path == "/v1/nas/cancel":
                with nas_control_lock:
                    if not nas_search_active:
                        self._send_json(
                            HTTPStatus.OK,
                            {"accepted": False, "message": "No NAS search is active."},
                        )
                        return
                    nas_cancel_event.set()
                    self._send_json(
                        HTTPStatus.ACCEPTED,
                        {"accepted": True, "message": "NAS search cancellation requested."},
                    )
            elif parsed.path == "/v1/memory/cleanup":
                recycle = parse_qs(parsed.query).get("recycle", ["0"])[0] == "1"
                try:
                    result = trainer.cleanup_memory(nas_lock, recycle=recycle)
                except WorkerMemoryBusyError as error:
                    self._send_json(HTTPStatus.CONFLICT, {"error": str(error)})
                except RuntimeError as error:
                    self._send_json(HTTPStatus.SERVICE_UNAVAILABLE, {"error": str(error)})
                else:
                    self._send_json(HTTPStatus.OK, result)
            elif parsed.path == "/v1/system/reboot":
                if self.headers.get("Content-Length", "0") != "0":
                    self._send_json(HTTPStatus.BAD_REQUEST, {"error": "request_body_not_allowed"})
                    return
                if not nas_lock.acquire(blocking=False):
                    self._send_json(
                        HTTPStatus.CONFLICT,
                        {"error": "Cannot reboot while NAS search is active"},
                    )
                    return
                try:
                    if not trainer.begin_reboot():
                        self._send_json(
                            HTTPStatus.CONFLICT,
                            {"error": "Cannot reboot while worker training is active"},
                        )
                        return
                    request = Request(
                        "http://127.0.0.1:5005/reboot",
                        data=b"",
                        headers={"Authorization": f"Bearer {config.api_token}"},
                        method="POST",
                    )
                    try:
                        with urlopen(request, timeout=5) as response:
                            payload = json.loads(response.read().decode("utf-8"))
                        if (
                            not isinstance(payload, dict)
                            or payload.get("status") != "accepted"
                            or payload.get("action") != "reboot"
                        ):
                            raise RuntimeError("Privileged reboot helper returned an invalid response")
                    except HTTPError as error:
                        detail = error.read().decode("utf-8", errors="replace").strip()[:300]
                        raise RuntimeError(
                            f"Privileged reboot helper returned HTTP {error.code}: {detail}"
                        ) from error
                    except (URLError, TimeoutError, OSError) as error:
                        raise RuntimeError(
                            f"Could not reach the privileged reboot helper: {error}"
                        ) from error
                    except (UnicodeDecodeError, json.JSONDecodeError) as error:
                        raise RuntimeError(
                            "Privileged reboot helper returned invalid JSON"
                        ) from error
                except RuntimeError as error:
                    trainer.cancel_reboot()
                    self._send_json(HTTPStatus.SERVICE_UNAVAILABLE, {"error": str(error)})
                else:
                    self._send_json(HTTPStatus.ACCEPTED, payload)
                finally:
                    nas_lock.release()
            else:
                self._send_json(HTTPStatus.NOT_FOUND, {"error": "not_found"})

        def do_PUT(self) -> None:
            parsed = urlparse(self.path)
            if not is_authorized(self.headers.get("Authorization"), config.api_token):
                self._discard_request_body(MAX_DATASET_SHARD_BYTES)
                self._unauthorized()
                return
            global_match = re.fullmatch(
                r"/v1/datasets/([A-Za-z0-9][A-Za-z0-9._-]{0,63})/global-validation",
                parsed.path,
            )
            if global_match is not None:
                self._install_dataset_shard(global_match.group(1), installer=install_global_validation)
                return
            match = re.fullmatch(
                r"/v1/datasets/([A-Za-z0-9][A-Za-z0-9._-]{0,63})/shard",
                parsed.path,
            )
            if match is None:
                self._discard_request_body(MAX_DATASET_SHARD_BYTES)
                self._send_json(HTTPStatus.NOT_FOUND, {"error": "not_found"})
                return
            self._install_dataset_shard(match.group(1), replace_existing=True)

        def do_DELETE(self) -> None:
            parsed = urlparse(self.path)
            if not is_authorized(self.headers.get("Authorization"), config.api_token):
                self._unauthorized()
                return
            run_match = re.fullmatch(
                r"/v1/runs/([A-Za-z0-9][A-Za-z0-9._-]{0,63})",
                parsed.path,
            )
            if run_match is not None:
                try:
                    result = trainer.cleanup_run(run_match.group(1))
                except RuntimeError as error:
                    self._send_json(HTTPStatus.CONFLICT, {"error": str(error)})
                except ValueError as error:
                    self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(error)})
                else:
                    self._send_json(HTTPStatus.OK, result)
                return
            match = re.fullmatch(
                r"/v1/datasets/([A-Za-z0-9][A-Za-z0-9._-]{0,63})/initial-model",
                parsed.path,
            )
            if match is None:
                self._send_json(HTTPStatus.NOT_FOUND, {"error": "not_found"})
                return
            dataset_name = match.group(1)
            with trainer._lock:
                if trainer._active_run is not None or trainer._scheduled_run is not None:
                    self._send_json(
                        HTTPStatus.CONFLICT,
                        {"error": "Cannot remove an initial model while training is active"},
                    )
                    return

                dataset_root = Path(config.dataset_root).resolve()
                dataset_directory = dataset_root / dataset_name
                models_directory = dataset_directory / "models"
                model_path = models_directory / "initial_model"
                if (
                    dataset_directory.is_symlink()
                    or dataset_directory.resolve().parent != dataset_root
                    or models_directory.is_symlink()
                    or model_path.is_symlink()
                ):
                    self._send_json(HTTPStatus.CONFLICT, {"error": "Initial-model path is unsafe"})
                    return
                if model_path.exists():
                    if not model_path.is_dir():
                        self._send_json(
                            HTTPStatus.CONFLICT,
                            {"error": "Initial-model path is not a directory"},
                        )
                        return
                    shutil.rmtree(model_path)
            self.send_response(HTTPStatus.NO_CONTENT)
            self.end_headers()

        def _configure(self) -> None:
            try:
                payload = self._read_json()
                result = trainer.configure(
                    rounds=payload["rounds"],
                    epochs=payload["epochs"],
                    topology=payload["topology"],
                    dataset_split=payload.get("dataset_split", "stratified"),
                    architecture=payload.get("architecture", FIXED_ARCHITECTURE),
                    dataset_name=payload.get("dataset_name"),
                    aggregation=payload.get("aggregation", "fedavg"),
                    sync_mode=payload.get("sync_mode", "sync"),
                    sync_deadline_seconds=payload.get("sync_deadline_seconds"),
                    min_peers=payload.get("min_peers", 1),
                    learned_senders=payload.get("learned_senders"),
                    measure_peers=payload.get("measure_peers", False),
                    augmentation=payload.get("augmentation"),
                    fresh_model=payload.get("fresh_model", False),
                )
            except RuntimeError as error:
                self._send_json(HTTPStatus.CONFLICT, {"error": str(error)})
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(error)})
            else:
                self._send_json(HTTPStatus.OK, result)

        def _configure_peers(self) -> None:
            try:
                payload = self._read_json()
                raw_peers = payload["peers"]
                if not isinstance(raw_peers, list):
                    raise ValueError("peers must be a list")
                peers = tuple(
                    Peer(
                        node_id=str(item["node_id"]),
                        base_url=str(item["base_url"]).rstrip("/"),
                        token=str(item["token"]),
                    )
                    for item in raw_peers
                    if isinstance(item, dict)
                )
                if len(peers) != len(raw_peers):
                    raise ValueError("Each peer must be an object")
                result = trainer.configure_peers(peers)
            except RuntimeError as error:
                self._send_json(HTTPStatus.CONFLICT, {"error": str(error)})
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(error)})
            else:
                self._send_json(HTTPStatus.OK, result)

        def _install_dataset_shard(
            self,
            dataset_name: str,
            *,
            replace_existing: bool = False,
            installer: Any = None,
        ) -> None:
            with trainer._lock:
                if trainer._active_run is not None or trainer._scheduled_run is not None:
                    self._discard_request_body(MAX_DATASET_SHARD_BYTES)
                    self._send_json(
                        HTTPStatus.CONFLICT,
                        {"error": "Cannot change a dataset while worker training is active"},
                    )
                    return
                if not nas_lock.acquire(blocking=False):
                    self._discard_request_body(MAX_DATASET_SHARD_BYTES)
                    self._send_json(
                        HTTPStatus.CONFLICT,
                        {"error": "Cannot change a dataset while NAS is active"},
                    )
                    return
                try:
                    try:
                        length = int(self.headers.get("Content-Length", ""))
                    except ValueError:
                        self._send_json(HTTPStatus.BAD_REQUEST, {"error": "Invalid Content-Length"})
                        return
                    if not 1 <= length <= MAX_DATASET_SHARD_BYTES:
                        self._send_json(
                            HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                            {"error": "Dataset shard payload size is out of range"},
                        )
                        return
                    payload = self.rfile.read(length)
                    if len(payload) != length:
                        self._send_json(HTTPStatus.BAD_REQUEST, {"error": "Dataset shard was truncated"})
                        return
                    try:
                        result = (
                            installer(config, dataset_name, payload)
                            if installer is not None
                            else install_dataset_shard(
                                config,
                                dataset_name,
                                payload,
                                replace_existing=replace_existing,
                            )
                        )
                    except FileExistsError as error:
                        self._send_json(HTTPStatus.CONFLICT, {"error": str(error)})
                    except ValueError as error:
                        self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(error)})
                    except OSError as error:
                        LOGGER.exception("Could not save uploaded dataset shard for %s", config.node_id)
                        self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(error)})
                    else:
                        self._send_json(HTTPStatus.OK, result)
                finally:
                    nas_lock.release()

        def _install_initial_model(self, dataset_name: str) -> None:
            with trainer._lock:
                if trainer._active_run is not None or trainer._scheduled_run is not None:
                    self._discard_request_body(MAX_DATASET_SHARD_BYTES)
                    self._send_json(
                        HTTPStatus.CONFLICT,
                        {"error": "Cannot install a model while worker training is active"},
                    )
                    return
                try:
                    length = int(self.headers.get("Content-Length", ""))
                except ValueError:
                    self._send_json(HTTPStatus.BAD_REQUEST, {"error": "Invalid Content-Length"})
                    return
                if not 1 <= length <= MAX_DATASET_SHARD_BYTES:
                    self._send_json(
                        HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                        {"error": "Initial model payload size is out of range"},
                    )
                    return
                payload = self.rfile.read(length)
                if len(payload) != length:
                    self._send_json(HTTPStatus.BAD_REQUEST, {"error": "Initial model was truncated"})
                    return
                try:
                    result = install_initial_model(config, dataset_name, payload)
                except FileNotFoundError as error:
                    self._send_json(HTTPStatus.NOT_FOUND, {"error": str(error)})
                except FileExistsError as error:
                    self._send_json(HTTPStatus.CONFLICT, {"error": str(error)})
                except ValueError as error:
                    self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(error)})
                except ImportError as error:
                    LOGGER.exception("Model-building dependency unavailable on %s", config.node_id)
                    self._send_json(
                        HTTPStatus.SERVICE_UNAVAILABLE,
                        {"error": f"Model-building dependency unavailable: {error}"},
                    )
                except OSError as error:
                    LOGGER.exception("Could not save initial model for %s", config.node_id)
                    self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(error)})
                else:
                    self._send_json(HTTPStatus.OK, result)

        def _nas_search(self) -> None:
            nonlocal nas_search_active
            dataset_name = "<unknown>"
            try:
                payload = self._read_json()
                dataset_name = payload["dataset_name"]
                if not isinstance(dataset_name, str) or not DATASET_NAME_PATTERN.fullmatch(
                    dataset_name
                ):
                    raise ValueError("NAS search requires a valid dataset_name")
                with trainer._lock:
                    if (
                        trainer._active_run is not None
                        or trainer._scheduled_run is not None
                        or trainer._reboot_requested
                    ):
                        raise RuntimeError("NAS search cannot run while training or reboot is active")
                with nas_control_lock:
                    if not nas_lock.acquire(blocking=False):
                        raise RuntimeError("Another NAS search is already active on this worker")
                    nas_cancel_event.clear()
                    nas_search_active = True
                try:
                    dataset_root = Path(config.dataset_root).resolve()
                    dataset_directory = dataset_root / dataset_name
                    dataset_path = dataset_directory.resolve()
                    if (
                        dataset_directory.is_symlink()
                        or dataset_path.parent != dataset_root
                    ):
                        raise ValueError("Selected dataset path is unsafe")
                    shard_path = dataset_path / "train.npz"
                    metadata_path = dataset_path / "metadata.json"
                    if not shard_path.is_file() or not metadata_path.is_file():
                        raise FileNotFoundError(
                            f"Preprocessed dataset shard is not installed: {dataset_name}"
                        )
                    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                    class_names = metadata.get("class_names")
                    shape = metadata.get("shape")
                    rows = metadata.get("rows")
                    if (
                        not isinstance(class_names, list)
                        or not class_names
                        or not isinstance(shape, list)
                        or len(shape) != 4
                        or any(not isinstance(dimension, int) or dimension < 1 for dimension in shape)
                        or not isinstance(rows, int)
                        or rows < 1
                    ):
                        raise ValueError("Installed dataset metadata is invalid for NAS search")
                    if shape[0] != rows:
                        raise ValueError(
                            "Installed dataset sample count does not match its shape"
                        )
                    dataset_samples = rows
                    dataset_image_shape = shape[1:]
                    smoke_test = payload.get("smoke_test", False)
                    if not isinstance(smoke_test, bool):
                        raise ValueError("NAS smoke_test must be a boolean")
                    architectures = payload.get("architectures")
                    epochs = payload.get("epochs", 1)
                    search_augmentation = payload.get("augmentation", "none")
                    if smoke_test:
                        architectures = [
                            smoke_test_architecture(
                                input_shape=tuple(dataset_image_shape),
                                num_classes=len(class_names),
                            )
                        ]
                        epochs = 1
                    if dataset_name == "cifar10":
                        images, labels, shard_classes = load_dataset_shard(shard_path)
                        if shard_classes != class_names:
                            raise ValueError("NAS shard classes do not match installed dataset metadata")
                        try:
                            result = search_dataset_architectures(
                                images,
                                labels,
                                shard_classes,
                                seed=payload.get("seed", 2026),
                                time_limit_seconds=payload.get(
                                    "time_limit_seconds", MAX_SEARCH_SECONDS
                                ),
                                node_id=config.node_id,
                                logger=_log_event,
                                architectures=architectures,
                                epochs=epochs,
                                cancel_event=nas_cancel_event,
                                reuse_tensorflow_runtime=True,
                                memory_constrained=dataset_name == "cifar10",
                                global_validation=load_global_validation(dataset_path),
                                augmentation=search_augmentation,
                            )
                        finally:
                            del images, labels, shard_classes
                            tensorflow = sys.modules.get("tensorflow")
                            if tensorflow is not None:
                                tensorflow.keras.backend.clear_session()
                            gc.collect()
                            _trim_process_heap()
                    else:
                        result = search_dataset_architectures_isolated(
                            str(shard_path),
                            seed=payload.get("seed", 2026),
                            time_limit_seconds=payload.get(
                                "time_limit_seconds", MAX_SEARCH_SECONDS
                            ),
                            node_id=config.node_id,
                            logger=_log_event,
                            architectures=architectures,
                            epochs=epochs,
                            cancel_event=nas_cancel_event,
                            augmentation=search_augmentation,
                        )
                    if result.get("dataset_classes") != class_names:
                        raise ValueError("NAS result classes do not match the dataset shard")
                    if smoke_test:
                        result["smoke_test"] = True
                        result["search_method"] = "single_candidate_smoke_test"
                    result["dataset_name"] = dataset_name
                    result["dataset_samples"] = dataset_samples
                    result["dataset_image_shape"] = dataset_image_shape
                finally:
                    with nas_control_lock:
                        nas_search_active = False
                        nas_lock.release()
            except FileNotFoundError as error:
                self._send_json(HTTPStatus.NOT_FOUND, {"error": str(error)})
                return
            except RuntimeError as error:
                self._send_json(HTTPStatus.CONFLICT, {"error": str(error)})
                return
            except (KeyError, TypeError, ValueError, TimeoutError, MemoryError) as error:
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(error)})
                return
            except OSError as error:
                LOGGER.exception("Dataset NAS failed to read %s on %s", dataset_name, config.node_id)
                self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(error)})
                return
            self._send_json(HTTPStatus.OK, result)

        def _start_run(self) -> None:
            try:
                payload = self._read_json()
                result = trainer.start_run(
                    run_id=payload["run_id"],
                    seed=payload["seed"],
                    topology=payload["topology"],
                )
            except RuntimeError as error:
                self._send_json(HTTPStatus.CONFLICT, {"error": str(error)})
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(error)})
            else:
                self._send_json(HTTPStatus.ACCEPTED, result)

        def _train_round(self) -> None:
            try:
                payload = self._read_json()
                result = trainer.run_round(
                    run_id=payload["run_id"],
                    round_index=payload["round"],
                    epochs=payload["epochs"],
                    seed=payload["seed"],
                    topology=payload["topology"],
                )
            except RuntimeError as error:
                self._send_json(HTTPStatus.CONFLICT, {"error": str(error)})
            except TimeoutError as error:
                self._send_json(HTTPStatus.GATEWAY_TIMEOUT, {"error": str(error)})
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(error)})
            else:
                self._send_json(HTTPStatus.OK, result)

        def _receive_peer_weights(self) -> None:
            sender = self.headers.get("X-Sender-Node", "")
            peer = trainer._peer_map.get(sender)
            if peer is None or not is_authorized(
                self.headers.get("Authorization"),
                peer.token,
            ):
                self._discard_request_body(MAX_WEIGHT_BYTES)
                self._unauthorized()
                return
            try:
                run_id = self.headers.get("X-Run-ID", "")
                round_index = int(self.headers.get("X-Round", ""))
                sample_count = int(self.headers.get("X-Sample-Count", ""))
                if not RUN_ID_PATTERN.fullmatch(run_id) or not 0 <= round_index < 100:
                    raise ValueError("Invalid run ID or round")
                length = int(self.headers.get("Content-Length", ""))
                if not 1 <= length <= MAX_WEIGHT_BYTES:
                    raise ValueError("Weight payload length is invalid")
                body = self.rfile.read(length)
                if len(body) != length:
                    raise ValueError("Weight payload was truncated")
                stored = trainer.accept_peer_weights(
                    run_id,
                    round_index,
                    sender,
                    sample_count,
                    body,
                    self.headers.get("X-Model-Signature"),
                )
            except (ValueError, OverflowError) as error:
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(error)})
            else:
                self._send_json(HTTPStatus.ACCEPTED, {"accepted": True, "stale": not stored})

        def _read_json(self) -> dict[str, Any]:
            length = int(self.headers.get("Content-Length", "0"))
            if not 1 <= length <= 16_384:
                raise ValueError("Request body must be between 1 and 16384 bytes")
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("Request body must be a JSON object")
            return payload

        def _unauthorized(self) -> None:
            self._send_json(
                HTTPStatus.UNAUTHORIZED,
                {"error": "unauthorized"},
                {"WWW-Authenticate": "Bearer"},
            )

        def _discard_request_body(self, maximum_bytes: int) -> None:
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                self.close_connection = True
                return
            if length < 0 or length > maximum_bytes:
                self.close_connection = True
                return
            if length and len(self.rfile.read(length)) != length:
                self.close_connection = True

        def _send_json(
            self,
            status: HTTPStatus,
            payload: dict[str, Any],
            headers: dict[str, str] | None = None,
        ) -> None:
            body = json.dumps(payload, separators=(",", ":"), allow_nan=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            for name, value in (headers or {}).items():
                self.send_header(name, value)
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format_string: str, *args: Any) -> None:
            LOGGER.info("%s - %s", self.client_address[0], format_string % args)

    return WorkerHandler


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config = WorkerConfig.from_env()
    log_path = Path(config.log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.handlers.RotatingFileHandler(
        log_path,
        maxBytes=10_000_000,
        backupCount=3,
        encoding="utf-8",
    )
    handler.setFormatter(logging.Formatter("%(message)s"))
    LOGGER.addHandler(handler)
    LOGGER.setLevel(logging.INFO)
    trainer = WorkerTrainer(config)
    server = ThreadingHTTPServer((config.bind_host, config.port), make_handler(config, trainer))
    server.daemon_threads = True
    LOGGER.info(
        "Starting authenticated training service for %s on %s:%s with %s samples",
        config.node_id,
        config.bind_host,
        config.port,
        trainer.labels.size,
    )
    try:
        server.serve_forever(poll_interval=0.5)
    except KeyboardInterrupt:
        LOGGER.info("Stopping worker training service")
    finally:
        server.shutdown()
        server.server_close()


if __name__ == "__main__":
    main()
