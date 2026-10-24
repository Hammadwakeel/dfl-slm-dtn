"""Authenticated worker API for local CIFAR-10 training and direct peer averaging."""

from __future__ import annotations

import hmac
import io
import base64
import json
import logging
import logging.handlers
import os
import re
import threading
import time
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, urlparse
from urllib.request import Request, urlopen

import numpy as np

from common.auth import is_authorized
from common.cifar_model import build_cifar10_model, compile_cifar10_model, model_size_bytes
from common.nas import FIXED_ARCHITECTURE, architecture_spec
from worker.nas_search import MAX_SEARCH_SECONDS, search_cifar10_architectures_isolated

LOGGER = logging.getLogger("dfl_slm.worker_training")
RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
MAX_WEIGHT_BYTES = 16 * 1024 * 1024


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
    bind_host: str = "0.0.0.0"
    peer_timeout_seconds: float = 60.0

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
        if node_id not in {"worker1", "worker2", "worker3"}:
            raise ValueError("DFL_NODE_ID must identify worker1, worker2, or worker3")
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
        if len(peers) != 2 or len({peer.node_id for peer in peers}) != 2:
            raise ValueError("Each worker must be configured with the other two worker peers")
        if any(len(peer.token) < 32 or not peer.base_url.startswith("http://") for peer in peers):
            raise ValueError("Peer URLs and tokens must be configured")
        port = int(os.environ.get("DFL_TRAINING_PORT", "5004"))
        timeout = float(os.environ.get("DFL_PEER_TIMEOUT_SECONDS", "60"))
        if not 1 <= port <= 65535 or timeout <= 0:
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
            bind_host=os.environ.get("DFL_BIND_HOST", "0.0.0.0"),
            peer_timeout_seconds=timeout,
        )


def ring_neighbours(node_id: str, worker_ids: tuple[str, ...]) -> tuple[str, ...]:
    if node_id not in worker_ids or len(worker_ids) < 3:
        raise ValueError("Ring topology requires a configured member of a 3-node worker ring")
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
    if node_id not in worker_ids or len(worker_ids) != 3:
        raise ValueError("Supported topologies require exactly three configured workers")
    ordered = tuple(sorted(worker_ids))
    index = ordered.index(node_id)
    if topology in {"ring", "directed_ring"}:
        return (ordered[(index + 1) % len(ordered)],)
    if topology == "bidirectional_ring":
        return tuple(sorted({ordered[(index - 1) % 3], ordered[(index + 1) % 3]}))
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


def topology_senders(
    node_id: str,
    worker_ids: tuple[str, ...],
    topology: str,
) -> tuple[str, ...]:
    if topology in {"ring", "directed_ring"}:
        ordered = tuple(sorted(worker_ids))
        return (ordered[(ordered.index(node_id) - 1) % len(ordered)],)
    return topology_neighbours(node_id, worker_ids, topology)


def weighted_average_weights(
    weighted_weights: list[tuple[list[np.ndarray], int]],
) -> list[np.ndarray]:
    if not weighted_weights or any(sample_count < 1 for _, sample_count in weighted_weights):
        raise ValueError("Weighted averaging requires non-empty models and positive sample counts")
    reference = weighted_weights[0][0]
    if not reference:
        raise ValueError("Model weights must not be empty")
    for weights, _ in weighted_weights:
        if len(weights) != len(reference):
            raise ValueError("Peer model weight counts do not match")
        for current, expected in zip(weights, reference):
            if current.shape != expected.shape or not np.isfinite(current).all():
                raise ValueError("Peer model weights have invalid shape or values")
    total_samples = sum(sample_count for _, sample_count in weighted_weights)
    return [
        (
            sum(
                weights[index].astype(np.float64) * sample_count
                for weights, sample_count in weighted_weights
            )
            / total_samples
        ).astype(reference[index].dtype)
        for index in range(len(reference))
    ]


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
    if any(array.dtype.kind not in "fiu" or array.nbytes > 4 * 1024 * 1024 for array in weights):
        raise ValueError("Model weight payload contains invalid arrays")
    return weights


def _memory_available_kb() -> int:
    with open("/proc/meminfo", encoding="ascii") as stream:
        for line in stream:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1])
    raise RuntimeError("MemAvailable is missing from /proc/meminfo")


def _resident_memory_mb() -> float:
    with open("/proc/self/status", encoding="ascii") as stream:
        for line in stream:
            if line.startswith("VmRSS:"):
                return round(int(line.split()[1]) / 1024, 1)
    raise RuntimeError("VmRSS is missing from /proc/self/status")


def _peak_process_rss_mb() -> float:
    import resource

    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)


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
        self._completed_rounds: dict[str, int] = {}
        self._round_weights: dict[tuple[str, int], list[np.ndarray]] = {}
        self._results: dict[str, list[dict[str, Any]]] = {}
        self._run_errors: dict[str, str] = {}
        self._active_run: str | None = None
        self._active_round: int | None = None
        self._scheduled_run: str | None = None
        self._run_thread: threading.Thread | None = None
        self._cancel = threading.Event()
        self._peer_map = {peer.node_id: peer for peer in config.peers}
        self.dataset_split = "stratified"
        self.images, self.labels = self._load_shard(config.data_path)
        self._rounds = 10
        self._epochs = 5
        self._topology = "ring"
        self._architecture = architecture_spec(FIXED_ARCHITECTURE)

    def status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "node_id": self.config.node_id,
                "sample_count": int(self.labels.size),
                "dataset_split": self.dataset_split,
                "active_run_id": self._active_run,
                "scheduled_run_id": self._scheduled_run,
                "active_round": self._active_round,
                "training_config": {
                    "rounds": self._rounds,
                    "epochs": self._epochs,
                    "topology": self._topology,
                    "architecture": self._architecture,
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
        return {"node_id": self.config.node_id, "runs": selected, "errors": errors}

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
    ) -> dict[str, Any]:
        if not 1 <= rounds <= 100 or not 1 <= epochs <= 50:
            raise ValueError("Rounds must be 1-100 and local epochs 1-50")
        if topology not in {
            "ring",
            "directed_ring",
            "bidirectional_ring",
            "mesh",
            "chain",
            "star",
        }:
            raise ValueError("Unsupported training topology")
        if dataset_split not in {"stratified", "dirichlet_a03", "dirichlet_a05"}:
            raise ValueError("Unsupported CIFAR-10 dataset split")
        selected_architecture = architecture_spec(architecture)
        shard_path = (
            self.config.data_path
            if dataset_split == "stratified"
            else self.config.shard_paths.get(dataset_split)
        )
        if not shard_path:
            raise ValueError(f"Worker shard for {dataset_split} is not installed")
        images, labels = self._load_shard(shard_path)
        with self._lock:
            if self._scheduled_run is not None or self._active_run is not None:
                raise RuntimeError("Cannot change worker configuration while training is active")
            self.images = images
            self.labels = labels
            self.dataset_split = dataset_split
            self._rounds = rounds
            self._epochs = epochs
            self._topology = topology
            self._architecture = selected_architecture
        return {
            "rounds": rounds,
            "epochs": epochs,
            "topology": topology,
            "dataset_split": dataset_split,
            "sample_count": int(labels.size),
            "architecture": selected_architecture,
        }

    def start_run(self, run_id: str, seed: int, topology: str) -> dict[str, Any]:
        if not RUN_ID_PATTERN.fullmatch(run_id) or not 0 <= seed <= 2**32 - 1:
            raise ValueError("run_id or seed is invalid")
        if topology not in {
            "ring",
            "directed_ring",
            "bidirectional_ring",
            "mesh",
            "chain",
            "star",
        }:
            raise ValueError("Unsupported training topology")
        with self._lock:
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

    def weights(self, run_id: str, round_index: int) -> tuple[bytes, int]:
        with self._lock:
            weights = self._round_weights.get((run_id, round_index))
            if weights is None:
                raise KeyError("Requested model is not available")
        return serialize_weights(weights), int(self.labels.size)

    def accept_peer_weights(
        self,
        run_id: str,
        round_index: int,
        sender: str,
        sample_count: int,
        payload: bytes,
    ) -> None:
        if sender not in self._peer_map or sample_count < 1:
            raise ValueError("Unknown peer or invalid sample count")
        weights = deserialize_weights(payload)
        key = (run_id, round_index)
        with self._condition:
            senders = self._incoming.setdefault(key, {})
            if sender in senders:
                raise ValueError("Duplicate weights received from a peer for this round")
            senders[sender] = (weights, sample_count, len(payload))
            self._condition.notify_all()
        _log_event(
            "peer_weights_received",
            run_id=run_id,
            round=round_index,
            peer_id=sender,
            sample_count=sample_count,
            bytes_received=len(payload),
        )

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
        neighbors = topology_neighbours(self.config.node_id, worker_ids, topology)
        expected_senders = topology_senders(self.config.node_id, worker_ids, topology)
        with self._lock:
            if self._active_run is not None:
                raise RuntimeError(f"Training is already active for {self._active_run}")
            current_round = self._completed_rounds.get(run_id, -1)
            if round_index != current_round + 1:
                raise ValueError(f"Expected round {current_round + 1}, received {round_index}")
            if run_id not in self._models:
                model = build_cifar10_model(seed, self._architecture)
                compile_cifar10_model(model)
                self._models[run_id] = model
                self._results[run_id] = []
            model = self._models[run_id]
            self._active_run = run_id
            self._active_round = round_index
        started = time.perf_counter()
        before_available_kb = _memory_available_kb()
        rss_before = _resident_memory_mb()
        train_started = time.perf_counter()
        try:
            import tensorflow as tf

            cancel_event = self._cancel

            class StopOnRequest(tf.keras.callbacks.Callback):
                def on_train_batch_end(self, batch: int, logs: Any = None) -> None:
                    if cancel_event.is_set():
                        self.model.stop_training = True

            history = model.fit(
                self.images,
                self.labels,
                epochs=epochs,
                batch_size=128,
                shuffle=True,
                verbose=0,
                callbacks=[StopOnRequest()],
            )
            train_seconds = time.perf_counter() - train_started
            if self._cancel.is_set():
                raise RuntimeError("Training was cancelled before peer exchange")
            local_weights = [np.asarray(item) for item in model.get_weights()]
            payload = serialize_weights(local_weights)
            sent_bytes = 0
            exchange_started = time.perf_counter()
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
                        "X-Sample-Count": str(self.labels.size),
                    },
                    method="POST",
                )
                try:
                    with urlopen(request, timeout=self.config.peer_timeout_seconds) as response:
                        if response.status != HTTPStatus.ACCEPTED:
                            raise RuntimeError(f"Peer {peer_id} returned HTTP {response.status}")
                    sent_bytes += len(payload)
                    _log_event(
                        "peer_weights_sent",
                        run_id=run_id,
                        round=round_index,
                        peer_id=peer_id,
                        bytes_sent=len(payload),
                        transfer_seconds=round(time.perf_counter() - transfer_started, 3),
                    )
                except (HTTPError, URLError, TimeoutError, OSError) as error:
                    raise RuntimeError(f"Peer exchange with {peer_id} failed: {error}") from error
            deadline = time.monotonic() + self.config.peer_timeout_seconds
            with self._condition:
                while not set(expected_senders).issubset(
                    self._incoming.get((run_id, round_index), {})
                ):
                    if self._cancel.is_set():
                        raise RuntimeError("Training was cancelled during peer exchange")
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        missing = sorted(
                            set(expected_senders)
                            - set(self._incoming.get((run_id, round_index), {}))
                        )
                        raise TimeoutError(f"Timed out waiting for peer weights from {missing}")
                    self._condition.wait(timeout=min(remaining, 1.0))
                incoming = self._incoming.pop((run_id, round_index))
            weighted_models = [(local_weights, int(self.labels.size))]
            bytes_received = 0
            for peer_id in expected_senders:
                peer_weights, sample_count, received_bytes = incoming[peer_id]
                weighted_models.append((peer_weights, sample_count))
                bytes_received += received_bytes
            averaged = weighted_average_weights(weighted_models)
            model.set_weights(averaged)
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
                "architecture": self._architecture,
                "dataset_split": self.dataset_split,
                "local_epochs": epochs,
                "sample_count": int(self.labels.size),
                "training_loss": history_values.get("loss"),
                "training_accuracy": history_values.get("accuracy"),
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
                "peer_messages_received": len(expected_senders),
                "exchange_seconds": round(exchange_seconds, 3),
                "neighbours": list(neighbors),
                "weight_bytes": model_size_bytes(model),
            }
            with self._lock:
                self._completed_rounds[run_id] = round_index
                self._round_weights[(run_id, round_index)] = [
                    np.asarray(item).copy() for item in averaged
                ]
                self._results[run_id].append(result)
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
                self._condition.notify_all()


def make_handler(config: WorkerConfig, trainer: WorkerTrainer) -> type[BaseHTTPRequestHandler]:
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
            elif parsed.path == "/v1/results":
                run_id = parse_qs(parsed.query).get("run_id", [None])[0]
                self._send_json(HTTPStatus.OK, trainer.results(run_id))
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
                self._unauthorized()
                return
            if parsed.path == "/v1/config":
                self._configure()
            elif parsed.path == "/v1/nas/search":
                self._nas_search()
            elif parsed.path == "/v1/start":
                self._start_run()
            elif parsed.path == "/v1/train-round":
                self._train_round()
            elif parsed.path == "/v1/cancel":
                self._send_json(HTTPStatus.OK, trainer.cancel())
            else:
                self._send_json(HTTPStatus.NOT_FOUND, {"error": "not_found"})

        def _configure(self) -> None:
            try:
                payload = self._read_json()
                result = trainer.configure(
                    rounds=payload["rounds"],
                    epochs=payload["epochs"],
                    topology=payload["topology"],
                    dataset_split=payload.get("dataset_split", "stratified"),
                    architecture=payload.get("architecture", FIXED_ARCHITECTURE),
                )
            except RuntimeError as error:
                self._send_json(HTTPStatus.CONFLICT, {"error": str(error)})
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(error)})
            else:
                self._send_json(HTTPStatus.OK, result)

        def _nas_search(self) -> None:
            try:
                payload = self._read_json()
                if trainer.dataset_split != "stratified":
                    raise ValueError("NAS search requires the stratified local worker shard")
                with trainer._lock:
                    if trainer._active_run is not None or trainer._scheduled_run is not None:
                        raise RuntimeError("NAS search cannot run while worker training is active")
                result = search_cifar10_architectures_isolated(
                    trainer.config.data_path,
                    seed=payload.get("seed", 2026),
                    time_limit_seconds=payload.get("time_limit_seconds", MAX_SEARCH_SECONDS),
                    node_id=config.node_id,
                    logger=_log_event,
                )
            except RuntimeError as error:
                self._send_json(HTTPStatus.CONFLICT, {"error": str(error)})
            except (KeyError, TypeError, ValueError, TimeoutError, MemoryError) as error:
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(error)})
            else:
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
                trainer.accept_peer_weights(run_id, round_index, sender, sample_count, body)
            except (ValueError, OverflowError) as error:
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(error)})
            else:
                self._send_json(HTTPStatus.ACCEPTED, {"accepted": True})

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
