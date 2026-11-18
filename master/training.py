"""Master-side orchestration for direct-peer distributed dataset training."""

from __future__ import annotations

import atexit
import base64
import json
import logging
import os
import re
import sqlite3
import subprocess
import statistics
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from common.dataset_nas import (
    CIFAR_INITIAL_POPULATION_SIZE,
    FINAL_EPOCHS,
    FINAL_POPULATION_SIZE,
    INITIAL_EPOCHS,
    INITIAL_POPULATION_SIZE,
    aggregate_worker_candidates,
    build_dataset_architecture,
    initial_population,
    mutate_population,
    pareto_front,
    select_best_architecture,
    select_evolutionary_parents,
)
from common.hpo import (
    HPO_EPOCHS,
    MEMORY_CONSTRAINED_HPO_EPOCHS,
    hpo_candidates,
    select_best_hyperparameters,
)
from topology_opt.reward import RewardConfig, calculate_run_reward
from topology_opt.selector import select_candidate
from common.model_design import (
    MAX_MODEL_WEIGHT_BYTES,
    ModelDesignError,
    validate_model_design,
    validate_normalization,
)
from common.nas import (
    ARCHITECTURE_SPACE,
    FIXED_ARCHITECTURE,
    architecture_spec,
    select_architecture,
)

LOGGER = logging.getLogger("dfl_slm.training")
TOPOLOGIES = (
    "directed_ring",
    "bidirectional_ring",
    "mesh",
    "chain",
    "star",
)
# "learned": a short mesh probe measures peer-model losses and waits on the real
# workers, APG picks each worker's senders, then training uses that graph.
TRAINING_TOPOLOGIES = (*TOPOLOGIES, "learned")
TOPOLOGY_PROBE_ROUNDS = 2
AGGREGATION_RULES = ("fedavg", "uniform", "metropolis", "trimmed_mean")
SYNC_MODES = ("sync", "semi_sync")
DATASET_SPLITS = {
    0.1: "dirichlet_a01",
    0.3: "dirichlet_a03",
    0.5: "dirichlet_a05",
    0.7: "dirichlet_a07",
}
OPTIMIZATION_ALPHAS = (0.1, 0.3, 0.5)
# Datasets the topology sweep supports. MNIST keeps the original fixed CNN so earlier
# sweeps stay comparable. b1_c8_k3 is too small for CIFAR-10 (about 35% accuracy),
# which would compress every topology into the same low-accuracy band, so CIFAR-10
# sweeps use one fixed 3-block CNN of about 62k parameters for every candidate.
OPTIMIZATION_ARCHITECTURES: dict[str, str | dict[str, Any]] = {
    "mnist": FIXED_ARCHITECTURE,
    "cifar10": build_dataset_architecture(
        input_shape=(32, 32, 3),
        num_classes=10,
        channels=(32, 64, 64),
        kernel_size=3,
        block_type="conv2d",
        head_units=64,
        dropout=0.2,
    ),
}


class TrainingError(RuntimeError):
    pass


class TrainingCancelled(TrainingError):
    pass


@dataclass(frozen=True)
class TrainingWorker:
    node_id: str
    base_url: str
    token: str


@dataclass(frozen=True)
class TrainingConfig:
    workers: tuple[TrainingWorker, ...]
    evaluator_python: str
    evaluator_script: str
    test_data_path: str
    train_data_path: str
    evaluator_log_path: str
    results_path: str
    rounds: int = 10
    epochs: int = 5
    topology: str = "directed_ring"
    peer_timeout_seconds: float = 60.0
    round_timeout_seconds: float = 7200.0

    @classmethod
    def from_env(cls) -> "TrainingConfig":
        encoded_workers = os.environ.get("DFL_WORKERS_JSON_B64", "")
        workers_value = (
            base64.b64decode(encoded_workers, validate=True)
            if encoded_workers
            else os.environ.get("DFL_WORKERS_JSON", "[]")
        )
        peers = json.loads(workers_value)
        workers = tuple(
            TrainingWorker(
                node_id=str(item["node_id"]),
                base_url=str(item["base_url"]).rstrip("/"),
                token=str(item["token"]),
            )
            for item in peers
        )
        worker_ids = [worker.node_id for worker in workers]
        expected_worker_ids = [
            f"worker{index}" for index in range(1, len(worker_ids) + 1)
        ]
        if len(worker_ids) < 3 or worker_ids != expected_worker_ids:
            raise ValueError(
                "DFL_WORKERS_JSON must configure at least three sequential workers in order"
            )
        if any(
            len(worker.token) < 32 or not worker.base_url.startswith("http://")
            for worker in workers
        ):
            raise ValueError("Worker training URLs and tokens must be configured")
        rounds = int(os.environ.get("DFL_TRAINING_ROUNDS", "10"))
        epochs = int(os.environ.get("DFL_LOCAL_EPOCHS", "5"))
        topology = os.environ.get("DFL_TOPOLOGY", "directed_ring")
        peer_timeout = float(os.environ.get("DFL_TRAINING_PEER_TIMEOUT_SECONDS", "60"))
        round_timeout = float(os.environ.get("DFL_TRAINING_ROUND_TIMEOUT_SECONDS", "7200"))
        if not 1 <= rounds <= 100 or not 1 <= epochs <= 50:
            raise ValueError("Training rounds or epochs are out of range")
        if (
            topology not in {"ring", *TRAINING_TOPOLOGIES}
            or peer_timeout <= 0
            or round_timeout <= 0
        ):
            raise ValueError("Training topology or timeout is invalid")
        return cls(
            workers=workers,
            evaluator_python=os.environ.get("DFL_EVALUATOR_PYTHON", "/usr/bin/python3"),
            evaluator_script=os.environ.get(
                "DFL_EVALUATOR_SCRIPT",
                "/opt/dfl-slm-dtn/master/cifar_evaluator.py",
            ),
            test_data_path=os.environ.get(
                "DFL_TEST_DATA_PATH",
                "/opt/dfl-slm-dtn/data/cifar10_test.npz",
            ),
            train_data_path=os.environ.get(
                "DFL_BASELINE_TRAIN_DATA_PATH",
                "/opt/dfl-slm-dtn/data/cifar10_train.npz",
            ),
            evaluator_log_path=os.environ.get(
                "DFL_EVALUATOR_LOG_PATH",
                "/var/log/dfl-slm-master/cifar-evaluator.log",
            ),
            results_path=os.environ.get(
                "DFL_TRAINING_RESULTS_PATH",
                "/var/lib/dfl-slm-master/experiments.sqlite3",
            ),
            rounds=rounds,
            epochs=epochs,
            topology=topology,
            peer_timeout_seconds=peer_timeout,
            round_timeout_seconds=round_timeout,
        )


class EvaluatorClient:
    def __init__(self, config: TrainingConfig) -> None:
        self.config = config
        self._lock = threading.Lock()
        self._process: subprocess.Popen[str] | None = None
        self._stderr_file: Any | None = None
        atexit.register(self.close)

    def _ensure_started(self) -> subprocess.Popen[str]:
        if self._process is not None and self._process.poll() is None:
            return self._process
        log_path = Path(self.config.evaluator_log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self._stderr_file = log_path.open("a", encoding="utf-8")
        self._process = subprocess.Popen(
            [
                self.config.evaluator_python,
                "-u",
                self.config.evaluator_script,
                self.config.test_data_path,
                self.config.train_data_path,
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self._stderr_file,
            text=True,
            encoding="utf-8",
            bufsize=1,
        )
        if self._process.stdin is None or self._process.stdout is None:
            raise TrainingError("Could not open evaluator process pipes")
        ready_line = self._process.stdout.readline()
        if not ready_line:
            return_code = self._process.poll()
            raise TrainingError(
                f"CIFAR evaluator exited during startup (exit code {return_code}); "
                f"see {log_path}"
            )
        ready = json.loads(ready_line)
        if ready.get("ready") is not True or ready.get("test_samples") != 10_000:
            raise TrainingError(f"CIFAR evaluator did not confirm the held-out test set: {ready}")
        LOGGER.info("CIFAR evaluator ready with %d held-out samples", ready["test_samples"])
        return self._process

    def _request(self, payload: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            process = self._ensure_started()
            if process.stdin is None or process.stdout is None:
                raise TrainingError("CIFAR evaluator pipes are unavailable")
            process.stdin.write(json.dumps(payload, separators=(",", ":")) + "\n")
            process.stdin.flush()
            line = process.stdout.readline()
            if not line:
                raise TrainingError(
                    f"CIFAR evaluator exited while processing a request (exit code {process.poll()})"
                )
            response = json.loads(line)
            if "error" in response:
                raise TrainingError(f"CIFAR evaluator failed: {response['error']}")
            return response

    def evaluate(
        self,
        models: dict[str, bytes],
        architecture: str | dict[str, Any] = FIXED_ARCHITECTURE,
    ) -> dict[str, Any]:
        return self._request(
            {
                "type": "evaluate",
                "architecture": architecture,
                "models": {
                    node_id: base64.b64encode(weights).decode("ascii")
                    for node_id, weights in models.items()
                },
            }
        )["results"]

    def train_baseline(
        self,
        seed: int,
        rounds: int,
        epochs: int,
        architecture: str | dict[str, Any] = FIXED_ARCHITECTURE,
    ) -> list[dict[str, Any]]:
        return self._request(
            {
                "type": "baseline",
                "seed": seed,
                "rounds": rounds,
                "epochs": epochs,
                "architecture": architecture,
            }
        )["baseline_rounds"]

    def close(self) -> None:
        process = self._process
        if process is not None and process.poll() is None:
            if process.stdin is not None:
                process.stdin.close()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.terminate()
                process.wait(timeout=5)
        if self._stderr_file is not None:
            self._stderr_file.close()
        self._process = None


class TrainingCoordinator:
    def __init__(
        self,
        config: TrainingConfig,
        *,
        evaluator: EvaluatorClient | None = None,
    ) -> None:
        self.config = config
        self.evaluator = evaluator or EvaluatorClient(config)
        self.reward_config = RewardConfig.load(
            os.environ.get(
                "DFL_TOPOLOGY_REWARD_CONFIG",
                str(Path(__file__).resolve().parents[1] / "topology_opt" / "reward_config.json"),
            )
        )
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._active_group_id: str | None = None
        self._started_worker_runs: set[str] = set()
        self._settings = {
            "rounds": config.rounds,
            "epochs": config.epochs,
            "topology": config.topology,
            "dataset_split": "stratified",
            "architecture": architecture_spec(FIXED_ARCHITECTURE),
            "aggregation": "fedavg",
            "sync_mode": "sync",
            "sync_deadline_seconds": None,
        }
        self._groups: list[dict[str, Any]] = self._load_groups()
        if self._recover_interrupted_groups():
            self._persist_locked()

    def configure_workers(self, workers: tuple[TrainingWorker, ...]) -> dict[str, Any]:
        worker_ids = [worker.node_id for worker in workers]
        expected_ids = [f"worker{index}" for index in range(1, len(workers) + 1)]
        if (
            len(workers) < 3
            or worker_ids != expected_ids
            or len(set(worker_ids)) != len(worker_ids)
            or any(len(worker.token) < 32 or not worker.base_url.startswith("http://") for worker in workers)
        ):
            raise ValueError("Workers must have sequential IDs, valid HTTP URLs, and API tokens")
        with self._lock:
            if self._active_group_id is not None:
                raise TrainingError("Cannot change workers while a training run is active")
            previous_workers = self.config.workers

        updated_workers: list[TrainingWorker] = []
        failures: list[str] = []
        with ThreadPoolExecutor(max_workers=len(workers)) as pool:
            futures = {
                pool.submit(
                    self._post,
                    worker,
                    "/v1/peers",
                    {
                        "peers": [
                            {
                                "node_id": peer.node_id,
                                "base_url": peer.base_url,
                                "token": peer.token,
                            }
                            for peer in workers
                            if peer.node_id != worker.node_id
                        ]
                    },
                ): worker
                for worker in workers
            }
            for future in as_completed(futures):
                worker = futures[future]
                try:
                    future.result()
                    updated_workers.append(worker)
                except Exception as error:
                    failures.append(f"{worker.node_id}: {error}")
        if failures:
            old_by_id = {worker.node_id: worker for worker in previous_workers}
            rollback_errors: list[str] = []
            for worker in updated_workers:
                old_worker = old_by_id.get(worker.node_id)
                if old_worker is None:
                    continue
                try:
                    self._post(
                        worker,
                        "/v1/peers",
                        {
                            "peers": [
                                {
                                    "node_id": peer.node_id,
                                    "base_url": peer.base_url,
                                    "token": peer.token,
                                }
                                for peer in previous_workers
                                if peer.node_id != old_worker.node_id
                            ]
                        },
                    )
                except Exception as error:
                    rollback_errors.append(f"{worker.node_id}: {error}")
            detail = "; ".join(failures)
            if rollback_errors:
                detail += "; rollback failed: " + "; ".join(rollback_errors)
            raise TrainingError(f"Could not synchronize worker peer lists: {detail}")

        with self._lock:
            if self._active_group_id is not None:
                raise TrainingError("A training run started while worker peers were synchronizing")
            self.config = replace(self.config, workers=workers)
        return {"configured_workers": worker_ids}

    def _recover_interrupted_groups(self) -> bool:
        recovered = False
        for group in self._groups:
            if group.get("status") != "running":
                continue
            message = (
                "Training was interrupted because the master coordinator restarted "
                "before this group completed."
            )
            group["status"] = "failed"
            group["completed_at"] = time.time()
            group["error"] = message
            for run in group.get("runs", []):
                if run.get("status") == "running":
                    run["status"] = "failed"
                    run["error"] = message
                run.pop("progress", None)
            recovered = True
        return recovered

    def _results_paths(self) -> tuple[Path, Path]:
        configured_path = Path(self.config.results_path)
        if configured_path.suffix.lower() == ".json":
            return configured_path.with_suffix(".sqlite3"), configured_path
        return configured_path, configured_path.with_suffix(".json")

    def configure(
        self,
        *,
        rounds: int | None = None,
        epochs: int | None = None,
        topology: str | None = None,
        dataset_split: str | None = None,
        aggregation: str | None = None,
        sync_mode: str | None = None,
        sync_deadline_seconds: int | None = None,
    ) -> dict[str, Any]:
        with self._lock:
            if self._active_group_id is not None:
                raise TrainingError("Cannot change training settings while a run is active")
            updated = dict(self._settings)
            if rounds is not None:
                if not 1 <= rounds <= 100:
                    raise ValueError("rounds must be between 1 and 100")
                updated["rounds"] = rounds
            if epochs is not None:
                if not 1 <= epochs <= 50:
                    raise ValueError("epochs must be between 1 and 50")
                updated["epochs"] = epochs
            if topology is not None:
                if topology == "ring":
                    topology = "directed_ring"
                if topology not in TRAINING_TOPOLOGIES:
                    raise ValueError(f"topology must be one of {', '.join(TRAINING_TOPOLOGIES)}")
                updated["topology"] = topology
            if dataset_split is not None:
                if dataset_split not in {"stratified", *DATASET_SPLITS.values()}:
                    raise ValueError("Unsupported CIFAR-10 training split")
                updated["dataset_split"] = dataset_split
            if aggregation is not None:
                if aggregation not in AGGREGATION_RULES:
                    raise ValueError(f"aggregation must be one of {', '.join(AGGREGATION_RULES)}")
                updated["aggregation"] = aggregation
            if sync_mode is not None:
                if sync_mode not in SYNC_MODES:
                    raise ValueError(f"sync_mode must be one of {', '.join(SYNC_MODES)}")
                updated["sync_mode"] = sync_mode
            if sync_deadline_seconds is not None:
                if not 5 <= sync_deadline_seconds <= 7200:
                    raise ValueError("sync_deadline_seconds must be between 5 and 7200")
                updated["sync_deadline_seconds"] = sync_deadline_seconds
            self._settings = updated
        return dict(updated)

    def start_training(
        self,
        *,
        rounds: int | None = None,
        epochs: int | None = None,
        seeds: tuple[int, ...] | None = None,
        dataset_split: str | None = None,
        dataset_name: str | None = None,
        architecture: str | dict[str, Any] = FIXED_ARCHITECTURE,
        aggregation: str | None = None,
        sync_mode: str | None = None,
        sync_deadline_seconds: int | None = None,
    ) -> dict[str, Any]:
        if dataset_name is not None and not re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}", dataset_name
        ):
            raise ValueError("A valid dataset_name is required")
        selected_seeds = (
            (2026,) if dataset_name is not None else (17, 42, 2026)
        ) if seeds is None else seeds
        if not 1 <= len(selected_seeds) <= 3 or any(
            seed < 0 for seed in selected_seeds
        ):
            raise ValueError("Provide between one and three non-negative training seeds")
        with self._lock:
            if self._active_group_id is not None:
                raise TrainingError(f"Training group {self._active_group_id} is already active")
            if isinstance(architecture, dict) and architecture.get("model_type") in {
                "custom",
                "pretrained_deit",
            }:
                if dataset_name is None:
                    raise ValueError("Custom MNIST architectures require a dataset_name")
                if architecture.get("dataset_name") != dataset_name:
                    raise ValueError("Model architecture does not match the selected dataset")
                raw_shape = architecture.get("input_shape")
                classes = architecture.get("classes")
                if (
                    not isinstance(raw_shape, list)
                    or len(raw_shape) != 3
                    or any(isinstance(value, bool) or not isinstance(value, int) for value in raw_shape)
                    or not isinstance(classes, list)
                    or len(classes) < 2
                    or any(not isinstance(value, str) or not value for value in classes)
                    or len(set(classes)) != len(classes)
                    or architecture.get("num_classes") != len(classes)
                ):
                    raise ValueError("Installed model dataset metadata is invalid")
                try:
                    normalized_design, parameter_count = validate_model_design(
                        architecture,
                        tuple(raw_shape),
                        len(classes),
                    )
                except ModelDesignError as error:
                    raise ValueError(str(error)) from error
                if architecture.get("parameter_count") != parameter_count:
                    raise ValueError("Installed model parameter count does not match its architecture")
                if architecture.get("model_type") == "pretrained_deit":
                    try:
                        normalized_normalization = validate_normalization(
                            architecture.get("normalization"),
                            raw_shape[2],
                        )
                    except ModelDesignError as error:
                        raise ValueError(str(error)) from error
                    if architecture.get("normalization") != normalized_normalization:
                        raise ValueError("Installed model normalization metadata is invalid")
                selected_architecture = {
                    **architecture,
                    **normalized_design,
                    "dataset_name": dataset_name,
                    "input_shape": raw_shape,
                    "classes": classes,
                    "num_classes": len(classes),
                    "parameter_count": parameter_count,
                }
            else:
                selected_architecture = architecture_spec(architecture)
            if self._settings["topology"] == "learned" and dataset_name is None:
                raise ValueError("The learned topology measures validation loss, so it needs a dataset run")
            settings = self.configure(
                rounds=rounds,
                epochs=epochs,
                dataset_split=dataset_split,
                aggregation=aggregation,
                sync_mode=sync_mode,
                sync_deadline_seconds=sync_deadline_seconds,
            )
            self._settings["architecture"] = selected_architecture
            settings["architecture"] = selected_architecture
            group_id = f"{settings['topology']}-{int(time.time())}-{uuid.uuid4().hex[:6]}"
            group = {
                "group_id": group_id,
                "kind": "decentralized",
                "topology": settings["topology"],
                "rounds": settings["rounds"],
                "epochs": settings["epochs"],
                "dataset_split": settings["dataset_split"],
                "architecture": selected_architecture,
                "aggregation": settings["aggregation"],
                "sync_mode": settings["sync_mode"],
                "sync_deadline_seconds": settings["sync_deadline_seconds"],
                "nas_enabled": False,
                "seeds": list(selected_seeds),
                "status": "running",
                "started_at": time.time(),
                "completed_at": None,
                "error": None,
                "runs": [],
            }
            if dataset_name is not None:
                group["dataset_name"] = dataset_name
            self._groups.append(group)
            self._active_group_id = group_id
            self._stop.clear()
            self._thread = threading.Thread(
                target=self._run_group,
                args=(group_id,),
                name=f"training-group-{group_id}",
                daemon=True,
            )
            self._thread.start()
            self._persist_locked()
        return {
            "accepted": True,
            "group_id": group_id,
            **settings,
            "seeds": list(selected_seeds),
            "dataset_name": dataset_name,
        }

    def start_nas_experiment(
        self,
        *,
        rounds: int = 10,
        epochs: int = 5,
        seeds: tuple[int, ...] = (17, 42, 2026),
        search_time_limit_seconds: int = 300,
    ) -> dict[str, Any]:
        if (
            not 1 <= len(seeds) <= 3
            or len(set(seeds)) != len(seeds)
            or any(seed < 0 for seed in seeds)
        ):
            raise ValueError("NAS requires one to three non-negative seeds")
        if not 1 <= rounds <= 100 or not 1 <= epochs <= 50:
            raise ValueError("NAS rounds must be 1-100 and epochs must be 1-50")
        if not 30 <= search_time_limit_seconds <= 300:
            raise ValueError("NAS search time limit must be between 30 and 300 seconds")
        with self._lock:
            if self._active_group_id is not None:
                raise TrainingError(f"Training group {self._active_group_id} is already active")
            group_id = f"nas-{int(time.time())}-{uuid.uuid4().hex[:6]}"
            group = {
                "group_id": group_id,
                "kind": "neural_architecture_search",
                "topology": self._settings["topology"],
                "dataset_split": "stratified",
                "rounds": rounds,
                "epochs": epochs,
                "seeds": list(seeds),
                "search_time_limit_seconds": search_time_limit_seconds,
                "architecture_space": list(ARCHITECTURE_SPACE),
                "fixed_architecture": architecture_spec(FIXED_ARCHITECTURE),
                "status": "running",
                "started_at": time.time(),
                "completed_at": None,
                "error": None,
                "runs": [],
                "worker_search": [],
            }
            self._groups.append(group)
            self._active_group_id = group_id
            self._stop.clear()
            self._thread = threading.Thread(
                target=self._run_nas_group,
                args=(group_id,),
                name=f"nas-group-{group_id}",
                daemon=True,
            )
            self._thread.start()
            self._persist_locked()
        return {
            "accepted": True,
            "group_id": group_id,
            "rounds": rounds,
            "epochs": epochs,
            "seeds": list(seeds),
            "topology": group["topology"],
            "dataset_split": "stratified",
            "candidate_count": len(ARCHITECTURE_SPACE),
            "search_time_limit_seconds": search_time_limit_seconds,
            "max_candidate_parameters": 12_000,
        }

    def start_dataset_nas(
        self,
        *,
        dataset_name: str,
        search_time_limit_seconds: int = 600,
        seed: int = 2026,
        smoke_test: bool = False,
    ) -> dict[str, Any]:
        if not isinstance(dataset_name, str) or not re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}", dataset_name
        ):
            raise ValueError("A valid dataset_name is required for NAS")
        if not 90 <= search_time_limit_seconds <= 600:
            raise ValueError("Dataset NAS search time limit must be between 90 and 600 seconds")
        if not isinstance(smoke_test, bool):
            raise ValueError("NAS smoke_test must be a boolean")
        if not 0 <= seed <= 2**32 - 1:
            raise ValueError("NAS seed is out of range")
        with self._lock:
            if self._active_group_id is not None:
                raise TrainingError(f"Training group {self._active_group_id} is already active")
            group_id = f"dataset-nas-{int(time.time())}-{uuid.uuid4().hex[:6]}"
            group = {
                "group_id": group_id,
                "kind": "dataset_architecture_search",
                "dataset_name": dataset_name,
                "topology": self._settings["topology"],
                "seed": seed,
                "search_time_limit_seconds": search_time_limit_seconds,
                "smoke_test": smoke_test,
                "search_method": (
                    "single_candidate_smoke_test"
                    if smoke_test
                    else "pareto_evolutionary_multifidelity"
                ),
                "nas_generations": [],
                "status": "running",
                "started_at": time.time(),
                "completed_at": None,
                "error": None,
                "worker_search": [],
                "worker_search_progress": {},
                "runs": [],
            }
            self._groups.append(group)
            self._active_group_id = group_id
            self._stop.clear()
            self._thread = threading.Thread(
                target=self._run_dataset_nas_group,
                args=(group_id,),
                name=f"dataset-nas-group-{group_id}",
                daemon=True,
            )
            self._thread.start()
            self._persist_locked()
        return {
            "accepted": True,
            "group_id": group_id,
            "dataset_name": dataset_name,
            "search_time_limit_seconds": search_time_limit_seconds,
            "seed": seed,
            "smoke_test": smoke_test,
        }

    def start_hpo(
        self,
        *,
        dataset_name: str,
        architecture: dict[str, Any],
        search_time_limit_seconds: int = 300,
        seed: int = 2026,
    ) -> dict[str, Any]:
        """Tune training settings of one architecture on every worker's own data."""
        if not isinstance(dataset_name, str) or not re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}", dataset_name
        ):
            raise ValueError("A valid dataset_name is required for HPO")
        if not isinstance(architecture, dict) or architecture.get("model_type") != "custom":
            raise ValueError("HPO requires a custom architecture")
        input_shape = architecture.get("input_shape")
        classes = architecture.get("classes")
        if (
            not isinstance(input_shape, list)
            or len(input_shape) != 3
            or not isinstance(classes, list)
            or len(classes) < 2
        ):
            raise ValueError("HPO architecture must carry its dataset input_shape and classes")
        # One worker search request is capped at MAX_SEARCH_SECONDS (300 s) on the worker.
        if not 90 <= search_time_limit_seconds <= 300:
            raise ValueError("HPO search time limit must be between 90 and 300 seconds")
        if not 0 <= seed <= 2**32 - 1:
            raise ValueError("HPO seed is out of range")
        candidates = hpo_candidates(
            architecture,
            input_shape=tuple(int(value) for value in input_shape),
            num_classes=len(classes),
            memory_constrained=dataset_name == "cifar10",
        )
        with self._lock:
            if self._active_group_id is not None:
                raise TrainingError(f"Training group {self._active_group_id} is already active")
            group_id = f"hpo-{int(time.time())}-{uuid.uuid4().hex[:6]}"
            group = {
                "group_id": group_id,
                "kind": "hyperparameter_search",
                "dataset_name": dataset_name,
                "topology": self._settings["topology"],
                "seed": seed,
                "search_time_limit_seconds": search_time_limit_seconds,
                "search_method": "grid_learning_rate_batch_size",
                "base_architecture": architecture,
                "hpo_candidates": candidates,
                "status": "running",
                "started_at": time.time(),
                "completed_at": None,
                "error": None,
                "worker_search": [],
                "runs": [],
            }
            self._groups.append(group)
            self._active_group_id = group_id
            self._stop.clear()
            self._thread = threading.Thread(
                target=self._run_hpo_group,
                args=(group_id,),
                name=f"hpo-group-{group_id}",
                daemon=True,
            )
            self._thread.start()
            self._persist_locked()
        return {
            "accepted": True,
            "group_id": group_id,
            "dataset_name": dataset_name,
            "candidate_count": len(candidates),
            "search_time_limit_seconds": search_time_limit_seconds,
        }

    def _run_hpo_group(self, group_id: str) -> None:
        group = self._get_group(group_id)
        try:
            base = group["base_architecture"]
            candidates = group["hpo_candidates"]
            epochs = MEMORY_CONSTRAINED_HPO_EPOCHS if group["dataset_name"] == "cifar10" else HPO_EPOCHS
            payload = {
                "dataset_name": group["dataset_name"],
                "seed": group["seed"],
                "time_limit_seconds": group["search_time_limit_seconds"],
                "epochs": epochs,
                "architectures": candidates,
            }

            def record(result: dict[str, Any]) -> None:
                with self._lock:
                    rows = {row["node_id"]: row for row in group["worker_search"]}
                    rows[result["node_id"]] = {
                        "node_id": result["node_id"],
                        "elapsed_seconds": result.get("elapsed_seconds"),
                        "candidate_count": len(result.get("candidates", [])),
                    }
                    group["worker_search"] = sorted(rows.values(), key=lambda row: row["node_id"])
                    self._persist_locked()

            searches = self._parallel_workers(
                lambda worker: self._post(
                    worker,
                    "/v1/nas/search",
                    payload,
                    timeout=group["search_time_limit_seconds"] + 60,
                ),
                on_result=record,
            )
            if self._stop.is_set():
                raise TrainingCancelled("HPO was cancelled")
            if any(result.get("fidelity_epochs") != epochs for result in searches):
                raise TrainingError("Workers evaluated HPO candidates at a different fidelity")
            measured = aggregate_worker_candidates(searches, worker_count=len(self.config.workers))
            if {row["architecture"]["id"] for row in measured} != {item["id"] for item in candidates}:
                raise TrainingError("HPO workers did not evaluate the same candidate list")
            best = select_best_hyperparameters(measured)
            selected_architecture = {
                **best["architecture"],
                "dataset_name": group["dataset_name"],
                "input_shape": list(base["input_shape"]),
                "classes": list(base["classes"]),
                "num_classes": len(base["classes"]),
            }
            if isinstance(base.get("normalization"), dict):
                selected_architecture["normalization"] = base["normalization"]
            with self._lock:
                group["hpo_results"] = measured
                group["selected_architecture"] = selected_architecture
                group["selected_hyperparameters"] = best["architecture"]["training"]
                self._persist_locked()
            self._update_group(
                group_id,
                status="completed",
                completed_at=time.time(),
                selection={
                    "method": group["search_method"],
                    "fidelity_epochs": epochs,
                    "selected_candidate": {**best, "architecture": selected_architecture},
                },
            )
        except TrainingCancelled as error:
            self._update_group(group_id, status="stopped", completed_at=time.time(), error=str(error))
        except Exception as error:
            LOGGER.exception("HPO group %s failed", group_id)
            self._update_group(
                group_id,
                status="failed",
                completed_at=time.time(),
                error=f"{type(error).__name__}: {error}",
            )
        finally:
            with self._lock:
                if self._active_group_id == group_id:
                    self._active_group_id = None
                    self._persist_locked()

    def start_optimization(
        self,
        *,
        rounds: int = 10,
        epochs: int = 5,
        seeds: tuple[int, ...] = (17, 42, 2026),
        alphas: tuple[float, ...] = (0.3,),
        dataset_name: str = "mnist",
    ) -> dict[str, Any]:
        if not 1 <= len(seeds) <= 3 or any(seed < 0 for seed in seeds):
            raise ValueError("Provide between one and three non-negative seeds")
        # Workers train on whichever shard the backend installed last, so one sweep
        # covers exactly one alpha: re-shard between sweeps, never inside one.
        if len(alphas) != 1 or alphas[0] not in OPTIMIZATION_ALPHAS:
            raise ValueError(
                "Topology optimization takes exactly one Dirichlet alpha from "
                f"{OPTIMIZATION_ALPHAS}; preprocess the shards for that alpha first"
            )
        if dataset_name not in OPTIMIZATION_ARCHITECTURES:
            raise ValueError(
                f"Topology optimization supports {sorted(OPTIMIZATION_ARCHITECTURES)}"
            )
        architecture = OPTIMIZATION_ARCHITECTURES[dataset_name]
        if not 1 <= rounds <= 100 or not 1 <= epochs <= 50:
            raise ValueError("rounds must be 1-100 and epochs must be 1-50")
        with self._lock:
            if self._active_group_id is not None:
                raise TrainingError(f"Training group {self._active_group_id} is already active")
            group_id = f"optimize-{int(time.time())}-{uuid.uuid4().hex[:6]}"
            group = {
                "group_id": group_id,
                "kind": "topology_optimization",
                "dataset_name": dataset_name,
                "architecture": architecture,
                "reward_metric_source": "validation_accuracy",
                "topology": "candidate_sweep",
                "rounds": rounds,
                "epochs": epochs,
                "seeds": list(seeds),
                "alphas": list(alphas),
                "candidates": [
                    {"topology": topology, "alpha": alpha}
                    for alpha in alphas
                    for topology in TOPOLOGIES
                ],
                "reward_config": self.reward_config.as_dict(),
                "status": "running",
                "started_at": time.time(),
                "completed_at": None,
                "error": None,
                "runs": [],
            }
            self._groups.append(group)
            self._active_group_id = group_id
            self._stop.clear()
            self._thread = threading.Thread(
                target=self._run_optimization_group,
                args=(group_id,),
                name=f"topology-optimization-{group_id}",
                daemon=True,
            )
            self._thread.start()
            self._persist_locked()
        return {
            "accepted": True,
            "group_id": group_id,
            "rounds": rounds,
            "epochs": epochs,
            "seeds": list(seeds),
            "alphas": list(alphas),
            "candidate_count": len(alphas) * len(TOPOLOGIES),
            "dataset_name": dataset_name,
            "architecture_id": architecture_spec(architecture)["id"]
            if isinstance(architecture, str)
            else architecture["id"],
            "reward_config": self.reward_config.as_dict(),
        }

    def start_baseline(
        self,
        *,
        rounds: int | None = None,
        epochs: int | None = None,
        seeds: tuple[int, ...] = (17, 42, 2026),
    ) -> dict[str, Any]:
        if len(seeds) != 3 or any(seed < 0 for seed in seeds):
            raise ValueError("Exactly three non-negative baseline seeds are required")
        with self._lock:
            if self._active_group_id is not None:
                raise TrainingError(f"Training group {self._active_group_id} is already active")
            settings = self.configure(rounds=rounds, epochs=epochs)
            group_id = f"baseline-{int(time.time())}-{uuid.uuid4().hex[:6]}"
            group = {
                "group_id": group_id,
                "kind": "centralized_baseline",
                "topology": "centralized",
                "rounds": settings["rounds"],
                "epochs": settings["epochs"],
                "seeds": list(seeds),
                "status": "running",
                "started_at": time.time(),
                "completed_at": None,
                "error": None,
                "runs": [],
            }
            self._groups.append(group)
            self._active_group_id = group_id
            self._stop.clear()
            self._thread = threading.Thread(
                target=self._run_baseline_group,
                args=(group_id,),
                name=f"baseline-group-{group_id}",
                daemon=True,
            )
            self._thread.start()
            self._persist_locked()
        return {"accepted": True, "group_id": group_id, **settings, "seeds": list(seeds)}

    def stop_training(self) -> dict[str, Any]:
        with self._lock:
            group_id = self._active_group_id
            if group_id is None:
                return {"accepted": False, "message": "No training group is active."}
            self._stop.set()
        cancel_errors = self._cancel_workers()
        return {
            "accepted": True,
            "group_id": group_id,
            "cancel_errors": cancel_errors,
        }

    def results(self) -> dict[str, Any]:
        with self._lock:
            worker_ids = [worker.node_id for worker in self.config.workers]
            ring = worker_ids + worker_ids[:1]
            bidirectional_ring = [
                [worker_ids[index], worker_ids[(index + 1) % len(worker_ids)]]
                for index in range(len(worker_ids))
            ]
            mesh = [
                [worker_ids[first], worker_ids[second]]
                for first in range(len(worker_ids))
                for second in range(first + 1, len(worker_ids))
            ]
            chain = [
                [worker_ids[index], worker_ids[index + 1]]
                for index in range(len(worker_ids) - 1)
            ]
            hub = worker_ids[0]
            return {
                "configuration": dict(self._settings),
                "active_group_id": self._active_group_id,
                "topologies": {
                    "ring": ring,
                    "directed_ring": ring,
                    "bidirectional_ring": bidirectional_ring,
                    "mesh": mesh,
                    "chain": chain,
                    "star": [[hub, worker_id] for worker_id in worker_ids[1:]],
                },
                "groups": json.loads(json.dumps(self._groups[-20:])),
            }

    def delete_group(self, group_id: str) -> dict[str, Any]:
        with self._lock:
            group = next(
                (item for item in self._groups if item["group_id"] == group_id),
                None,
            )
            if group is None:
                raise TrainingError(f"Experiment group {group_id} was not found")
            if group_id == self._active_group_id or group.get("status") == "running":
                raise TrainingError("An active experiment group cannot be deleted")
            database_path, _ = self._results_paths()
            with sqlite3.connect(database_path, timeout=30) as connection:
                connection.execute(
                    "DELETE FROM experiment_groups WHERE group_id = ?",
                    (group_id,),
                )
            self._groups.remove(group)
            return {"deleted": True, "group_id": group_id}

    def _get_group(self, group_id: str) -> dict[str, Any]:
        with self._lock:
            for group in self._groups:
                if group["group_id"] == group_id:
                    return group
        raise TrainingError(f"Training group {group_id} was not found")

    def _update_group(self, group_id: str, **updates: Any) -> None:
        with self._lock:
            group = next(group for group in self._groups if group["group_id"] == group_id)
            group.update(updates)
            self._persist_locked()

    def _append_run_row(self, group_id: str, run: dict[str, Any], row: dict[str, Any]) -> None:
        with self._lock:
            group = next(item for item in self._groups if item["group_id"] == group_id)
            current_run = next(item for item in group["runs"] if item["run_id"] == run["run_id"])
            current_run["rounds"].append(row)
            current_run.pop("progress", None)
            self._persist_locked()

    def _run_group(self, group_id: str) -> None:
        group = self._get_group(group_id)
        try:
            if group["topology"] == "learned" and not group.get("learned_senders"):
                self._learn_topology(group_id, group)
            for seed in group["seeds"]:
                if self._stop.is_set():
                    raise TrainingCancelled("Training was cancelled")
                run_id = f"{group['topology']}-seed-{seed}-{uuid.uuid4().hex[:8]}"
                run = {
                    "run_id": run_id,
                    "seed": seed,
                    "topology": group["topology"],
                    "status": "running",
                    "rounds": [],
                }
                with self._lock:
                    self._get_group(group_id)["runs"].append(run)
                    self._persist_locked()
                self._run_distributed_seed(group_id, group, run)
                run["status"] = "completed"
                with self._lock:
                    self._persist_locked()
            self._update_group(group_id, status="completed", completed_at=time.time())
        except TrainingCancelled as error:
            self._cancel_workers()
            with self._lock:
                for run in self._get_group(group_id)["runs"]:
                    if run["status"] == "running":
                        run.update(status="stopped", error=str(error))
            self._update_group(
                group_id,
                status="stopped",
                completed_at=time.time(),
                error=str(error),
            )
        except Exception as error:
            self._cancel_workers()
            LOGGER.exception("Distributed training group %s failed", group_id)
            with self._lock:
                for run in self._get_group(group_id)["runs"]:
                    if run["status"] == "running":
                        run.update(status="failed", error=f"{type(error).__name__}: {error}")
            self._update_group(
                group_id,
                status="failed",
                completed_at=time.time(),
                error=f"{type(error).__name__}: {error}",
            )
        finally:
            with self._lock:
                if self._active_group_id == group_id:
                    self._active_group_id = None
                self._persist_locked()

    def _learn_topology(self, group_id: str, group: dict[str, Any]) -> None:
        """Probe on a mesh, then let APG choose every worker's senders."""
        probe_group = {
            **group,
            "topology": "mesh",
            "rounds": TOPOLOGY_PROBE_ROUNDS,
            "epochs": 1,
            "measure_peers": True,
        }
        probe = {
            "run_id": f"probe-mesh-{uuid.uuid4().hex[:8]}",
            "seed": group["seeds"][0],
            "topology": "mesh",
            "phase": "topology_probe",
            "status": "running",
            "rounds": [],
        }
        with self._lock:
            group["runs"].append(probe)
            self._persist_locked()
        self._run_distributed_seed(group_id, probe_group, probe)
        probe["status"] = "completed"
        # numpy-based; imported here so the coordinator starts without it.
        from topology_opt.learning import learn_senders

        worker_ids = [worker.node_id for worker in self.config.workers]
        senders, diagnostics = learn_senders(worker_ids, probe["rounds"][-1]["workers"])
        with self._lock:
            group["learned_senders"] = senders
            group["topology_learning"] = {**diagnostics, "probe_run_id": probe["run_id"], "senders": senders}
            self._persist_locked()

    def _run_distributed_seed(
        self,
        group_id: str,
        group: dict[str, Any],
        run: dict[str, Any],
    ) -> None:
        try:
            self._run_distributed_seed_impl(group_id, group, run)
        except Exception:
            with self._lock:
                workers_started = run["run_id"] in self._started_worker_runs
                self._started_worker_runs.discard(run["run_id"])
            if workers_started:
                cancel_errors = self._cancel_workers()
                if cancel_errors:
                    LOGGER.error(
                        "Could not cancel all workers after run %s failed: %s",
                        run["run_id"],
                        cancel_errors,
                    )
            raise
        finally:
            with self._lock:
                self._started_worker_runs.discard(run["run_id"])
            self._cleanup_worker_run(group_id, run)

    def _run_distributed_seed_impl(
        self,
        group_id: str,
        group: dict[str, Any],
        run: dict[str, Any],
    ) -> None:
        configuration = {
            "rounds": group["rounds"],
            "epochs": group["epochs"],
            "topology": group["topology"],
            "dataset_split": group.get("dataset_split", "stratified"),
            "architecture": group.get("architecture", FIXED_ARCHITECTURE),
            "aggregation": group.get("aggregation", "fedavg"),
            "sync_mode": group.get("sync_mode", "sync"),
        }
        if group.get("sync_deadline_seconds") is not None:
            configuration["sync_deadline_seconds"] = group["sync_deadline_seconds"]
        if group.get("dataset_name") is not None:
            configuration["dataset_name"] = group["dataset_name"]
        if group.get("learned_senders"):
            configuration["learned_senders"] = group["learned_senders"]
        if group.get("measure_peers"):
            configuration["measure_peers"] = True
        self._parallel_workers(
            lambda worker: self._post(worker, "/v1/config", configuration)
        )
        started: set[str] = set()
        try:
            def start(worker: TrainingWorker) -> tuple[str, dict[str, Any]]:
                return worker.node_id, self._post(
                    worker,
                    "/v1/start",
                    {
                        "run_id": run["run_id"],
                        "seed": run["seed"],
                        "topology": run["topology"],
                        "architecture": group.get("architecture", FIXED_ARCHITECTURE),
                    },
                )

            for node_id, _ in self._parallel_workers(start):
                started.add(node_id)
            with self._lock:
                self._started_worker_runs.add(run["run_id"])
        except Exception:
            self._cancel_workers(started or None)
            raise

        for round_index in range(group["rounds"]):
            self._wait_for_worker_rounds(run["run_id"], round_index, group_id)
            summaries = self._parallel_workers(
                lambda worker: (
                    worker.node_id,
                    self._get_run_results(worker, run["run_id"]),
                )
            )
            local_rows = {
                node_id: payload["runs"][run["run_id"]][round_index]
                for node_id, payload in summaries
            }
            scores: dict[str, Any] = {}
            if group.get("dataset_name") is None:
                # Only the legacy CIFAR path evaluates held-out accuracy on the master;
                # dataset runs report worker-local validation metrics, so their weights
                # are not downloaded every round.
                binary_weights = self._parallel_workers(
                    lambda worker: (
                        worker.node_id,
                        self._get_bytes(
                            worker,
                            "/v1/weights?" + urlencode(
                                {"run_id": run["run_id"], "round": round_index}
                            ),
                        ),
                    )
                )
                scores = self.evaluator.evaluate(
                    dict(binary_weights),
                    group.get("architecture", FIXED_ARCHITECTURE),
                )
            worker_metrics = {}
            for node_id in local_rows:
                metrics = {
                    **local_rows[node_id],
                    **scores.get(node_id, {}),
                }
                for legacy_name, validation_name in (
                    ("accuracy", "validation_accuracy"),
                    ("macro_f1", "validation_macro_f1"),
                ):
                    if legacy_name not in metrics and validation_name in metrics:
                        metrics[legacy_name] = metrics[validation_name]
                worker_metrics[node_id] = metrics
            row = {
                "round": round_index + 1,
                "topology": group["topology"],
                "completed_at": time.time(),
                "workers": worker_metrics,
            }
            self._append_run_row(group_id, run, row)

    def _cleanup_worker_run(
        self,
        group_id: str,
        run: dict[str, Any],
    ) -> None:
        run_id = run["run_id"]

        def cleanup(worker: TrainingWorker) -> tuple[str, dict[str, Any]]:
            try:
                self._delete(
                    worker,
                    f"/v1/runs/{run_id}",
                    timeout=self.config.peer_timeout_seconds,
                )
            except TrainingError as error:
                LOGGER.error(
                    "Could not clean run %s memory on %s: %s",
                    run_id,
                    worker.node_id,
                    error,
                )
                return worker.node_id, {"status": "failed", "error": str(error)}
            return worker.node_id, {"status": "cleaned"}

        cleanup_results = self._parallel_workers(cleanup)
        cleanup_status = {
            worker_id: result
            for worker_id, result in cleanup_results
        }
        run["worker_cleanup"] = {
            "status": (
                "completed"
                if all(result["status"] == "cleaned" for result in cleanup_status.values())
                and len(cleanup_status) == len(self.config.workers)
                else "incomplete"
            ),
            "workers": cleanup_status,
        }
        with self._lock:
            self._get_group(group_id)
            self._persist_locked()

    def _wait_for_worker_rounds(self, run_id: str, round_index: int, group_id: str) -> None:
        deadline = time.monotonic() + self.config.round_timeout_seconds
        while time.monotonic() < deadline:
            if self._stop.is_set():
                raise TrainingCancelled("Training was cancelled")
            snapshots = self._parallel_workers(
                lambda worker: (
                    worker.node_id,
                    self._get_run_results(worker, run_id),
                )
            )
            worker_progress: dict[str, dict[str, Any]] = {}
            round_complete = True
            for node_id, payload in snapshots:
                if run_id in payload.get("errors", {}):
                    raise TrainingError(
                        f"{node_id} failed in run {run_id}: {payload['errors'][run_id]}"
                    )
                rows = payload.get("runs", {}).get(run_id, [])
                if len(rows) <= round_index:
                    round_complete = False
                progress = payload.get("progress")
                if (
                    isinstance(progress, dict)
                    and progress.get("run_id") == run_id
                    and progress.get("round") == round_index + 1
                ):
                    worker_progress[node_id] = {
                        "epoch": progress.get("epoch"),
                        "epochs": progress.get("epochs"),
                        "phase": progress.get("phase"),
                    }
            self._update_run_progress(
                group_id,
                run_id,
                {"round": round_index + 1, "workers": worker_progress},
            )
            if round_complete:
                return
            time.sleep(1)
        raise TrainingError(
            f"Timed out waiting for all workers to complete round {round_index + 1} "
            f"for group {group_id}"
        )

    def _update_run_progress(
        self,
        group_id: str,
        run_id: str,
        progress: dict[str, Any],
    ) -> None:
        with self._lock:
            group = self._get_group(group_id)
            run = next(
                (item for item in group["runs"] if item["run_id"] == run_id),
                None,
            )
            if run is None or run.get("progress") == progress:
                return
            run["progress"] = progress
            self._persist_locked()

    def _summarize_topology_candidates(
        self,
        group: dict[str, Any],
        candidate_specs: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        # Score with the criteria the group was started with, so history never shifts.
        reward_config = (
            RewardConfig.from_dict(group["reward_config"])
            if isinstance(group.get("reward_config"), dict)
            else self.reward_config
        )
        summaries = []
        for candidate in candidate_specs:
            matching_runs = [
                run
                for run in group["runs"]
                if run["topology"] == candidate["topology"]
                and run["alpha"] == candidate["alpha"]
                and run["status"] == "completed"
            ]
            per_run = [
                calculate_run_reward(run["rounds"], reward_config)
                for run in matching_runs
            ]
            summary: dict[str, Any] = {
                "topology": candidate["topology"],
                "alpha": candidate["alpha"],
                "completed_runs": len(per_run),
            }
            if per_run:
                summary.update(
                    {
                        "mean_reward": round(
                            sum(value["reward"] for value in per_run) / len(per_run),
                            6,
                        ),
                        "mean_final_accuracy": round(
                            sum(value["metrics"]["final_accuracy"] for value in per_run)
                            / len(per_run),
                            6,
                        ),
                        "mean_accuracy_gain_available": all(
                            value["metrics"]["accuracy_gain_available"]
                            for value in per_run
                        ),
                        "mean_components": {
                            name: round(
                                sum(value["components"][name] for value in per_run)
                                / len(per_run),
                                6,
                            )
                            for name in reward_config.weights
                        },
                        "std_reward": round(
                            statistics.pstdev(value["reward"] for value in per_run), 6
                        ),
                        "per_run_rewards": per_run,
                    }
                )
                if reward_config.version == 2:
                    summary.update(
                        {
                            "reward_version": 2,
                            "mean_min_worker_accuracy": round(
                                sum(value["metrics"]["min_worker_accuracy"] for value in per_run)
                                / len(per_run),
                                6,
                            ),
                            "passes_quality_gate": all(
                                value["metrics"]["passes_quality_gate"] for value in per_run
                            ),
                            "quality_source": per_run[-1]["metrics"]["quality_source"],
                        }
                    )
            summaries.append(summary)
        return summaries

    def _run_optimization_group(self, group_id: str) -> None:
        group = self._get_group(group_id)
        candidate_specs = [dict(candidate) for candidate in group["candidates"]]
        try:
            for candidate in candidate_specs:
                for seed in group["seeds"]:
                    if self._stop.is_set():
                        raise TrainingCancelled("Topology optimization was cancelled")
                    alpha = candidate["alpha"]
                    run_id = (
                        f"{candidate['topology']}-a{str(alpha).replace('.', '')}-"
                        f"seed-{seed}-{uuid.uuid4().hex[:8]}"
                    )
                    run = {
                        "run_id": run_id,
                        "seed": seed,
                        "topology": candidate["topology"],
                        "dataset_split": DATASET_SPLITS[alpha],
                        "alpha": alpha,
                        "status": "running",
                        "rounds": [],
                    }
                    with self._lock:
                        self._get_group(group_id)["runs"].append(run)
                        self._persist_locked()
                    candidate_group = {
                        **group,
                        "topology": candidate["topology"],
                        "dataset_split": DATASET_SPLITS[alpha],
                    }
                    self._run_distributed_seed(group_id, candidate_group, run)
                    run["status"] = "completed"
                    with self._lock:
                        current_group = self._get_group(group_id)
                        current_group["candidates"] = self._summarize_topology_candidates(
                            current_group,
                            candidate_specs,
                        )
                        self._persist_locked()

            candidate_summaries = self._summarize_topology_candidates(
                group,
                candidate_specs,
            )
            measured_candidates = [
                candidate
                for candidate in candidate_summaries
                if candidate["completed_runs"] > 0
            ]
            if not measured_candidates:
                raise TrainingError("No topology candidates completed successfully")
            selection = select_candidate(measured_candidates)
            chosen = selection["selected_candidate"]
            with self._lock:
                self._settings["topology"] = chosen["topology"]
                self._settings["dataset_split"] = DATASET_SPLITS[chosen["alpha"]]
            self._update_group(
                group_id,
                candidates=candidate_summaries,
                selection=selection,
                selected_configuration={
                    "topology": chosen["topology"],
                    "alpha": chosen["alpha"],
                    "dataset_split": DATASET_SPLITS[chosen["alpha"]],
                },
                status="completed",
                completed_at=time.time(),
            )
        except TrainingCancelled as error:
            self._cancel_workers()
            self._update_group(
                group_id,
                status="stopped",
                completed_at=time.time(),
                error=str(error),
            )
        except Exception as error:
            self._cancel_workers()
            LOGGER.exception("Topology optimization group %s failed", group_id)
            self._update_group(
                group_id,
                status="failed",
                completed_at=time.time(),
                error=f"{type(error).__name__}: {error}",
            )
        finally:
            with self._lock:
                if self._active_group_id == group_id:
                    self._active_group_id = None
                self._persist_locked()

    def _run_nas_group(self, group_id: str) -> None:
        group = self._get_group(group_id)
        try:
            base_configuration = {
                "rounds": group["rounds"],
                "epochs": group["epochs"],
                "topology": group["topology"],
                "dataset_split": "stratified",
                "architecture": FIXED_ARCHITECTURE,
            }
            self._parallel_workers(
                lambda worker: self._post(worker, "/v1/config", base_configuration)
            )
            searches = self._parallel_workers(
                lambda worker: self._post(
                    worker,
                    "/v1/nas/search",
                    {
                        "seed": 2026,
                        "time_limit_seconds": group["search_time_limit_seconds"],
                    },
                    timeout=group["search_time_limit_seconds"] + 30,
                )
            )
            searches = sorted(searches, key=lambda row: row["node_id"])
            candidate_rows = [
                result
                for worker_result in searches
                for result in worker_result["candidates"]
            ]
            selection = select_architecture(candidate_rows)
            group["worker_search"] = searches
            group["nas_candidates"] = selection["candidate_means"]
            group["selected_architecture"] = selection["architecture"]
            group["search_elapsed_seconds"] = max(
                float(result["elapsed_seconds"]) for result in searches
            )
            group["search_peak_rss_mb"] = max(
                float(result["peak_rss_mb"]) for result in searches
            )
            with self._lock:
                self._persist_locked()

            variants = (
                ("fixed", architecture_spec(FIXED_ARCHITECTURE)),
                ("nas", selection["architecture"]),
            )
            for variant, architecture in variants:
                run_group = {
                    **group,
                    "architecture": architecture,
                    "variant": variant,
                }
                for seed in group["seeds"]:
                    if self._stop.is_set():
                        raise TrainingCancelled("NAS experiment was cancelled")
                    run = {
                        "run_id": f"nas-{variant}-seed-{seed}-{uuid.uuid4().hex[:8]}",
                        "seed": seed,
                        "topology": group["topology"],
                        "dataset_split": "stratified",
                        "architecture": architecture,
                        "variant": variant,
                        "status": "running",
                        "rounds": [],
                    }
                    with self._lock:
                        self._get_group(group_id)["runs"].append(run)
                        self._persist_locked()
                    self._run_distributed_seed(group_id, run_group, run)
                    run["status"] = "completed"
                    with self._lock:
                        self._persist_locked()

            for seed in group["seeds"]:
                if self._stop.is_set():
                    raise TrainingCancelled("NAS experiment was cancelled")
                run = {
                    "run_id": f"nas-centralized-seed-{seed}-{uuid.uuid4().hex[:8]}",
                    "seed": seed,
                    "topology": "centralized",
                    "dataset_split": "stratified",
                    "architecture": architecture_spec(FIXED_ARCHITECTURE),
                    "variant": "centralized_fixed",
                    "status": "running",
                    "rounds": [],
                }
                rows = self.evaluator.train_baseline(
                    seed,
                    group["rounds"],
                    group["epochs"],
                    architecture=FIXED_ARCHITECTURE,
                )
                run["rounds"] = [
                    {
                        **row,
                        "topology": "centralized",
                        "workers": {"dfl-master": row},
                    }
                    for row in rows
                ]
                run["status"] = "completed"
                with self._lock:
                    self._get_group(group_id)["runs"].append(run)
                    self._persist_locked()

            comparisons = {
                variant: self._summarize_nas_runs(
                    [
                        run
                        for run in group["runs"]
                        if run["variant"] == variant and run["status"] == "completed"
                    ],
                    search_seconds=(
                        group["search_elapsed_seconds"] if variant == "nas" else 0.0
                    ),
                    search_peak_rss_mb=(
                        group["search_peak_rss_mb"] if variant == "nas" else 0.0
                    ),
                )
                for variant in ("fixed", "nas", "centralized_fixed")
            }
            group["comparison"] = comparisons
            group["nas_beats_fixed"] = (
                comparisons["nas"]["mean_accuracy"]
                > comparisons["fixed"]["mean_accuracy"]
            )
            self._update_group(
                group_id,
                status="completed",
                completed_at=time.time(),
            )
        except TrainingCancelled as error:
            self._cancel_workers()
            self._update_group(
                group_id,
                status="stopped",
                completed_at=time.time(),
                error=str(error),
            )
        except Exception as error:
            self._cancel_workers()
            LOGGER.exception("NAS experiment group %s failed", group_id)
            self._update_group(
                group_id,
                status="failed",
                completed_at=time.time(),
                error=f"{type(error).__name__}: {error}",
            )
        finally:
            with self._lock:
                if self._active_group_id == group_id:
                    self._active_group_id = None
                self._persist_locked()

    def _run_dataset_nas_group(self, group_id: str) -> None:
        group = self._get_group(group_id)
        try:
            if self._stop.is_set():
                raise TrainingCancelled("Dataset NAS was cancelled")

            started_at = time.time()
            search_started = time.monotonic()
            search_deadline = search_started + group["search_time_limit_seconds"]
            smoke_test = group.get("smoke_test") is True
            initial_stage_limit = (
                min(300, group["search_time_limit_seconds"])
                if smoke_test
                else min(300, max(30, int(group["search_time_limit_seconds"] * 0.5)))
            )
            input_shape: tuple[int, int, int] | None = None
            class_names: list[str] | None = None
            population: list[dict[str, Any]] | None = None
            final_candidates: list[dict[str, Any]] = []
            final_searches: list[dict[str, Any]] = []
            fidelities = (
                (1,)
                if smoke_test
                # CIFAR-10 also evolves: a one-epoch screen alone always favoured the
                # smallest, fastest-learning model.
                else (INITIAL_EPOCHS, FINAL_EPOCHS)
            )
            with self._lock:
                group["worker_search"] = []
                group["worker_search_progress"] = {
                    worker.node_id: {
                        "status": "searching",
                        "started_at": started_at,
                        "generation": 1,
                        "fidelity_epochs": fidelities[0],
                    }
                    for worker in self.config.workers
                }
                self._persist_locked()

            for generation, fidelity_epochs in enumerate(fidelities, start=1):
                if self._stop.is_set():
                    raise TrainingCancelled("Dataset NAS was cancelled")
                remaining_seconds = int(search_deadline - time.monotonic())
                if remaining_seconds < 30:
                    raise TimeoutError(
                        "Dataset NAS needs at least 30 seconds remaining for the next "
                        "worker evaluation stage"
                    )
                if generation == 2:
                    if input_shape is None or class_names is None or population is None:
                        raise TrainingError("NAS evolution is missing first-generation results")
                    parents = [
                        candidate["architecture"]
                        for candidate in select_evolutionary_parents(
                            final_candidates,
                            count=min(2, INITIAL_POPULATION_SIZE),
                        )
                    ]
                    population = mutate_population(
                        parents,
                        seed=group["seed"] + generation,
                        input_shape=input_shape,
                        num_classes=len(class_names),
                        size=FINAL_POPULATION_SIZE - 2,
                        memory_constrained=group["dataset_name"] == "cifar10",
                    )

                with self._lock:
                    group["worker_search_progress"] = {
                        worker.node_id: {
                            "status": "searching",
                            "started_at": started_at,
                            "generation": generation,
                            "fidelity_epochs": fidelity_epochs,
                            "candidate_count": (
                                len(population)
                                if population
                                else 1
                                if smoke_test
                                else CIFAR_INITIAL_POPULATION_SIZE
                                if group["dataset_name"] == "cifar10"
                                else INITIAL_POPULATION_SIZE
                            ),
                        }
                        for worker in self.config.workers
                    }
                    self._persist_locked()

                def record_worker_search(result: dict[str, Any]) -> None:
                    with self._lock:
                        current = {
                            row["node_id"]: row for row in group.get("worker_search", [])
                        }
                        current[result["node_id"]] = result
                        group["worker_search"] = sorted(current.values(), key=lambda row: row["node_id"])
                        group["worker_search_progress"][result["node_id"]] = {
                            "status": "completed",
                            "started_at": started_at,
                            "completed_at": time.time(),
                            "generation": generation,
                            "fidelity_epochs": fidelity_epochs,
                            "candidate_count": len(result.get("candidates", [])),
                            "elapsed_seconds": result.get("elapsed_seconds"),
                        }
                        self._persist_locked()

                payload: dict[str, Any] = {
                    "dataset_name": group["dataset_name"],
                    "seed": group["seed"],
                    "time_limit_seconds": min(
                        remaining_seconds,
                        initial_stage_limit if generation == 1 else 300,
                    ),
                    "epochs": fidelity_epochs,
                }
                if smoke_test:
                    payload["smoke_test"] = True
                if population is not None:
                    payload["architectures"] = population
                searches = self._parallel_workers(
                    lambda worker: self._post(
                        worker,
                        "/v1/nas/search",
                        payload,
                        timeout=min(remaining_seconds, 300) + 30,
                    ),
                    on_result=record_worker_search,
                )
                searches = sorted(searches, key=lambda row: row["node_id"])
                class_sets = {
                    tuple(result.get("dataset_classes", [])) for result in searches
                }
                dataset_names = {result.get("dataset_name") for result in searches}
                image_shapes = {
                    tuple(result.get("dataset_image_shape", [])) for result in searches
                }
                if (
                    len(class_sets) != 1
                    or dataset_names != {group["dataset_name"]}
                    or len(image_shapes) != 1
                    or any(
                        result.get("fidelity_epochs") != fidelity_epochs
                        for result in searches
                    )
                ):
                    raise TrainingError(
                        "Workers searched inconsistent dataset shards or fidelity settings"
                    )
                resolved_shape = next(iter(image_shapes))
                if len(resolved_shape) != 3 or resolved_shape[-1] not in {1, 3}:
                    raise TrainingError("Workers reported an unsupported dataset image shape")
                resolved_classes = list(next(iter(class_sets)))
                if input_shape is not None and (
                    input_shape != resolved_shape or class_names != resolved_classes
                ):
                    raise TrainingError("Workers searched inconsistent dataset shards between NAS stages")
                input_shape = resolved_shape
                class_names = resolved_classes
                if generation == 1 and population is None:
                    if smoke_test:
                        candidate_rows = searches[0].get("candidates")
                        if not isinstance(candidate_rows, list) or not candidate_rows:
                            raise TrainingError("NAS smoke test returned no candidate architecture")
                        population = [
                            row["architecture"]
                            for row in candidate_rows
                            if isinstance(row, dict)
                            and isinstance(row.get("architecture"), dict)
                        ]
                        if len(population) != 1:
                            raise TrainingError(
                                "NAS smoke test must evaluate exactly one lightweight candidate"
                            )
                    else:
                        population = initial_population(
                            seed=group["seed"],
                            input_shape=input_shape,
                            num_classes=len(class_names),
                            size=(
                                CIFAR_INITIAL_POPULATION_SIZE
                                if group["dataset_name"] == "cifar10"
                                else INITIAL_POPULATION_SIZE
                            ),
                            memory_constrained=group["dataset_name"] == "cifar10",
                        )
                measured = aggregate_worker_candidates(
                    searches,
                    worker_count=len(self.config.workers),
                )
                expected_ids = {candidate["id"] for candidate in population or []}
                if {row["architecture"]["id"] for row in measured} != expected_ids:
                    raise TrainingError(
                        "NAS workers did not evaluate the same architecture population"
                    )
                for candidate in measured:
                    candidate["generation"] = generation
                    candidate["fidelity_epochs"] = fidelity_epochs
                    candidate["is_pareto_optimal"] = False
                front_ids = {
                    item["architecture"]["id"] for item in pareto_front(measured)
                }
                for candidate in measured:
                    candidate["is_pareto_optimal"] = candidate["architecture"]["id"] in front_ids
                with self._lock:
                    group.setdefault("nas_generations", []).append(
                        {
                            "generation": generation,
                            "fidelity_epochs": fidelity_epochs,
                            "candidates": measured,
                        }
                    )
                    group["nas_candidates"] = measured
                    group["dataset_classes"] = class_names
                    group["dataset_image_shape"] = list(input_shape)
                    group["search_method"] = (
                        "single_candidate_smoke_test"
                        if smoke_test
                        else "pareto_evolutionary_multifidelity"
                    )
                    self._persist_locked()
                final_candidates = measured
                final_searches = searches

            selected = select_best_architecture(final_candidates)
            pareto_ids = {
                item["architecture"]["id"] for item in pareto_front(final_candidates)
            }
            selection = {
                "method": (
                    "single_candidate_smoke_test"
                    if smoke_test
                    else "pareto_evolutionary_multifidelity"
                ),
                "fidelity_epochs": fidelities[-1],
                "selected_candidate": selected,
                "pareto_front": [
                    candidate
                    for candidate in final_candidates
                    if candidate["architecture"]["id"] in pareto_ids
                ],
            }
            selected_architecture = {
                **selected["architecture"],
                "dataset_name": group["dataset_name"],
                "input_shape": list(input_shape),
                "classes": class_names,
                "num_classes": len(class_names),
            }
            selection["selected_candidate"] = {
                **selected,
                "architecture": selected_architecture,
            }
            group["worker_search"] = final_searches
            group["selected_architecture"] = selected_architecture
            group["selected_parameter_count"] = selected["parameter_count"]
            group["search_elapsed_seconds"] = round(
                time.monotonic() - search_started,
                3,
            )
            group["search_peak_rss_mb"] = max(
                float(result["peak_rss_mb"]) for result in final_searches
            )
            self._update_group(
                group_id,
                status="completed",
                completed_at=time.time(),
                selection=selection,
            )
        except TrainingCancelled as error:
            with self._lock:
                for progress in group.get("worker_search_progress", {}).values():
                    if progress["status"] == "searching":
                        progress.update(status="stopped", error=str(error))
                self._persist_locked()
            self._update_group(
                group_id,
                status="stopped",
                completed_at=time.time(),
                error=str(error),
            )
        except Exception as error:
            LOGGER.exception("Dataset NAS group %s failed", group_id)
            with self._lock:
                for progress in group.get("worker_search_progress", {}).values():
                    if progress["status"] == "searching":
                        progress.update(status="failed", error=str(error))
                self._persist_locked()
            self._update_group(
                group_id,
                status="failed",
                completed_at=time.time(),
                error=f"{type(error).__name__}: {error}",
            )
        finally:
            with self._lock:
                if self._active_group_id == group_id:
                    self._active_group_id = None
                self._persist_locked()

    @staticmethod
    def _summarize_nas_runs(
        runs: list[dict[str, Any]],
        *,
        search_seconds: float,
        search_peak_rss_mb: float,
    ) -> dict[str, Any]:
        if not runs:
            raise TrainingError("NAS comparison is missing completed experiment runs")

        def worker_metric(
            metrics: dict[str, Any],
            *,
            primary: str,
            fallback: str,
        ) -> float:
            value = metrics.get(primary)
            if value is None:
                value = metrics.get(fallback)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TrainingError(
                    f"NAS comparison worker metrics are missing {primary} "
                    f"(or legacy {fallback})"
                )
            return float(value)

        accuracies = []
        macro_f1_values = []
        runtime_values = []
        peak_rss = []
        bytes_sent = []
        for run in runs:
            final_workers = run["rounds"][-1]["workers"]
            accuracies.append(
                sum(
                    worker_metric(
                        row,
                        primary="validation_accuracy",
                        fallback="accuracy",
                    )
                    for row in final_workers.values()
                )
                / len(final_workers)
            )
            macro_f1_values.append(
                sum(
                    worker_metric(
                        row,
                        primary="validation_macro_f1",
                        fallback="macro_f1",
                    )
                    for row in final_workers.values()
                )
                / len(final_workers)
            )
            runtime_values.append(
                sum(
                    max(
                        float(metrics.get("round_seconds", metrics.get("runtime_seconds", 0)))
                        for metrics in row["workers"].values()
                    )
                    for row in run["rounds"]
                )
            )
            peak_rss.append(
                max(
                    float(metrics.get("peak_process_rss_mb", 0))
                    for row in run["rounds"]
                    for metrics in row["workers"].values()
                )
            )
            bytes_sent.append(
                sum(
                    int(metrics.get("bytes_sent", 0))
                    for row in run["rounds"]
                    for metrics in row["workers"].values()
                )
            )
        mean_runtime = sum(runtime_values) / len(runtime_values)
        return {
            "completed_runs": len(runs),
            "mean_accuracy": sum(accuracies) / len(accuracies),
            "mean_macro_f1": sum(macro_f1_values) / len(macro_f1_values),
            "mean_training_runtime_seconds": mean_runtime,
            "mean_runtime_including_search_seconds": (
                mean_runtime + search_seconds / len(runs)
            ),
            "total_runtime_including_search_seconds": sum(runtime_values) + search_seconds,
            "search_seconds": search_seconds,
            "peak_rss_mb": max(max(peak_rss), search_peak_rss_mb),
            "mean_communication_bytes": sum(bytes_sent) / len(bytes_sent),
        }

    def _run_baseline_group(self, group_id: str) -> None:
        group = self._get_group(group_id)
        try:
            for seed in group["seeds"]:
                if self._stop.is_set():
                    raise TrainingCancelled("Baseline training was cancelled")
                run_id = f"centralized-seed-{seed}-{uuid.uuid4().hex[:8]}"
                rows = self.evaluator.train_baseline(
                    seed,
                    group["rounds"],
                    group["epochs"],
                )
                run = {
                    "run_id": run_id,
                    "seed": seed,
                    "topology": "centralized",
                    "status": "completed",
                    "rounds": [
                        {
                            **row,
                            "workers": {"dfl-master": row},
                        }
                        for row in rows
                    ],
                }
                with self._lock:
                    self._get_group(group_id)["runs"].append(run)
                    self._persist_locked()
            self._update_group(group_id, status="completed", completed_at=time.time())
        except TrainingCancelled as error:
            self._update_group(group_id, status="stopped", completed_at=time.time(), error=str(error))
        except Exception as error:
            LOGGER.exception("Centralized baseline group %s failed", group_id)
            self._update_group(
                group_id,
                status="failed",
                completed_at=time.time(),
                error=f"{type(error).__name__}: {error}",
            )
        finally:
            with self._lock:
                if self._active_group_id == group_id:
                    self._active_group_id = None
                self._persist_locked()

    def _cancel_workers(self, node_ids: set[str] | None = None) -> dict[str, str]:
        targets = [
            worker for worker in self.config.workers if node_ids is None or worker.node_id in node_ids
        ]
        errors = {}
        with ThreadPoolExecutor(max_workers=len(targets) or 1) as pool:
            futures = {
                pool.submit(self._post, worker, "/v1/cancel", {}): worker.node_id
                for worker in targets
            }
            for future in as_completed(futures):
                node_id = futures[future]
                try:
                    future.result()
                except Exception as error:
                    errors[node_id] = str(error)
                    LOGGER.warning(
                        "Could not cancel training on %s: %s",
                        node_id,
                        error,
                    )
        return errors

    def _parallel_workers(
        self,
        operation: Any,
        *,
        on_result: Callable[[Any], None] | None = None,
    ) -> list[Any]:
        with ThreadPoolExecutor(max_workers=len(self.config.workers)) as pool:
            futures = {pool.submit(operation, worker): worker.node_id for worker in self.config.workers}
            results = []
            errors = []
            for future in as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)
                    if on_result is not None:
                        on_result(result)
                except Exception as error:
                    errors.append(f"{futures[future]}: {error}")
            if errors:
                raise TrainingError("Worker request failed: " + "; ".join(errors))
            return results

    @staticmethod
    def _request(
        worker: TrainingWorker,
        path: str,
        *,
        payload: dict[str, Any] | None = None,
        timeout: float = 30.0,
        method: str | None = None,
    ) -> tuple[bytes, dict[str, str]]:
        body = json.dumps(payload).encode("utf-8") if payload is not None else None
        request = Request(
            worker.base_url + path,
            data=body,
            headers={
                "Authorization": f"Bearer {worker.token}",
                **({"Content-Type": "application/json"} if body is not None else {}),
            },
            method=method or ("POST" if body is not None else "GET"),
        )
        try:
            with urlopen(request, timeout=timeout) as response:
                return response.read(), dict(response.headers.items())
        except HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace").strip()[:500]
            raise TrainingError(
                f"{worker.node_id} returned HTTP {error.code} for {path}: {detail}"
            ) from error
        except (URLError, TimeoutError, OSError) as error:
            raise TrainingError(f"Could not reach {worker.node_id} for {path}: {error}") from error

    def _delete(
        self,
        worker: TrainingWorker,
        path: str,
        *,
        timeout: float = 30,
    ) -> dict[str, Any]:
        body, _ = self._request(worker, path, timeout=timeout, method="DELETE")
        try:
            response = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise TrainingError(
                f"{worker.node_id} returned an invalid cleanup response"
            ) from error
        if not isinstance(response, dict) or response.get("cleaned") is not True:
            raise TrainingError(f"{worker.node_id} did not confirm run memory cleanup")
        return response

    def _post(
        self,
        worker: TrainingWorker,
        path: str,
        payload: dict[str, Any],
        *,
        timeout: float = 30,
    ) -> dict[str, Any]:
        body, _ = self._request(worker, path, payload=payload, timeout=timeout)
        response = json.loads(body.decode("utf-8"))
        if not isinstance(response, dict):
            raise TrainingError(f"{worker.node_id} returned a non-object response")
        return response

    def _get_json(
        self,
        worker: TrainingWorker,
        path: str,
        *,
        timeout: float = 10,
    ) -> dict[str, Any]:
        body, _ = self._request(worker, path, timeout=timeout)
        response = json.loads(body.decode("utf-8"))
        if not isinstance(response, dict):
            raise TrainingError(f"{worker.node_id} returned a non-object response")
        return response

    def _get_run_results(self, worker: TrainingWorker, run_id: str) -> dict[str, Any]:
        path = "/v1/results?" + urlencode({"run_id": run_id})
        retryable_errors = (
            "timed out",
            "no route to host",
            "network is unreachable",
            "connection refused",
            "connection reset",
            "remote end closed connection",
        )
        for attempt in range(3):
            try:
                return self._get_json(
                    worker,
                    path,
                    timeout=self.config.peer_timeout_seconds,
                )
            except TrainingError as error:
                message = str(error).lower()
                if attempt == 2 or not any(
                    retryable_error in message
                    for retryable_error in retryable_errors
                ):
                    raise
                delay = (1, 3)[attempt]
                LOGGER.warning(
                    "Transient error reading run %s results from %s; retry %s/2 in %s seconds: %s",
                    run_id,
                    worker.node_id,
                    attempt + 1,
                    delay,
                    error,
                )
                time.sleep(delay)
        raise AssertionError("unreachable")

    def _get_bytes(self, worker: TrainingWorker, path: str) -> bytes:
        body, headers = self._request(worker, path, timeout=20)
        if not body or len(body) > MAX_MODEL_WEIGHT_BYTES or "X-Sample-Count" not in headers:
            raise TrainingError(f"{worker.node_id} returned invalid model weights")
        return body

    def _load_groups(self) -> list[dict[str, Any]]:
        database_path, legacy_path = self._results_paths()
        database_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(database_path, timeout=30) as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS experiment_groups (
                    group_id TEXT PRIMARY KEY,
                    payload_json TEXT NOT NULL
                )
                """
            )
            rows = connection.execute(
                "SELECT payload_json FROM experiment_groups ORDER BY rowid"
            ).fetchall()
            if rows:
                groups = [json.loads(row[0]) for row in rows]
                if any(not isinstance(group, dict) for group in groups):
                    raise ValueError(f"Training database contains an invalid group: {database_path}")
                return groups

            if not legacy_path.exists():
                return []
            payload = json.loads(legacy_path.read_text(encoding="utf-8"))
            if not isinstance(payload, list) or any(
                not isinstance(group, dict) or not isinstance(group.get("group_id"), str)
                for group in payload
            ):
                raise ValueError(f"Training results file must contain valid groups: {legacy_path}")
            connection.executemany(
                "INSERT INTO experiment_groups (group_id, payload_json) VALUES (?, ?)",
                [
                    (
                        group["group_id"],
                        json.dumps(group, allow_nan=False, separators=(",", ":")),
                    )
                    for group in payload
                ],
            )
            return payload

    def _persist_locked(self) -> None:
        database_path, _ = self._results_paths()
        database_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(database_path, timeout=30) as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS experiment_groups (
                    group_id TEXT PRIMARY KEY,
                    payload_json TEXT NOT NULL
                )
                """
            )
            connection.executemany(
                """
                INSERT INTO experiment_groups (group_id, payload_json)
                VALUES (?, ?)
                ON CONFLICT(group_id) DO UPDATE SET payload_json = excluded.payload_json
                """,
                [
                    (
                        group["group_id"],
                        json.dumps(group, allow_nan=False, separators=(",", ":")),
                    )
                    for group in self._groups
                ],
            )
