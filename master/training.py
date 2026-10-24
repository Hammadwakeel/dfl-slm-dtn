"""Master-side orchestration for direct-peer CIFAR-10 distributed training."""

from __future__ import annotations

import atexit
import base64
import json
import logging
import os
import subprocess
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from topology_opt.reward import RewardConfig, calculate_run_reward
from topology_opt.selector import select_candidate

LOGGER = logging.getLogger("dfl_slm.training")
TOPOLOGIES = (
    "directed_ring",
    "bidirectional_ring",
    "mesh",
    "chain",
    "star",
)
DATASET_SPLITS = {
    0.3: "dirichlet_a03",
    0.5: "dirichlet_a05",
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
        if [worker.node_id for worker in workers] != ["worker1", "worker2", "worker3"]:
            raise ValueError("DFL_WORKERS_JSON must configure worker1, worker2, and worker3 in order")
        if any(
            len(worker.token) < 32 or not worker.base_url.startswith("http://")
            for worker in workers
        ):
            raise ValueError("Worker training URLs and tokens must be configured")
        rounds = int(os.environ.get("DFL_TRAINING_ROUNDS", "10"))
        epochs = int(os.environ.get("DFL_LOCAL_EPOCHS", "5"))
        topology = os.environ.get("DFL_TOPOLOGY", "directed_ring")
        peer_timeout = float(os.environ.get("DFL_TRAINING_PEER_TIMEOUT_SECONDS", "60"))
        if not 1 <= rounds <= 100 or not 1 <= epochs <= 50:
            raise ValueError("Training rounds or epochs are out of range")
        if topology not in {"ring", *TOPOLOGIES} or peer_timeout <= 0:
            raise ValueError("Training topology or peer timeout is invalid")
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
                "/var/lib/dfl-slm-master/experiments.json",
            ),
            rounds=rounds,
            epochs=epochs,
            topology=topology,
            peer_timeout_seconds=peer_timeout,
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

    def evaluate(self, models: dict[str, bytes]) -> dict[str, Any]:
        return self._request(
            {
                "type": "evaluate",
                "models": {
                    node_id: base64.b64encode(weights).decode("ascii")
                    for node_id, weights in models.items()
                },
            }
        )["results"]

    def train_baseline(self, seed: int, rounds: int, epochs: int) -> list[dict[str, Any]]:
        return self._request(
            {"type": "baseline", "seed": seed, "rounds": rounds, "epochs": epochs}
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
        self._settings = {
            "rounds": config.rounds,
            "epochs": config.epochs,
            "topology": config.topology,
            "dataset_split": "stratified",
        }
        self._groups: list[dict[str, Any]] = self._load_groups()

    def configure(
        self,
        *,
        rounds: int | None = None,
        epochs: int | None = None,
        topology: str | None = None,
        dataset_split: str | None = None,
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
                if topology not in TOPOLOGIES:
                    raise ValueError(f"topology must be one of {', '.join(TOPOLOGIES)}")
                updated["topology"] = topology
            if dataset_split is not None:
                if dataset_split not in {"stratified", *DATASET_SPLITS.values()}:
                    raise ValueError("Unsupported CIFAR-10 training split")
                updated["dataset_split"] = dataset_split
            self._settings = updated
        return dict(updated)

    def start_training(
        self,
        *,
        rounds: int | None = None,
        epochs: int | None = None,
        seeds: tuple[int, ...] = (17, 42, 2026),
        dataset_split: str | None = None,
    ) -> dict[str, Any]:
        if len(seeds) != 3 or any(seed < 0 for seed in seeds):
            raise ValueError("Exactly three non-negative training seeds are required")
        with self._lock:
            if self._active_group_id is not None:
                raise TrainingError(f"Training group {self._active_group_id} is already active")
            settings = self.configure(
                rounds=rounds,
                epochs=epochs,
                dataset_split=dataset_split,
            )
            group_id = f"{settings['topology']}-{int(time.time())}-{uuid.uuid4().hex[:6]}"
            group = {
                "group_id": group_id,
                "kind": "decentralized",
                "topology": settings["topology"],
                "rounds": settings["rounds"],
                "epochs": settings["epochs"],
                "dataset_split": settings["dataset_split"],
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
                target=self._run_group,
                args=(group_id,),
                name=f"training-group-{group_id}",
                daemon=True,
            )
            self._thread.start()
            self._persist_locked()
        return {"accepted": True, "group_id": group_id, **settings, "seeds": list(seeds)}

    def start_optimization(
        self,
        *,
        rounds: int = 10,
        epochs: int = 5,
        seeds: tuple[int, ...] = (17, 42, 2026),
        alphas: tuple[float, ...] = (0.3, 0.5),
    ) -> dict[str, Any]:
        if not 1 <= len(seeds) <= 3 or any(seed < 0 for seed in seeds):
            raise ValueError("Provide between one and three non-negative seeds")
        if not alphas or any(alpha not in DATASET_SPLITS for alpha in alphas):
            raise ValueError("Dirichlet alphas must be selected from 0.3 and 0.5")
        if not 1 <= rounds <= 100 or not 1 <= epochs <= 50:
            raise ValueError("rounds must be 1-100 and epochs must be 1-50")
        with self._lock:
            if self._active_group_id is not None:
                raise TrainingError(f"Training group {self._active_group_id} is already active")
            group_id = f"optimize-{int(time.time())}-{uuid.uuid4().hex[:6]}"
            group = {
                "group_id": group_id,
                "kind": "topology_optimization",
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
            return {
                "configuration": dict(self._settings),
                "active_group_id": self._active_group_id,
                "topologies": {
                    "ring": ["worker1", "worker2", "worker3", "worker1"],
                    "directed_ring": ["worker1", "worker2", "worker3", "worker1"],
                    "bidirectional_ring": [
                        ["worker1", "worker2"],
                        ["worker2", "worker3"],
                        ["worker3", "worker1"],
                    ],
                    "mesh": [
                        ["worker1", "worker2"],
                        ["worker1", "worker3"],
                        ["worker2", "worker3"],
                    ],
                    "chain": [["worker1", "worker2"], ["worker2", "worker3"]],
                    "star": [["worker1", "worker2"], ["worker1", "worker3"]],
                },
                "groups": json.loads(json.dumps(self._groups[-20:])),
            }

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
            self._persist_locked()

    def _run_group(self, group_id: str) -> None:
        group = self._get_group(group_id)
        try:
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

    def _run_distributed_seed(
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
        }
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
                    },
                )

            for node_id, _ in self._parallel_workers(start):
                started.add(node_id)
        except Exception:
            self._cancel_workers(started or None)
            raise

        for round_index in range(group["rounds"]):
            self._wait_for_worker_rounds(run["run_id"], round_index, group_id)
            summaries = self._parallel_workers(
                lambda worker: (
                    worker.node_id,
                    self._get_json(
                        worker,
                        "/v1/results?" + urlencode({"run_id": run["run_id"]}),
                    ),
                )
            )
            local_rows = {
                node_id: payload["runs"][run["run_id"]][round_index]
                for node_id, payload in summaries
            }
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
            scores = self.evaluator.evaluate(dict(binary_weights))
            row = {
                "round": round_index + 1,
                "topology": group["topology"],
                "completed_at": time.time(),
                "workers": {
                    node_id: {
                        **local_rows[node_id],
                        **scores[node_id],
                    }
                    for node_id in local_rows
                },
            }
            self._append_run_row(group_id, run, row)

    def _wait_for_worker_rounds(self, run_id: str, round_index: int, group_id: str) -> None:
        deadline = time.monotonic() + self.config.peer_timeout_seconds * 6
        while time.monotonic() < deadline:
            if self._stop.is_set():
                raise TrainingCancelled("Training was cancelled")
            snapshots = self._parallel_workers(
                lambda worker: (
                    worker.node_id,
                    self._get_json(
                        worker,
                        "/v1/results?" + urlencode({"run_id": run_id}),
                    ),
                )
            )
            for node_id, payload in snapshots:
                if run_id in payload.get("errors", {}):
                    raise TrainingError(
                        f"{node_id} failed in run {run_id}: {payload['errors'][run_id]}"
                    )
                rows = payload.get("runs", {}).get(run_id, [])
                if len(rows) <= round_index:
                    break
            else:
                return
            time.sleep(1)
        raise TrainingError(
            f"Timed out waiting for all workers to complete round {round_index + 1} "
            f"for group {group_id}"
        )

    def _run_optimization_group(self, group_id: str) -> None:
        group = self._get_group(group_id)
        try:
            for candidate in group["candidates"]:
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
                        self._persist_locked()

            candidates = []
            for candidate in group["candidates"]:
                matching_runs = [
                    run
                    for run in group["runs"]
                    if run["topology"] == candidate["topology"]
                    and run["alpha"] == candidate["alpha"]
                    and run["status"] == "completed"
                ]
                per_run = [
                    calculate_run_reward(run["rounds"], self.reward_config)
                    for run in matching_runs
                ]
                if not per_run:
                    continue
                candidates.append(
                    {
                        "topology": candidate["topology"],
                        "alpha": candidate["alpha"],
                        "completed_runs": len(per_run),
                        "mean_reward": round(
                            sum(value["reward"] for value in per_run) / len(per_run),
                            6,
                        ),
                        "mean_final_accuracy": round(
                            sum(value["metrics"]["final_accuracy"] for value in per_run)
                            / len(per_run),
                            6,
                        ),
                        "mean_components": {
                            name: round(
                                sum(value["components"][name] for value in per_run)
                                / len(per_run),
                                6,
                            )
                            for name in self.reward_config.weights
                        },
                        "per_run_rewards": per_run,
                    }
                )
            if not candidates:
                raise TrainingError("No topology candidates completed successfully")
            selection = select_candidate(candidates)
            chosen = selection["selected_candidate"]
            with self._lock:
                self._settings["topology"] = chosen["topology"]
                self._settings["dataset_split"] = DATASET_SPLITS[chosen["alpha"]]
            self._update_group(
                group_id,
                candidates=candidates,
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
                except TrainingError as error:
                    errors[node_id] = str(error)
        return errors

    def _parallel_workers(self, operation: Any) -> list[Any]:
        with ThreadPoolExecutor(max_workers=len(self.config.workers)) as pool:
            futures = {pool.submit(operation, worker): worker.node_id for worker in self.config.workers}
            results = []
            errors = []
            for future in as_completed(futures):
                try:
                    results.append(future.result())
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
    ) -> tuple[bytes, dict[str, str]]:
        body = json.dumps(payload).encode("utf-8") if payload is not None else None
        request = Request(
            worker.base_url + path,
            data=body,
            headers={
                "Authorization": f"Bearer {worker.token}",
                **({"Content-Type": "application/json"} if body is not None else {}),
            },
            method="POST" if body is not None else "GET",
        )
        try:
            with urlopen(request, timeout=timeout) as response:
                return response.read(), dict(response.headers.items())
        except HTTPError as error:
            raise TrainingError(f"{worker.node_id} returned HTTP {error.code} for {path}") from error
        except (URLError, TimeoutError, OSError) as error:
            raise TrainingError(f"Could not reach {worker.node_id} for {path}: {error}") from error

    def _post(
        self,
        worker: TrainingWorker,
        path: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        body, _ = self._request(worker, path, payload=payload, timeout=30)
        response = json.loads(body.decode("utf-8"))
        if not isinstance(response, dict):
            raise TrainingError(f"{worker.node_id} returned a non-object response")
        return response

    def _get_json(self, worker: TrainingWorker, path: str) -> dict[str, Any]:
        body, _ = self._request(worker, path, timeout=10)
        response = json.loads(body.decode("utf-8"))
        if not isinstance(response, dict):
            raise TrainingError(f"{worker.node_id} returned a non-object response")
        return response

    def _get_bytes(self, worker: TrainingWorker, path: str) -> bytes:
        body, headers = self._request(worker, path, timeout=20)
        if not body or len(body) > 16 * 1024 * 1024 or "X-Sample-Count" not in headers:
            raise TrainingError(f"{worker.node_id} returned invalid model weights")
        return body

    def _load_groups(self) -> list[dict[str, Any]]:
        path = Path(self.config.results_path)
        if not path.exists():
            return []
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError(f"Training results file must contain a JSON array: {path}")
        return payload

    def _persist_locked(self) -> None:
        path = Path(self.config.results_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(self._groups, allow_nan=False) + "\n", encoding="utf-8")
        temporary.replace(path)
