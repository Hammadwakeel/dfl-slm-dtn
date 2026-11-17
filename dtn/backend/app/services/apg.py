"""Read-only APG result access and user-started PC simulation jobs."""

from __future__ import annotations

import json
import logging
import math
import statistics
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from sqlalchemy.orm import Session, sessionmaker

from app.db.models import AutonomyExecutionEvent

LOGGER = logging.getLogger(__name__)
NODES = tuple(f"node_{index}" for index in range(8))
TOPOLOGIES = frozenset({"apg", "ring", "mesh"})
DATA_SETTINGS = frozenset({"iid_stratified", "constrained_dirichlet_a01"})
SEEDS = frozenset({17, 42, 2026})
ACTIVE_JOB_STATES = frozenset({"queued", "running"})


class APGResultsError(RuntimeError):
    pass


class APGJobStopped(RuntimeError):
    pass


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise APGResultsError(f"Could not read APG result file {path.name}: {error}") from error
    if not isinstance(value, dict):
        raise APGResultsError(f"APG result file {path.name} must contain a JSON object")
    return value


def _source_label(run: dict[str, Any]) -> str:
    if run.get("simulation_mode") == "hybrid_shaped_simulation":
        return "HYBRID-SHAPED"
    if run.get("simulation_mode") == "all_simulated":
        return "SIMULATED"
    source = run.get("source_label")
    return source if source in {"SIMULATED", "HYBRID-SHAPED", "HARDWARE"} else "NOT AVAILABLE"


def _numeric_stats(values: Any) -> dict[str, float | None]:
    flattened: list[float] = []

    def visit(value: Any) -> None:
        if isinstance(value, list):
            for item in value:
                visit(item)
        elif isinstance(value, (int, float)) and math.isfinite(float(value)):
            flattened.append(float(value))

    visit(values)
    if not flattened:
        return {"min": None, "mean": None, "max": None}
    return {
        "min": min(flattened),
        "mean": statistics.fmean(flattened),
        "max": max(flattened),
    }


def _input_summary(rounds: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    previous: dict[str, str] | None = None
    for round_info in rounds:
        inputs = round_info.get("apg_inputs")
        if not isinstance(inputs, dict):
            summaries.append(
                {
                    "round": round_info.get("round"),
                    "loss_changed": None,
                    "waiting_time_changed": None,
                    "theta_changed": None,
                    "loss_stats": None,
                    "waiting_time_stats": None,
                    "theta_stats": None,
                    "inputs_available": False,
                }
            )
            previous = None
            continue
        current = {
            "loss": json.dumps(inputs.get("loss_matrix_normalized"), sort_keys=True),
            "waiting_time": json.dumps(
                inputs.get("wait_matrix_normalized"), sort_keys=True
            ),
            "theta": json.dumps(inputs.get("theta_train_only"), sort_keys=True),
        }
        summaries.append(
            {
                "round": round_info.get("round"),
                "loss_changed": None if previous is None else current["loss"] != previous["loss"],
                "waiting_time_changed": (
                    None
                    if previous is None
                    else current["waiting_time"] != previous["waiting_time"]
                ),
                "theta_changed": (
                    None if previous is None else current["theta"] != previous["theta"]
                ),
                "loss_stats": _numeric_stats(inputs.get("loss_matrix_normalized")),
                "waiting_time_stats": _numeric_stats(inputs.get("wait_matrix_normalized")),
                "theta_stats": _numeric_stats(inputs.get("theta_train_only")),
                "inputs_available": True,
            }
        )
        previous = current
    return summaries


def _round_summary(round_info: dict[str, Any], inputs: dict[str, Any]) -> dict[str, Any]:
    per_node = round_info.get("per_node") or {}
    losses = [
        float(node["validation_loss"])
        for node in per_node.values()
        if isinstance(node, dict) and isinstance(node.get("validation_loss"), (int, float))
    ]
    return {
        "round": round_info.get("round"),
        "split": "VALIDATION",
        "source_label": round_info.get("source_label", "NOT AVAILABLE"),
        "mean_validation_accuracy": round_info.get("mean_validation_accuracy"),
        "mean_validation_macro_f1": round_info.get("mean_validation_macro_f1"),
        "mean_validation_loss": statistics.fmean(losses) if losses else None,
        "edges_used": round_info.get("edges_used", []),
        "edge_count_used": round_info.get("edge_count_used"),
        "edges_changed_vs_previous_round": round_info.get("edges_changed_vs_previous_round"),
        "safeguard_hits": round_info.get("safeguard_hits"),
        "g": round_info.get("g"),
        "iterations_used": round_info.get("iterations_used"),
        "inputs": inputs,
        "bytes_sent_simulated": round_info.get("bytes_sent_simulated"),
    }


def summarize_run(run: dict[str, Any]) -> dict[str, Any]:
    source = _source_label(run)
    rounds = run.get("rounds", [])
    input_summaries = _input_summary(rounds) if isinstance(rounds, list) else []
    round_summaries = [
        _round_summary(round_info, input_summaries[index])
        for index, round_info in enumerate(rounds)
        if isinstance(round_info, dict)
    ]
    final_test = run.get("final_test")
    final_test_summary = (
        final_test.get("summary")
        if isinstance(final_test, dict) and isinstance(final_test.get("summary"), dict)
        else None
    )
    return {
        "run_id": run.get("run_id", "NOT AVAILABLE"),
        "simulation_mode": run.get("simulation_mode", "NOT AVAILABLE"),
        "source_label": source,
        "data_setting": run.get("data_setting", "NOT AVAILABLE"),
        "topology": run.get("topology", "NOT AVAILABLE"),
        "seed": run.get("seed"),
        "status": "finished" if final_test_summary is not None else "training_only",
        "split_labels": {"rounds": "VALIDATION", "final_metrics": "TEST"},
        "duration_seconds": run.get("job_wall_seconds", run.get("pc_simulation_wall_seconds")),
        "rounds": round_summaries,
        "final_test_summary": final_test_summary,
    }


def detailed_run(run: dict[str, Any]) -> dict[str, Any]:
    result = dict(run)
    rounds = run.get("rounds", [])
    if isinstance(rounds, list):
        inputs = _input_summary(rounds)
        result["rounds"] = [
            {**round_info, "inputs": inputs[index]}
            for index, round_info in enumerate(rounds)
            if isinstance(round_info, dict)
        ]
    return result


def _mesh_edges() -> set[tuple[str, str]]:
    return {(receiver, source) for receiver in NODES for source in NODES if receiver != source}


def _run_edges(run: dict[str, Any]) -> list[set[tuple[str, str]]]:
    edges_per_round: list[set[tuple[str, str]]] = []
    for round_info in run.get("rounds", []):
        edges_per_round.append(
            {
                (str(edge[0]), str(edge[1]))
                for edge in round_info.get("edges_used", [])
                if isinstance(edge, list) and len(edge) == 2
            }
        )
    return edges_per_round


def build_verdict(
    runs: list[dict[str, Any]],
    data_setting: str,
    simulation_mode: str,
) -> dict[str, Any]:
    matching = [
        run
        for run in runs
        if run.get("data_setting") == data_setting
        and run.get("simulation_mode") == simulation_mode
    ]
    apg_runs = [run for run in matching if run.get("topology") == "apg"]
    mesh_edges = _mesh_edges()
    adapt_flags: list[bool] = []
    follow_flags: list[bool] = []
    validation_gains: list[float] = []
    for run in apg_runs:
        rounds = run.get("rounds", [])
        edges = _run_edges(run)
        graph_differs_from_mesh = any(edge_set != mesh_edges for edge_set in edges)
        changed_after_initial_update = any(
            int(round_info.get("edges_changed_vs_previous_round") or 0) > 0
            for round_info in rounds
            if int(round_info.get("round", 0)) > 1
        )
        adapt_flags.append(graph_differs_from_mesh and changed_after_initial_update)
        input_summaries = _input_summary(rounds)
        follows = False
        for index in range(1, min(len(rounds), len(input_summaries))):
            previous_inputs = input_summaries[index - 1]
            current_inputs = input_summaries[index]
            input_changed = any(
                current_inputs.get(key) is True
                for key in ("loss_changed", "waiting_time_changed", "theta_changed")
            )
            edges_changed = (
                int(rounds[index].get("edges_changed_vs_previous_round") or 0) > 0
            )
            follows = follows or (input_changed and edges_changed)
        follow_flags.append(follows)
        validation_values = [
            round_info.get("mean_validation_accuracy")
            for round_info in rounds
            if isinstance(round_info.get("mean_validation_accuracy"), (int, float))
        ]
        if len(validation_values) >= 2:
            validation_gains.append(float(validation_values[-1]) - float(validation_values[0]))

    matched_seeds: dict[int, dict[str, dict[str, Any]]] = {}
    for run in matching:
        if run.get("topology") not in {"apg", "mesh", "ring"}:
            continue
        seed = run.get("seed")
        if isinstance(seed, int) and isinstance(run.get("final_test"), dict):
            matched_seeds.setdefault(seed, {})[str(run["topology"])] = run
    paired_deltas: list[float] = []
    for seed, by_topology in matched_seeds.items():
        if not {"apg", "mesh", "ring"}.issubset(by_topology):
            continue
        apg_accuracy = by_topology["apg"]["final_test"]["summary"].get("accuracy")
        fixed_accuracies = [
            by_topology[topology]["final_test"]["summary"].get("accuracy")
            for topology in ("mesh", "ring")
        ]
        if isinstance(apg_accuracy, (float, int)) and all(
            isinstance(value, (float, int)) for value in fixed_accuracies
        ):
            paired_deltas.append(float(apg_accuracy) - max(map(float, fixed_accuracies)))

    topology_adapts = bool(adapt_flags) and any(adapt_flags)
    topology_follows_inputs = bool(follow_flags) and any(follow_flags)
    n_validation = len(validation_gains)
    validation_mean_gain = statistics.fmean(validation_gains) if validation_gains else None
    validation_sd_gain = (
        statistics.stdev(validation_gains) if len(validation_gains) >= 2 else None
    )
    model_learns = (
        validation_mean_gain is not None
        and validation_sd_gain is not None
        and validation_mean_gain > validation_sd_gain
    )
    n_test_seeds = len(paired_deltas)
    paired_mean = statistics.fmean(paired_deltas) if paired_deltas else None
    paired_sd = statistics.stdev(paired_deltas) if len(paired_deltas) >= 2 else None
    benefit = (
        n_test_seeds >= 3
        and paired_mean is not None
        and paired_sd is not None
        and paired_mean > paired_sd
    )

    if not topology_adapts:
        verdict = "Topology is not adapting (same graph every round)"
    elif n_test_seeds < 3 or n_validation < 2:
        verdict = "Not enough data (run more seeds)"
    elif topology_follows_inputs and model_learns and benefit:
        verdict = f"Topology is adapting and improves over ring/mesh ({n_test_seeds} seeds)"
    else:
        verdict = "Topology is adapting, no demonstrated benefit over ring/mesh"
    return {
        "data_setting": data_setting,
        "simulation_mode": simulation_mode,
        "source_label": (
            "HYBRID-SHAPED"
            if simulation_mode == "hybrid_shaped_simulation"
            else "SIMULATED"
        ),
        "verdict": verdict,
        "rules": {
            "topology_adapts": topology_adapts,
            "topology_follows_inputs": topology_follows_inputs,
            "model_learns": model_learns if n_validation >= 2 else None,
            "validation_mean_gain": validation_mean_gain,
            "validation_seed_sd": validation_sd_gain,
            "validation_seed_count": n_validation,
            "benefit_over_best_fixed_topology": benefit if n_test_seeds >= 3 else None,
            "paired_test_mean_delta": paired_mean,
            "paired_test_sample_sd": paired_sd,
            "test_seed_count": n_test_seeds,
            "test_split": "TEST",
            "validation_split": "VALIDATION",
            "benefit_rule": (
                "paired APG minus per-seed best of Mesh/Ring TEST accuracy mean "
                "must exceed sample SD (ddof=1), with at least 3 matched seeds"
            ),
        },
        "runs": [run.get("run_id") for run in apg_runs],
    }


@dataclass
class APGJob:
    job_id: str
    topology: str
    data_setting: str
    seed: int
    status: str = "queued"
    rounds_completed: int = 0
    total_rounds: int = 30
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    started_at: str | None = None
    completed_at: str | None = None
    duration_seconds: float | None = None
    result_file: str | None = None
    error: str | None = None
    cancel_event: threading.Event = field(default_factory=threading.Event, repr=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "run_id": self.job_id,
            "topology": self.topology,
            "data_setting": self.data_setting,
            "seed": self.seed,
            "simulation_mode": "all_simulated",
            "source_label": "SIMULATED",
            "status": self.status,
            "rounds_completed": self.rounds_completed,
            "total_rounds": self.total_rounds,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "duration_seconds": self.duration_seconds,
            "result_file": self.result_file,
            "error": self.error,
        }


class APGJobManager:
    def __init__(
        self,
        results_root: Path,
        sessions: sessionmaker[Session],
        stop_requested: Callable[[], bool],
        runner: Callable[..., dict[str, Any]] | None = None,
    ) -> None:
        self.results_root = results_root
        self.sessions = sessions
        self.stop_requested = stop_requested
        self.runner = runner or self._run_driver
        self._jobs: dict[str, APGJob] = {}
        self._lock = threading.RLock()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="apg-simulation")

    def start(self, topology: str, data_setting: str, seed: int) -> dict[str, Any]:
        if topology not in TOPOLOGIES:
            raise ValueError(f"Unsupported APG topology: {topology}")
        if data_setting not in DATA_SETTINGS:
            raise ValueError(f"Unsupported APG data setting: {data_setting}")
        if seed not in SEEDS:
            raise ValueError(f"Unsupported APG seed: {seed}")
        with self._lock:
            if self.stop_requested():
                raise PermissionError("Global STOP is active; APG simulation cannot be started")
            if any(job.status in ACTIVE_JOB_STATES for job in self._jobs.values()):
                raise RuntimeError("An APG simulation job is already active")
            job_id = f"apg-ui-{uuid.uuid4().hex}"
            job = APGJob(job_id=job_id, topology=topology, data_setting=data_setting, seed=seed)
            self._jobs[job_id] = job
            self._audit(job, "queued")
            self._executor.submit(self._run_job, job)
            return job.to_dict()

    def request_stop_all(self) -> None:
        with self._lock:
            for job in self._jobs.values():
                if job.status in ACTIVE_JOB_STATES:
                    job.cancel_event.set()

    def get(self, job_id: str) -> dict[str, Any] | None:
        with self._lock:
            job = self._jobs.get(job_id)
            return job.to_dict() if job else None

    def list_jobs(self) -> list[dict[str, Any]]:
        with self._lock:
            return [job.to_dict() for job in self._jobs.values()]

    def shutdown(self) -> None:
        self.request_stop_all()
        self._executor.shutdown(wait=False, cancel_futures=True)

    def load_runs(self) -> list[dict[str, Any]]:
        runs: list[dict[str, Any]] = []
        matrix_path = self.results_root / "final_metrics.json"
        if matrix_path.exists():
            result = _read_json(matrix_path)
            source_runs = result.get("runs")
            if not isinstance(source_runs, list):
                raise APGResultsError("The APG full-matrix result has no runs list")
            runs.extend(run for run in source_runs if isinstance(run, dict))
        else:
            matrix_dir = self.results_root / "matrix_runs"
            if matrix_dir.exists():
                for path in sorted(matrix_dir.glob("*.json")):
                    if path.name.endswith("_convergence.csv"):
                        continue
                    run = _read_json(path)
                    if "run_id" in run:
                        runs.append(run)
        ui_dir = self.results_root / "ui_runs"
        if ui_dir.exists():
            for path in sorted(ui_dir.glob("apg-ui-*.json")):
                runs.append(_read_json(path))
        return runs

    def find_run(self, run_id: str) -> dict[str, Any] | None:
        if not run_id or any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for character in run_id):
            return None
        ui_path = self.results_root / "ui_runs" / f"{run_id}.json"
        if ui_path.exists():
            return _read_json(ui_path)
        for run in self.load_runs():
            if run.get("run_id") == run_id:
                return run
        return None

    def _audit(self, job: APGJob, status: str) -> None:
        try:
            with self.sessions.begin() as session:
                session.add(
                    AutonomyExecutionEvent(
                        action="run_apg_simulation",
                        mode="FULL_ACCESS",
                        timestamp=datetime.now(timezone.utc),
                        result_status=status,
                        dataset_name=job.data_setting,
                        run_id=job.job_id,
                        details={
                            "topology": job.topology,
                            "data_setting": job.data_setting,
                            "seed": job.seed,
                            "source_label": "SIMULATED",
                        },
                    )
                )
        except Exception:
            LOGGER.warning(
                "Could not persist APG simulation audit event for job=%s status=%s",
                job.job_id,
                status,
                exc_info=True,
            )

    def _run_job(self, job: APGJob) -> None:
        started = time.perf_counter()
        with self._lock:
            if job.cancel_event.is_set() or self.stop_requested():
                job.status = "stopped"
                job.completed_at = datetime.now(timezone.utc).isoformat()
                job.duration_seconds = 0.0
                self._audit(job, "stopped")
                return
            job.status = "running"
            job.started_at = datetime.now(timezone.utc).isoformat()
            self._audit(job, "running")

        def should_stop() -> bool:
            return job.cancel_event.is_set() or self.stop_requested()

        def progress(completed: int, total: int) -> None:
            with self._lock:
                job.rounds_completed = completed
                job.total_rounds = total

        try:
            run = self.runner(
                job=job,
                should_stop=should_stop,
                progress_callback=progress,
            )
            if should_stop():
                raise APGJobStopped("Global STOP requested before TEST evaluation")
            if "final_test" not in run:
                raise APGResultsError("The completed APG run is missing its TEST evaluation")
            run["job_wall_seconds"] = time.perf_counter() - started
            run["source_label"] = "SIMULATED"
            run["simulation_mode"] = "all_simulated"
            output_dir = self.results_root / "ui_runs"
            output_dir.mkdir(parents=True, exist_ok=True)
            path = output_dir / f"{job.job_id}.json"
            serialized = json.dumps(run, indent=2, sort_keys=True, allow_nan=False) + "\n"
            with path.open("x", encoding="utf-8", newline="\n") as result_file:
                result_file.write(serialized)
            with self._lock:
                job.status = "finished"
                job.rounds_completed = job.total_rounds
                job.result_file = path.name
                job.completed_at = datetime.now(timezone.utc).isoformat()
                job.duration_seconds = time.perf_counter() - started
                self._audit(job, "finished")
        except (APGJobStopped, InterruptedError) as error:
            with self._lock:
                job.status = "stopped"
                job.error = str(error)
                job.completed_at = datetime.now(timezone.utc).isoformat()
                job.duration_seconds = time.perf_counter() - started
                self._audit(job, "stopped")
        except Exception as error:
            LOGGER.exception("APG simulation job failed: %s", job.job_id)
            with self._lock:
                job.status = "failed"
                job.error = f"{type(error).__name__}: {error}"
                job.completed_at = datetime.now(timezone.utc).isoformat()
                job.duration_seconds = time.perf_counter() - started
                self._audit(job, "failed")

    @staticmethod
    def _run_driver(
        *,
        job: APGJob,
        should_stop: Callable[[], bool],
        progress_callback: Callable[[int, int], None],
    ) -> dict[str, Any]:
        from app.core.config import REPO_ROOT

        repo_root = str(REPO_ROOT)
        if repo_root not in sys.path:
            sys.path.insert(0, repo_root)
        from scripts import run_apg_phase_b as driver

        driver.SIMULATED_PROFILE = driver._read_json(driver.PROFILE_PATH)
        report_path, _ = driver._latest_training_run()
        train_images, train_labels, train_hash = driver._load_train_only(report_path)
        architecture, initial_weights, model_hashes = driver._load_initial_model()
        seed_splits, shard_records = driver._make_data_setting(
            train_images,
            train_labels,
            job.data_setting,
        )
        seed_split = next(item for item in seed_splits if item["seed"] == job.seed)
        profile, _, bandwidths = driver._profiles()
        base_time = float(profile["reference_measurements"]["base_round_time"]["value_seconds"])
        if should_stop():
            raise APGJobStopped("Global STOP requested before APG training")
        run = driver._run_one(
            simulation_mode="all_simulated",
            data_setting=job.data_setting,
            topology=job.topology,
            seed=job.seed,
            nodes=seed_split["nodes"],
            architecture=architecture,
            initial_weights=initial_weights,
            apg_parameters=driver.APGParameters(),
            base_time_seconds=base_time,
            bandwidths=bandwidths,
            round_count=30,
            local_epochs=3,
            should_stop=should_stop,
            progress_callback=progress_callback,
            run_id_override=job.job_id,
        )
        if should_stop():
            raise APGJobStopped("Global STOP requested after APG training")
        run.pop("convergence_records", None)
        test_path = report_path.parent / "test.npz"
        test_info = driver._evaluate_test_once([run], architecture, test_path)
        run["test_data"] = test_info
        run["rounds_requested"] = 30
        run["local_epochs_per_round"] = 3
        run["test_split_opened"] = True
        run["train_npz_sha256"] = train_hash
        run["initial_model"] = model_hashes
        run["shard_class_counts"] = {
            record["node_id"]: record["shard_class_counts"] for record in shard_records
        }
        return run
