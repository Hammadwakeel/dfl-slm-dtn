import json
import time
from pathlib import Path

from master.training import TrainingConfig, TrainingCoordinator, TrainingWorker


class FakeEvaluator:
    def evaluate(self, models):
        return {}

    def close(self):
        pass


def make_config(tmp_path: Path) -> TrainingConfig:
    return TrainingConfig(
        workers=tuple(
            TrainingWorker(f"worker{index}", f"http://127.0.0.1:{5003 + index}", "x" * 32)
            for index in range(1, 4)
        ),
        evaluator_python="python",
        evaluator_script="evaluator.py",
        test_data_path="test.npz",
        train_data_path="train.npz",
        evaluator_log_path=str(tmp_path / "evaluator.log"),
        results_path=str(tmp_path / "experiments.json"),
    )


def test_failed_distributed_run_is_persisted_as_failed(tmp_path: Path) -> None:
    coordinator = TrainingCoordinator(make_config(tmp_path), evaluator=FakeEvaluator())

    def fail_before_rounds(group_id, group, run):
        raise RuntimeError("worker disconnected")

    coordinator._run_distributed_seed = fail_before_rounds
    accepted = coordinator.start_training(rounds=1, epochs=1)
    deadline = time.monotonic() + 3
    while coordinator.results()["active_group_id"] is not None and time.monotonic() < deadline:
        time.sleep(0.01)

    group = next(item for item in coordinator.results()["groups"] if item["group_id"] == accepted["group_id"])
    assert group["status"] == "failed"
    assert group["runs"][0]["status"] == "failed"
    assert group["runs"][0]["error"] == "RuntimeError: worker disconnected"

    persisted = json.loads((tmp_path / "experiments.json").read_text(encoding="utf-8"))
    assert persisted[0]["runs"][0]["status"] == "failed"


def test_optimizer_runs_real_candidate_sweep_and_persists_reward_selection(tmp_path: Path) -> None:
    coordinator = TrainingCoordinator(make_config(tmp_path), evaluator=FakeEvaluator())

    def record_candidate_run(group_id, group, run):
        accuracy = {
            "directed_ring": 0.54,
            "bidirectional_ring": 0.55,
            "mesh": 0.56,
            "chain": 0.57,
            "star": 0.58,
        }[group["topology"]]
        row = {
            "round": 1,
            "workers": {
                worker.node_id: {
                    "accuracy": accuracy,
                    "bytes_sent": 100,
                    "exchange_seconds": 0.1,
                    "round_seconds": 2,
                }
                for worker in coordinator.config.workers
            },
        }
        run["rounds"].append(row)

    coordinator._run_distributed_seed = record_candidate_run
    accepted = coordinator.start_optimization(
        rounds=1,
        epochs=1,
        seeds=(17,),
        alphas=(0.3,),
    )
    deadline = time.monotonic() + 3
    while coordinator.results()["active_group_id"] is not None and time.monotonic() < deadline:
        time.sleep(0.01)

    group = next(
        item
        for item in coordinator.results()["groups"]
        if item["group_id"] == accepted["group_id"]
    )
    assert group["status"] == "completed"
    assert len(group["candidates"]) == 5
    assert group["selection"]["selected_candidate"]["topology"] == "directed_ring"
    assert group["selection"]["baselines"]["best_observed"]["topology"] == "star"
    assert group["reward_config"]["weights"]["accuracy_gain"] == 0.5
    assert coordinator.results()["configuration"]["dataset_split"] == "dirichlet_a03"
