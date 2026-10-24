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
