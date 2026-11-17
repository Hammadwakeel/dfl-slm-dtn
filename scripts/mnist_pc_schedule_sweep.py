"""PC-only schedule sensitivity for the real MNIST worker shards."""

from __future__ import annotations

import json
from pathlib import Path

from mnist_pc_training_check import (
    OUTPUT_PATH,
    _dataset,
    _simulate_worker_schedule,
)

ROOT = Path(__file__).resolve().parents[1]
SWEEP_PATH = ROOT / "data" / "mnist_pc_schedule_sweep.json"


def main() -> None:
    _, _, _, _, per_worker = _dataset()
    baselines = json.loads(OUTPUT_PATH.read_text(encoding="utf-8"))
    previous = baselines["federated_worker_schedule"]
    schedules = [
        (5, 10),
        (10, 5),
        (5, 20),
    ]
    rows = [
        {
            "rounds": previous["rounds"],
            "local_epochs_per_round": previous["local_epochs_per_round"],
            "train_steps_per_worker": sum(previous["train_rows_per_worker"])
            // previous["workers"]
            // previous["batch_size"]
            * previous["rounds"]
            * previous["local_epochs_per_round"],
            "mean_final_accuracy": previous["mean_final_accuracy"],
            "mean_final_macro_f1": previous["mean_final_macro_f1"],
            "final_worker_metrics": previous["final_worker_metrics"],
            "source": "initial 5x2 PC run",
        }
    ]
    for rounds, local_epochs in schedules:
        print(
            f"PC schedule sweep starting: rounds={rounds}; local_epochs={local_epochs}; "
            f"optimizer_steps_per_worker={rounds * local_epochs * 5}"
        )
        result = _simulate_worker_schedule(
            per_worker,
            seed=2026,
            rounds=rounds,
            local_epochs=local_epochs,
        )
        result["train_steps_per_worker"] = rounds * local_epochs * 5
        result["source"] = "PC-only weighted-averaging simulation"
        rows.append(result)
        SWEEP_PATH.write_text(
            json.dumps({"schedules": rows}, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    summary = {
        "reason": "Find a PC-validated schedule that rises above the 5x2 10% result.",
        "chance_accuracy": 0.1,
        "schedules": rows,
    }
    SWEEP_PATH.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("schedule_summary=" + json.dumps(summary, sort_keys=True))
    print(f"schedule_sweep_json={SWEEP_PATH}")


if __name__ == "__main__":
    main()
