"""Shared loading of fair-campaign evidence, honouring docs/evidence/excluded-runs.json.

A run listed with scope "run" is left out entirely; one with scope "baseline" keeps
everything except its baseline value (see excluded-runs.json for why).
"""

from __future__ import annotations

import json
from pathlib import Path

EVIDENCE = Path(__file__).resolve().parents[1] / "docs" / "evidence"


def _excluded() -> dict[str, str]:
    path = EVIDENCE / "excluded-runs.json"
    if not path.is_file():
        return {}
    return {run_id: item["scope"] for run_id, item in json.loads(path.read_text(encoding="utf-8"))["runs"].items()}


EXCLUDED = _excluded()


def run_excluded(run: dict) -> bool:
    return EXCLUDED.get(str(run.get("run_id"))) == "run"


def baseline_excluded(run: dict) -> bool:
    return str(run.get("run_id")) in EXCLUDED


# Trainings that use the baseline's budget (rounds x local epochs from the run's initial
# weights). Pruning fine-tunes already trained weights, so it is extra training.
EQUAL_BUDGET_STEPS = ("baseline_training", "measure_step", "learn_topology")


def fair_final(run: dict) -> dict:
    """The run's best equal-budget model, chosen by global validation accuracy like the supervisor.

    Returns {"test", "validation", "step", "parameter_count"}; a run whose best model is a
    pruned and fine-tuned one is scored by its best model before pruning.
    """
    steps = [
        s for s in run["steps"]
        if s["step"] in EQUAL_BUDGET_STEPS and s.get("status", "completed") == "completed"
        and isinstance(s.get("validation_accuracy"), (int, float)) and isinstance(s.get("test_accuracy"), (int, float))
    ]
    best = max(steps, key=lambda s: s["validation_accuracy"])
    return {
        "test": best["test_accuracy"],
        "validation": best["validation_accuracy"],
        "step": best.get("measures") or best["step"],
        "parameter_count": best.get("parameter_count"),
    }


def campaign_runs(dataset: str, split: str, date: str | None = None) -> list[dict]:
    """Completed fair-campaign runs of one setting, newest report per seed, sorted by seed.

    Each run gains ``final_test_accuracy`` (and validation, step, parameter count): its best
    equal-budget model (``fair_final``), which the comparison with the baseline uses.
    ``best_test_accuracy`` may be a pruned model that was fine-tuned for extra rounds.

    Reads ``fair-<dataset>-<split>-3seed-<date>.json`` (any date when ``date`` is None)
    and the extra-seed and re-run reports ``fair-<dataset>-<split>-{extra-seeds,rerun}-*.json``.
    """
    main = f"fair-{dataset}-{split}-3seed-{date}.json" if date else f"fair-{dataset}-{split}-3seed-*.json"
    paths = sorted(EVIDENCE.glob(main)) + sorted(EVIDENCE.glob(f"fair-{dataset}-{split}-extra-seeds-*.json")) + sorted(
        EVIDENCE.glob(f"fair-{dataset}-{split}-rerun-*.json")
    )
    by_seed: dict[int, dict] = {}
    for path in paths:
        for run in json.loads(path.read_text(encoding="utf-8"))["runs"]:
            if run["status"] == "completed" and not run_excluded(run):
                final = fair_final(run)
                by_seed[run["seed"]] = {
                    **run,
                    "final_test_accuracy": final["test"],
                    "final_validation_accuracy": final["validation"],
                    "final_step": final["step"],
                    "final_parameter_count": final["parameter_count"] or run.get("best_parameter_count"),
                }
    return [by_seed[seed] for seed in sorted(by_seed)]
