"""Paper tables across the fair step-by-step campaigns (all datasets, alphas and seeds).

    python scripts/fair_summary.py --date 2026-11-21

Reads every ``docs/evidence/fair-<dataset>-<split>-3seed-<date>.json`` written by
``scripts/paper_campaign.py --measure-each-step`` and writes one Markdown summary.

Every model in these runs is trained with the same budget as the baseline (same
rounds x local epochs, same installed initial weights). A step's effect is measured
against the configuration that was active just before it: the baseline, or the last
measurement whose change was kept. The supervisor reverts a change whose measurement
falls more than 0.5 points below the best so far, so the active configuration never
includes a harmful change. Steps run in the order the on-device SLM chose, so effects
are grouped by step type, not by position.
"""

from __future__ import annotations

import argparse
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs" / "evidence"
DATASETS = (("fashion_mnist", "Fashion-MNIST"), ("femnist", "FEMNIST"))
SPLITS = (("dirichlet_a01", "0.1"), ("dirichlet_a05", "0.5"), ("dirichlet_a09", "0.9"))
STEPS = (
    ("run_nas", "NAS"),
    ("optimize_topology", "Topology sweep"),
    ("optimize_aggregation", "Aggregation sweep"),
    ("run_hpo", "HPO"),
    ("optimize_sync", "Synchronization sweep"),
    ("learn_topology", "Learned topology (APG)"),
)


def _pct(values: list[float]) -> str:
    if not values:
        return "n/a"
    mean = 100 * statistics.fmean(values)
    if len(values) == 1:
        return f"{mean:.2f}"
    return f"{mean:.2f} ± {100 * statistics.stdev(values):.2f}"


def _signed(values: list[float]) -> str:
    if not values:
        return "n/a"
    mean = 100 * statistics.fmean(values)
    text = f"{mean:+.2f}"
    return text if len(values) == 1 else f"{text} ± {100 * statistics.stdev(values):.2f}"


def step_effects(run: dict) -> dict[str, dict]:
    """Effect of each step type relative to the configuration active before it."""
    effects: dict[str, dict] = {}
    active_val = active_test = None
    for step in run["steps"]:
        if step.get("status") != "completed" or not isinstance(step.get("validation_accuracy"), (int, float)):
            continue
        name = step["step"]
        val, test = step["validation_accuracy"], step.get("test_accuracy")
        if name == "baseline_training":
            active_val, active_test = val, test
            continue
        kind = step.get("measures") if name == "measure_step" else name
        if kind not in dict(STEPS) or active_val is None:
            continue
        effects[kind] = {
            "d_val": val - active_val,
            "d_test": test - active_test if isinstance(test, (int, float)) and isinstance(active_test, (int, float)) else None,
            "kept": step.get("accepted") is not False if name == "measure_step" else None,
            "choice": step.get("architecture") if kind in {"run_nas", "run_hpo"} else None,
        }
        if name == "measure_step" and step.get("accepted") is not False:
            active_val, active_test = val, test
    return effects


def load(date: str) -> dict[tuple[str, str], list[dict]]:
    found = {}
    for dataset, _ in DATASETS:
        for split, _ in SPLITS:
            path = EVIDENCE / f"fair-{dataset}-{split}-3seed-{date}.json"
            if path.is_file():
                runs = json.loads(path.read_text(encoding="utf-8"))["runs"]
                found[(dataset, split)] = sorted((r for r in runs if r["status"] == "completed"), key=lambda r: r["seed"])
    return found


def build(date: str) -> str:
    data = load(date)
    lines = [
        "# Fair step-by-step evaluation: Fashion-MNIST and FEMNIST",
        "",
        f"Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} by `scripts/fair_summary.py`.",
        "",
        "Eight Jetson workers, decentralized training, Dirichlet label split with α = 0.1, 0.5 and 0.9,",
        "seeds 17, 42 and 2026. Every trained model (baseline and each step's measurement) uses the",
        "same budget: 10 rounds × 5 local epochs from the same installed initial weights. Accuracy is",
        "the mean over the 8 workers on the dataset's shared test split. Values are mean ± std over seeds.",
        "",
        "## Baseline vs. final model (same training budget)",
        "",
        "| Dataset | α | Seeds | Baseline test (%) | Final model test (%) | Gain (points) | Final params |",
        "|---|---|---|---|---|---|---|",
    ]
    for dataset, dname in DATASETS:
        for split, alpha in SPLITS:
            runs = data.get((dataset, split), [])
            if not runs:
                continue
            base = [r["baseline_test_accuracy"] for r in runs if isinstance(r.get("baseline_test_accuracy"), (int, float))]
            best = [r["best_test_accuracy"] for r in runs if isinstance(r.get("best_test_accuracy"), (int, float))]
            gains = [r["best_test_accuracy"] - r["baseline_test_accuracy"] for r in runs
                     if isinstance(r.get("best_test_accuracy"), (int, float)) and isinstance(r.get("baseline_test_accuracy"), (int, float))]
            params = sorted({f"{r['best_parameter_count']:,}" for r in runs if r.get("best_parameter_count")})
            lines.append(f"| {dname} | {alpha} | {len(runs)} | {_pct(base)} | {_pct(best)} | {_signed(gains)} | {', '.join(params)} |")

    lines += [
        "",
        "## Effect of each step",
        "",
        "Change in validation and test accuracy (points) caused by each step, measured against the",
        "configuration active just before it. \"Kept\" counts seeds where the change was kept; a change",
        "more than 0.5 points below the best so far is reverted. The learned topology is the last step",
        "and is reported without a keep/revert decision.",
        "",
    ]
    for dataset, dname in DATASETS:
        lines += [f"### {dname}", "", "| Step | " + " | ".join(f"α {a}: Δ val" + f" | α {a}: Δ test | α {a}: kept" for _, a in SPLITS) + " |",
                  "|---|" + "---|" * (3 * len(SPLITS))]
        per_split = {split: [step_effects(r) for r in data.get((dataset, split), [])] for split, _ in SPLITS}
        for kind, label in STEPS:
            cells = []
            for split, _ in SPLITS:
                effects = [e[kind] for e in per_split[split] if kind in e]
                d_val = [e["d_val"] for e in effects]
                d_test = [e["d_test"] for e in effects if e["d_test"] is not None]
                kept = [e["kept"] for e in effects if e["kept"] is not None]
                cells += [_signed(d_val), _signed(d_test), f"{sum(kept)}/{len(kept)}" if kept else "–"]
            lines.append(f"| {label} | " + " | ".join(cells) + " |")
        lines.append("")

    lines += ["## Per-seed accuracy after each step (validation %, test %)", ""]
    for dataset, dname in DATASETS:
        for split, alpha in SPLITS:
            runs = data.get((dataset, split), [])
            if not runs:
                continue
            lines += [f"### {dname}, α {alpha}", "", "| Seed | Order of steps (validation % → kept?) | Final test |", "|---|---|---|"]
            for run in runs:
                parts = []
                for step in run["steps"]:
                    if step.get("status") != "completed" or not isinstance(step.get("validation_accuracy"), (int, float)):
                        continue
                    name = step["step"]
                    kind = step.get("measures") if name == "measure_step" else name
                    label = "Baseline" if name == "baseline_training" else dict(STEPS).get(kind)
                    if label is None:
                        continue
                    mark = " (reverted)" if step.get("accepted") is False else ""
                    parts.append(f"{label} {100 * step['validation_accuracy']:.2f}{mark}")
                best = run.get("best_test_accuracy")
                lines.append(f"| {run['seed']} | {' → '.join(parts)} | {100 * best:.2f} |" if isinstance(best, (int, float)) else f"| {run['seed']} | {' → '.join(parts)} | n/a |")
            lines.append("")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--date", default="2026-11-21")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    out = args.out or EVIDENCE / f"fair-summary-{args.date}.md"
    out.write_text(build(args.date), encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
