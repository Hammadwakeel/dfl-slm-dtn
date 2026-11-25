"""How many of the fair campaign's keep-or-revert decisions stand out from noise?

    python scripts/decision_robustness.py --date 2026-11-21

In fair mode the supervisor keeps a search step's change when its equal-budget
measurement is at least the best validation accuracy so far minus 0.5 points. The
same configuration trained twice with the same seed does not give the same accuracy
(thread scheduling, network timing and semi-synchronous deadlines are not
deterministic), so a decision is only informative when the measured difference is
larger than that run-to-run noise.

This script estimates the noise per dataset and Dirichlet split from the repeated
trainings (the campaign baseline plus every ablation pass's baseline and full
configuration, same seed): the pooled within-seed standard deviation sigma of the
global validation accuracy. A decision compares two such measurements, so its noise
is sigma * sqrt(2); a difference beyond two of those (about a 95 % band) is called
clear. It writes ``docs/evidence/decision-robustness-<date>.md``.

It also gives a paired test of the components that only one seed kept (HPO, sync):
each ablation pass trained the full configuration and the configuration without the
component back to back, so their per-pass differences are paired samples.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evidence_runs import baseline_excluded, campaign_runs  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs" / "evidence"
DATASETS = {"fashion_mnist": "Fashion-MNIST", "femnist": "FEMNIST"}
SPLITS = {"dirichlet_a01": "0.1", "dirichlet_a05": "0.5", "dirichlet_a09": "0.9"}
STEPS = {
    "run_nas": "NAS",
    "run_hpo": "HPO",
    "optimize_topology": "topology",
    "optimize_aggregation": "aggregation",
    "optimize_sync": "sync",
}
TOLERANCE = 0.005
# Two-sided 97.5 % Student t quantiles by degrees of freedom.
T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 8: 2.306, 10: 2.228, 20: 2.086}


def t975(dof: int) -> float:
    eligible = [key for key in T975 if key <= dof]
    return T975[max(eligible)] if eligible else float("inf")


def load_campaign(date: str) -> dict[tuple[str, str], list[dict[str, Any]]]:
    runs: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for dataset in DATASETS:
        for split in SPLITS:
            found = campaign_runs(dataset, split, date)
            if found:
                runs[(dataset, split)] = found
    return runs


def load_ablations(date: str) -> list[dict[str, Any]]:
    runs = []
    for path in sorted(EVIDENCE.glob(f"fair-ablation-{date}*.json")):
        runs += [r for r in json.loads(path.read_text(encoding="utf-8"))["runs"] if r["status"] == "completed"]
    return runs


def pooled_sigma(campaign: dict, ablations: list[dict]) -> dict[tuple[str, str], tuple[float, int]]:
    """Pooled within-seed std of validation accuracy, and its degrees of freedom."""
    groups: dict[tuple[str, str, int, str], list[float]] = defaultdict(list)
    for (dataset, split), runs in campaign.items():
        for run in runs:
            groups[(dataset, split, run["seed"], "baseline")].append(run["baseline_validation_accuracy"])
    for run in ablations:
        for label in ("baseline", "full"):
            if label == "baseline" and baseline_excluded(run):
                continue
            value = run["accuracy"].get(label, {}).get("validation")
            if value is not None:
                groups[(run["dataset"], run["split"], run["seed"], label)].append(value)
    sigma: dict[tuple[str, str], tuple[float, int]] = {}
    for dataset in DATASETS:
        for split in SPLITS:
            squares, dof = 0.0, 0
            for (d, s, _seed, _label), values in groups.items():
                if (d, s) == (dataset, split) and len(values) >= 2:
                    squares += statistics.variance(values) * (len(values) - 1)
                    dof += len(values) - 1
            if dof:
                sigma[(dataset, split)] = (math.sqrt(squares / dof), dof)
    return sigma


def decisions(run: dict[str, Any]) -> list[dict[str, Any]]:
    """Each measured step with the best validation accuracy it was compared with."""
    best = run["baseline_validation_accuracy"]
    rows = []
    for step in run["steps"]:
        if step["step"] != "measure_step" or step.get("validation_accuracy") is None:
            continue
        value = step["validation_accuracy"]
        rows.append({"step": step["measures"], "accuracy": value, "best": best, "delta": value - best, "kept": bool(step.get("accepted"))})
        if step.get("accepted") and value > best:
            best = value
    return rows


def paired_components(ablations: list[dict]) -> list[dict[str, Any]]:
    """Per (dataset, split, seed, component): full minus without, one value per ablation pass."""
    groups: dict[tuple[str, str, int, str], list[float]] = defaultdict(list)
    for run in ablations:
        full = run["accuracy"].get("full", {}).get("test")
        for label in ("without HPO", "without sync", "without aggregation"):
            other = run["accuracy"].get(label, {}).get("test")
            if full is not None and other is not None:
                groups[(run["dataset"], run["split"], run["seed"], label)].append(full - other)
    rows = []
    for (dataset, split, seed, label), deltas in sorted(groups.items()):
        mean = statistics.fmean(deltas)
        if len(deltas) >= 2:
            half = t975(len(deltas) - 1) * statistics.stdev(deltas) / math.sqrt(len(deltas))
        else:
            half = None
        rows.append({"dataset": dataset, "split": split, "seed": seed, "component": label.replace("without ", ""), "deltas": deltas, "mean": mean, "half_width": half})
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--date", default="2026-11-21")
    args = parser.parse_args()
    campaign = load_campaign(args.date)
    ablations = load_ablations(args.date)
    sigma = pooled_sigma(campaign, ablations)

    pct = lambda value: f"{100 * value:.2f}"  # noqa: E731
    lines = [
        "# Robustness of the keep-or-revert decisions",
        "",
        f"Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} by `scripts/decision_robustness.py`.",
        "",
        "The fair campaign keeps a step's change when its measurement is at least the best global",
        f"validation accuracy so far minus {100 * TOLERANCE:.1f} points. σ is the pooled within-seed standard",
        "deviation of global validation accuracy over repeated trainings of the same configuration with",
        "the same seed (campaign baseline and every ablation pass's baseline and full configuration). A",
        "decision compares two measurements, so its noise band is 2·σ·√2 (about 95 %). A decision is",
        "*clear* when the measured difference lies outside that band and *within noise* otherwise.",
        "",
        "## Noise and decisions per setting",
        "",
        "| Dataset | α | σ (points) | dof | 95 % band (points) | Decisions | Kept | Reverted | Clear gain | Clear loss | Within noise |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    detail = []
    totals = defaultdict(int)
    for (dataset, split), runs in campaign.items():
        s, dof = sigma.get((dataset, split), (float("nan"), 0))
        band = 2 * s * math.sqrt(2)
        counts = defaultdict(int)
        for run in runs:
            for row in decisions(run):
                verdict = "clear gain" if row["delta"] > band else "clear loss" if row["delta"] < -band else "within noise"
                counts["decisions"] += 1
                counts["kept" if row["kept"] else "reverted"] += 1
                counts[verdict] += 1
                detail.append((DATASETS[dataset], SPLITS[split], run["seed"], STEPS.get(row["step"], row["step"]), row, verdict))
        for key, value in counts.items():
            totals[key] += value
        lines.append(
            f"| {DATASETS[dataset]} | {SPLITS[split]} | {pct(s)} | {dof} | ±{pct(band)} | {counts['decisions']} | {counts['kept']} | "
            f"{counts['reverted']} | {counts['clear gain']} | {counts['clear loss']} | {counts['within noise']} |"
        )
    lines.append(
        f"| **All** | | | | | {totals['decisions']} | {totals['kept']} | {totals['reverted']} | {totals['clear gain']} | "
        f"{totals['clear loss']} | {totals['within noise']} |"
    )
    by_step: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for *_rest, step, row, verdict in detail:
        by_step[step][verdict] += 1
        by_step[step]["kept within noise" if row["kept"] and verdict == "within noise" else "other"] += 0
    lines += [
        "",
        "## Decisions per step",
        "",
        "| Step | Decisions | Clear gain | Clear loss | Within noise |",
        "|---|---|---|---|---|",
    ]
    for step in STEPS.values():
        counts = by_step.get(step)
        if counts:
            total = counts["clear gain"] + counts["clear loss"] + counts["within noise"]
            lines.append(f"| {step} | {total} | {counts['clear gain']} | {counts['clear loss']} | {counts['within noise']} |")
    lines += [
        "",
        "## Paired test of the components a single seed kept",
        "",
        "Each ablation pass trains the full configuration and the configuration without the component",
        "back to back with the same seed and initial weights. The table gives the per-pass differences",
        "(full minus without, test points), their mean and a 95 % t interval when there are two or more",
        "passes. An interval that contains 0 means the component's effect is not distinguishable from noise.",
        "",
        "| Dataset | α | Seed | Component | Per-pass differences | Mean | 95 % interval |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in paired_components(ablations):
        interval = "n = 1" if row["half_width"] is None else f"[{100 * (row['mean'] - row['half_width']):+.2f}, {100 * (row['mean'] + row['half_width']):+.2f}]"
        lines.append(
            f"| {DATASETS[row['dataset']]} | {SPLITS[row['split']]} | {row['seed']} | {row['component']} | "
            + ", ".join(f"{100 * d:+.2f}" for d in row["deltas"])
            + f" | {100 * row['mean']:+.2f} | {interval} |"
        )
    lines += [
        "",
        "## Every decision",
        "",
        "| Dataset | α | Seed | Step | Measured (%) | Best before (%) | Difference (points) | Kept | Verdict |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for dname, alpha, seed, step, row, verdict in detail:
        lines.append(
            f"| {dname} | {alpha} | {seed} | {step} | {pct(row['accuracy'])} | {pct(row['best'])} | "
            f"{100 * row['delta']:+.2f} | {'yes' if row['kept'] else 'no'} | {verdict} |"
        )
    out = EVIDENCE / f"decision-robustness-{args.date}.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
