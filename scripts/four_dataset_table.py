"""One table of the fair (equal-budget) campaign across MNIST, Fashion-MNIST, FEMNIST and CIFAR-10.

    python scripts/four_dataset_table.py

Reads every ``docs/evidence/fair-<dataset>-<split>-3seed-<date>.json`` (any date) written
by ``scripts/paper_campaign.py --measure-each-step``, plus extra-seed reports
``fair-<dataset>-<split>-extra-seeds-<date>.json``, and writes
``docs/evidence/fair-four-datasets.md``. Every trained model in these runs uses the same
budget (10 rounds x 5 local epochs from the same initial weights); accuracy is the mean over
the 8 workers on the dataset's official test split; values are mean ± std over seeds.
"""

from __future__ import annotations

import json
import statistics
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evidence_runs import campaign_runs  # noqa: E402
from fair_summary import STEPS, _pct, _signed, step_effects  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs" / "evidence"
DATASETS = (("mnist", "MNIST"), ("fashion_mnist", "Fashion-MNIST"), ("femnist", "FEMNIST"), ("cifar10", "CIFAR-10"))
SPLITS = (("dirichlet_a01", "0.1"), ("dirichlet_a05", "0.5"), ("dirichlet_a09", "0.9"))
OUT = EVIDENCE / "fair-four-datasets.md"


def load() -> dict[tuple[str, str], list[dict]]:
    """Completed runs per (dataset, split), with extra seeds and re-runs, minus excluded runs."""
    found = {}
    for dataset, _ in DATASETS:
        for split, _ in SPLITS:
            runs = campaign_runs(dataset, split)
            if runs:
                found[(dataset, split)] = runs
    return found


def _num(value) -> bool:
    return isinstance(value, (int, float))


def build() -> str:
    data = load()
    lines = [
        "# Fair evaluation across four datasets",
        "",
        f"Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} by `scripts/four_dataset_table.py`.",
        "",
        "Eight Jetson workers, decentralized training, Dirichlet label skew α (smaller = more skewed).",
        "The on-device SLM chooses the order of the search steps; every model it compares, the baseline",
        "included, is trained with the same budget (10 rounds × 5 local epochs, same initial weights).",
        "Test accuracy is the mean over the 8 workers on the dataset's official test split, mean ± std",
        "over seeds (n = number of seeds). Gain = final model minus baseline, per seed.",
        "",
        "## Baseline vs. final model",
        "",
        "| Dataset | α | n | Baseline test (%) | Final test (%) | Gain (points) | Selected topology | Final params |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for dataset, dname in DATASETS:
        for split, alpha in SPLITS:
            runs = data.get((dataset, split), [])
            if not runs:
                lines.append(f"| {dname} | {alpha} | 0 | pending | pending | pending | – | – |")
                continue
            base = [r["baseline_test_accuracy"] for r in runs if _num(r.get("baseline_test_accuracy"))]
            best = [r["best_test_accuracy"] for r in runs if _num(r.get("best_test_accuracy"))]
            gains = [r["best_test_accuracy"] - r["baseline_test_accuracy"] for r in runs
                     if _num(r.get("best_test_accuracy")) and _num(r.get("baseline_test_accuracy"))]
            topologies = Counter(r.get("best_topology") or "?" for r in runs)
            topology = ", ".join(f"{name} {count}/{len(runs)}" for name, count in topologies.most_common())
            params = [r["best_parameter_count"] for r in runs if r.get("best_parameter_count")]
            params_text = f"{statistics.fmean(params) / 1000:.0f}k (mean)" if params else "–"
            lines.append(f"| {dname} | {alpha} | {len(runs)} | {_pct(base)} | {_pct(best)} | {_signed(gains)} | {topology} | {params_text} |")
    lines += [
        "",
        "## Effect of each step (test points)",
        "",
        "Change in test accuracy caused by each step, against the configuration active just before it",
        "(the baseline or the last kept change); kept = seeds where the change was kept. The learned",
        "topology (APG) is measured last, without a keep/revert decision.",
        "",
        "| Dataset | α | " + " | ".join(label for _, label in STEPS) + " |",
        "|---|---|" + "---|" * len(STEPS),
    ]
    for dataset, dname in DATASETS:
        for split, alpha in SPLITS:
            runs = data.get((dataset, split), [])
            if not runs:
                continue
            effects = [step_effects(r) for r in runs]
            cells = []
            for kind, _label in STEPS:
                items = [e[kind] for e in effects if kind in e]
                d_test = [item["d_test"] for item in items if item["d_test"] is not None]
                kept = [item["kept"] for item in items if item["kept"] is not None]
                cell = _signed(d_test)
                if kept:
                    cell += f" ({sum(kept)}/{len(kept)} kept)"
                cells.append(cell)
            lines.append(f"| {dname} | {alpha} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "## Per seed (test %)",
        "",
        "| Dataset | α | Seed | Baseline | Final | Final configuration |",
        "|---|---|---|---|---|---|",
    ]
    for dataset, dname in DATASETS:
        for split, alpha in SPLITS:
            for run in data.get((dataset, split), []):
                base, best = run.get("baseline_test_accuracy"), run.get("best_test_accuracy")
                config = f"{run.get('best_architecture') or '?'} on {run.get('best_topology') or '?'}"
                lines.append(
                    f"| {dname} | {alpha} | {run['seed']} | {100 * base:.2f} | {100 * best:.2f} | {config} |"
                    if _num(base) and _num(best) else f"| {dname} | {alpha} | {run['seed']} | n/a | n/a | {config} |"
                )
    return "\n".join(lines) + "\n"


def main() -> None:
    OUT.write_text(build(), encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
