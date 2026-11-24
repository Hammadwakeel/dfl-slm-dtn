"""Is the mesh's gain topology, or just more communication?

    python scripts/topology_degree_study.py --date 2026-11-21

The ablation found that the selected mesh beats the baseline's directed ring by the
largest margin of any component. But in a mesh every worker receives 7 models per
round and in the ring only 1, so the gain could come from communication volume
rather than from the graph. For every finished fair run at the given splits this
script retrains the run's final configuration (architecture, hyperparameters,
aggregation and synchronization unchanged) on:

* ``in-degree 2`` and ``in-degree 4``: circulant graphs where worker i receives from
  workers i-1 .. i-k, the same compute and 2 or 4 models per worker per round;
* ``ring 70x1``: the directed ring for 70 rounds of 1 local epoch, so every worker
  receives exactly as many models as in the mesh (70) with 70 local epochs instead of 50.

In-degree 1 (the directed ring) and 7 (the mesh) come from the ablation passes. Each
graph's mixing matrix is also characterized by its spectral gap (1 - |lambda_2| of the
uniform averaging matrix), which bounds how fast decentralized averaging reaches
consensus. Results go to ``docs/evidence/topology-degree-<date>.json`` / ``.md``;
re-running resumes finished seeds. The API token is read from ``.env`` and never printed.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ablation_campaign import _accuracy, ablation_configs  # noqa: E402
from paper_campaign import Api, api_token, log, summarize_run, wait_for_run  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs" / "evidence"
DATASETS = {"fashion_mnist": "Fashion-MNIST", "femnist": "FEMNIST"}
ALPHAS = {"dirichlet_a01": "0.1", "dirichlet_a05": "0.5", "dirichlet_a09": "0.9"}
WORKERS = tuple(f"worker{index}" for index in range(1, 9))
ROUNDS, EPOCHS = 10, 5
# label -> (in-degree per round, rounds, local epochs)
GRAPHS = {
    "in-degree 1 (ring)": (1, ROUNDS, EPOCHS),
    "in-degree 2": (2, ROUNDS, EPOCHS),
    "in-degree 4": (4, ROUNDS, EPOCHS),
    "in-degree 7 (mesh)": (7, ROUNDS, EPOCHS),
    "ring 70x1": (1, 70, 1),
}
NEW = ("in-degree 2", "in-degree 4", "ring 70x1")


def circulant(k: int) -> dict[str, list[str]]:
    """Worker i receives from workers i-1 .. i-k (indices modulo 8)."""
    n = len(WORKERS)
    return {WORKERS[i]: [WORKERS[(i - offset) % n] for offset in range(1, k + 1)] for i in range(n)}


def spectral_gap(k: int) -> float:
    """1 - second-largest eigenvalue modulus of uniform averaging over self and k senders."""
    n = len(WORKERS)
    mixing = np.zeros((n, n))
    for i in range(n):
        for offset in range(0, k + 1):
            mixing[i, (i - offset) % n] = 1.0 / (k + 1)
    moduli = sorted(np.abs(np.linalg.eigvals(mixing)), reverse=True)
    return float(1.0 - moduli[1])


def study_configs(full: dict[str, Any]) -> list[dict[str, Any]]:
    base = {key: full[key] for key in ("architecture", "aggregation", "sync_mode")}
    return [
        {"label": "in-degree 2", **base, "topology": "learned", "senders": circulant(2)},
        {"label": "in-degree 4", **base, "topology": "learned", "senders": circulant(4)},
        {"label": "ring 70x1", **base, "topology": "directed_ring", "rounds": 70, "local_epochs": 1},
    ]


def ablation_tests(dataset: str, split: str, seed: int, date: str) -> dict[str, list[float]]:
    """Test accuracies of the full (mesh) and without-topology (ring) trainings in every ablation pass."""
    found: dict[str, list[float]] = {"in-degree 7 (mesh)": [], "in-degree 1 (ring)": []}
    for path in sorted(EVIDENCE.glob(f"fair-ablation-{date}*.json")):
        for run in json.loads(path.read_text(encoding="utf-8"))["runs"]:
            if (run["dataset"], run["split"], run["seed"], run["status"]) != (dataset, split, seed, "completed"):
                continue
            if run["final_config"]["topology"] != "mesh":
                continue
            for label, key in (("in-degree 7 (mesh)", "full"), ("in-degree 1 (ring)", "without topology")):
                test = run["accuracy"].get(key, {}).get("test")
                if test is not None:
                    found[label].append(test)
    return found


def _fmt(values: list[float]) -> str:
    if not values:
        return "–"
    mean = 100 * statistics.fmean(values)
    return f"{mean:.2f}" if len(values) == 1 else f"{mean:.2f} ± {100 * statistics.stdev(values):.2f}"


def write_report(path: Path, results: list[dict[str, Any]], date: str) -> None:
    lines = [
        "# Topology or communication volume? In-degree study",
        "",
        f"Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} by `scripts/topology_degree_study.py`.",
        "",
        "Every seed's final fair-campaign configuration (architecture, hyperparameters, aggregation and",
        "synchronization unchanged) is trained on directed graphs of increasing in-degree with the same",
        f"budget ({ROUNDS} rounds x {EPOCHS} local epochs, same seed and initial weights). In-degree k means",
        "worker i averages its model with the models of workers i-1 .. i-k each round (a circulant graph);",
        "k = 1 is the baseline's directed ring and k = 7 the mesh. `ring 70x1` is the ring with 70 rounds of",
        "1 local epoch: every worker receives as many models as in the mesh (70), with more local compute",
        "(70 epochs instead of 50). Models received counts model transfers into one worker over the run;",
        "bytes scale with it times the model size. The spectral gap (1 - |λ2| of the uniform averaging",
        "matrix) measures how fast one round of averaging mixes information across the 8 workers.",
        f"In-degree 1 and 7 come from the leave-one-out ablation passes (`fair-ablation-{date}*.json`), so",
        "they are averaged over every repeat. Accuracy is the mean test accuracy over the 8 workers (%).",
        "",
        "## Mean over seeds",
        "",
        "| Dataset | α | Graph | Models received per worker | Spectral gap | Test accuracy (%) | Seeds |",
        "|---|---|---|---|---|---|---|",
    ]
    completed = [r for r in results if r["status"] == "completed"]
    for dataset, name in DATASETS.items():
        for split, alpha in ALPHAS.items():
            rows = [r for r in completed if (r["dataset"], r["split"]) == (dataset, split)]
            if not rows:
                continue
            for label, (degree, rounds, _epochs) in GRAPHS.items():
                per_seed = [statistics.fmean(r["tests"][label]) for r in rows if r["tests"].get(label)]
                gap = spectral_gap(degree) if rounds == ROUNDS else spectral_gap(1)
                gap_text = f"{gap:.3f}" if rounds == ROUNDS else f"{gap:.3f} per round, 7x the rounds"
                lines.append(f"| {name} | {alpha} | {label} | {degree * rounds} | {gap_text} | {_fmt(per_seed)} | {len(per_seed)} |")
    lines += [
        "",
        "## Per seed",
        "",
        "| Dataset | α | Seed | " + " | ".join(GRAPHS) + " |",
        "|---|---|---|" + "---|" * len(GRAPHS),
    ]
    for r in results:
        if r["status"] != "completed":
            lines.append(f"| {r['dataset']} | {r['split']} | {r['seed']} | {r['status']}: {r.get('error') or ''} |")
            continue
        cells = [_fmt(r["tests"].get(label, [])) for label in GRAPHS]
        lines.append(f"| {DATASETS[r['dataset']]} | {ALPHAS[r['split']]} | {r['seed']} | " + " | ".join(cells) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--date", default="2026-11-21", help="date stamp of the fair campaign and ablation reports")
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS))
    parser.add_argument("--splits", nargs="+", default=["dirichlet_a01"])
    parser.add_argument("--time-budget", type=int, default=240, help="minutes per run")
    parser.add_argument("--api", default="http://127.0.0.1:8000")
    parser.add_argument("--dry-run", action="store_true", help="print the configurations and exit")
    args = parser.parse_args(argv)

    out = EVIDENCE / f"topology-degree-{args.date}.md"
    progress_path = out.with_suffix(".json")
    progress = json.loads(progress_path.read_text(encoding="utf-8")) if progress_path.is_file() else {"runs": [], "active": None}
    api = Api(args.api, api_token())
    for dataset in args.datasets:
        for split in args.splits:
            report = EVIDENCE / f"fair-{dataset}-{split}-3seed-{args.date}.json"
            if not report.is_file():
                continue
            for source in json.loads(report.read_text(encoding="utf-8"))["runs"]:
                if source["status"] != "completed":
                    continue
                key = (dataset, split, source["seed"])
                if any((r["dataset"], r["split"], r["seed"]) == key and r["status"] == "completed" for r in progress["runs"]):
                    log(f"{dataset} {split} seed {source['seed']}: already done")
                    continue
                state = api.call("GET", f"/api/supervisor/runs/{source['run_id']}")["state"]
                configs, final_config = ablation_configs(state)
                full = configs[0]
                if full["topology"] != "mesh":
                    log(f"{dataset} {split} seed {source['seed']}: final topology {full['topology']}, skipped")
                    continue
                configs = study_configs(full)
                if args.dry_run:
                    log(f"{dataset} {split} seed {source['seed']}: {[c['label'] for c in configs]} on {final_config['architecture']}, {full['aggregation']}, {full['sync_mode']}")
                    continue
                active = progress.get("active")
                if active and tuple(active["key"]) == key:
                    run_id = active["run_id"]
                    log(f"{dataset} {split} seed {source['seed']}: resuming run {run_id}")
                else:
                    goal = {
                        "dataset": dataset,
                        "dataset_split": split,
                        "seed": source["seed"],
                        "target_accuracy": 0.99,
                        "max_experiments": 20,
                        "time_budget_minutes": args.time_budget,
                        "rounds": ROUNDS,
                        "local_epochs": EPOCHS,
                        "topology": "directed_ring",
                        "aggregation": "fedavg",
                        "sync_mode": "semi_sync",
                        "measure_each_step": True,
                        "ablation_configs": configs,
                    }
                    run_id = api.call("POST", "/api/supervisor/runs", goal)["id"]
                    progress["active"] = {"key": list(key), "run_id": run_id}
                    progress_path.write_text(json.dumps(progress, indent=2), encoding="utf-8")
                    log(f"{dataset} {split} seed {source['seed']}: started run {run_id}")
                run = wait_for_run(api, run_id)
                summary = summarize_run(source["seed"], run)
                tests = ablation_tests(dataset, split, source["seed"], args.date)
                for label in NEW:
                    _val, test = _accuracy(summary, label)
                    if test is not None:
                        tests[label] = [test]
                _val, baseline = _accuracy(summary, "baseline")
                complete = summary["status"] == "completed" and all(label in tests for label in NEW)
                progress["runs"] = [r for r in progress["runs"] if (r["dataset"], r["split"], r["seed"]) != key]
                progress["runs"].append({
                    "dataset": dataset,
                    "split": split,
                    "seed": source["seed"],
                    "source_run_id": source["run_id"],
                    "run_id": run_id,
                    "status": "completed" if complete else "failed",
                    "error": summary.get("error"),
                    "final_config": final_config,
                    "baseline_test": baseline,
                    "tests": tests,
                    "steps": summary["steps"],
                })
                progress["active"] = None
                progress_path.write_text(json.dumps(progress, indent=2), encoding="utf-8")
                write_report(out, progress["runs"], args.date)
                log(f"{dataset} {split} seed {source['seed']}: {'completed' if complete else 'failed'} "
                    + ", ".join(f"{k} {_fmt(v)}" for k, v in tests.items()))
                if not complete:
                    log("stopping so the failure can be fixed")
                    return 1
    if not args.dry_run and progress["runs"]:
        write_report(out, progress["runs"], args.date)
        log(f"report written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
