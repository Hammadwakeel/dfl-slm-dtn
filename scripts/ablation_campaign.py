"""Leave-one-out ablation of the fair campaign's final configurations.

    python scripts/ablation_campaign.py --date 2026-11-21

In the fair campaign the on-device SLM chose the order of the steps, so a step's
measured effect depends on what ran before it. This ablation removes that order
dependence. For every finished run (dataset x Dirichlet split x seed) it reads the
final configuration from the supervisor (the architecture with its training
hyperparameters, the topology, the aggregation rule and the synchronization mode, all
after keep-or-revert) and trains, with the baseline's budget (same rounds x local
epochs, same seed, same installed initial weights):

* ``full``: the final configuration itself (a repeat, which also gives run-to-run noise);
* ``without NAS``: the baseline CNN with the final training hyperparameters;
* ``without HPO``: the final architecture with the baseline's learning rate and batch size;
* ``without topology``: the directed ring of the baseline instead of the selected topology;
* ``without aggregation``: FedAvg instead of the selected rule;
* ``without sync``: the baseline's semi-synchronous mode instead of the selected one.

A variant is skipped when the run kept the default for that component (for example
the sweep chose FedAvg, or the HPO change was reverted). The supervisor also re-trains
the baseline first. Results go to ``docs/evidence/fair-ablation-<date>.json`` / ``.md``;
re-running resumes finished seeds. ``--repeat N`` runs the same ablation again into
``fair-ablation-<date>-repeat<N>.json`` / ``.md``, to measure run-to-run noise. The API token is read from ``.env`` and never printed.
"""

from __future__ import annotations

import argparse
import copy
import json
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paper_campaign import Api, api_token, log, summarize_run, wait_for_run  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs" / "evidence"
DATASETS = (("fashion_mnist", "Fashion-MNIST"), ("femnist", "FEMNIST"))
SPLITS = (("dirichlet_a01", "0.1"), ("dirichlet_a05", "0.5"), ("dirichlet_a09", "0.9"))
BASE_TOPOLOGY, BASE_AGGREGATION, BASE_SYNC = "directed_ring", "fedavg", "semi_sync"
VARIANTS = ("without NAS", "without HPO", "without topology", "without aggregation", "without sync")


def _training(architecture: dict[str, Any]) -> dict[str, Any]:
    return dict(architecture.get("training") or {})


def ablation_configs(state: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Leave-one-out configurations from a finished fair run's state."""
    baseline = state["baseline_architecture"]
    final = state.get("nas_architecture") or baseline
    topology = state.get("selected_topology") or BASE_TOPOLOGY
    aggregation = state.get("selected_aggregation") or BASE_AGGREGATION
    sync_mode = state.get("selected_sync_mode") or BASE_SYNC
    nas_step = next(
        (item for item in state.get("experiments", []) if item.get("step") == "measure_step" and item.get("measures") == "run_nas" and item.get("accepted") is not False),
        None,
    )
    nas_kept = bool(nas_step) and not str(nas_step.get("architecture") or "").startswith(("seed-", "baseline"))
    hpo_kept = _training(final) != _training(baseline)

    def config(label: str, architecture: dict[str, Any], topo: str, agg: str, sync: str) -> dict[str, Any]:
        return {"label": label, "architecture": architecture, "topology": topo, "aggregation": agg, "sync_mode": sync}

    configs = [config("full", final, topology, aggregation, sync_mode)]
    if nas_kept:
        plain = copy.deepcopy(baseline)
        plain["training"] = _training(final)
        plain["id"] = "baseline-final-training"
        configs.append(config("without NAS", plain, topology, aggregation, sync_mode))
    if hpo_kept:
        default = copy.deepcopy(final)
        default["training"] = _training(baseline)
        default["id"] = f"{final.get('id', 'model')}-default-training"
        configs.append(config("without HPO", default, topology, aggregation, sync_mode))
    if topology != BASE_TOPOLOGY:
        configs.append(config("without topology", final, BASE_TOPOLOGY, aggregation, sync_mode))
    if aggregation != BASE_AGGREGATION:
        configs.append(config("without aggregation", final, topology, BASE_AGGREGATION, sync_mode))
    if sync_mode != BASE_SYNC:
        configs.append(config("without sync", final, topology, aggregation, BASE_SYNC))
    final_config = {
        "architecture": final.get("id"),
        "parameter_count": final.get("parameter_count"),
        "training": _training(final),
        "topology": topology,
        "aggregation": aggregation,
        "sync_mode": sync_mode,
        "nas_kept": nas_kept,
        "hpo_kept": hpo_kept,
    }
    return configs, final_config


def _accuracy(summary: dict[str, Any], label: str) -> tuple[float | None, float | None]:
    step = next((s for s in summary["steps"] if s["step"] == "ablation_training" and s.get("measures") == label and s.get("status") == "completed"), None)
    if step is None and label == "baseline":
        step = next((s for s in summary["steps"] if s["step"] == "baseline_training" and s.get("status") == "completed"), None)
    return (step.get("validation_accuracy"), step.get("test_accuracy")) if step else (None, None)


def _fmt(values: list[float], signed: bool = False) -> str:
    if not values:
        return "–"
    mean = 100 * statistics.fmean(values)
    text = f"{mean:+.2f}" if signed else f"{mean:.2f}"
    return text if len(values) == 1 else f"{text} ± {100 * statistics.stdev(values):.2f}"


def write_report(path: Path, results: list[dict[str, Any]]) -> None:
    lines = [
        "# Leave-one-out ablation of the fair campaign",
        "",
        f"Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} by `scripts/ablation_campaign.py`.",
        "",
        "For every finished fair run, the final configuration (after keep-or-revert) is retrained",
        "with the baseline's budget (10 rounds x 5 local epochs, same seed and initial weights), once",
        "as it is and once without each component the run kept. A component's contribution is the",
        "test accuracy of the full configuration minus that of the configuration without it, so it",
        "does not depend on the order in which the SLM ran the steps. Values are mean ± std over the",
        "seeds that kept the component (n). Accuracy is the mean over the 8 workers on the test split.",
        "",
        "## Contribution of each component (test accuracy points, full minus without)",
        "",
        "| Dataset | α | Full (%) | " + " | ".join(f"{v.replace('without ', '').capitalize()} (n)" for v in VARIANTS) + " |",
        "|---|---|---|" + "---|" * len(VARIANTS),
    ]
    for dataset, dname in DATASETS:
        for split, alpha in SPLITS:
            rows = [r for r in results if r["dataset"] == dataset and r["split"] == split and r["status"] == "completed"]
            if not rows:
                continue
            full = [r["accuracy"]["full"]["test"] for r in rows if r["accuracy"].get("full", {}).get("test") is not None]
            cells = []
            for variant in VARIANTS:
                deltas = [
                    r["accuracy"]["full"]["test"] - r["accuracy"][variant]["test"]
                    for r in rows
                    if r["accuracy"].get(variant, {}).get("test") is not None and r["accuracy"].get("full", {}).get("test") is not None
                ]
                cells.append(f"{_fmt(deltas, signed=True)} ({len(deltas)})" if deltas else "– (0)")
            lines.append(f"| {dname} | {alpha} | {_fmt(full)} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "## Repeatability",
        "",
        "The baseline and the full configuration are each trained twice: once in the campaign and",
        "once here, with the same seed. The difference is run-to-run noise on the real cluster",
        "(thread scheduling, network timing and semi-synchronous deadlines are not deterministic).",
        "",
        "| Dataset | α | Seed | Baseline test: campaign / repeat | Full test: campaign / repeat |",
        "|---|---|---|---|---|",
    ]
    for r in results:
        if r["status"] != "completed":
            continue
        dname = dict(DATASETS)[r["dataset"]]
        alpha = dict(SPLITS)[r["split"]]
        b0, b1 = r.get("campaign_baseline_test"), r["accuracy"].get("baseline", {}).get("test")
        f0, f1 = r.get("campaign_final_test"), r["accuracy"].get("full", {}).get("test")
        pct = lambda v: f"{100 * v:.2f}" if isinstance(v, (int, float)) else "n/a"  # noqa: E731
        lines.append(f"| {dname} | {alpha} | {r['seed']} | {pct(b0)} / {pct(b1)} | {pct(f0)} / {pct(f1)} |")
    lines += ["", "## Per run", "", "| Dataset | α | Seed | Final configuration | " + " | ".join(["Baseline", "Full", *VARIANTS]) + " |",
              "|---|---|---|---|" + "---|" * (2 + len(VARIANTS))]
    for r in results:
        if r["status"] != "completed":
            lines.append(f"| {r['dataset']} | {r['split']} | {r['seed']} | {r['status']}: {r.get('error') or ''} |")
            continue
        fc = r["final_config"]
        desc = f"{fc['architecture']}, lr {fc['training'].get('learning_rate')}, batch {fc['training'].get('batch_size')}, {fc['topology']}, {fc['aggregation']}, {fc['sync_mode']}"
        cells = []
        for label in ("baseline", "full", *VARIANTS):
            test = r["accuracy"].get(label, {}).get("test")
            cells.append(f"{100 * test:.2f}" if isinstance(test, (int, float)) else "–")
        lines.append(f"| {dict(DATASETS)[r['dataset']]} | {dict(SPLITS)[r['split']]} | {r['seed']} | {desc} | " + " | ".join(cells) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--date", default="2026-11-21", help="date stamp of the fair campaign reports")
    parser.add_argument("--datasets", nargs="+", default=[d for d, _ in DATASETS])
    parser.add_argument("--splits", nargs="+", default=[s for s, _ in SPLITS])
    parser.add_argument("--time-budget", type=int, default=240, help="minutes per ablation run")
    parser.add_argument("--api", default="http://127.0.0.1:8000")
    parser.add_argument("--repeat", type=int, default=0,
                        help="repeat number N > 0 writes fair-ablation-<date>-repeat<N>.json/.md, to measure run-to-run noise")
    parser.add_argument("--seeds", type=int, nargs="+", default=None, help="only these seeds (default: every seed)")
    parser.add_argument("--dry-run", action="store_true", help="print the configurations and exit")
    args = parser.parse_args(argv)

    if args.repeat < 0:
        parser.error("--repeat must be 0 or positive")
    suffix = f"-repeat{args.repeat}" if args.repeat else ""
    out = EVIDENCE / f"fair-ablation-{args.date}{suffix}.md"
    progress_path = out.with_suffix(".json")
    progress = json.loads(progress_path.read_text(encoding="utf-8")) if progress_path.is_file() else {"runs": [], "active": None}
    api = Api(args.api, api_token())
    for dataset in args.datasets:
        for split in args.splits:
            report = EVIDENCE / f"fair-{dataset}-{split}-3seed-{args.date}.json"
            if not report.is_file():
                continue
            for source in json.loads(report.read_text(encoding="utf-8"))["runs"]:
                if source["status"] != "completed" or (args.seeds and source["seed"] not in args.seeds):
                    continue
                key = (dataset, split, source["seed"])
                if any((r["dataset"], r["split"], r["seed"]) == key and r["status"] == "completed" for r in progress["runs"]):
                    log(f"{dataset} {split} seed {source['seed']}: already done")
                    continue
                state = api.call("GET", f"/api/supervisor/runs/{source['run_id']}")["state"]
                configs, final_config = ablation_configs(state)
                if args.dry_run:
                    log(f"{dataset} {split} seed {source['seed']}: {[c['label'] for c in configs]} {json.dumps({k: v for k, v in final_config.items() if k != 'training'})} training {final_config['training']}")
                    continue
                active = progress.get("active")
                if active and tuple(active["key"]) == key:
                    run_id = active["run_id"]
                    log(f"{dataset} {split} seed {source['seed']}: resuming ablation run {run_id}")
                else:
                    goal = {
                        "dataset": dataset,
                        "dataset_split": split,
                        "seed": source["seed"],
                        "target_accuracy": 0.99,
                        "max_experiments": 20,
                        "time_budget_minutes": args.time_budget,
                        "rounds": 10,
                        "local_epochs": 5,
                        "topology": BASE_TOPOLOGY,
                        "aggregation": BASE_AGGREGATION,
                        "sync_mode": BASE_SYNC,
                        "measure_each_step": True,
                        "ablation_configs": configs,
                    }
                    run_id = api.call("POST", "/api/supervisor/runs", goal)["id"]
                    progress["active"] = {"key": list(key), "run_id": run_id}
                    progress_path.write_text(json.dumps(progress, indent=2), encoding="utf-8")
                    log(f"{dataset} {split} seed {source['seed']}: started ablation run {run_id} ({len(configs)} configurations)")
                run = wait_for_run(api, run_id)
                summary = summarize_run(source["seed"], run)
                accuracy = {}
                for label in ("baseline", *(c["label"] for c in configs)):
                    val, test = _accuracy(summary, label)
                    if test is not None:
                        accuracy[label] = {"validation": val, "test": test}
                complete = summary["status"] == "completed" and all(c["label"] in accuracy for c in configs)
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
                    "campaign_baseline_test": source.get("baseline_test_accuracy"),
                    "campaign_final_test": source.get("best_test_accuracy"),
                    "accuracy": accuracy,
                    "steps": summary["steps"],
                })
                progress["active"] = None
                progress_path.write_text(json.dumps(progress, indent=2), encoding="utf-8")
                write_report(out, progress["runs"])
                log(f"{dataset} {split} seed {source['seed']}: {'completed' if complete else 'failed'} "
                    + ", ".join(f"{k} {100 * v['test']:.2f}" for k, v in accuracy.items()))
                if not complete:
                    log("stopping so the failure can be fixed")
                    return 1
    if not args.dry_run and progress["runs"]:
        write_report(out, progress["runs"])
        log(f"report written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
