"""Run the agentic supervisor end to end once per seed and write a paper report.

Each seed is one independent supervisor run: the agent prepares the data,
trains a baseline, chooses its own improvement steps (NAS, HPO, topology,
retraining, pruning) and stops by itself. Accuracy is the coordinator's global
evaluation on the official test split (never used for decisions).

    python scripts/paper_campaign.py --dataset mnist --target 0.95
    python scripts/paper_campaign.py --dataset cifar10 --target 0.70 --time-budget 900

Progress is saved after every run, so re-running the same command resumes.
The API token is read from ``.env`` (``DTN_API_TOKEN``) and never printed.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SEEDS = (17, 42, 2026)
POLL_SECONDS = 60
ACTIVE = {"queued", "running", "stopping"}
STEP_LABELS = {
    "baseline_training": "Baseline",
    "run_nas": "NAS",
    "run_hpo": "HPO",
    "optimize_topology": "Topology sweep",
    "learn_topology": "APG learned topology",
    "train_final": "Final training",
    "continue_training": "Continued training",
    "prune_model": "Pruning + fine-tune",
}


def api_token() -> str:
    text = (REPO_ROOT / ".env").read_text(encoding="utf-8")
    match = re.search(r"^\s*DTN_API_TOKEN\s*=\s*(.+)$", text, re.M)
    if not match:
        raise SystemExit("DTN_API_TOKEN is missing from .env")
    return match.group(1).strip().strip('"').strip("'")


class Api:
    def __init__(self, base_url: str, token: str) -> None:
        self.base_url = base_url.rstrip("/")
        self._headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

    def call(self, method: str, path: str, body: dict[str, Any] | None = None) -> Any:
        data = json.dumps(body).encode("utf-8") if body is not None else None
        request = urllib.request.Request(self.base_url + path, data=data, method=method, headers=self._headers)
        for attempt in range(5):
            try:
                with urllib.request.urlopen(request, timeout=120) as response:
                    return json.loads(response.read().decode("utf-8"))
            except urllib.error.HTTPError as error:
                detail = error.read().decode("utf-8", errors="replace")[:300]
                if error.code < 500 or attempt == 4:
                    raise RuntimeError(f"{method} {path} -> HTTP {error.code}: {detail}") from error
            except (urllib.error.URLError, TimeoutError, OSError):
                if attempt == 4:
                    raise
            time.sleep(15 * (attempt + 1))
        raise AssertionError("unreachable")


def log(message: str) -> None:
    print(f"{datetime.now().strftime('%H:%M:%S')} {message}", flush=True)


def wait_for_run(api: Api, run_id: str) -> dict[str, Any]:
    last_event_id = 0
    while True:
        run = api.call("GET", f"/api/supervisor/runs/{run_id}")
        # The API returns the newest 200 events; track them by id, not position.
        for event in run.get("events") or []:
            if int(event.get("id") or 0) > last_event_id:
                last_event_id = int(event["id"])
                log(f"  {event.get('event_type')}: {str(event.get('message'))[:220]}")
        if run.get("status") not in ACTIVE:
            return run
        time.sleep(POLL_SECONDS)


def _test(experiment: dict[str, Any]) -> tuple[float | None, float | None]:
    test = experiment.get("test") or {}
    return test.get("mean_accuracy"), test.get("std_accuracy")


def summarize_run(seed: int, run: dict[str, Any]) -> dict[str, Any]:
    state = run.get("state") or {}
    experiments = state.get("experiments") or []
    trained = [item for item in experiments if item.get("kind") == "decentralized" and item.get("status") == "completed"]
    best = state.get("best") or {}
    best_experiment = next((item for item in trained if item.get("group_id") == best.get("group_id")), None)
    baseline = next((item for item in trained if item.get("step") == "baseline_training"), None)
    pruning = state.get("pruning") or {}
    pruned = next((item for item in reversed(trained) if item.get("step") == "prune_model"), None)
    events = run.get("events") or []
    decisions = [event for event in events if event.get("event_type") == "decision"]
    return {
        "seed": seed,
        "run_id": run.get("id"),
        "status": run.get("status"),
        "outcome": (run.get("result") or {}).get("outcome"),
        "error": run.get("error"),
        "created_at": run.get("created_at"),
        "finished_at": run.get("completed_at") or run.get("updated_at"),
        "steps": [
            {
                "step": item.get("step"),
                "status": item.get("status"),
                "topology": item.get("topology"),
                "architecture": item.get("architecture"),
                "parameter_count": item.get("parameter_count"),
                "validation_accuracy": item.get("validation_accuracy"),
                "local_validation_accuracy": item.get("local_validation_accuracy"),
                "test_accuracy": _test(item)[0],
                "test_std": _test(item)[1],
            }
            for item in experiments
        ],
        "baseline_test_accuracy": _test(baseline)[0] if baseline else None,
        "best_step": best.get("step"),
        "best_topology": best.get("topology"),
        "best_architecture": best.get("architecture"),
        "best_validation_accuracy": best.get("validation_accuracy"),
        "best_test_accuracy": _test(best_experiment)[0] if best_experiment else None,
        "best_test_std": _test(best_experiment)[1] if best_experiment else None,
        "best_parameter_count": best_experiment.get("parameter_count") if best_experiment else None,
        "pruning_accepted": pruning.get("accepted"),
        "pruning_attempts": [
            {"ratio": item.get("ratio"), "validation_accuracy": item.get("validation_accuracy"), "accepted": item.get("accepted")}
            for item in state.get("pruning_attempts") or []
        ],
        "pruned_test_accuracy": _test(pruned)[0] if pruned else None,
        "pruned_parameter_count": pruned.get("parameter_count") if pruned else None,
        "warm_start": bool(state.get("warm_start")),
        "continuations": [
            {key: item.get(key) for key in ("validation_accuracy", "gain", "improved")}
            for item in state.get("continuations") or []
        ],
        "deployed_model": (run.get("result") or {}).get("deployed_model"),
        "deployed_test_accuracy": ((run.get("result") or {}).get("deployed_model") or {}).get("test_accuracy"),
        "decisions": [
            {"message": str(event.get("message"))[:300]}
            for event in decisions
        ],
        "slm_decisions": sum(1 for event in decisions if " by slm" in str(event.get("message"))),
        "rule_decisions": sum(1 for event in decisions if " by rule" in str(event.get("message"))),
        "fallback_decisions": sum(1 for event in decisions if " by fallback" in str(event.get("message"))),
        "errors": [
            str(event.get("message"))[:300]
            for event in events
            if event.get("event_type") in {"step_failed", "error", "failed"}
        ],
    }


def _pct(value: float | None) -> str:
    return f"{100 * value:.2f}%" if isinstance(value, (int, float)) else "n/a"


def _mean_std(values: list[float]) -> str:
    if not values:
        return "n/a"
    if len(values) == 1:
        return _pct(values[0])
    return f"{100 * statistics.fmean(values):.2f}% ± {100 * statistics.pstdev(values):.2f}"


def _hours(start: str | None, end: str | None) -> str:
    try:
        delta = datetime.fromisoformat(end) - datetime.fromisoformat(start)  # type: ignore[arg-type]
        return f"{delta.total_seconds() / 3600:.2f} h"
    except (TypeError, ValueError):
        return "n/a"


def write_report(path: Path, dataset: str, goal: dict[str, Any], runs: list[dict[str, Any]]) -> None:
    def values(key: str) -> list[float]:
        return [item[key] for item in runs if isinstance(item.get(key), (int, float))]

    lines = [
        f"# Agentic end-to-end supervisor runs: {dataset.upper()}, {len(runs)} seed{'s' if len(runs) != 1 else ''}",
        "",
        f"Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} by `scripts/paper_campaign.py`.",
        "",
        "Each seed is one independent run of the autonomous supervisor on the real",
        "8-Jetson cluster. The on-device SLM (Qwen3.5 4B on the master Jetson) chooses",
        "every improvement step from the allowed tools; fixed rules only start the",
        "baseline, compress a model that met the target and stop the run. Accuracy is",
        "the coordinator's global evaluation of every worker's model on the official",
        "test split (mean over the 8 workers); decisions use the global validation split.",
        "",
        "## Goal",
        "",
        "| Setting | Value |",
        "|---|---|",
    ]
    for key in ("dataset", "target_accuracy", "max_experiments", "time_budget_minutes", "dataset_split", "rounds", "local_epochs"):
        if key in goal:
            lines.append(f"| {key} | {goal[key]} |")
    lines += [
        "",
        "## Summary over seeds",
        "",
        "| Metric | Mean ± std over seeds |",
        "|---|---|",
        f"| Baseline test accuracy | {_mean_std(values('baseline_test_accuracy'))} |",
        f"| Best model test accuracy | {_mean_std(values('best_test_accuracy'))} |",
        f"| Pruned model test accuracy | {_mean_std(values('pruned_test_accuracy'))} |",
        f"| Model installed on the workers at the end (test) | {_mean_std(values('deployed_test_accuracy'))} |",
        f"| Target met (seeds) | {sum(1 for item in runs if (item.get('best_validation_accuracy') or 0) >= goal.get('target_accuracy', 1))} / {len(runs)} |",
        "",
        "## Per seed",
        "",
        "| Seed | Status | Baseline test | Best step | Topology | Params | Best test | Pruned test (params) | SLM / rule / fallback decisions | Wall time |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for item in runs:
        pruned = (
            f"{_pct(item['pruned_test_accuracy'])} ({item['pruned_parameter_count']:,})"
            if isinstance(item.get("pruned_parameter_count"), int)
            else _pct(item.get("pruned_test_accuracy"))
        )
        params = f"{item['best_parameter_count']:,}" if isinstance(item.get("best_parameter_count"), int) else "n/a"
        lines.append(
            f"| {item['seed']} | {item['status']} | {_pct(item.get('baseline_test_accuracy'))} | "
            f"{STEP_LABELS.get(item.get('best_step') or '', item.get('best_step'))} | {item.get('best_topology')} | {params} | "
            f"{_pct(item.get('best_test_accuracy'))} ± {_pct(item.get('best_test_std'))} | {pruned} | "
            f"{item['slm_decisions']} / {item['rule_decisions']} / {item['fallback_decisions']} | "
            f"{_hours(item.get('created_at'), item.get('finished_at'))} |"
        )
    for item in runs:
        lines += [
            "",
            f"### Seed {item['seed']} (run `{item['run_id']}`)",
            "",
            "| Step | Status | Topology | Architecture | Params | Global val | Global test | Worker-reported |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for step in item["steps"]:
            params = f"{step['parameter_count']:,}" if isinstance(step.get("parameter_count"), int) else ""
            test = f"{_pct(step['test_accuracy'])} ± {_pct(step['test_std'])}" if step.get("test_accuracy") is not None else ""
            lines.append(
                f"| {STEP_LABELS.get(step['step'], step['step'])} | {step['status']} | {step.get('topology') or ''} | "
                f"{step.get('architecture') or ''} | {params} | {_pct(step.get('validation_accuracy')) if step.get('validation_accuracy') is not None else ''} | "
                f"{test} | {_pct(step.get('local_validation_accuracy')) if step.get('local_validation_accuracy') is not None else ''} |"
            )
        lines += ["", "Agent decisions:", ""]
        lines += [f"1. {decision['message']}" for decision in item["decisions"]] or ["(none)"]
        if item["pruning_attempts"]:
            attempts = ", ".join(
                f"{int(round(100 * attempt['ratio']))}% → {_pct(attempt['validation_accuracy'])} ({'kept' if attempt['accepted'] else 'rejected'})"
                for attempt in item["pruning_attempts"]
                if isinstance(attempt.get("ratio"), (int, float))
            )
            lines += ["", f"Pruning attempts: {attempts}."]
        if item.get("continuations"):
            continued = ", ".join(
                f"{_pct(entry.get('validation_accuracy'))} ({'improved' if entry.get('improved') else 'no gain'})"
                for entry in item["continuations"]
            )
            lines += ["", f"Continued training from the best checkpoint: {continued}."]
        deployed = item.get("deployed_model") or {}
        if deployed:
            lines += [
                "",
                f"Model left on every worker: {STEP_LABELS.get(deployed.get('step') or '', deployed.get('step'))} "
                f"({deployed.get('source_worker')}), global val {_pct(deployed.get('validation_accuracy'))}, "
                f"test {_pct(deployed.get('test_accuracy'))}, "
                + ("installed." if deployed.get("installed") else f"not installed: {deployed.get('error')}."),
            ]
        if item["errors"]:
            lines += ["", "Errors recorded during the run:", ""] + [f"- {error}" for error in item["errors"]]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", choices=("mnist", "cifar10"), required=True)
    parser.add_argument("--target", type=float, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--max-experiments", type=int, default=8)
    parser.add_argument("--time-budget", type=int, default=480, help="minutes per run")
    parser.add_argument("--api", default="http://127.0.0.1:8000")
    parser.add_argument("--out", type=Path, default=None, help="report path (markdown)")
    args = parser.parse_args(argv)

    date = datetime.now().strftime("%Y-%m-%d")
    report = args.out or REPO_ROOT / "docs" / "evidence" / f"agentic-{args.dataset}-3seed-{date}.md"
    progress_path = report.with_suffix(".json")
    progress = json.loads(progress_path.read_text(encoding="utf-8")) if progress_path.is_file() else {"runs": []}
    api = Api(args.api, api_token())
    goal = {
        "dataset": args.dataset,
        "target_accuracy": args.target,
        "max_experiments": args.max_experiments,
        "time_budget_minutes": args.time_budget,
    }
    for seed in args.seeds:
        done = next((item for item in progress["runs"] if item["seed"] == seed and item["status"] == "completed"), None)
        if done:
            log(f"seed {seed}: already completed (run {done['run_id']})")
            continue
        pending = progress.get("active")
        if pending and pending.get("seed") == seed:
            run_id = pending["run_id"]
            log(f"seed {seed}: resuming run {run_id}")
        else:
            created = api.call("POST", "/api/supervisor/runs", {**goal, "seed": seed})
            run_id = created["id"]
            progress["active"] = {"seed": seed, "run_id": run_id}
            progress_path.parent.mkdir(parents=True, exist_ok=True)
            progress_path.write_text(json.dumps(progress, indent=2), encoding="utf-8")
            log(f"seed {seed}: started run {run_id}")
        run = wait_for_run(api, run_id)
        summary = summarize_run(seed, run)
        progress["runs"] = [item for item in progress["runs"] if item["seed"] != seed] + [summary]
        progress["active"] = None
        progress["goal"] = run.get("goal") or goal
        progress_path.write_text(json.dumps(progress, indent=2), encoding="utf-8")
        write_report(report, args.dataset, progress["goal"], sorted(progress["runs"], key=lambda item: args.seeds.index(item["seed"]) if item["seed"] in args.seeds else 99))
        log(f"seed {seed}: {summary['status']} best test {_pct(summary['best_test_accuracy'])}")
        if summary["status"] != "completed":
            log(f"seed {seed} did not complete; stopping the campaign so the failure can be fixed")
            return 1
    log(f"report written to {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
