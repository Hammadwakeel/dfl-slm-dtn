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


def campaign_runs(dataset: str, split: str, date: str | None = None) -> list[dict]:
    """Completed fair-campaign runs of one setting, newest report per seed, sorted by seed.

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
                by_seed[run["seed"]] = run
    return [by_seed[seed] for seed in sorted(by_seed)]
