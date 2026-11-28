"""Paper tables across the fair step-by-step campaigns (all datasets, alphas and seeds).

    python scripts/fair_summary.py --date 2026-11-21

Reads every ``docs/evidence/fair-<dataset>-<split>-3seed-<date>.json`` written by
``scripts/paper_campaign.py --measure-each-step``, plus the leave-one-out ablation
(``fair-ablation-<date>.json`` and any ``fair-ablation-<date>-repeat<N>.json``) written
by ``scripts/ablation_campaign.py``, and writes one Markdown summary.

Every model in these runs is trained with the same budget as the baseline (same
rounds x local epochs, same installed initial weights). A step's effect is measured
against the configuration that was active just before it: the baseline, or the last
measurement whose change was kept. The supervisor reverts a change whose measurement
falls more than 0.5 points below the best so far. Steps run in the order the on-device
SLM chose, so effects are grouped by step type, not by position.

When present, it also folds in the in-degree study (``topology-degree-<date>.json``,
``scripts/topology_degree_study.py``) and the noise analysis of the keep-or-revert
decisions (``scripts/decision_robustness.py``), and ends with the limitations.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evidence_runs import baseline_excluded, campaign_runs  # noqa: E402

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


WORKERS = 8
FLOAT_BYTES = 4
# Directed edges per round (one model sent per edge), eight workers.
TOPOLOGY_EDGES = {"directed_ring": WORKERS, "bidirectional_ring": 2 * WORKERS, "chain": 2 * (WORKERS - 1),
                  "star": 2 * (WORKERS - 1), "mesh": WORKERS * (WORKERS - 1)}


def _apg_runs(date: str) -> list[dict]:
    path = EVIDENCE / f"fair-apg-probes-{date}.json"
    return json.loads(path.read_text(encoding="utf-8"))["runs"] if path.is_file() else []


def _learned_edges(apg: list[dict], dataset: str, split: str, seed: int) -> int | None:
    for item in apg:
        if (item["dataset"], item["split"], item["seed"]) == (dataset, split, seed):
            return sum(len(senders) for senders in item["senders"].values())
    return None


def communication_section(data: dict, date: str) -> list[str]:
    """Models sent per training (rounds x directed edges) and the bytes they carry."""
    apg = _apg_runs(date)
    lines = [
        "## Communication per training",
        "",
        "Every training runs 10 rounds; in each round every directed edge carries one model",
        f"(float32 weights). Directed edges per round: directed ring {TOPOLOGY_EDGES['directed_ring']},",
        f"mesh {TOPOLOGY_EDGES['mesh']}; the learned (APG) graph is measured per run. The baseline uses",
        "the baseline CNN on the directed ring; the final model is the run's best model on its topology.",
        "The search steps (NAS, sweeps) are not counted; they cost the same for every seed.",
        "",
        "| Dataset | α | Baseline: models / MB | Final model: models / MB | Learned (APG): edges / models / MB | Final test (%) | Learned test (%) |",
        "|---|---|---|---|---|---|---|",
    ]
    for dataset, dname in DATASETS:
        for split, alpha in SPLITS:
            runs = data.get((dataset, split), [])
            if not runs:
                continue
            base_mb, final_mb, final_msgs, learned_mb, learned_edges, learned_test = [], [], [], [], [], []
            base_msgs = 10 * TOPOLOGY_EDGES["directed_ring"]
            for run in runs:
                rounds = 10
                baseline = next((s for s in run["steps"] if s["step"] == "baseline_training"), {})
                base_mb.append(base_msgs * (baseline.get("parameter_count") or 0) * FLOAT_BYTES / 1e6)
                edges = TOPOLOGY_EDGES.get(run.get("best_topology") or "", 0)
                final_msgs.append(rounds * edges)
                final_mb.append(rounds * edges * (run.get("final_parameter_count") or 0) * FLOAT_BYTES / 1e6)
                learned = next((s for s in run["steps"] if s["step"] == "learn_topology" and s.get("status") == "completed"), None)
                count = _learned_edges(apg, dataset, split, run["seed"])
                if learned and count:
                    learned_edges.append(count)
                    learned_mb.append(rounds * count * (learned.get("parameter_count") or 0) * FLOAT_BYTES / 1e6)
                    if isinstance(learned.get("test_accuracy"), (int, float)):
                        learned_test.append(learned["test_accuracy"])
            best = [r["final_test_accuracy"] for r in runs if isinstance(r.get("final_test_accuracy"), (int, float))]
            mean = statistics.fmean
            learned_cell = (f"{mean(learned_edges):.1f} / {10 * mean(learned_edges):.0f} / {mean(learned_mb):.1f}"
                            if learned_edges else "n/a")
            lines.append(
                f"| {dname} | {alpha} | {base_msgs} / {mean(base_mb):.1f} | {mean(final_msgs):.0f} / {mean(final_mb):.1f} | "
                f"{learned_cell} | {_pct(best)} | {_pct(learned_test)} |"
            )
    return lines + [""]


def apg_section(date: str) -> list[str]:
    """Why the learned topology trails the mesh: whom APG keeps, measured on the probe."""
    apg = _apg_runs(date)
    if not apg:
        return []
    import numpy as np

    lines = [
        "## Learned topology (APG) diagnostics",
        "",
        "APG runs on a 2-round mesh probe. It keeps, for each receiver, the senders whose models",
        "have a low loss on the receiver's own validation data and arrive quickly; a small balance",
        "term pulls the neighbourhood label mix toward the global one. The table compares the kept",
        "and dropped senders and the label gap (total-variation distance) between the data a worker",
        "aggregates over (itself and its senders) and the global label mix; on the mesh that gap is 0.",
        "",
        "| Dataset | α | Senders kept per worker | Peer loss: kept / dropped | Label cosine: kept / dropped | Label gap vs global |",
        "|---|---|---|---|---|---|",
    ]
    groups: dict[tuple[str, str], list[dict]] = {}
    for item in apg:
        groups.setdefault((item["dataset"], item["split"]), []).append(item)
    for dataset, dname in DATASETS:
        for split, alpha in SPLITS:
            items = groups.get((dataset, split))
            if not items:
                continue
            kept_n, loss_k, loss_d, sim_k, sim_d, gap = [], [], [], [], [], []
            for item in items:
                ids = sorted(item["probe"])
                index = {worker: i for i, worker in enumerate(ids)}
                hist = np.array([item["probe"][w]["class_histogram"] for w in ids], dtype=float)
                unit = hist / np.linalg.norm(hist, axis=1, keepdims=True)
                global_mix = hist.sum(axis=0) / hist.sum()
                for receiver in ids:
                    i = index[receiver]
                    kept = [index[s] for s in item["senders"][receiver]]
                    dropped = [j for j in range(len(ids)) if j != i and j not in kept]
                    losses = item["probe"][receiver].get("peer_losses") or {}
                    kept_n.append(len(kept))
                    loss_k += [losses[ids[j]] for j in kept if ids[j] in losses]
                    loss_d += [losses[ids[j]] for j in dropped if ids[j] in losses]
                    sim_k += [float(unit[i] @ unit[j]) for j in kept]
                    sim_d += [float(unit[i] @ unit[j]) for j in dropped]
                    neighbourhood = hist[[i, *kept]].sum(axis=0)
                    gap.append(0.5 * float(np.abs(neighbourhood / neighbourhood.sum() - global_mix).sum()))
            m = statistics.fmean
            lines.append(
                f"| {dname} | {alpha} | {m(kept_n):.1f} of {WORKERS - 1} | {m(loss_k):.2f} / {m(loss_d):.2f} | "
                f"{m(sim_k):.2f} / {m(sim_d):.2f} | {m(gap):.2f} |"
            )
    return lines + [
        "",
        "APG optimizes each worker's local fit and waiting time, not global accuracy. It keeps",
        "peers whose models already suit the receiver's data, so each worker averages over only part",
        "of the label space. The label gap and the accuracy loss of the learned topology are both",
        "largest at α 0.1, where the label skew is strongest. The learned graph sends about half as",
        "many models as the mesh, so it trades accuracy for communication.",
        "",
    ]

ABLATION_VARIANTS = (
    ("without NAS", "NAS"),
    ("without HPO", "HPO"),
    ("without topology", "Topology"),
    ("without aggregation", "Aggregation"),
    ("without sync", "Sync"),
)


def _ablation_passes(date: str) -> list[list[dict]]:
    """Completed runs of the leave-one-out ablation and of each later repeat of it."""
    paths = [EVIDENCE / f"fair-ablation-{date}.json", *sorted(EVIDENCE.glob(f"fair-ablation-{date}-repeat*.json"))]
    return [
        [r for r in json.loads(path.read_text(encoding="utf-8"))["runs"] if r["status"] == "completed"]
        for path in paths
        if path.is_file()
    ]


def _test(run: dict, label: str) -> float | None:
    if label == "baseline" and baseline_excluded(run):
        return None
    value = run["accuracy"].get(label, {}).get("test")
    return value if isinstance(value, (int, float)) else None


def ablation_section(data: dict, date: str) -> list[str]:
    """Order-independent component contributions and run-to-run noise."""
    passes = _ablation_passes(date)
    if not passes:
        return []
    by_seed: dict[tuple[str, str, int], list[dict]] = {}
    for runs in passes:
        for run in runs:
            by_seed.setdefault((run["dataset"], run["split"], run["seed"]), []).append(run)
    lines = [
        "## Leave-one-out ablation (order-independent)",
        "",
        "The step effects above depend on the order the SLM chose. To remove that dependence, each",
        "run's final configuration is retrained with the same budget, once as it is (full) and once",
        "without each component the run kept. A component's contribution is full minus without (test",
        "points). When a seed was ablated more than once, its contribution is the mean over those",
        "repeats; values are mean ± std over the seeds that kept the component (n). Details per run are",
        f"in `fair-ablation-{date}.md`.",
        "",
        "| Dataset | α | Ablation passes | Full test (%) | " + " | ".join(f"{name} (n)" for _, name in ABLATION_VARIANTS) + " |",
        "|---|---|---|---|" + "---|" * len(ABLATION_VARIANTS),
    ]
    for dataset, dname in DATASETS:
        for split, alpha in SPLITS:
            seeds = {k: v for k, v in by_seed.items() if k[:2] == (dataset, split)}
            if not seeds:
                continue
            full = [statistics.fmean(t for r in runs if (t := _test(r, "full")) is not None) for runs in seeds.values()]
            cells = []
            for variant, _ in ABLATION_VARIANTS:
                deltas = []
                for runs in seeds.values():
                    pairs = [(_test(r, "full"), _test(r, variant)) for r in runs]
                    pairs = [f - w for f, w in pairs if f is not None and w is not None]
                    if pairs:
                        deltas.append(statistics.fmean(pairs))
                cells.append(f"{_signed(deltas)} ({len(deltas)})" if deltas else "– (0)")
            count = max(len(runs) for runs in seeds.values())
            lines.append(f"| {dname} | {alpha} | {count} | {_pct(full)} | " + " | ".join(cells) + " |")

    lines += [
        "",
        "## Run-to-run noise",
        "",
        "The baseline and the full configuration are retrained with the same seed in the campaign",
        "and in every ablation pass. The spread between those repeats is run-to-run noise on the",
        "cluster (thread scheduling, network timing and semi-synchronous deadlines are not",
        "deterministic). The table gives the mean over seeds of the per-seed standard deviation and",
        "the largest per-seed range (max − min), in test points.",
        "",
        "| Dataset | α | Trainings per seed | Baseline: mean std / max range | Full: mean std / max range |",
        "|---|---|---|---|---|",
    ]
    detail: list[str] = []
    for dataset, dname in DATASETS:
        for split, alpha in SPLITS:
            seeds = {k: v for k, v in by_seed.items() if k[:2] == (dataset, split)}
            if not seeds:
                continue
            source = {r["seed"]: r for r in data.get((dataset, split), [])}
            spread: dict[str, tuple[list[float], list[float]]] = {"baseline": ([], []), "full": ([], [])}
            counts = []
            for (_, _, seed), runs in sorted(seeds.items()):
                campaign = source.get(seed, {})
                values = {
                    "baseline": [campaign.get("baseline_test_accuracy"), *(_test(r, "baseline") for r in runs)],
                    "full": [campaign.get("final_test_accuracy"), *(_test(r, "full") for r in runs)],
                }
                row = []
                for label, vals in values.items():
                    vals = [v for v in vals if isinstance(v, (int, float))]
                    if len(vals) >= 2:
                        spread[label][0].append(statistics.stdev(vals))
                        spread[label][1].append(max(vals) - min(vals))
                    row.append(f"{_pct(vals)} ({', '.join(f'{100 * v:.2f}' for v in vals)})")
                counts.append(len(runs) + 1)
                detail.append(f"| {dname} | {alpha} | {seed} | {row[0]} | {row[1]} |")
            cells = [
                f"{100 * statistics.fmean(s):.2f} / {100 * max(rng):.2f}" if s else "n/a"
                for s, rng in spread.values()
            ]
            span = str(min(counts)) if min(counts) == max(counts) else f"{min(counts)}–{max(counts)}"
            lines.append(f"| {dname} | {alpha} | {span} | {cells[0]} | {cells[1]} |")
    lines += [
        "",
        "Per seed (mean ± std over the repeats; the campaign value comes first):",
        "",
        "| Dataset | α | Seed | Baseline test (%) | Full test (%) |",
        "|---|---|---|---|---|",
        *detail,
        "",
    ]
    return lines


def topology_degree_section(data: dict, date: str) -> list[str]:
    """Graph or communication volume: accuracy against in-degree, and the APG graph at a similar budget."""
    path = EVIDENCE / f"topology-degree-{date}.json"
    if not path.is_file():
        return []
    import topology_degree_study as study

    runs = [r for r in json.loads(path.read_text(encoding="utf-8"))["runs"] if r["status"] == "completed"]
    if not runs:
        return []
    apg = _apg_runs(date)
    lines = [
        "## Topology or communication volume? (in-degree study)",
        "",
        "The mesh's gain could come from receiving 7 models per round instead of 1. Each seed's final",
        "configuration (everything else unchanged) is trained on circulant graphs where worker i averages",
        "with workers i-1 .. i-k, with the same 10 × 5 budget, and on the directed ring for 70 rounds × 1",
        "epoch, which receives as many models as the mesh (70 per worker) with more local compute.",
        "k = 1 and k = 7 come from the ablation passes. The spectral gap (1 − |λ2| of the uniform averaging",
        "matrix) measures how much one round mixes the workers' models. The learned (APG) graph of the",
        "same seed, which keeps about 4 senders per worker, is listed for comparison with k = 4. Test %,",
        f"mean ± std over seeds; details in `topology-degree-{date}.md`.",
        "",
        "| Dataset | α | Graph | Models received per worker | Spectral gap | Test (%) | Seeds |",
        "|---|---|---|---|---|---|---|",
    ]
    for dataset, dname in DATASETS:
        for split, alpha in SPLITS:
            rows = [r for r in runs if (r["dataset"], r["split"]) == (dataset, split)]
            if not rows:
                continue
            for label, (degree, rounds, _epochs) in study.GRAPHS.items():
                values = [statistics.fmean(r["tests"][label]) for r in rows if r["tests"].get(label)]
                gap = f"{study.spectral_gap(degree):.3f}" + ("" if rounds == study.ROUNDS else " (7× rounds)")
                lines.append(f"| {dname} | {alpha} | {label} | {degree * rounds} | {gap} | {_pct(values)} | {len(values)} |")
            learned, edges = [], []
            for run in data.get((dataset, split), []):
                if not any(r["seed"] == run["seed"] for r in rows):
                    continue
                step = next((s for s in run["steps"] if s["step"] == "learn_topology" and s.get("status") == "completed"), None)
                count = _learned_edges(apg, dataset, split, run["seed"])
                if step and isinstance(step.get("test_accuracy"), (int, float)) and count:
                    learned.append(step["test_accuracy"])
                    edges.append(count / WORKERS)
            if learned:
                lines.append(
                    f"| {dname} | {alpha} | learned (APG), {statistics.fmean(edges):.1f} senders | {10 * statistics.fmean(edges):.0f} | "
                    f"varies | {_pct(learned)} | {len(learned)} |"
                )
    return lines + [""]


def decision_section(date: str) -> list[str]:
    """How many keep-or-revert decisions stand out from run-to-run noise."""
    import decision_robustness as robust

    campaign = robust.load_campaign(date)
    ablations = robust.load_ablations(date)
    sigma = robust.pooled_sigma(campaign, ablations)
    if not campaign or not sigma:
        return []
    lines = [
        "## Are the keep-or-revert decisions above noise?",
        "",
        "σ is the pooled within-seed standard deviation of global validation accuracy over repeated",
        "trainings of the same configuration and seed. A decision compares two measurements, so a",
        "difference beyond 2·σ·√2 (about 95 %) is *clear*; smaller differences are *within noise*. The",
        "default rule keeps changes down to 0.5 points below the best, well inside that band, so a single",
        "kept HPO, aggregation or synchronization decision is not evidence of an improvement; the paired",
        "test below shows where repeats do settle it. The supervisor now takes the tolerance from the goal",
        "(`accept_tolerance`), so a run can require a",
        f"gain larger than the noise. Every decision is listed in `decision-robustness-{date}.md`.",
        "",
        "| Dataset | α | σ (points) | 95 % band | Decisions | Clear gain | Clear loss | Within noise |",
        "|---|---|---|---|---|---|---|---|",
    ]
    per_step: dict[str, list[int]] = {}
    for dataset, dname in DATASETS:
        for split, alpha in SPLITS:
            runs = campaign.get((dataset, split))
            if not runs or (dataset, split) not in sigma:
                continue
            s, _dof = sigma[(dataset, split)]
            band = 2 * s * 2 ** 0.5
            counts = [0, 0, 0]
            for run in runs:
                for row in robust.decisions(run):
                    index = 0 if row["delta"] > band else 1 if row["delta"] < -band else 2
                    counts[index] += 1
                    per_step.setdefault(robust.STEPS.get(row["step"], row["step"]), [0, 0, 0])[index] += 1
            lines.append(f"| {dname} | {alpha} | {100 * s:.2f} | ±{100 * band:.2f} | {sum(counts)} | {counts[0]} | {counts[1]} | {counts[2]} |")
    lines += ["", "| Step | Decisions | Clear gain | Clear loss | Within noise |", "|---|---|---|---|---|"]
    for step in robust.STEPS.values():
        if step in per_step:
            gain, loss, noise = per_step[step]
            lines.append(f"| {step} | {gain + loss + noise} | {gain} | {loss} | {noise} |")
    paired = [row for row in robust.paired_components(ablations) if row["half_width"] is not None]
    if paired:
        lines += [
            "",
            "Components that only one seed kept, tested over the repeated ablation passes (full minus without,",
            "test points, 95 % t interval):",
            "",
            "| Dataset | α | Seed | Component | Passes | Mean | 95 % interval |",
            "|---|---|---|---|---|---|---|",
        ]
        for row in paired:
            lines.append(
                f"| {dict(DATASETS)[row['dataset']]} | {dict(SPLITS)[row['split']]} | {row['seed']} | {row['component']} | "
                f"{len(row['deltas'])} | {100 * row['mean']:+.2f} | "
                f"[{100 * (row['mean'] - row['half_width']):+.2f}, {100 * (row['mean'] + row['half_width']):+.2f}] |"
            )
    return lines + [""]


def limitations_section(data: dict) -> list[str]:
    """What these results do not show."""
    best = [
        r["final_test_accuracy"]
        for runs in data.values()
        for r in runs
        if isinstance(r.get("final_test_accuracy"), (int, float))
    ]
    target = next((r.get("target_accuracy") for runs in data.values() for r in runs if r.get("target_accuracy")), 0.99)
    return [
        "## Limitations",
        "",
        f"* **Target accuracy.** Every run's goal was {100 * target:.0f} % so that the supervisor would never stop",
        f"  early and every step would be measured. No run reached it (best final test {100 * max(best):.2f} %);",
        "  the target is a stopping rule, not a claim. Results are reported as gains over the baseline at",
        "  an equal training budget.",
        "* **Run-to-run noise.** Training on the real cluster is not deterministic. At Fashion-MNIST α 0.1 the",
        "  same configuration and seed vary by up to about 9 points, and the spread between seeds is larger",
        "  still, so that setting is reported over repeats and its per-step effects are wide.",
        "* **Small effects.** HPO, aggregation and synchronization change accuracy by about ±1 point on",
        "  average, inside the noise band; only NAS and the topology change it clearly.",
        "* **Communication.** The selected mesh sends 7 times as many models per round as the directed ring.",
        "  The in-degree study separates graph connectivity from the number of models received.",
        "* **Learned topology (APG).** APG halves the mesh's communication but loses accuracy, most at α 0.1;",
        "  it optimizes each worker's local fit and waiting time, not global accuracy.",
        "* **Scope.** One cluster of 8 Jetson workers on Wi-Fi, two 28 × 28 grayscale datasets and three",
        "  label-skew levels. CIFAR-10 was run with the non-fair (search-budget) supervisor, not in this",
        "  equal-budget mode, so it is not part of these tables.",
        "",
    ]


def load(date: str) -> dict[tuple[str, str], list[dict]]:
    """Completed campaign runs per setting, with extra seeds and re-runs, minus excluded runs."""
    found = {}
    for dataset, _ in DATASETS:
        for split, _ in SPLITS:
            runs = campaign_runs(dataset, split, date)
            if runs:
                found[(dataset, split)] = runs
    return found


def build(date: str) -> str:
    data = load(date)
    lines = [
        "# Fair step-by-step evaluation: Fashion-MNIST and FEMNIST",
        "",
        f"Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} by `scripts/fair_summary.py`.",
        "",
        "Eight Jetson workers, decentralized training, Dirichlet label split with α = 0.1, 0.5 and 0.9,",
        "seeds 17, 42 and 2026 (Fashion-MNIST α 0.1 also 7 and 99). Runs whose baseline started from",
        "trained weights are excluded and re-run (`excluded-runs.json`). Every trained model (baseline",
        "and each step's measurement) uses the",
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
            best = [r["final_test_accuracy"] for r in runs if isinstance(r.get("final_test_accuracy"), (int, float))]
            gains = [r["final_test_accuracy"] - r["baseline_test_accuracy"] for r in runs
                     if isinstance(r.get("final_test_accuracy"), (int, float)) and isinstance(r.get("baseline_test_accuracy"), (int, float))]
            params = sorted({f"{r['final_parameter_count']:,}" for r in runs if r.get("final_parameter_count")})
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

    lines += communication_section(data, date)
    lines += apg_section(date)
    lines += ablation_section(data, date)
    lines += topology_degree_section(data, date)
    lines += decision_section(date)
    lines += limitations_section(data)
    lines += ["## Per-seed accuracy after each step (validation %)", ""]
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
                best = run.get("final_test_accuracy")
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
