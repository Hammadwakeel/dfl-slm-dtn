"""Slide-ready figures (16:9 PNG) from the fair-campaign evidence.

    python scripts/paper_figures.py

Writes docs/figures/*.png. Every number is read from docs/evidence (the same
loaders as the summary tables, so excluded runs stay excluded).
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import decision_robustness as robust  # noqa: E402
import topology_degree_study as degree  # noqa: E402
from fair_summary import ABLATION_VARIANTS, _ablation_passes, _apg_runs, _learned_edges, _test, step_effects  # noqa: E402
from four_dataset_table import DATASETS, SPLITS  # noqa: E402
from four_dataset_table import load as load_campaigns  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "figures"
DATE = "2026-11-21"

# Reference palette (light mode), validated with the dataviz validator.
SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]  # blue, orange, aqua, yellow
BASELINE_GRAY = "#b9b7af"
NEUTRAL, COOL, WARM = "#f0efec", "#2a78d6", "#e34948"
ALPHAS = [alpha for _, alpha in SPLITS]

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 14,
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "axes.edgecolor": MUTED,
    "axes.labelcolor": INK2,
    "axes.facecolor": SURFACE,
    "figure.facecolor": SURFACE,
    "xtick.color": INK2,
    "ytick.color": INK2,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.grid.axis": "y",
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "axes.axisbelow": True,
    "legend.frameon": False,
})
SIZE = (13.33, 7.5)


def save(fig, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.png", dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print(OUT / f"{name}.png")


def title(fig, text: str, sub: str) -> None:
    fig.suptitle(text, x=0.01, ha="left", fontsize=20, fontweight="bold", color=INK, y=1.0)
    fig.text(0.01, 0.935, sub, ha="left", fontsize=13, color=INK2)


def mean_std(values):
    values = [100 * v for v in values]
    return (statistics.fmean(values), statistics.stdev(values) if len(values) > 1 else 0.0) if values else (None, None)


def fig_baseline_vs_final(data) -> None:
    fig, axes = plt.subplots(1, 4, figsize=SIZE, sharey=True)
    width = 0.36
    for ax, (dataset, name) in zip(axes, DATASETS):
        for i, (split, alpha) in enumerate(SPLITS):
            runs = data.get((dataset, split), [])
            base = mean_std([r["baseline_test_accuracy"] for r in runs])
            final = mean_std([r["best_test_accuracy"] for r in runs])
            if base[0] is None:
                continue
            for offset, (value, std), color in ((-width / 2 - 0.01, base, BASELINE_GRAY), (width / 2 + 0.01, final, SERIES[0])):
                ax.bar(i + offset, value, width, color=color, yerr=std, ecolor=INK2, capsize=3, error_kw={"linewidth": 1})
                ax.text(i + offset, value + std + 1.2, f"{value:.0f}", ha="center", va="bottom", fontsize=11, color=INK)
        ax.set_title(name, color=INK, loc="left")
        counts = [len(data.get((dataset, split), [])) for split, _ in SPLITS]
        ax.set_xticks(range(len(ALPHAS)), [f"α {a}\nn={n}" for a, n in zip(ALPHAS, counts)])
        ax.set_ylim(0, 110)
    axes[0].set_ylabel("Test accuracy (%)")
    handles = [plt.Rectangle((0, 0), 1, 1, color=BASELINE_GRAY), plt.Rectangle((0, 0), 1, 1, color=SERIES[0])]
    fig.legend(handles, ["Baseline (directed ring, baseline CNN)", "Final model chosen by the agent"], loc="lower center",
               ncol=2, bbox_to_anchor=(0.5, -0.03), fontsize=12)
    title(fig, "The agent improves every dataset, most where data is most skewed",
          "Mean test accuracy over the 8 workers, equal training budget (10 rounds × 5 epochs); bars ± std over seeds (n); smaller α = more label skew")
    fig.tight_layout(rect=(0, 0.04, 1, 0.92))
    save(fig, "01-baseline-vs-final")


def fig_gain_vs_alpha(data) -> None:
    fig, ax = plt.subplots(figsize=SIZE)
    x = [float(a) for a in ALPHAS]
    labels = []
    for color, (dataset, name) in zip(SERIES, DATASETS):
        gains, stds = [], []
        for split, _ in SPLITS:
            runs = data.get((dataset, split), [])
            g = mean_std([r["best_test_accuracy"] - r["baseline_test_accuracy"] for r in runs])
            gains.append(g[0])
            stds.append(g[1])
        ax.errorbar(x, gains, yerr=stds, color=color, linewidth=2, marker="o", markersize=9, capsize=4,
                    markeredgecolor=SURFACE, markeredgewidth=2, label=name)
        labels.append([gains[0], f"{name}  +{gains[0]:.1f}"])
    # Direct labels at α 0.1, pushed apart so close values do not collide.
    labels.sort(key=lambda item: item[0])
    for k in range(1, len(labels)):
        labels[k][0] = max(labels[k][0], labels[k - 1][0] + 2.6)
    for y, text in labels:
        ax.text(x[0] - 0.03, y, text, ha="right", va="center", fontsize=12, color=INK)
    ax.set_xticks(x, [f"α {a}" for a in ALPHAS])
    ax.set_xlim(-0.25, 1.0)
    ax.set_ylim(-5, 50)
    ax.axhline(0, color=MUTED, linewidth=1)
    ax.set_ylabel("Gain over baseline (test points)")
    ax.set_xlabel("Dirichlet α (label skew decreases →)")
    ax.legend(loc="upper right", ncol=4, fontsize=12)
    title(fig, "The harder the data split, the larger the agent's gain",
          "Final model minus baseline, test points, mean ± std over seeds (Fashion-MNIST α 0.1: 5 seeds, otherwise 3)")
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    save(fig, "02-gain-vs-alpha")


def fig_step_heatmap(data) -> None:
    steps = [("optimize_topology", "Topology"), ("run_nas", "NAS"), ("optimize_aggregation", "Aggregation"),
             ("run_hpo", "HPO"), ("optimize_sync", "Sync"), ("learn_topology", "Learned topology (APG)")]
    cols, matrix = [], []
    for dataset, name in DATASETS:
        for split, alpha in SPLITS:
            cols.append(f"{name}\nα {alpha}")
    for kind, _ in steps:
        row = []
        for dataset, _ in DATASETS:
            for split, _ in SPLITS:
                effects = [step_effects(r).get(kind) for r in data.get((dataset, split), [])]
                values = [e["d_test"] for e in effects if e and e["d_test"] is not None]
                row.append(100 * statistics.fmean(values) if values else float("nan"))
        matrix.append(row)
    limit = 30
    cmap = LinearSegmentedColormap.from_list("div", [WARM, NEUTRAL, COOL])
    fig, ax = plt.subplots(figsize=SIZE)
    ax.grid(False)
    image = ax.imshow(matrix, cmap=cmap, vmin=-limit, vmax=limit, aspect="auto")
    for i, row in enumerate(matrix):
        for j, value in enumerate(row):
            strong = abs(value) > 18
            ax.text(j, i, f"{value:+.1f}", ha="center", va="center", fontsize=11, color="#ffffff" if strong else INK)
    ax.set_xticks(range(len(cols)), [f"α {alpha}" for _ in DATASETS for _s, alpha in SPLITS], fontsize=11)
    for block, (_dataset, name) in enumerate(DATASETS):
        ax.text(block * 3 + 1, -0.62, name, ha="center", va="bottom", fontsize=14, fontweight="bold", color=INK)
    ax.set_yticks(range(len(steps)), [label for _, label in steps])
    for boundary in (2.5, 5.5, 8.5):
        ax.axvline(boundary, color=SURFACE, linewidth=4)
    for side in ("left", "bottom"):
        ax.spines[side].set_visible(False)
    bar = fig.colorbar(image, ax=ax, fraction=0.025, pad=0.01)
    bar.set_label("Δ test accuracy (points)", color=INK2)
    bar.outline.set_visible(False)
    title(fig, "Topology is the step that matters; aggregation, HPO and sync stay near zero",
          "Change in test accuracy caused by each step against the configuration before it, mean over seeds (blue = gain, red = loss)")
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    save(fig, "03-step-effects-heatmap")


def fig_degree() -> None:
    path = ROOT / "docs" / "evidence" / f"topology-degree-{DATE}.json"
    runs = [r for r in json.loads(path.read_text(encoding="utf-8"))["runs"] if r["status"] == "completed"]
    campaign = load_campaigns()
    apg = _apg_runs(DATE)
    fig, axes = plt.subplots(1, 2, figsize=SIZE, sharey=True)
    names = {"fashion_mnist": "Fashion-MNIST, α 0.1", "femnist": "FEMNIST, α 0.1"}
    for ax, (dataset, name) in zip(axes, names.items()):
        rows = [r for r in runs if r["dataset"] == dataset]
        xs, ys, es = [], [], []
        for label, (deg, rounds, _e) in degree.GRAPHS.items():
            if rounds != degree.ROUNDS:
                continue
            m = mean_std([statistics.fmean(r["tests"][label]) for r in rows if r["tests"].get(label)])
            xs.append(deg * rounds)
            ys.append(m[0])
            es.append(m[1])
        ax.errorbar(xs, ys, yerr=es, color=SERIES[0], linewidth=2, marker="o", markersize=9, capsize=4,
                    markeredgecolor=SURFACE, markeredgewidth=2, label="Graph with 1, 2, 4, 7 peers (10 rounds × 5 epochs)")
        for x, y, peers in zip(xs, ys, (1, 2, 4, 7)):
            ax.annotate(f"{peers} peer{'s' if peers > 1 else ''}: {y:.1f}", (x, y), xytext=(8, -18), textcoords="offset points",
                        fontsize=11, color=INK)
        ring = mean_std([statistics.fmean(r["tests"]["ring 70x1"]) for r in rows if r["tests"].get("ring 70x1")])
        ax.errorbar([70], [ring[0]], yerr=[ring[1]], color=SERIES[1], marker="s", markersize=10, capsize=4, linestyle="none",
                    markeredgecolor=SURFACE, markeredgewidth=2, label="Ring with mesh's communication (70 rounds × 1 epoch)")
        ax.annotate(f"ring 70×1: {ring[0]:.1f}", (70, ring[0]), xytext=(-12, 0), textcoords="offset points", ha="right",
                    va="center", fontsize=11, color=INK)
        learned, edges = [], []
        for run in campaign.get((dataset, "dirichlet_a01"), []):
            if not any(r["seed"] == run["seed"] for r in rows):
                continue
            step = next((s for s in run["steps"] if s["step"] == "learn_topology" and s.get("status") == "completed"), None)
            count = _learned_edges(apg, dataset, "dirichlet_a01", run["seed"])
            if step and count and isinstance(step.get("test_accuracy"), (int, float)):
                learned.append(step["test_accuracy"])
                edges.append(10 * count / 8)
        if learned:
            m = mean_std(learned)
            ax.errorbar([statistics.fmean(edges)], [m[0]], yerr=[m[1]], color=SERIES[2], marker="D", markersize=10, capsize=4,
                        linestyle="none", markeredgecolor=SURFACE, markeredgewidth=2, label="Learned graph (APG)")
            ax.annotate(f"APG: {m[0]:.1f}", (statistics.fmean(edges), m[0]), xytext=(10, -4), textcoords="offset points",
                        fontsize=11, color=INK)
        ax.set_title(name, color=INK, loc="left")
        ax.set_xticks([10, 20, 40, 70])
        ax.set_xlabel("Models each worker receives during training")
        ax.set_ylim(30, 85)
    axes[0].set_ylabel("Test accuracy (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=11, bbox_to_anchor=(0.5, -0.04))
    title(fig, "The mesh wins through connectivity, not just more communication",
          "Same final configuration per seed; mean ± std over 3 seeds. With as many models as the mesh, the ring is still ~15 points lower")
    fig.tight_layout(rect=(0, 0.05, 1, 0.9))
    save(fig, "04-topology-or-communication")


def fig_ablation() -> None:
    passes = _ablation_passes(DATE)
    by_seed: dict[tuple, list] = {}
    for runs in passes:
        for run in runs:
            by_seed.setdefault((run["dataset"], run["split"], run["seed"]), []).append(run)
    comps = [("without topology", "Topology"), ("without NAS", "NAS"), ("without aggregation", "Aggregation"), ("without sync", "Sync")]
    settings = [(d, n, s, a) for d, n in DATASETS for s, a in SPLITS if any(k[:2] == (d, s) for k in by_seed)]
    fig, ax = plt.subplots(figsize=SIZE)
    width = 0.2
    for c, ((variant, label), color) in enumerate(zip(comps, SERIES)):
        means, stds = [], []
        for dataset, _n, split, _a in settings:
            deltas = []
            for key, runs in by_seed.items():
                if key[:2] != (dataset, split):
                    continue
                pairs = [f - w for r in runs if (f := _test(r, "full")) is not None and (w := _test(r, variant)) is not None]
                if pairs:
                    deltas.append(statistics.fmean(pairs))
            m = mean_std(deltas)
            means.append(m[0] if m[0] is not None else 0)
            stds.append(m[1] or 0)
        xs = [i + (c - 1.5) * (width + 0.02) for i in range(len(settings))]
        ax.bar(xs, means, width, yerr=stds, color=color, ecolor=INK2, capsize=2, error_kw={"linewidth": 1}, label=label)
        if label == "Topology":
            for x, v, s in zip(xs, means, stds):
                ax.text(x, v + s + 0.8, f"{v:+.0f}", ha="center", fontsize=10, color=INK)
    ax.axhline(0, color=MUTED, linewidth=1)
    ax.set_xticks(range(len(settings)), [f"{n}\nα {a}" for _d, n, _s, a in settings], fontsize=11)
    ax.set_ylabel("Contribution (test points, full − without)")
    fig.legend(*ax.get_legend_handles_labels(), loc="lower center", ncol=4, fontsize=12, bbox_to_anchor=(0.5, -0.03))
    title(fig, "Leave-one-out ablation: removing the topology costs the most",
          "Each run's final configuration retrained without one component (same budget, seed and initial weights); mean ± std over seeds that kept it")
    fig.tight_layout(rect=(0, 0.05, 1, 0.92))
    save(fig, "05-ablation-contributions")


def fig_decisions() -> None:
    campaign = robust.load_campaign(DATE)
    sigma = robust.pooled_sigma(campaign, robust.load_ablations(DATE))
    counts: dict[str, list[int]] = {}
    for key, runs in campaign.items():
        if key not in sigma:
            continue
        band = 2 * sigma[key][0] * 2 ** 0.5
        for run in runs:
            for row in robust.decisions(run):
                index = 0 if row["delta"] > band else 2 if row["delta"] < -band else 1
                counts.setdefault(robust.STEPS.get(row["step"], row["step"]), [0, 0, 0])[index] += 1
    order = [s for s in ("topology", "NAS", "aggregation", "HPO", "sync") if s in counts]
    names = {"topology": "Topology", "NAS": "NAS", "aggregation": "Aggregation", "HPO": "HPO", "sync": "Sync"}
    fig, ax = plt.subplots(figsize=SIZE)
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    left = [0] * len(order)
    for index, (label, color) in enumerate((("Clear gain", COOL), ("Within noise", BASELINE_GRAY), ("Clear loss", WARM))):
        values = [counts[s][index] for s in order]
        ax.barh([names[s] for s in order], values, left=left, color=color, edgecolor=SURFACE, linewidth=2, label=label, height=0.6)
        for y, (l, v) in enumerate(zip(left, values)):
            if v:
                ax.text(l + v / 2, y, str(v), ha="center", va="center", fontsize=13,
                        color="#ffffff" if color != BASELINE_GRAY else INK)
        left = [l + v for l, v in zip(left, values)]
    ax.invert_yaxis()
    ax.set_xlabel("Keep-or-revert decisions (Fashion-MNIST and FEMNIST, all α and seeds)")
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    fig.legend(*ax.get_legend_handles_labels(), loc="lower center", ncol=3, fontsize=12, bbox_to_anchor=(0.5, -0.03))
    title(fig, "Topology and NAS decisions are clearly above run-to-run noise",
          "A decision is clear when its measured difference exceeds the 95 % noise band (2·σ·√2) estimated from repeated trainings")
    fig.tight_layout(rect=(0, 0.05, 1, 0.92))
    save(fig, "06-decisions-vs-noise")


def main() -> None:
    data = load_campaigns()
    fig_baseline_vs_final(data)
    fig_gain_vs_alpha(data)
    fig_step_heatmap(data)
    fig_degree()
    fig_ablation()
    fig_decisions()


if __name__ == "__main__":
    main()
