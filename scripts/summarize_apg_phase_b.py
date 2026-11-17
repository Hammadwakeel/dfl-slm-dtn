"""Summarize saved APG Phase B results and create read-only diagnostics."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results_apg"
RESULTS_JSON = RESULTS / "final_metrics.json"
SUMMARY_MD = RESULTS / "SUMMARY_TABLES.md"
NODES = tuple(f"node_{index}" for index in range(8))
TOPOLOGIES = ("ring", "mesh", "apg")
MODES = ("all_simulated", "hybrid_shaped_simulation")
SETTINGS = ("iid_stratified", "constrained_dirichlet_a01")
SEEDS = (17, 42, 2026)
MASK = ~np.eye(len(NODES), dtype=bool)


def _read_results() -> dict[str, Any]:
    result = json.loads(RESULTS_JSON.read_text(encoding="utf-8"))
    if result.get("status") != "COMPLETED" or len(result.get("runs", [])) != 36:
        raise ValueError("Expected the accepted 36-run full-matrix result")
    return result


def _mean_sd(values: list[float]) -> tuple[float, float]:
    return float(np.mean(values)), float(np.std(values, ddof=1))


def _fmt(mean: float, sd: float) -> str:
    return f"{mean:.4f} +/- {sd:.4f}"


def _run_map(results: dict[str, Any]) -> dict[tuple[str, str, str, int], dict[str, Any]]:
    mapped = {
        (
            run["simulation_mode"],
            run["data_setting"],
            run["topology"],
            int(run["seed"]),
        ): run
        for run in results["runs"]
    }
    expected = len(MODES) * len(SETTINGS) * len(TOPOLOGIES) * len(SEEDS)
    if len(mapped) != expected:
        raise ValueError(f"Expected {expected} unique runs; found {len(mapped)}")
    return mapped


def _classification(mean_delta: float, sample_sd: float) -> str:
    if mean_delta > sample_sd:
        return "better beyond seed noise"
    if mean_delta < -sample_sd:
        return "worse beyond seed noise"
    return "equal / within seed noise"


def _summary_table(
    run_map: dict[tuple[str, str, str, int], dict[str, Any]],
) -> tuple[list[str], list[str]]:
    rows = [
        "| Mode | Data setting | Topology | TEST accuracy (mean +/- SD) | TEST macro-F1 | "
        "TEST macro precision | TEST macro recall | TEST accuracy by seed (17 / 42 / 2026) | "
        "VALIDATION accuracy, final round (mean +/- SD) | SIMULATED bytes total (3 seeds) | "
        "SIMULATED bytes/run mean | SIMULATED mean edges used |",
        "|---|---|---|---:|---:|---:|---:|---|---:|---:|---:|---:|",
    ]
    comparisons = [
        "| Mode | Data setting | Comparison | TEST accuracy paired mean delta (APG - fixed) | "
        "Paired SD (ddof=1) | n | Assessment |",
        "|---|---|---|---:|---:|---:|---|",
    ]
    for mode in MODES:
        for setting in SETTINGS:
            for topology in TOPOLOGIES:
                runs = [run_map[(mode, setting, topology, seed)] for seed in SEEDS]
                metric_values = {
                    metric: [float(run["final_test"]["summary"][metric]) for run in runs]
                    for metric in ("accuracy", "macro_f1", "macro_precision", "macro_recall")
                }
                metrics = {name: _fmt(*_mean_sd(values)) for name, values in metric_values.items()}
                val_last = [float(run["rounds"][-1]["mean_validation_accuracy"]) for run in runs]
                val_summary = _fmt(*_mean_sd(val_last))
                bytes_per_run = [
                    sum(int(round_info["bytes_sent_simulated"]) for round_info in run["rounds"])
                    for run in runs
                ]
                edge_counts = [
                    int(round_info["edge_count_used"])
                    for run in runs
                    for round_info in run["rounds"]
                ]
                rows.append(
                    f"| {mode} | {setting} | {topology} | {metrics['accuracy']} | "
                    f"{metrics['macro_f1']} | {metrics['macro_precision']} | "
                    f"{metrics['macro_recall']} | "
                    + " / ".join(f"{value:.4f}" for value in metric_values["accuracy"])
                    + f" | {val_summary} | {sum(bytes_per_run):,} | "
                    f"{np.mean(bytes_per_run):,.0f} | {np.mean(edge_counts):.2f} |"
                )

            apg_runs = [run_map[(mode, setting, "apg", seed)] for seed in SEEDS]
            for comparator in ("ring", "mesh"):
                fixed_runs = [run_map[(mode, setting, comparator, seed)] for seed in SEEDS]
                deltas = [
                    float(apg["final_test"]["summary"]["accuracy"])
                    - float(fixed["final_test"]["summary"]["accuracy"])
                    for apg, fixed in zip(apg_runs, fixed_runs)
                ]
                mean_delta, sd_delta = _mean_sd(deltas)
                adapting = any(
                    int(round_info.get("apg_edges_changed_for_next_round") or 0) > 0
                    for run in apg_runs
                    for round_info in run["rounds"]
                )
                assessment = _classification(mean_delta, sd_delta)
                comparisons.append(
                    f"| {mode} | {setting} | APG - {comparator} | {mean_delta:+.4f} | "
                    f"{sd_delta:.4f} | {len(deltas)} | {assessment}; APG adapted={adapting} |"
                )
    return rows, comparisons


def _strongly_connected(edges: list[list[str]]) -> bool:
    adjacency = {node: [] for node in NODES}
    for receiver, source in edges:
        adjacency[source].append(receiver)

    def reachable(start: str) -> set[str]:
        seen = {start}
        stack = [start]
        while stack:
            node = stack.pop()
            for neighbor in adjacency[node]:
                if neighbor not in seen:
                    seen.add(neighbor)
                    stack.append(neighbor)
        return seen

    return all(len(reachable(node)) == len(NODES) for node in NODES)


def _graph_diagnostics(
    run_map: dict[tuple[str, str, str, int], dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[str]]:
    records: list[dict[str, Any]] = []
    summary = [
        "| Data setting | Simulation mode | APG runs | APG rounds | Graph-changing rounds | "
        "Changed after round 2 | Mean nodes with out-degree 0 (min-max) | "
        "Out-degree-zero node-rounds | Strongly connected rounds | In-degree min-max | "
        "Out-degree min-max |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for setting in SETTINGS:
        for mode in MODES:
            runs = [run_map[(mode, setting, "apg", seed)] for seed in SEEDS]
            zero_out = 0
            connected = 0
            in_degrees: list[int] = []
            out_degrees: list[int] = []
            zero_out_per_round: list[int] = []
            graph_changes = 0
            changes_after_round2 = 0
            for run in runs:
                for round_info in run["rounds"]:
                    edges = round_info["edges_used"]
                    indegree = {node: 0 for node in NODES}
                    outdegree = {node: 0 for node in NODES}
                    for receiver, source in edges:
                        indegree[receiver] += 1
                        outdegree[source] += 1
                    changed = int(round_info.get("edges_changed_vs_previous_round") or 0) > 0
                    if changed:
                        graph_changes += 1
                    if round_info["round"] >= 3 and changed:
                        changes_after_round2 += 1
                    zero_count = sum(degree == 0 for degree in outdegree.values())
                    zero_out += zero_count
                    zero_out_per_round.append(zero_count)
                    connected_round = _strongly_connected(edges)
                    connected += int(connected_round)
                    in_degrees.extend(indegree.values())
                    out_degrees.extend(outdegree.values())
                    records.append(
                        {
                            "run_id": run["run_id"],
                            "simulation_mode": mode,
                            "data_setting": setting,
                            "seed": run["seed"],
                            "round": round_info["round"],
                            "out_degree_zero_nodes": sum(
                                degree == 0 for degree in outdegree.values()
                            ),
                            "strongly_connected": connected_round,
                            "min_in_degree": min(indegree.values()),
                            "max_in_degree": max(indegree.values()),
                            "min_out_degree": min(outdegree.values()),
                            "max_out_degree": max(outdegree.values()),
                            "edges_changed_vs_previous_round": round_info[
                                "edges_changed_vs_previous_round"
                            ],
                            "edge_count_used": round_info["edge_count_used"],
                        }
                    )
            total_rounds = len(runs) * len(runs[0]["rounds"])
            summary.append(
                f"| {setting} | {mode} | {len(runs)} | {total_rounds} | "
                f"{graph_changes} | {changes_after_round2} | "
                f"{np.mean(zero_out_per_round):.2f} ({min(zero_out_per_round)}-"
                f"{max(zero_out_per_round)}) | {zero_out} | {connected}/{total_rounds} | "
                f"{min(in_degrees)}-{max(in_degrees)} | {min(out_degrees)}-{max(out_degrees)} |"
            )
    return records, summary


def _matrix(values: list[list[float | None]]) -> np.ndarray:
    return np.asarray(
        [[0.0 if value is None else float(value) for value in row] for row in values],
        dtype=np.float64,
    )


def _term_magnitudes(
    run_map: dict[tuple[str, str, str, int], dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[str]]:
    records: list[dict[str, Any]] = []
    summary = [
        "| Data setting | Simulation mode | Gradient contribution (off-diagonal Frobenius norm) | Min | Median | Max |",
        "|---|---|---|---:|---:|---:|",
    ]
    for setting in SETTINGS:
        for mode in MODES:
            for seed in SEEDS:
                run = run_map[(mode, setting, "apg", seed)]
                previous_weights = np.full((8, 8), 1.0 / 7.0, dtype=np.float64)
                np.fill_diagonal(previous_weights, 0.0)
                for round_info in run["rounds"]:
                    inputs = round_info["apg_inputs"]
                    loss = _matrix(inputs["loss_matrix_normalized"])
                    wait = _matrix(inputs["wait_matrix_normalized"])
                    theta = np.asarray(inputs["theta_train_only"], dtype=np.float64)
                    weights = np.asarray(round_info["apg_weights"], dtype=np.float64)
                    n = len(NODES)
                    gamma, mu, lambda_ = 0.6, 0.2, 0.2
                    alpha, beta = 0.6, 0.4
                    global_mix = np.tile(theta.mean(axis=0, keepdims=True), (n, 1))
                    term_loss = gamma * alpha * loss
                    term_time = gamma * beta * wait
                    term_f = term_loss + term_time
                    term_h = (2.0 * mu / n) * ((weights @ theta - global_mix) @ theta.T)
                    term_r = 2.0 * lambda_ * (weights - previous_weights)
                    row = {
                        "run_id": run["run_id"],
                        "simulation_mode": mode,
                        "data_setting": setting,
                        "seed": seed,
                        "round": round_info["round"],
                        "gamma_F_norm": float(np.linalg.norm(term_f[MASK])),
                        "gamma_alpha_L_norm": float(np.linalg.norm(term_loss[MASK])),
                        "gamma_beta_T_synthetic_norm": float(np.linalg.norm(term_time[MASK])),
                        "mu_H_theta_norm": float(np.linalg.norm(term_h[MASK])),
                        "lambda_R_regularizer_norm": float(np.linalg.norm(term_r[MASK])),
                        "objective_abs_change_at_stop": float(
                            round_info["stop_abs_objective_change"]
                        ),
                        "stop_psi": float(round_info["stop_tolerance_psi"]),
                        "iterations_used": int(round_info["iterations_used"]),
                    }
                    records.append(row)
                    previous_weights = weights
            for key, label in (
                ("gamma_F_norm", "gamma*F (loss + synthetic T)"),
                ("gamma_alpha_L_norm", "gamma*alpha*L contribution"),
                ("gamma_beta_T_synthetic_norm", "gamma*beta*T synthetic contribution"),
                ("mu_H_theta_norm", "mu*H (Theta balance)"),
                ("lambda_R_regularizer_norm", "lambda*R (regularizer)"),
            ):
                values = [
                    float(record[key])
                    for record in records
                    if record["data_setting"] == setting
                    and record["simulation_mode"] == mode
                ]
                summary.append(
                    f"| {setting} | {mode} | {label} | {min(values):.6g} | "
                    f"{float(np.median(values)):.6g} | {max(values):.6g} |"
                )
    return records, summary


def _write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    if not records:
        raise ValueError(f"Refusing to write an empty diagnostic CSV: {path}")
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def _plot_validation(
    run_map: dict[tuple[str, str, str, int], dict[str, Any]],
) -> list[str]:
    names = []
    for mode in MODES:
        for setting in SETTINGS:
            for topology in TOPOLOGIES:
                series = [
                    run_map[(mode, setting, topology, seed)]["rounds"] for seed in SEEDS
                ]
                rounds = np.arange(1, len(series[0]) + 1)
                for metric, title, filename_metric in (
                    ("mean_validation_accuracy", "accuracy", "accuracy"),
                    ("mean_validation_macro_f1", "macro-F1", "macro_f1"),
                ):
                    values = np.asarray(
                        [[float(info[metric]) for info in run] for run in series],
                        dtype=np.float64,
                    )
                    mean = values.mean(axis=0)
                    sd = values.std(axis=0, ddof=1)
                    figure, axis = plt.subplots(figsize=(8, 4.5))
                    axis.plot(rounds, mean, color="#176b87", label="Mean across 3 seeds")
                    axis.fill_between(rounds, mean - sd, mean + sd, color="#176b87", alpha=0.2, label=" +/- 1 sample SD")
                    axis.set(
                        title=f"VALIDATION {title} · {setting} · {topology} · {mode} · SIMULATED",
                        xlabel="Round",
                        ylabel=f"VALIDATION {title}",
                    )
                    axis.set_ylim(0, 1)
                    axis.grid(alpha=0.2)
                    axis.legend(frameon=False)
                    figure.tight_layout()
                    name = f"validation_{filename_metric}_{mode}_{setting}_{topology}.png"
                    figure.savefig(RESULTS / name, dpi=150)
                    plt.close(figure)
                    names.append(name)
    return names


def _plot_test_bars(
    run_map: dict[tuple[str, str, str, int], dict[str, Any]],
) -> list[str]:
    names = []
    for mode in MODES:
        for setting in SETTINGS:
            means, deviations = [], []
            for topology in TOPOLOGIES:
                values = [
                    float(
                        run_map[(mode, setting, topology, seed)]["final_test"]["summary"][
                            "accuracy"
                        ]
                    )
                    for seed in SEEDS
                ]
                mean, sd = _mean_sd(values)
                means.append(mean)
                deviations.append(sd)
            figure, axis = plt.subplots(figsize=(7, 4.5))
            axis.bar(TOPOLOGIES, means, yerr=deviations, capsize=5, color=["#4986a8", "#62a37c", "#bb8056"])
            axis.set_ylim(0, 1)
            axis.set_ylabel("TEST accuracy")
            axis.set_title(f"TEST accuracy +/- sample SD · {setting} · {mode} · SIMULATED (n=3)")
            axis.grid(axis="y", alpha=0.2)
            figure.tight_layout()
            name = f"test_accuracy_{mode}_{setting}.png"
            figure.savefig(RESULTS / name, dpi=150)
            plt.close(figure)
            names.append(name)
    return names


def _plot_confusions(
    run_map: dict[tuple[str, str, str, int], dict[str, Any]],
) -> list[str]:
    names = []
    for mode in MODES:
        for setting in SETTINGS:
            for topology in TOPOLOGIES:
                confusion = np.zeros((10, 10), dtype=np.int64)
                for seed in SEEDS:
                    test = run_map[(mode, setting, topology, seed)]["final_test"]
                    true = np.asarray(test["y_true"], dtype=np.int64)
                    for node in NODES:
                        predicted = np.asarray(test["y_pred_by_node"][node], dtype=np.int64)
                        np.add.at(confusion, (true, predicted), 1)
                row_sums = confusion.sum(axis=1, keepdims=True)
                normalized = np.divide(
                    confusion * 100.0,
                    row_sums,
                    out=np.zeros_like(confusion, dtype=np.float64),
                    where=row_sums > 0,
                )
                figure, axes = plt.subplots(1, 2, figsize=(12, 5))
                for axis, matrix, label, fmt in (
                    (axes[0], confusion, "Counts, seed-summed", "d"),
                    (axes[1], normalized, "Row-normalized percent", ".1f"),
                ):
                    image = axis.imshow(matrix, cmap="Blues")
                    axis.set(title=label, xlabel="Predicted class", ylabel="True class")
                    axis.set_xticks(range(10), [str(i) for i in range(10)], rotation=45, ha="right")
                    axis.set_yticks(range(10), [str(i) for i in range(10)])
                    threshold = float(np.max(matrix)) / 2.0
                    for i in range(10):
                        for j in range(10):
                            axis.text(j, i, format(matrix[i, j], fmt), ha="center", va="center", fontsize=7, color="white" if matrix[i, j] > threshold else "black")
                    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
                figure.suptitle(f"TEST confusion · {topology} · {setting} · {mode} · SIMULATED (seed-summed)")
                figure.tight_layout()
                name = f"test_confusion_{topology}_{mode}_{setting}.png"
                figure.savefig(RESULTS / name, dpi=150)
                plt.close(figure)
                names.append(name)
    return names


def _plot_apg_edges(
    run_map: dict[tuple[str, str, str, int], dict[str, Any]],
) -> list[str]:
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for axis, setting in zip(axes, SETTINGS):
        for mode in MODES:
            arrays = np.asarray(
                [
                    [
                        float(run_map[(mode, setting, "apg", seed)]["rounds"][i]["edge_count_used"])
                        for i in range(30)
                    ]
                    for seed in SEEDS
                ],
                dtype=np.float64,
            )
            mean = arrays.mean(axis=0)
            sd = arrays.std(axis=0, ddof=1)
            rounds = np.arange(1, 31)
            axis.plot(rounds, mean, label=mode)
            axis.fill_between(rounds, mean - sd, mean + sd, alpha=0.15)
        axis.set(title=f"{setting} · APG · SIMULATED", xlabel="Round", ylabel="Edges used")
        axis.grid(alpha=0.2)
        axis.legend(frameon=False)
    figure.suptitle("APG edges used per round, mean +/- sample SD across seeds · SIMULATED")
    figure.tight_layout()
    name = "apg_edges_per_round.png"
    figure.savefig(RESULTS / name, dpi=150)
    plt.close(figure)
    return [name]


def _write_summary(
    results: dict[str, Any],
    run_map: dict[tuple[str, str, str, int], dict[str, Any]],
    graph_summary: list[str],
    term_summary: list[str],
    charts: list[str],
) -> None:
    main_rows, comparisons = _summary_table(run_map)
    verdicts = []
    for item in results["apg_benefit_checks"]:
        verdicts.append(
            f"| {item['simulation_mode']} | {item['data_setting']} | "
            f"{item['verdict']} |"
        )
    sensitivity_section = ""
    sensitivity_path = RESULTS / "smoke_apg_all_simulated_iid_seed17_psi000001.json"
    if sensitivity_path.exists():
        baseline_path = RESULTS / "smoke_apg_all_simulated_iid_seed17.json"
        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
        sensitivity = json.loads(sensitivity_path.read_text(encoding="utf-8"))
        sensitivity_rows = [
            "| Round | psi=0.01 iterations | psi=1e-6 iterations | "
            "Selected edges differ |",
            "|---:|---:|---:|---|",
        ]
        for baseline_round, sensitivity_round in zip(
            baseline["rounds"], sensitivity["rounds"]
        ):
            differs = (
                baseline_round["apg_selected_edges_for_next_round"]
                != sensitivity_round["apg_selected_edges_for_next_round"]
            )
            sensitivity_rows.append(
                f"| {baseline_round['round']} | {baseline_round['iterations_used']} | "
                f"{sensitivity_round['iterations_used']} | {differs} |"
            )
        sensitivity_section = (
            "## Smoke-only VALIDATION sensitivity (SIMULATED)\n\n"
            "Separate three-round TRAIN/VALIDATION smoke; TEST was not opened. "
            "Informational only; the full matrix remains at psi=0.01.\n\n"
            + "\n".join(sensitivity_rows)
            + "\n\n"
        )
    report = f"""# APG full-matrix summary

All reported training, communication, TEST, and VALIDATION results are **SIMULATED**. TEST metrics are from the saved TEST predictions after the full training matrix completed. TEST results were not used for tuning. Mean +/- SD uses three seeds and sample SD (ddof=1). Communication bytes and graph edges are simulation outputs, not physical measurements.

## TEST, VALIDATION, and communication summary (SIMULATED)

`VALIDATION accuracy, final round` is reported separately from TEST. `SIMULATED mean edges used` averages all 30 rounds and three seeds. `SIMULATED bytes total` sums sent bytes across the three seed runs; mean bytes/run is also shown.

{chr(10).join(main_rows)}

## Paired TEST accuracy comparisons (SIMULATED)

Each APG-minus-fixed delta pairs the same three seeds. Assessment labels **better** if mean delta > paired sample SD, **worse** if mean delta < -paired sample SD, otherwise **equal / within seed noise**. The report's stricter "no demonstrated benefit" rule requires APG to adapt and mean paired accuracy delta to exceed paired SD against both Ring and Mesh.

{chr(10).join(comparisons)}

## APG graph diagnostics (SIMULATED)

Receiver-centric convention: each edge [receiver, source] means the receiver receives from that source. In-degree counts selected sources; out-degree counts receivers for which the node is a source. Strong connectivity uses source-to-receiver direction. Each setting/mode has 3 APG runs x 30 rounds.

{chr(10).join(graph_summary)}

The "rounds changed after round 2" count includes round >=3 with a nonzero change in the graph used relative to the prior round.

## APG gradient term magnitudes (SIMULATED)

Norms are off-diagonal Frobenius norms of gradient contributions at the saved final APG weights for each round, using the prior round's saved APG weights (initially uniform mesh weights). `gamma*F` is decomposed into its loss and synthetic-T contributions for interpretation. T is synthetic; it is unavailable as a per-edge hardware measurement. The optimization uses accelerated extrapolation, so these final-weight norms are diagnostic term magnitudes, not a causal attribution of selected edges.

{chr(10).join(term_summary)}

Each APG run/round's exact norms, objective change at stop, psi, and iterations are in `apg_gradient_terms.csv`.

## Generated charts

{chr(10).join(f"- `{name}` — VALIDATION / TEST and SIMULATED as indicated in the chart title." for name in charts)}

## Verdicts and limitations

| Simulation mode | Data setting | TEST paired-seed verdict |
|---|---|---|
{chr(10).join(verdicts)}

{sensitivity_section}

The four matrix-level verdicts above follow the predeclared paired-seed SD rule. The per-comparator table distinguishes mean differences beyond seed noise from differences within the paired SD; it does not claim statistical significance. The hybrid-shaped run is an in-process emulation, not a physical hybrid run. Real per-edge T, real Jetson timing, hardware communication, and PSO-GNN results are **NOT AVAILABLE**. APG's T timing input is synthetic; psi remains the spec default 0.01, which stops APG after very few iterations in this matrix.
"""
    SUMMARY_MD.write_text(report, encoding="utf-8")


def main() -> None:
    results = _read_results()
    run_map = _run_map(results)
    graph_records, graph_summary = _graph_diagnostics(run_map)
    term_records, term_summary = _term_magnitudes(run_map)
    _write_csv(RESULTS / "apg_graph_diagnostics.csv", graph_records)
    _write_csv(RESULTS / "apg_gradient_terms.csv", term_records)
    charts = (
        _plot_validation(run_map)
        + _plot_test_bars(run_map)
        + _plot_confusions(run_map)
        + _plot_apg_edges(run_map)
    )
    _write_summary(results, run_map, graph_summary, term_summary, charts)
    print(f"summary={SUMMARY_MD.relative_to(ROOT)}")
    print(f"graph_diagnostics_rows={len(graph_records)}")
    print(f"gradient_term_rows={len(term_records)}")
    print(f"charts_created={len(charts)}")
    for path in [SUMMARY_MD, RESULTS / "apg_graph_diagnostics.csv", RESULTS / "apg_gradient_terms.csv"]:
        print(f"created={path.relative_to(ROOT)}")
    for name in charts:
        print(f"created=results_apg/{name}")


if __name__ == "__main__":
    main()
