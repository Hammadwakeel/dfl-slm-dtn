import { useState } from "react";

import { deleteExperimentGroup } from "../api/client";
import type { ExperimentGroup, ExperimentResults, TrainingTopology, WorkerRoundMetrics } from "../api/client";

interface Props {
  results: ExperimentResults | null;
  error: string;
  token: string;
  onRefresh: () => void;
  onOpenModelStudio: () => void;
}

function averageFinalAccuracy(group: ExperimentGroup): number | null {
  const values = group.runs.flatMap((run) => {
    const finalRound = run.rounds[run.rounds.length - 1];
    if (!finalRound) return [];
    const accuracies = Object.values(finalRound.workers)
      .map((metrics) => metrics.validation_accuracy ?? metrics.accuracy)
      .filter((value): value is number => typeof value === "number" && Number.isFinite(value));
    if (accuracies.length === 0) return [];
    return [accuracies.reduce((sum, accuracy) => sum + accuracy, 0) / accuracies.length];
  });
  return values.length ? values.reduce((sum, value) => sum + value, 0) / values.length : null;
}

function formatPercent(value: number | undefined): string {
  return value === undefined || !Number.isFinite(value) ? "—" : `${(value * 100).toFixed(2)}%`;
}

function formatFixed(value: number | undefined, digits: number): string {
  return value === undefined || !Number.isFinite(value) ? "—" : value.toFixed(digits);
}

function runtime(metrics: WorkerRoundMetrics): number | undefined {
  return metrics.round_seconds ?? metrics.runtime_seconds ?? metrics.training_seconds;
}

function AccuracyChart({ run }: { run: ExperimentGroup["runs"][number] }) {
  const workerIds = [...new Set(run.rounds.flatMap((round) => Object.keys(round.workers)))];
  const series = workerIds.map((workerId, seriesIndex) => ({
    workerId,
    color: ["#79ddb0", "#78b7ef", "#f0bf72"][seriesIndex % 3],
    path: run.rounds.flatMap((round, roundIndex) => {
        const score = round.workers[workerId]?.accuracy ?? round.workers[workerId]?.validation_accuracy;
        if (score === undefined || !Number.isFinite(score)) return null;
        const x = 56 + (roundIndex / Math.max(run.rounds.length - 1, 1)) * 520;
        const y = 176 - score * 144;
        return [{ x, y }];
      })
      .filter((point): point is { x: number; y: number } => point !== null)
      .map((point, index) => `${index === 0 ? "M" : "L"}${point.x.toFixed(1)},${point.y.toFixed(1)}`)
      .join(" "),
  })).filter((item) => item.path);
  if (series.length === 0) return null;

  const downloadableSvg = [
    '<svg xmlns="http://www.w3.org/2000/svg" width="640" height="220" viewBox="0 0 640 220">',
    '<rect width="100%" height="100%" fill="#ffffff"/>',
    '<g stroke="#cbd5e1"><line x1="56" y1="32" x2="56" y2="176"/><line x1="56" y1="176" x2="576" y2="176"/></g>',
    '<g fill="#64748b" font-family="sans-serif" font-size="11"><text x="8" y="38">100%</text><text x="20" y="180">0%</text><text x="56" y="202">Round 1</text><text x="520" y="202">Round ' + run.rounds.length + "</text></g>",
    ...series.map((item) => `<path d="${item.path}" fill="none" stroke="${item.color}" stroke-width="3"/>`),
    "</svg>",
  ].join("");

  return (
    <div className="experiment-chart">
      <div className="experiment-chart__header">
        <h4>Validation accuracy by round</h4>
        <a
          download={`${run.run_id}-accuracy.svg`}
          href={`data:image/svg+xml;charset=utf-8,${encodeURIComponent(downloadableSvg)}`}
        >
          Download graph
        </a>
      </div>
      <svg aria-label="Validation accuracy by round" role="img" viewBox="0 0 640 220">
        <line x1="56" y1="32" x2="56" y2="176" />
        <line x1="56" y1="176" x2="576" y2="176" />
        {series.map((item) => <path d={item.path} key={item.workerId} stroke={item.color} />)}
        <text x="8" y="38">100%</text>
        <text x="20" y="180">0%</text>
        <text x="56" y="202">Round 1</text>
        <text x="520" y="202">Round {run.rounds.length}</text>
      </svg>
      <div className="experiment-chart__legend">
        {series.map((item) => (
          <span key={item.workerId}><i style={{ backgroundColor: item.color }} />{item.workerId}</span>
        ))}
      </div>
    </div>
  );
}

function ValidationConfusionMatrices({ run }: { run: ExperimentGroup["runs"][number] }) {
  const finalRound = run.rounds[run.rounds.length - 1];
  if (!finalRound) return null;
  const workers = Object.entries(finalRound.workers);
  const validMatrices = workers.flatMap(([, metrics]) => {
    const matrix = metrics.validation_confusion_matrix_counts;
    if (
      !matrix?.length ||
      matrix.some((row) => row.length !== matrix.length || row.some((value) => !Number.isFinite(value)))
    ) return [];
    return [{ matrix }];
  });
  const matrixSize = validMatrices[0]?.matrix.length ?? 0;
  const compatibleMatrices = validMatrices.filter(({ matrix }) => matrix.length === matrixSize);
  const combinedMatrix = Array.from({ length: matrixSize }, (_, row) =>
    Array.from({ length: matrixSize }, (_, column) =>
      compatibleMatrices.reduce((sum, item) => sum + item.matrix[row][column], 0),
    ),
  );
  const rowTotals = combinedMatrix.map((row) => row.reduce((sum, value) => sum + value, 0));
  const columnTotals = Array.from({ length: matrixSize }, (_, column) =>
    combinedMatrix.reduce((sum, row) => sum + row[column], 0),
  );
  const totalSamples = rowTotals.reduce((sum, count) => sum + count, 0);
  const correctPredictions = combinedMatrix.reduce(
    (sum, row, index) => sum + (row[index] ?? 0),
    0,
  );
  const perClassRows = combinedMatrix.map((row, classIndex) => {
    const truePositives = row[classIndex] ?? 0;
    const precision = columnTotals[classIndex] ? truePositives / columnTotals[classIndex] : 0;
    const recall = rowTotals[classIndex] ? truePositives / rowTotals[classIndex] : 0;
    return {
      classIndex,
      precision,
      recall,
      f1: precision + recall ? (2 * precision * recall) / (precision + recall) : 0,
    };
  });
  const macroAverage = (key: "precision" | "recall" | "f1") =>
    perClassRows.length
      ? perClassRows.reduce((sum, row) => sum + row[key], 0) / perClassRows.length
      : undefined;
  const normalizedMatrix = combinedMatrix.map((row, index) =>
    row.map((count) => rowTotals[index] ? (count / rowTotals[index]) * 100 : 0),
  );
  return (
    <div className="confusion-matrices">
      {compatibleMatrices.length > 0 && (
        <section className="combined-validation" aria-label="Combined final validation results">
          <div className="combined-validation__heading">
            <h4>Final validation results · all workers together</h4>
            <span>
              Combined from {compatibleMatrices.length} worker confusion matrices
              {compatibleMatrices.length !== validMatrices.length ? ` · ${validMatrices.length - compatibleMatrices.length} incompatible matrix excluded` : ""}
            </span>
          </div>
          <p className="combined-validation__method">
            Summed worker validation matrices · each sample counted once.
          </p>
          <div className="final-worker-metrics">
            <div><span>Overall accuracy</span><strong>{formatPercent(totalSamples ? correctPredictions / totalSamples : undefined)}</strong></div>
            <div><span>Macro precision</span><strong>{formatPercent(macroAverage("precision"))}</strong></div>
            <div><span>Macro recall</span><strong>{formatPercent(macroAverage("recall"))}</strong></div>
            <div><span>Macro F1</span><strong>{formatPercent(macroAverage("f1"))}</strong></div>
            <div><span>Combined validation samples</span><strong>{totalSamples.toLocaleString()}</strong></div>
          </div>
          <div className="table-scroll">
            <table className="metrics-table confusion-matrix">
              <caption>Combined final-round validation confusion matrix · counts</caption>
              <thead>
                <tr><th>Actual / Predicted</th>{combinedMatrix.map((_, index) => <th key={index}>{index}</th>)}</tr>
              </thead>
              <tbody>
                {combinedMatrix.map((row, rowIndex) => (
                  <tr key={rowIndex}>
                    <th>{rowIndex}</th>
                    {row.map((count, columnIndex) => <td key={columnIndex}>{count}</td>)}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <details className="normalized-confusion">
            <summary>Combined row-normalized confusion matrix (%)</summary>
            <div className="table-scroll">
              <table className="metrics-table confusion-matrix">
                <thead>
                  <tr><th>Actual / Predicted</th>{normalizedMatrix.map((_, index) => <th key={index}>{index}</th>)}</tr>
                </thead>
                <tbody>
                  {normalizedMatrix.map((row, rowIndex) => (
                    <tr key={rowIndex}>
                      <th>{rowIndex}</th>
                      {row.map((value, columnIndex) => <td key={columnIndex}>{value.toFixed(1)}%</td>)}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </details>
          <div className="table-scroll">
            <table className="metrics-table per-class-metrics">
              <caption>Combined per-class validation metrics</caption>
              <thead><tr><th>Class</th><th>Precision</th><th>Recall</th><th>F1</th></tr></thead>
              <tbody>
                {perClassRows.map((row) => (
                  <tr key={row.classIndex}>
                    <th>{row.classIndex}</th>
                    <td>{formatPercent(row.precision)}</td>
                    <td>{formatPercent(row.recall)}</td>
                    <td>{formatPercent(row.f1)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      )}
      <div className="table-scroll">
        <table className="metrics-table worker-validation-comparison">
          <caption>Final-round worker metrics · individual breakdown</caption>
          <thead>
            <tr>
              <th>Worker</th><th>Accuracy</th><th>Macro precision</th><th>Macro recall</th>
              <th>Macro F1</th><th>Samples</th><th>Training loss</th><th>Eval time</th><th>Peak RAM</th>
            </tr>
          </thead>
          <tbody>
            {workers.map(([workerId, metrics]) => (
              <tr key={workerId}>
                <th>{workerId}</th>
                <td>{formatPercent(metrics.validation_accuracy ?? metrics.accuracy)}</td>
                <td>{formatPercent(metrics.validation_macro_precision ?? metrics.macro_precision)}</td>
                <td>{formatPercent(metrics.validation_macro_recall ?? metrics.macro_recall)}</td>
                <td>{formatPercent(metrics.validation_macro_f1 ?? metrics.macro_f1)}</td>
                <td>{metrics.validation_sample_count?.toLocaleString() ?? "—"}</td>
                <td>{formatFixed(metrics.loss ?? metrics.training_loss, 4)}</td>
                <td>{metrics.evaluation_seconds === undefined ? "—" : `${metrics.evaluation_seconds.toFixed(2)} s`}</td>
                <td>{metrics.peak_process_rss_mb === undefined ? "—" : `${metrics.peak_process_rss_mb.toFixed(1)} MB`}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {compatibleMatrices.length === 0 && (
        <p className="combined-validation__method">
          No compatible worker confusion matrices.
        </p>
      )}
    </div>
  );
}

function groupDetails(group: ExperimentGroup): string {
  const details: string[] = [];
  if (typeof group.rounds === "number") {
    details.push(`${group.rounds} ${group.rounds === 1 ? "round" : "rounds"}`);
  }
  if (typeof group.epochs === "number") {
    details.push(`${group.epochs} local ${group.epochs === 1 ? "epoch" : "epochs"}`);
  }
  const seeds = group.seeds ?? (typeof group.seed === "number" ? [group.seed] : []);
  if (seeds.length > 0) {
    details.push(`${seeds.length === 1 ? "seed" : "seeds"} ${seeds.join(", ")}`);
  }
  return details.join(" · ");
}

function groupDatasetName(group: ExperimentGroup): string {
  return group.dataset_name ?? group.architecture?.dataset_name ?? "Dataset not reported";
}

function TrainingTopology({ topology }: { topology: TrainingTopology | "centralized" }) {
  let links: string[];
  if (topology === "centralized") {
    links = ["dfl-master · centralized"];
  } else if (topology === "chain") {
    links = ["worker1 ↔ worker2", "worker2 ↔ worker3"];
  } else if (topology === "star") {
    links = ["worker1 ↔ worker2", "worker1 ↔ worker3"];
  } else if (topology === "mesh") {
    links = ["worker1 ↔ worker2", "worker1 ↔ worker3", "worker2 ↔ worker3"];
  } else if (topology === "bidirectional_ring") {
    links = ["worker1 ↔ worker2", "worker2 ↔ worker3", "worker3 ↔ worker1"];
  } else {
    links = ["worker1 → worker2", "worker2 → worker3", "worker3 → worker1"];
  }
  return (
    <div className="training-topology" aria-label={`${topology} training topology`}>
      <strong>{topology} topology</strong>
      <div>{links.map((link) => <span key={link}>{link}</span>)}</div>
    </div>
  );
}

export default function ExperimentsPage({ token, results, error, onRefresh, onOpenModelStudio }: Props) {
  const [deletingGroupId, setDeletingGroupId] = useState<string | null>(null);
  const [deletingSelectedGroups, setDeletingSelectedGroups] = useState(false);
  const [deleteError, setDeleteError] = useState("");
  const [selectedGroupId, setSelectedGroupId] = useState<string | null>(null);
  const [selectedGroupIds, setSelectedGroupIds] = useState<Set<string>>(new Set());
  const allGroups = results?.groups ?? [];
  const mnistGroups = allGroups.filter((group) =>
    group.dataset_name === "mnist" || group.architecture?.dataset_name === "mnist",
  );
  const groups = [...(mnistGroups.length > 0 ? mnistGroups : allGroups)].reverse();
  const deletableGroups = groups.filter((group) =>
    group.status !== "running" && results?.active_group_id !== group.group_id,
  );
  const selectedGroups = groups.filter((group) => selectedGroupIds.has(group.group_id));
  const selectedGroup = groups.find((group) => group.group_id === selectedGroupId) ?? null;
  const bestGroup = groups
    .filter((group) => group.status === "completed" && averageFinalAccuracy(group) !== null)
    .reduce<ExperimentGroup | null>((best, group) => {
      if (best === null || (averageFinalAccuracy(group) ?? -Infinity) > (averageFinalAccuracy(best) ?? -Infinity)) {
        return group;
      }
      return best;
    }, null);
  const baseline = groups.find((group) => group.kind === "centralized_baseline");
  const decentralized = groups.find((group) => group.kind === "decentralized" && group.status === "completed");

  async function deleteGroup(group: ExperimentGroup) {
    if (!window.confirm(`Permanently delete experiment ${group.group_id} and all its saved results?`)) return;
    setDeletingGroupId(group.group_id);
    setDeleteError("");
    try {
      await deleteExperimentGroup(token, group.group_id);
      setSelectedGroupId(null);
      onRefresh();
    } catch (cause) {
      setDeleteError(cause instanceof Error ? cause.message : "Could not delete the experiment.");
    } finally {
      setDeletingGroupId(null);
    }
  }

  async function deleteSelectedGroups() {
    if (selectedGroups.length === 0 || deletingSelectedGroups) return;
    if (!window.confirm(`Permanently delete ${selectedGroups.length} selected experiment${selectedGroups.length === 1 ? "" : "s"} and all their saved results?`)) return;
    setDeletingSelectedGroups(true);
    setDeleteError("");
    try {
      const outcomes = await Promise.allSettled(
        selectedGroups.map((group) => deleteExperimentGroup(token, group.group_id)),
      );
      const deletedIds = selectedGroups
        .filter((_group, index) => outcomes[index].status === "fulfilled")
        .map((group) => group.group_id);
      const failures = outcomes.filter((outcome): outcome is PromiseRejectedResult => (
        outcome.status === "rejected"
      ));
      setSelectedGroupIds((current) => new Set(
        [...current].filter((groupId) => !deletedIds.includes(groupId)),
      ));
      if (deletedIds.length > 0) onRefresh();
      if (failures.length > 0) {
        const messages = failures
          .map(({ reason }) => reason instanceof Error ? reason.message : String(reason))
          .filter(Boolean);
        setDeleteError(
          `${deletedIds.length} experiment${deletedIds.length === 1 ? "" : "s"} deleted; ${failures.length} could not be deleted.${messages.length > 0 ? ` ${messages.join(" ")}` : ""}`,
        );
      }
    } finally {
      setDeletingSelectedGroups(false);
    }
  }

  return (
    <section className="content experiments-page">
      <div className="page-heading">
        <h1>Experiments</h1>
        <button className="quiet-button" onClick={onRefresh} type="button">Refresh</button>
      </div>

      {groups.length === 0 ? (
        <div className="panel-card empty-experiments">
          <h2>No experiment results yet</h2>
        </div>
      ) : selectedGroup ? (
        <>
          <div className="experiment-detail-actions">
            <button className="quiet-button" onClick={() => setSelectedGroupId(null)} type="button">Back to experiments</button>
            <button className="primary-button" onClick={onOpenModelStudio} type="button">Open Model Studio</button>
          </div>
          {error && <div className="alert inline-alert" role="alert">{error}</div>}
          {deleteError && <div className="alert inline-alert" role="alert">{deleteError}</div>}
          {baseline && decentralized && (
            <div className="comparison-grid">
              <article className="summary-card">
                <div className="summary-card__top"><span>{decentralized.topology} · mean final accuracy · workers/seeds</span></div>
                <strong>{formatPercent(averageFinalAccuracy(decentralized) ?? undefined)}</strong>
              </article>
              <article className="summary-card">
                <div className="summary-card__top"><span>Centralized · final accuracy · 50,000 training images</span></div>
                <strong>{formatPercent(averageFinalAccuracy(baseline) ?? undefined)}</strong>
              </article>
            </div>
          )}
          <article className="experiment-group panel-card" key={selectedGroup.group_id}>
            <header className="experiment-group__header">
              <div>
                <span className="eyebrow">{selectedGroup.kind.replace(/_/g, " ")}</span>
                <h2>{selectedGroup.group_id}</h2>
                <p>{groupDatasetName(selectedGroup)} · {groupDetails(selectedGroup) || "Experiment configuration unavailable"}</p>
              </div>
              <span className={`run-status ${selectedGroup.status}`}>{selectedGroup.status}</span>
            </header>
            <div className="experiment-group__actions">
              <button
                className="remove-device-button"
                disabled={selectedGroup.status === "running" || results?.active_group_id === selectedGroup.group_id || deletingGroupId !== null}
                onClick={() => void deleteGroup(selectedGroup)}
                type="button"
              >
                {deletingGroupId === selectedGroup.group_id ? "Deleting…" : "Delete experiment"}
              </button>
            </div>
            <TrainingTopology topology={selectedGroup.topology === "candidate_sweep" ? selectedGroup.selected_configuration?.topology ?? "directed_ring" : selectedGroup.topology} />
            {selectedGroup.kind === "decentralized" && (
              <p className="experiment-sync-note">
                Aggregation: <strong>{selectedGroup.aggregation ?? "fedavg"}</strong> · Synchronization:{" "}
                <strong>
                  {selectedGroup.sync_mode === "semi_sync"
                    ? `semi-synchronous (${selectedGroup.sync_deadline_seconds ?? "default"} s peer deadline)`
                    : "synchronous"}
                </strong>
              </p>
            )}
            {selectedGroup.error && <div className="alert inline-alert" role="alert">{selectedGroup.error}</div>}
            {selectedGroup.kind === "hyperparameter_search" && selectedGroup.hpo_results && (
              <section className="experiment-run" aria-label="Hyperparameter results">
                <h3>
                  Hyperparameter candidates (mean of every worker)
                  {selectedGroup.selected_hyperparameters
                    ? ` · selected lr ${selectedGroup.selected_hyperparameters.learning_rate}, batch ${selectedGroup.selected_hyperparameters.batch_size}`
                    : ""}
                </h3>
                <div className="table-scroll">
                  <table className="metrics-table">
                    <thead>
                      <tr><th>Candidate</th><th>Optimizer</th><th>Learning rate</th><th>Batch</th><th>Val. accuracy</th><th>Macro F1</th><th>Time</th></tr>
                    </thead>
                    <tbody>
                      {[...selectedGroup.hpo_results]
                        .sort((left, right) => right.validation_accuracy - left.validation_accuracy)
                        .map((row) => (
                          <tr key={row.architecture.id}>
                            <td>{row.architecture.id}</td>
                            <td>{row.architecture.training?.optimizer ?? "—"}</td>
                            <td>{row.architecture.training?.learning_rate ?? "—"}</td>
                            <td>{row.architecture.training?.batch_size ?? "—"}</td>
                            <td>{formatPercent(row.validation_accuracy)}</td>
                            <td>{formatPercent(row.validation_macro_f1)}</td>
                            <td>{`${row.elapsed_seconds.toFixed(1)} s`}</td>
                          </tr>
                        ))}
                    </tbody>
                  </table>
                </div>
              </section>
            )}
            {selectedGroup.runs.map((run) => (
              <section className="experiment-run" key={run.run_id}>
                <h3>Seed {run.seed} · {run.run_id}</h3>
                <AccuracyChart run={run} />
                <ValidationConfusionMatrices run={run} />
                {run.rounds.map((round) => (
                  <div className="round-block" key={`${run.run_id}-${round.round}`}>
                    <h4>Round {round.round}</h4>
                    <div className="table-scroll">
                      <table className="metrics-table">
                        <thead>
                          <tr>
                            <th>Node</th><th>Accuracy</th><th>Precision</th><th>Recall</th>
                            <th>Loss</th><th>Macro F1</th><th>Runtime</th><th>Peak RAM</th>
                            <th>Bytes exchanged</th><th>Exchange</th><th>Peer wait</th><th>Peers used</th>
                          </tr>
                        </thead>
                        <tbody>
                          {Object.entries(round.workers).map(([nodeId, metrics]) => (
                            <tr key={nodeId}>
                              <td>{nodeId}</td>
                              <td>{formatPercent(metrics.accuracy ?? metrics.validation_accuracy)}</td>
                              <td>{formatPercent(metrics.macro_precision ?? metrics.validation_macro_precision)}</td>
                              <td>{formatPercent(metrics.macro_recall ?? metrics.validation_macro_recall)}</td>
                              <td>{formatFixed(metrics.loss ?? metrics.training_loss, 4)}</td>
                              <td>{formatPercent(metrics.macro_f1 ?? metrics.validation_macro_f1)}</td>
                              <td>{runtime(metrics) === undefined ? "—" : `${runtime(metrics)?.toFixed(2)} s`}</td>
                              <td>{metrics.peak_process_rss_mb === undefined ? "—" : `${metrics.peak_process_rss_mb.toFixed(1)} MB`}</td>
                              <td>{(metrics.bytes_sent ?? 0) + (metrics.bytes_received ?? 0)}</td>
                              <td>{metrics.exchange_seconds === undefined ? "—" : `${metrics.exchange_seconds.toFixed(3)} s`}</td>
                              <td>{metrics.peer_wait_seconds === undefined ? "—" : `${metrics.peer_wait_seconds.toFixed(3)} s`}</td>
                              <td title={metrics.missing_peers?.length ? `Missing: ${metrics.missing_peers.join(", ")}` : undefined}>
                                {metrics.peers_used === undefined
                                  ? "—"
                                  : `${metrics.peers_used.length}${metrics.missing_peers?.length ? ` (missing ${metrics.missing_peers.join(", ")})` : ""}`}
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </div>
                ))}
              </section>
            ))}
          </article>
          <div className="data-note"><span>i</span> MNIST scores shown here use each worker&apos;s local validation split. Peer transfers contain model weights, not image examples.</div>
        </>
      ) : (
        <div aria-label="Saved experiments" className="experiment-list">
          <div className="experiment-bulk-actions">
            <label>
              <input
                aria-label="Select all experiments"
                checked={deletableGroups.length > 0 && deletableGroups.every((group) => selectedGroupIds.has(group.group_id))}
                disabled={deletableGroups.length === 0 || deletingSelectedGroups || deletingGroupId !== null}
                onChange={(event) => {
                  setSelectedGroupIds((current) => {
                    const next = new Set(current);
                    for (const group of deletableGroups) {
                      if (event.target.checked) next.add(group.group_id);
                      else next.delete(group.group_id);
                    }
                    return next;
                  });
                }}
                type="checkbox"
              />
              Select all
            </label>
            <span>{selectedGroups.length} selected</span>
            <button
              className="remove-device-button"
              disabled={selectedGroups.length === 0 || deletingSelectedGroups || deletingGroupId !== null}
              onClick={() => void deleteSelectedGroups()}
              type="button"
            >
              {deletingSelectedGroups ? "Deleting…" : "Delete selected"}
            </button>
          </div>
          {groups.map((group) => (
            <div
              className="experiment-list__entry"
              key={group.group_id}
            >
              <input
                aria-label={`Select experiment ${group.group_id}`}
                checked={selectedGroupIds.has(group.group_id)}
                disabled={group.status === "running" || results?.active_group_id === group.group_id || deletingSelectedGroups || deletingGroupId !== null}
                onChange={(event) => setSelectedGroupIds((current) => {
                  const next = new Set(current);
                  if (event.target.checked) next.add(group.group_id);
                  else next.delete(group.group_id);
                  return next;
                })}
                type="checkbox"
              />
              <button
                aria-label={`View experiment details for ${group.group_id}`}
                className="experiment-list__item"
                onClick={() => {
                  setSelectedGroupId(group.group_id);
                  setDeleteError("");
                }}
                type="button"
              >
                <span>{group.group_id}</span>
                {bestGroup?.group_id === group.group_id && <span className="best-result-badge">Best experiment</span>}
              </button>
            </div>
          ))}
          {error && <div className="alert inline-alert" role="alert">{error}</div>}
          {deleteError && <div className="alert inline-alert" role="alert">{deleteError}</div>}
        </div>
      )}
    </section>
  );
}
