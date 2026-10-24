import { useState } from "react";

import {
  sendSlmCommand,
  startCentralizedBaseline,
} from "../api/client";
import type { ExperimentGroup, ExperimentResults, WorkerRoundMetrics } from "../api/client";
import type { TrainingTopology } from "../api/client";

interface Props {
  token: string;
  results: ExperimentResults | null;
  error: string;
  onRefresh: () => void;
}

function averageFinalAccuracy(group: ExperimentGroup): number | null {
  const values = group.runs.flatMap((run) => {
    const finalRound = run.rounds[run.rounds.length - 1];
    if (!finalRound) return [];
    const workerMetrics = Object.values(finalRound.workers);
    if (workerMetrics.length === 0) return [];
    return [workerMetrics.reduce((sum, metrics) => sum + metrics.accuracy, 0) / workerMetrics.length];
  });
  return values.length ? values.reduce((sum, value) => sum + value, 0) / values.length : null;
}

function formatPercent(value: number | undefined): string {
  return value === undefined ? "—" : `${(value * 100).toFixed(2)}%`;
}

function runtime(metrics: WorkerRoundMetrics): number | undefined {
  return metrics.round_seconds ?? metrics.runtime_seconds ?? metrics.training_seconds;
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

export default function ExperimentsPage({ token, results, error, onRefresh }: Props) {
  const [topology, setTopology] = useState<TrainingTopology>(
    results?.configuration.topology ?? "directed_ring",
  );
  const [datasetSplit, setDatasetSplit] = useState("dirichlet_a03");
  const [rounds, setRounds] = useState(10);
  const [epochs, setEpochs] = useState(5);
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState("");
  const [commandError, setCommandError] = useState("");
  const groups = [...(results?.groups ?? [])].reverse();
  const active = Boolean(results?.active_group_id);
  const baseline = groups.find((group) => group.kind === "centralized_baseline");
  const decentralized = groups.find((group) => group.kind === "decentralized" && group.status === "completed");

  async function command(action: string, parameters: Record<string, unknown>) {
    const response = await sendSlmCommand(token, { action, parameters });
    if (!response.result.available) {
      throw new Error(response.result.message ?? `${action} is not available.`);
    }
    return response;
  }

  async function startDistributed(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setBusy(true);
    setMessage("");
    setCommandError("");
    try {
      await command("set_topology", { topology });
      await command("set_rounds", { rounds });
      await command("set_epochs", { epochs });
      const response = await command("start_training", { rounds, epochs, dataset_split: datasetSplit });
      setMessage(`Distributed ${topology} run accepted. It will execute three seeds on ${datasetSplit}.`);
      if (response.result.data?.group_id) setMessage(`Run group ${String(response.result.data.group_id)} accepted.`);
      onRefresh();
    } catch (cause) {
      setCommandError(cause instanceof Error ? cause.message : "Could not start distributed training.");
    } finally {
      setBusy(false);
    }
  }

  async function startBaseline() {
    setBusy(true);
    setMessage("");
    setCommandError("");
    try {
      await startCentralizedBaseline(token, rounds, epochs);
      setMessage("Centralized baseline accepted on dfl-master using all 50,000 training images.");
      onRefresh();
    } catch (cause) {
      setCommandError(cause instanceof Error ? cause.message : "Could not start the baseline.");
    } finally {
      setBusy(false);
    }
  }

  async function stopTraining() {
    setBusy(true);
    setCommandError("");
    try {
      const response = await command("stop_training", {});
      setMessage(response.result.message ?? "Stop requested for the active training group.");
      onRefresh();
    } catch (cause) {
      setCommandError(cause instanceof Error ? cause.message : "Could not stop training.");
    } finally {
      setBusy(false);
    }
  }

  return (
    <section className="content experiments-page">
      <div className="page-heading">
        <div>
          <span className="eyebrow">CIFAR-10 · REAL DEVICE TRAINING</span>
          <h1>Experiments</h1>
          <p>Per-worker held-out test metrics, peer exchange, and centralized comparison.</p>
        </div>
        <button className="quiet-button" onClick={onRefresh} type="button">Refresh</button>
      </div>

      <div className="experiment-controls panel-card">
        <div>
          <h2>Start experiment</h2>
          <p>Distributed runs use three fixed seeds: 17, 42, and 2026.</p>
        </div>
        <form onSubmit={(event) => void startDistributed(event)}>
          <label>
            Topology
            <select value={topology} onChange={(event) => setTopology(event.target.value as TrainingTopology)}>
              <option value="directed_ring">Directed ring</option>
              <option value="bidirectional_ring">Bidirectional ring</option>
              <option value="mesh">Mesh</option>
              <option value="chain">Line / chain</option>
              <option value="star">Star (worker1 hub)</option>
            </select>
          </label>
          <label>
            Training split
            <select value={datasetSplit} onChange={(event) => setDatasetSplit(event.target.value)}>
              <option value="dirichlet_a03">Dirichlet α=0.3</option>
              <option value="dirichlet_a05">Dirichlet α=0.5</option>
              <option value="stratified">Stratified IID</option>
            </select>
          </label>
          <label>
            Rounds
            <input min={1} max={100} onChange={(event) => setRounds(Number(event.target.value))} type="number" value={rounds} />
          </label>
          <label>
            Local epochs
            <input min={1} max={50} onChange={(event) => setEpochs(Number(event.target.value))} type="number" value={epochs} />
          </label>
          <button className="primary-button" disabled={busy || active} type="submit">
            {busy ? "Working…" : "Start distributed"}
          </button>
          <button className="quiet-button" disabled={busy || active} onClick={() => void startBaseline()} type="button">
            Start centralized baseline
          </button>
          {active && (
            <button className="danger-button" disabled={busy} onClick={() => void stopTraining()} type="button">
              Stop active run
            </button>
          )}
        </form>
        {message && <div className="success-note" role="status">{message}</div>}
        {commandError && <div className="alert inline-alert" role="alert">{commandError}</div>}
        {error && <div className="alert inline-alert" role="alert">{error}</div>}
      </div>

      {baseline && decentralized && (
        <div className="comparison-grid">
          <article className="summary-card">
            <div className="summary-card__top"><span>{decentralized.topology} · final test accuracy</span></div>
            <strong>{formatPercent(averageFinalAccuracy(decentralized) ?? undefined)}</strong>
            <small>Mean of final per-worker scores across completed seeds</small>
          </article>
          <article className="summary-card">
            <div className="summary-card__top"><span>Centralized · final test accuracy</span></div>
            <strong>{formatPercent(averageFinalAccuracy(baseline) ?? undefined)}</strong>
            <small>Single dfl-master model trained on all 50,000 training images</small>
          </article>
        </div>
      )}

      {groups.length === 0 ? (
        <div className="panel-card empty-experiments">
          <h2>No experiment results yet</h2>
          <p>Start a Ring or Mesh run to collect live metrics from the workers.</p>
        </div>
      ) : (
        groups.map((group) => (
          <article className="experiment-group panel-card" key={group.group_id}>
            <header className="experiment-group__header">
              <div>
                <span className="eyebrow">{group.kind.replace(/_/g, " ")}</span>
                <h2>{group.group_id}</h2>
                <p>{group.rounds} rounds · {group.epochs} local epochs · seeds {group.seeds.join(", ")}</p>
              </div>
              <span className={`run-status ${group.status}`}>{group.status}</span>
            </header>
            <TrainingTopology topology={group.topology === "candidate_sweep" ? group.selected_configuration?.topology ?? "directed_ring" : group.topology} />
            {group.error && <div className="alert inline-alert" role="alert">{group.error}</div>}
            {group.runs.map((run) => (
              <section className="experiment-run" key={run.run_id}>
                <h3>Seed {run.seed} · {run.run_id}</h3>
                {run.rounds.map((round) => (
                  <div className="round-block" key={`${run.run_id}-${round.round}`}>
                    <h4>Round {round.round}</h4>
                    <div className="table-scroll">
                      <table className="metrics-table">
                        <thead>
                          <tr>
                            <th>Node</th><th>Test accuracy</th><th>Test loss</th><th>Macro F1</th>
                            <th>Runtime</th><th>Peak RAM</th><th>Bytes exchanged</th><th>Exchange</th>
                          </tr>
                        </thead>
                        <tbody>
                          {Object.entries(round.workers).map(([nodeId, metrics]) => (
                            <tr key={nodeId}>
                              <td>{nodeId}</td>
                              <td>{formatPercent(metrics.accuracy)}</td>
                              <td>{metrics.loss.toFixed(4)}</td>
                              <td>{formatPercent(metrics.macro_f1)}</td>
                              <td>{(runtime(metrics) ?? 0).toFixed(2)} s</td>
                              <td>{metrics.peak_process_rss_mb.toFixed(1)} MB</td>
                              <td>{(metrics.bytes_sent ?? 0) + (metrics.bytes_received ?? 0)}</td>
                              <td>{metrics.exchange_seconds === undefined ? "—" : `${metrics.exchange_seconds.toFixed(3)} s`}</td>
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
        ))
      )}
      <div className="data-note"><span>i</span> Held-out test data remains on the master evaluator; workers receive only their disjoint training shard.</div>
    </section>
  );
}
