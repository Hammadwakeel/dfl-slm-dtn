import { useState } from "react";

import { sendSlmCommand } from "../api/client";
import type { ExperimentResults } from "../api/client";

interface Props {
  token: string;
  results: ExperimentResults | null;
  error: string;
  onRefresh: () => void;
}

function architectureLabel(architecture: { id: string; blocks: number; channels: number; kernel_size: number }) {
  return `${architecture.id} · ${architecture.blocks} block${architecture.blocks === 1 ? "" : "s"}, ${architecture.channels} channels, ${architecture.kernel_size}×${architecture.kernel_size}`;
}

export default function NASPage({ token, results, error, onRefresh }: Props) {
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState("");
  const [commandError, setCommandError] = useState("");
  const groups = [...(results?.groups ?? [])].reverse();
  const group = groups.find((item) => item.kind === "neural_architecture_search");
  const active = Boolean(results?.active_group_id);

  async function start(rounds: number, epochs: number, seeds: number[]) {
    setBusy(true);
    setMessage("");
    setCommandError("");
    try {
      const response = await sendSlmCommand(token, {
        action: "enable_nas",
        parameters: {
          enabled: true,
          rounds,
          epochs,
          seeds,
          search_time_limit_seconds: 300,
        },
      });
      if (!response.result.available) {
        throw new Error(response.result.message ?? "NAS is unavailable.");
      }
      const groupId = response.result.data?.group_id;
      setMessage(groupId ? `NAS experiment ${String(groupId)} started.` : "NAS experiment started.");
      onRefresh();
    } catch (cause) {
      setCommandError(cause instanceof Error ? cause.message : "Could not start NAS.");
    } finally {
      setBusy(false);
    }
  }

  async function stop() {
    setBusy(true);
    setCommandError("");
    try {
      await sendSlmCommand(token, { action: "stop_training", parameters: {} });
      setMessage("Stop requested for the active run.");
      onRefresh();
    } catch (cause) {
      setCommandError(cause instanceof Error ? cause.message : "Could not stop NAS.");
    } finally {
      setBusy(false);
    }
  }

  const candidates = group?.nas_candidates ?? [];
  const comparison = group?.comparison;
  return (
    <section className="content nas-page">
      <div className="page-heading">
        <div>
          <span className="eyebrow">CIFAR-10 · REAL WORKER GPU SEARCH</span>
          <h1>Neural architecture search</h1>
          <p>Workers evaluate the same bounded CNN space on their own shard validation data; the master selects one common model before DFL.</p>
        </div>
        <button className="quiet-button" onClick={onRefresh} type="button">Refresh</button>
      </div>

      <article className="panel-card">
        <h2>Search space and limits</h2>
        <p>1–2 convolution blocks · 8 or 16 channels · 3×3 or 5×5 kernels · one local search epoch per candidate.</p>
        <p>Maximum candidate: <strong>11,466 parameters</strong> (below the 12,000 cap). Each worker is limited to 300 seconds and 2,048 MiB peak process RSS per search.</p>
        <p>Validation uses a deterministic 20% split of each worker’s stratified training shard. The held-out CIFAR-10 test set is not used for search or selection.</p>
        <div className="topology-actions">
          <button className="quiet-button" disabled={busy || active} onClick={() => void start(1, 1, [17])} type="button">
            {busy ? "Starting…" : "Run search + 1-round pilot"}
          </button>
          <button className="primary-button" disabled={busy || active} onClick={() => void start(10, 5, [17, 42, 2026])} type="button">
            Run full 3-seed comparison
          </button>
          {active && <button className="danger-button" disabled={busy} onClick={() => void stop()} type="button">Stop active run</button>}
        </div>
        {message && <div className="success-note" role="status">{message}</div>}
        {commandError && <div className="alert inline-alert" role="alert">{commandError}</div>}
        {error && <div className="alert inline-alert" role="alert">{error}</div>}
      </article>

      {group && (
        <>
          <article className="panel-card">
            <div className="section-heading">
              <div>
                <h2>Latest NAS run</h2>
                <p>{group.group_id} · {group.status} · {group.rounds} rounds × {group.epochs} epochs · {group.topology}</p>
              </div>
            </div>
            {group.error && <div className="alert inline-alert" role="alert">{group.error}</div>}
            {group.selected_architecture ? (
              <p>Selected common architecture: <strong>{architectureLabel(group.selected_architecture)}</strong></p>
            ) : (
              <p>Waiting for all worker-local searches to complete.</p>
            )}
            {group.search_elapsed_seconds !== undefined && (
              <p>Parallel search wall time: {group.search_elapsed_seconds.toFixed(1)} s · peak worker RSS: {group.search_peak_rss_mb?.toFixed(1)} MiB</p>
            )}
            <div className="table-wrap">
              <table>
                <thead><tr><th>Candidate</th><th>Parameters</th><th>Mean validation accuracy</th><th>Worker accuracies</th><th>Mean candidate time</th><th>Peak worker RSS</th></tr></thead>
                <tbody>
                  {candidates.map((candidate) => (
                    <tr key={candidate.architecture.id}>
                      <td>{architectureLabel(candidate.architecture)}</td>
                      <td>{candidate.parameter_count.toLocaleString()}</td>
                      <td>{(candidate.mean_validation_accuracy * 100).toFixed(2)}%</td>
                      <td>{candidate.worker_accuracies.map((value) => `${(value * 100).toFixed(1)}%`).join(" / ")}</td>
                      <td>{candidate.mean_elapsed_seconds.toFixed(2)} s</td>
                      <td>{candidate.max_peak_rss_mb.toFixed(1)} MiB</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </article>
          {comparison && (
            <article className="panel-card">
              <h2>Fixed vs NAS vs centralized reference</h2>
              <p>{group.nas_beats_fixed ? "NAS-selected model outperformed the fixed DFL model on mean held-out accuracy." : "NAS did not beat the fixed model on mean held-out accuracy."}</p>
              <div className="table-wrap">
                <table>
                  <thead><tr><th>Variant</th><th>Accuracy</th><th>Macro-F1</th><th>Mean runtime incl. search</th><th>Peak RSS</th><th>Communication bytes</th></tr></thead>
                  <tbody>
                    {(["fixed", "nas", "centralized_fixed"] as const).map((variant) => {
                      const row = comparison[variant];
                      return (
                        <tr key={variant}>
                          <td>{variant === "centralized_fixed" ? "Centralized fixed" : variant.toUpperCase()}</td>
                          <td>{(row.mean_accuracy * 100).toFixed(2)}%</td>
                          <td>{row.mean_macro_f1.toFixed(4)}</td>
                          <td>{row.mean_runtime_including_search_seconds.toFixed(1)} s mean</td>
                          <td>{row.peak_rss_mb.toFixed(1)} MiB</td>
                          <td>{Math.round(row.mean_communication_bytes).toLocaleString()}</td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </article>
          )}
        </>
      )}
    </section>
  );
}
