import { Fragment, useState } from "react";

import {
  deleteExperimentGroup,
  dispatchAutonomyAction,
  sendSlmCommand,
} from "../api/client";
import LayeredArchitectureView from "../components/LayeredArchitectureView";
import WorkerMemoryCleanupControl from "../components/WorkerMemoryCleanupControl";
import { layersFromNASArchitecture } from "../components/architecture";
import type {
  ExperimentGroup,
  ExperimentResults,
  NASCandidate,
  NASArchitecture,
} from "../api/client";

interface Props {
  token: string;
  results: ExperimentResults | null;
  error: string;
  onRefresh: () => void;
  onUseSelectedArchitecture: (architecture: NASArchitecture) => void;
  onViewCandidate?: (candidate: NASCandidate, group: ExperimentGroup) => void;
}

function architectureLabel(architecture: NASArchitecture) {
  const channels = "channel_schedule" in architecture
    ? architecture.channel_schedule.join("/")
    : String(architecture.channels);
  const blockType = "block_type" in architecture && architecture.block_type === "separable_conv2d"
    ? "separable conv"
    : "conv";
  const attention = "attention" in architecture && architecture.attention ? " · attention" : "";
  return `${architecture.id} · ${architecture.blocks} blocks, ${channels} channels, ${architecture.kernel_size}×${architecture.kernel_size} ${blockType}${attention}`;
}

function elapsedLabel(startedAt: number | undefined, completedAt: number | null | undefined, running: boolean): string {
  if (startedAt === undefined) return "—";
  const end = completedAt ?? (running ? Date.now() / 1000 : startedAt);
  const seconds = Math.max(0, Math.floor(end - startedAt));
  const minutes = Math.floor(seconds / 60);
  return minutes > 0 ? `${minutes}m ${seconds % 60}s` : `${seconds}s`;
}

export default function NASPage({
  token,
  results,
  error,
  onRefresh,
  onUseSelectedArchitecture,
  onViewCandidate,
}: Props) {
  const [busy, setBusy] = useState(false);
  const [memoryCleanupBusy, setMemoryCleanupBusy] = useState(false);
  const [message, setMessage] = useState("");
  const [commandError, setCommandError] = useState("");
  const [deletingGroup, setDeletingGroup] = useState(false);
  const [openArchitectureId, setOpenArchitectureId] = useState<string | null>(null);
  const nasGroups = [...(results?.groups ?? [])].reverse().filter((item) =>
    item.kind === "dataset_architecture_search" || item.kind === "neural_architecture_search",
  );
  const group = nasGroups[0];
  const previousGroups = nasGroups.slice(1);
  const active = Boolean(results?.active_group_id);

  async function start(searchTimeLimitSeconds: number, smokeTest = false) {
    setBusy(true);
    setMessage("");
    setCommandError("");
    try {
      const dispatched = await dispatchAutonomyAction(token, {
        action: "enable_nas",
        parameters: {
          enabled: true,
          dataset_name: "mnist",
          search_time_limit_seconds: searchTimeLimitSeconds,
          ...(smokeTest ? { smoke_test: true } : {}),
        },
      }, smokeTest
        ? "Run a 90-second MNIST NAS smoke test using one lightweight candidate for one epoch on all registered workers."
        : "Run the full 10-minute MNIST Pareto NAS search on all registered workers using validation data.", {
        dataset_name: "mnist",
        search_time_limit_seconds: searchTimeLimitSeconds,
        ...(smokeTest ? { smoke_test: true } : {}),
      });
      if (!dispatched.result.available) {
        throw new Error(dispatched.result.message ?? "NAS is unavailable.");
      }
      const data = dispatched.result.data;
      const groupId = data?.group_id;
      const description = smokeTest ? "MNIST NAS smoke test" : "MNIST architecture search";
      setMessage(groupId ? `${description} ${String(groupId)} started on the workers.` : `${description} started on the workers.`);
      onRefresh();
    } catch (cause) {
      setCommandError(cause instanceof Error ? cause.message : "Could not start NAS.");
      onRefresh();
    } finally {
      setBusy(false);
    }
  }

  async function stop() {
    setBusy(true);
    setCommandError("");
    try {
      const response = await sendSlmCommand(token, { action: "stop_training", parameters: {} });
      if (response.result?.available) {
        setMessage("Stop requested for the active run.");
      } else {
        throw new Error(response.result?.message ?? "Stop request is unavailable.");
      }
      onRefresh();
    } catch (cause) {
      setCommandError(cause instanceof Error ? cause.message : "Could not stop NAS.");
    } finally {
      setBusy(false);
    }
  }

  async function deleteNasResult() {
    if (!group || !window.confirm(`Permanently delete NAS result ${group.group_id}?`)) return;
    setDeletingGroup(true);
    setCommandError("");
    try {
      await deleteExperimentGroup(token, group.group_id);
      setMessage(`NAS result ${group.group_id} deleted.`);
      onRefresh();
    } catch (cause) {
      setCommandError(cause instanceof Error ? cause.message : "Could not delete the NAS result.");
    } finally {
      setDeletingGroup(false);
    }
  }

  const candidates = group?.nas_candidates ?? [];
  const workerSearches = group?.worker_search ?? [];
  const workerSearchProgress = group?.worker_search_progress ?? {};
  const workers = [...new Set([
    "worker1",
    "worker2",
    "worker3",
    ...Object.keys(workerSearchProgress),
    ...workerSearches.map((worker) => worker.node_id),
  ])];
  const activeProgress = workers
    .map((worker) => workerSearchProgress[worker])
    .find((progress) => progress?.status === "searching");
  const completedWorkers = workers.filter((worker) =>
    workerSearchProgress[worker]
      ? workerSearchProgress[worker].status === "completed"
      : workerSearches.some((result) => result.node_id === worker),
  ).length;
  const reportedWorkers = workers.filter((worker) =>
    workerSearchProgress[worker] !== undefined
    || workerSearches.some((result) => result.node_id === worker),
  ).length;
  const comparison = group?.comparison;
  const bestComparisonVariant = comparison
    ? (["fixed", "nas", "centralized_fixed"] as const).reduce((best, variant) =>
        comparison[variant].mean_accuracy > comparison[best].mean_accuracy ? variant : best,
      "fixed" as const)
    : undefined;
  const searchStartError = /approved NAS search did not start/i.test(commandError);
  const visibleCommandError = group?.status === "running" && searchStartError
    ? ""
    : commandError;
  return (
    <section className="content nas-page">
      <div className="page-heading">
        <h1>Neural architecture search</h1>
        <button className="quiet-button" onClick={onRefresh} type="button">Refresh</button>
      </div>

      <article className="panel-card">
        <h2>Search controls</h2>
        <p>Selection: validation · final comparison: held-out test.</p>
        <div className="topology-actions">
          <button className="primary-button" disabled={busy || active || memoryCleanupBusy} onClick={() => void start(600)} type="button">
            {busy ? "Searching…" : "Search architectures"}
          </button>
          <button className="secondary-button" disabled={busy || active || memoryCleanupBusy} onClick={() => void start(90, true)} type="button">
            {busy ? "Searching…" : "Run 90-second smoke test"}
          </button>
          {active && <button className="danger-button" disabled={busy} onClick={() => void stop()} type="button">Stop active run</button>}
        </div>
        <WorkerMemoryCleanupControl
          token={token}
          disabled={busy || active}
          onBusyChange={setMemoryCleanupBusy}
          onComplete={onRefresh}
        />
        {message && <div className="success-note" role="status">{message}</div>}
        {visibleCommandError && <div className="alert inline-alert" role="alert">{visibleCommandError}</div>}
        {error && <div className="alert inline-alert" role="alert">{error}</div>}
      </article>

      {group && (
        <>
          <article className="panel-card">
            <div className="section-heading">
              <div>
                <h2>Latest NAS run</h2>
                <p>{group.group_id} · {group.dataset_name ?? "MNIST"} · {group.status} · {group.topology} topology · seed {group.seed ?? "—"}</p>
              </div>
              <button
                className="remove-device-button"
                disabled={group.status === "running" || results?.active_group_id === group.group_id || deletingGroup}
                onClick={() => void deleteNasResult()}
                type="button"
              >
                {deletingGroup ? "Deleting…" : "Delete NAS result"}
              </button>
            </div>
            {group.error && <div className="alert inline-alert" role="alert">{group.error}</div>}
            <section className={`nas-run-status nas-run-status--${group.status}`} aria-label="NAS run status">
              <div className="nas-run-status__heading">
                <span className={`nas-run-status__badge nas-run-status__badge--${group.status}`}>
                  <span aria-hidden="true" />
                  {group.status === "running" ? "RUNNING" : group.status.toUpperCase()}
                </span>
                <div>
                  <h3>{group.status === "running" ? "Architecture search is in progress" : `Search ${group.status}`}</h3>
                  <p>
                    {group.status === "running"
                      ? reportedWorkers > 0
                        ? `${reportedWorkers}/${workers.length} workers reported · ${completedWorkers} results received.`
                        : "Run accepted · waiting for worker updates."
                      : group.status === "completed"
                        ? "Worker evaluation complete."
                        : group.error ?? `This search is ${group.status}.`}
                  </p>
                </div>
                {group.status === "running" && (
                  <span className="nas-auto-refresh">LIVE</span>
                )}
              </div>

              <div className="nas-run-facts" aria-label="Run details">
                <div><span>Dataset</span><strong>{group.dataset_name ?? "MNIST"}</strong></div>
                <div><span>Topology</span><strong>{group.topology}</strong></div>
                <div><span>Seed</span><strong>{group.seed ?? "—"}</strong></div>
                <div><span>Search budget</span><strong>{group.search_time_limit_seconds ? `${group.search_time_limit_seconds} s` : "—"}</strong></div>
                <div><span>Elapsed</span><strong>{elapsedLabel(group.started_at, group.completed_at, group.status === "running")}</strong></div>
                <div><span>Combined candidates</span><strong>{candidates.length}</strong></div>
                <div><span>Search method</span><strong>{group.search_method === "pareto_evolutionary_multifidelity" ? "Pareto + multi-fidelity" : group.search_method === "single_candidate_smoke_test" ? "Single-candidate smoke test" : group.search_method ?? "—"}</strong></div>
              </div>

              {group.status === "running" && (
                <div className="nas-worker-activity">
                  <div className="nas-worker-activity__summary">
                    <strong>Worker activity</strong>
                    <span>
                      {completedWorkers}/{workers.length} results received
                      {activeProgress?.generation
                        ? ` · generation ${activeProgress.generation}, ${activeProgress.fidelity_epochs} epoch fidelity`
                        : ""}
                    </span>
                  </div>
                  <div
                    className="nas-worker-activity__track"
                    role="progressbar"
                    aria-label="Worker results received"
                    aria-valuemin={0}
                    aria-valuemax={workers.length}
                    aria-valuenow={completedWorkers}
                  >
                    <span style={{ width: `${workers.length ? (completedWorkers / workers.length) * 100 : 0}%` }} />
                  </div>
                  {reportedWorkers === 0 && (
                    <p className="nas-worker-activity__note">
                      Progress: worker results received, not training percentage.
                    </p>
                  )}
                  <div className="nas-worker-grid" role="group" aria-label="Worker status details">
                    {workers.map((worker) => {
                      const progress = workerSearchProgress[worker];
                      const workerResult = workerSearches.find((result) => result.node_id === worker);
                      const status = progress?.status ?? (workerResult ? "completed" : "waiting");
                      const statusLabel = status === "searching"
                        ? "Searching candidates"
                        : status === "completed"
                          ? "Results received"
                          : status === "failed"
                            ? "Search failed"
                            : status === "stopped"
                              ? "Search stopped"
                              : "No update received";
                      return (
                        <article className={`nas-worker-card nas-worker-card--${status}`} key={worker}>
                          <div className="nas-worker-card__title">
                            <span className="nas-worker-card__dot" aria-hidden="true" />
                            <strong>{worker}</strong>
                            <span>{statusLabel}</span>
                          </div>
                          <p>
                            {workerResult
                              ? `${workerResult.candidates.length} candidates · ${workerResult.elapsed_seconds.toFixed(1)} s`
                              : progress?.status === "searching"
                                ? "Searching."
                                : progress?.error ?? "Waiting for update."}
                          </p>
                          {progress?.candidate_count !== undefined && (
                            <small>
                              Generation {progress.generation ?? "—"} · {progress.fidelity_epochs ?? "—"} epoch fidelity · {progress.candidate_count} candidates
                            </small>
                          )}
                        </article>
                      );
                    })}
                  </div>
                </div>
              )}
            </section>
            {group.selected_architecture ? (
              <>
                <p className="nas-selected-model">
                  <span className="best-result-badge">BEST MODEL · VALIDATION SELECTED</span>
                  <strong>{architectureLabel(group.selected_architecture)}</strong>
                </p>
                <LayeredArchitectureView
                  layers={layersFromNASArchitecture(group.selected_architecture)}
                  note={"layers" in group.selected_architecture
                    ? undefined
                    : "Legacy layer sizes are illustrative."}
                  title={`Selected architecture · ${group.selected_architecture.id}`}
                />
              </>
            ) : (
              group.status !== "running" && (
                <p>
                  {workerSearches.length > 0
                    ? `${workerSearches.length} worker${workerSearches.length === 1 ? "" : "s"} returned results.`
                    : "No worker candidate results saved."}
                </p>
              )
            )}
            {group.search_elapsed_seconds !== undefined && (
              <p>Parallel search wall time: {group.search_elapsed_seconds.toFixed(1)} s · peak worker RSS: {group.search_peak_rss_mb?.toFixed(1)} MiB</p>
            )}
            {candidates.length > 0 ? <div className="table-wrap">
              <table>
                <thead><tr><th>Candidate</th><th>Generation / fidelity</th><th>Parameters</th><th>Mean validation accuracy</th><th>Macro-F1</th><th>Worker accuracies</th><th>Mean candidate time</th><th>Peak worker RSS</th><th>Details</th></tr></thead>
                <tbody>
                  {candidates.map((candidate) => {
                    const isSelected = candidate.architecture.id === group.selected_architecture?.id;
                    return (
                    <Fragment key={candidate.architecture.id}>
                    <tr className={isSelected ? "best-result-row" : undefined} key={candidate.architecture.id}>
                      <td>
                        {architectureLabel(candidate.architecture)}
                        {isSelected && <span className="best-result-badge">BEST MODEL</span>}
                        {candidate.is_pareto_optimal && <span className="best-result-badge">PARETO FRONT</span>}
                      </td>
                      <td>{candidate.generation ?? "—"} / {candidate.fidelity_epochs ?? "—"} epochs</td>
                      <td>{candidate.parameter_count.toLocaleString()}</td>
                      <td>{(candidate.mean_validation_accuracy * 100).toFixed(2)}%</td>
                      <td>{candidate.mean_validation_macro_f1?.toFixed(4) ?? "—"}</td>
                      <td>{candidate.worker_accuracies.map((value) => `${(value * 100).toFixed(1)}%`).join(" / ")}</td>
                      <td>{candidate.mean_elapsed_seconds === undefined ? "—" : `${candidate.mean_elapsed_seconds.toFixed(2)} s`}</td>
                      <td>{candidate.max_peak_rss_mb === undefined ? "—" : `${candidate.max_peak_rss_mb.toFixed(1)} MiB`}</td>
                      <td>
                        {onViewCandidate && group && (
                          <button
                            className="nas-details-button"
                            type="button"
                            onClick={() => onViewCandidate(candidate, group)}
                            aria-label={`View details for ${candidate.architecture.id}`}
                          >
                            View details
                          </button>
                        )}
                        <button
                          className="nas-details-button"
                          type="button"
                          onClick={() => setOpenArchitectureId((current) =>
                            current === candidate.architecture.id ? null : candidate.architecture.id,
                          )}
                          aria-label={`${openArchitectureId === candidate.architecture.id ? "Hide" : "Show"} architecture for ${candidate.architecture.id}`}
                        >
                          {openArchitectureId === candidate.architecture.id ? "Hide architecture" : "Show architecture"}
                        </button>
                      </td>
                    </tr>
                    {openArchitectureId === candidate.architecture.id && (
                      <tr className="nas-architecture-row" key={`${candidate.architecture.id}-architecture`}>
                        <td colSpan={9}>
                          <LayeredArchitectureView
                            layers={layersFromNASArchitecture(candidate.architecture)}
                            note={"layers" in candidate.architecture
                              ? undefined
                              : "Legacy CNN layer sequence is inferred from block metadata; block sizes are illustrative."}
                            title={`Candidate architecture · ${candidate.architecture.id}`}
                          />
                        </td>
                      </tr>
                    )}
                    </Fragment>
                    );
                  })}
                </tbody>
              </table>
            </div> : (
              <div className="nas-empty-results">
                <strong>No combined candidate scores yet</strong>
                <span>
                  {workerSearches.length > 0
                    ? "Worker results available · awaiting combined scores."
                    : group.status === "running"
                      ? "Awaiting worker results."
                      : "No combined scores saved."}
                </span>
              </div>
            )}
            {workerSearches.length > 0 && (
              <>
                <h3>Worker search results</h3>
                <div className="table-wrap">
                  <table>
                    <thead><tr><th>Worker</th><th>Candidate</th><th>Parameters</th><th>Validation accuracy</th><th>Search time</th><th>Peak RSS</th></tr></thead>
                    <tbody>
                      {workerSearches.flatMap((worker) =>
                        worker.candidates.map((candidate) => (
                          <tr key={`${worker.node_id}-${candidate.architecture.id}`}>
                            <td>{worker.node_id}</td>
                            <td>{architectureLabel(candidate.architecture)}</td>
                            <td>{candidate.parameter_count.toLocaleString()}</td>
                            <td>{(candidate.validation_accuracy * 100).toFixed(2)}%</td>
                            <td>{candidate.elapsed_seconds.toFixed(2)} s</td>
                            <td>{candidate.peak_rss_mb.toFixed(1)} MiB</td>
                          </tr>
                        )),
                      )}
                    </tbody>
                  </table>
                </div>
              </>
            )}
            {group.status === "completed" && group.selected_architecture && (
              <button
                className="primary-button"
                onClick={() => onUseSelectedArchitecture(group.selected_architecture!)}
                type="button"
              >
                Use selected architecture in Model Studio
              </button>
            )}
          </article>
          {comparison && (
            <article className="panel-card">
              <h2>Fixed vs NAS vs centralized reference</h2>
              <p>NAS vs fixed · held-out accuracy: {group.nas_beats_fixed ? "NAS higher" : "fixed higher or tied"}.</p>
              <div className="table-wrap">
                <table>
                  <thead><tr><th>Variant</th><th>Accuracy</th><th>Macro-F1</th><th>Mean runtime incl. search</th><th>Peak RSS</th><th>Communication bytes</th></tr></thead>
                  <tbody>
                    {(["fixed", "nas", "centralized_fixed"] as const).map((variant) => {
                      const row = comparison[variant];
                      const isBest = variant === bestComparisonVariant;
                      return (
                        <tr className={isBest ? "best-result-row" : undefined} key={variant}>
                          <td>
                            {variant === "centralized_fixed" ? "Centralized fixed" : variant.toUpperCase()}
                            {isBest && <span className="best-result-badge">BEST HELD-OUT ACCURACY</span>}
                          </td>
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
      {previousGroups.length > 0 && (
        <article className="panel-card">
          <h2>Previous NAS results</h2>
          {previousGroups.map((previous) => (
            <section key={previous.group_id}>
              <h3>{previous.group_id}</h3>
              <p>
                {previous.dataset_name ?? "MNIST"} · {previous.status} · {previous.topology} topology · seed {previous.seed ?? "—"}
              </p>
              {previous.error && <div className="alert inline-alert" role="alert">{previous.error}</div>}
              {previous.selected_architecture && (
                <p>Selected architecture: <strong>{architectureLabel(previous.selected_architecture)}</strong></p>
              )}
              {(previous.nas_candidates?.length ?? 0) > 0 && (
                <div className="table-wrap">
                  <table>
                    <thead><tr><th>Candidate</th><th>Parameters</th><th>Mean validation accuracy</th><th>Worker accuracies</th><th>Details</th></tr></thead>
                    <tbody>
                      {previous.nas_candidates?.map((candidate) => (
                        <tr key={candidate.architecture.id}>
                          <td>{architectureLabel(candidate.architecture)}</td>
                          <td>{candidate.parameter_count.toLocaleString()}</td>
                          <td>{(candidate.mean_validation_accuracy * 100).toFixed(2)}%</td>
                          <td>{candidate.worker_accuracies.map((value) => `${(value * 100).toFixed(1)}%`).join(" / ")}</td>
                  <td>
                    {onViewCandidate && (
                      <button
                        className="nas-details-button"
                        type="button"
                        onClick={() => onViewCandidate(candidate, previous)}
                        aria-label={`View details for ${candidate.architecture.id}`}
                      >
                        View details
                      </button>
                    )}
                  </td>
                </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
              {(previous.worker_search?.length ?? 0) > 0 && (
                <div className="table-wrap">
                  <table>
                    <thead><tr><th>Worker</th><th>Candidates returned</th><th>Search time</th></tr></thead>
                    <tbody>
                      {previous.worker_search?.map((worker) => (
                        <tr key={worker.node_id}>
                          <td>{worker.node_id}</td>
                          <td>{worker.candidates.length}</td>
                          <td>{worker.elapsed_seconds.toFixed(2)} s</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
              {(previous.nas_candidates?.length ?? 0) === 0
                && (previous.worker_search?.length ?? 0) === 0
                && <p>No worker candidate results saved.</p>}
            </section>
          ))}
        </article>
      )}
      {!group && (
        <article className="panel-card">
          <h2>No MNIST architecture-search results yet</h2>
        </article>
      )}
    </section>
  );
}
