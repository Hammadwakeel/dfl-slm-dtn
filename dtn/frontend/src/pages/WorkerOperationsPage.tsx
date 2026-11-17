import { useCallback, useEffect, useState } from "react";

import {
  CoordinatorApiError,
  getWorkerProcesses,
  rebootWorker,
  stopWorkerNasSearch,
  stopWorkerTraining,
} from "../api/client";
import type { WorkerProcessSnapshot } from "../api/client";
import WorkerMemoryCleanupControl from "../components/WorkerMemoryCleanupControl";
import type { Device } from "../types";

interface Props {
  token: string;
  devices: Device[];
}

export default function WorkerOperationsPage({ token, devices }: Props) {
  const [snapshots, setSnapshots] = useState<WorkerProcessSnapshot[]>([]);
  const [busyNode, setBusyNode] = useState("");
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const [loading, setLoading] = useState(false);
  const workers = devices.filter((device) => device.role === "worker");

  const refresh = useCallback(async () => {
    setLoading(true);
    try {
      setSnapshots(await getWorkerProcesses(token));
      setError("");
    } catch (cause) {
      const operationsApiMissing = cause instanceof CoordinatorApiError && cause.status === 404;
      setError(operationsApiMissing
        ? "The coordinator returned 404 for worker process monitoring. Install the updated backend and refresh this page."
        : cause instanceof Error
          ? cause.message
          : "Could not load worker processes.");
    } finally {
      setLoading(false);
    }
  }, [token]);

  useEffect(() => {
    const initialRefresh = window.setTimeout(() => void refresh(), 0);
    const timer = window.setInterval(() => void refresh(), 5000);
    return () => {
      window.clearTimeout(initialRefresh);
      window.clearInterval(timer);
    };
  }, [refresh]);

  async function stopWorkerActivity(worker: Device) {
    const snapshot = snapshots.find((item) => item.node_id === worker.node_id);
    const nasActive = snapshot && "nas_active" in snapshot ? snapshot.nas_active : false;
    const action = nasActive ? "NAS search" : "DFL training";
    if (!window.confirm(
      `Request cancellation of the active ${action} on ${worker.hostname}? This does not stop unrelated operating-system processes.`,
    )) return;
    setBusyNode(worker.node_id);
    setError("");
    setNotice("");
    try {
      const result = nasActive
        ? await stopWorkerNasSearch(token, worker.node_id)
        : await stopWorkerTraining(token, worker.node_id);
      setNotice(nasActive
        ? result.message ?? `NAS cancellation requested on ${worker.node_id}.`
        : result.accepted
          ? `Cancellation requested for ${result.run_id ?? worker.node_id}.`
          : result.message ?? `No training run is active on ${worker.node_id}.`);
      await refresh();
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Could not stop worker training.");
    } finally {
      setBusyNode("");
    }
  }

  async function restartWorker(worker: Device) {
    if (!window.confirm(
      `Reboot ${worker.hostname} (${worker.node_id}) now? The worker will be unavailable until it starts again.`,
    )) return;
    setBusyNode(worker.node_id);
    setError("");
    setNotice("");
    try {
      await rebootWorker(token, worker.node_id);
      setNotice(`Reboot accepted for ${worker.node_id}. Waiting for it to come back online.`);
      await refresh();
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Could not reboot the worker.");
    } finally {
      setBusyNode("");
    }
  }

  return (
    <section className="content worker-operations-page">
      <div className="page-heading">
        <h1>Worker operations</h1>
        <button className="quiet-button" disabled={loading} onClick={() => void refresh()} type="button">
          {loading ? "Refreshing…" : "Refresh"}
        </button>
      </div>

      <div className="worker-operations-notice">
        <strong>Safe controls</strong>
        <span>
          Stops affect DFL training and NAS only. Reboot is blocked while either runs. Cleanup preserves datasets, models, history, and logs.
        </span>
      </div>
      <WorkerMemoryCleanupControl
        token={token}
        disabled={workers.every((worker) => !worker.online)}
        onComplete={() => void refresh()}
      />
      {error && <div className="alert inline-alert" role="alert">{error}</div>}
      {notice && <div className="alert inline-alert" role="status">{notice}</div>}

      <div className="worker-operations-grid">
        {workers.map((worker) => {
          const snapshot = snapshots.find((item) => item.node_id === worker.node_id);
          const processSnapshot = snapshot && "training" in snapshot ? snapshot : undefined;
          const snapshotError = snapshot && "error" in snapshot ? snapshot.error : undefined;
          const trainingActive = Boolean(
            processSnapshot?.training.active_run_id || processSnapshot?.training.scheduled_run_id,
          );
          const nasActive = Boolean(processSnapshot?.nas_active);
          const trainingRun = processSnapshot?.training.active_run_id
            ?? processSnapshot?.training.scheduled_run_id;
          const memoryTotal = worker.latest_status?.memory.total_bytes;
          const memoryUsed = worker.latest_status?.memory.used_bytes;
          const memoryPercent = memoryTotal && memoryTotal > 0 && memoryUsed !== undefined
            ? Math.round((memoryUsed / memoryTotal) * 100)
            : null;

          return (
            <article className="worker-operations-card" key={worker.node_id}>
              <header className="worker-operations-card__header">
                <div>
                  <h2>{worker.hostname}</h2>
                  <span>{worker.node_id} · {worker.ip}</span>
                </div>
                <span className={`status-pill ${worker.online && !snapshotError ? "online" : "offline"}`}>
                  {snapshotError ? "Unavailable" : worker.online ? "Online" : "Offline"}
                </span>
              </header>
              {snapshotError ? (
                <div className="alert inline-alert" role="alert">{snapshotError}</div>
              ) : processSnapshot ? (
                <>
                  <div className="worker-operations-metrics">
                    <div><span>Memory used</span><strong>{memoryPercent === null ? "—" : `${memoryPercent}%`}</strong></div>
                    <div><span>Available RAM</span><strong>{processSnapshot.memory_available_mb.toFixed(0)} MB</strong></div>
                    <div><span>NAS</span><strong>{processSnapshot.nas_active ? "Running" : "Idle"}</strong></div>
                    <div>
                      <span>Training</span>
                      <strong>{trainingActive ? `Running · ${trainingRun}` : "Idle"}</strong>
                    </div>
                  </div>
                  {trainingActive && (
                    <p className="worker-operations-progress">
                      {processSnapshot.training.active_phase?.replace(/_/g, " ") ?? "Training"}
                      {processSnapshot.training.active_round !== null && ` · round ${processSnapshot.training.active_round}`}
                      {processSnapshot.training.active_epoch !== null && processSnapshot.training.total_epochs !== null
                        ? ` · epoch ${processSnapshot.training.active_epoch}/${processSnapshot.training.total_epochs}`
                        : ""}
                    </p>
                  )}
                  <div className="worker-operations-actions">
                    <button
                      className="quiet-button"
                      disabled={(!trainingActive && !nasActive) || busyNode === worker.node_id}
                      onClick={() => void stopWorkerActivity(worker)}
                      type="button"
                    >
                      {busyNode === worker.node_id
                        ? "Working…"
                        : nasActive
                          ? "Stop NAS search"
                          : "Stop DFL training"}
                    </button>
                    <button
                      className="danger-button"
                      disabled={!worker.online || processSnapshot.nas_active || trainingActive || busyNode === worker.node_id}
                      onClick={() => void restartWorker(worker)}
                      type="button"
                    >
                      Reboot Jetson
                    </button>
                  </div>
                  <div className="worker-process-table-wrap">
                    <h3>Top processes by RAM</h3>
                    <table className="worker-process-table">
                      <thead>
                        <tr><th>Process</th><th>PID</th><th>CPU</th><th>RAM</th></tr>
                      </thead>
                      <tbody>
                        {processSnapshot.processes.slice(0, 10).map((process) => (
                          <tr key={process.pid}>
                            <td title={`User: ${process.user}`}>{process.name}</td>
                            <td>{process.pid}</td>
                            <td>{process.cpu_percent.toFixed(1)}%</td>
                            <td>{process.rss_mb.toFixed(0)} MB</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                    <small>Only DFL training and NAS jobs can be stopped.</small>
                    <small>Snapshot: {new Date(processSnapshot.sampled_at).toLocaleTimeString()}</small>
                  </div>
                </>
              ) : (
                <p className="worker-operations-progress">
                  {worker.online ? "Loading process data…" : "Worker is offline."}
                </p>
              )}
            </article>
          );
        })}
      </div>
    </section>
  );
}
