import { useState } from "react";

import { cleanupWorkerMemory } from "../api/client";
import type { WorkerMemoryCleanupResult } from "../api/client";

interface Props {
  token: string;
  disabled?: boolean;
  onComplete?: () => void;
  onBusyChange?: (busy: boolean) => void;
}

function change(before: number | undefined, after: number | undefined): string {
  if (before === undefined || after === undefined) return "—";
  const delta = after - before;
  return `${delta >= 0 ? "+" : ""}${delta.toFixed(0)} MB`;
}

function released(before: number | undefined, after: number | undefined): string {
  if (before === undefined || after === undefined) return "—";
  const delta = before - after;
  return `${delta >= 0 ? "+" : ""}${delta.toFixed(0)} MB`;
}

export default function WorkerMemoryCleanupControl({
  token,
  disabled = false,
  onComplete,
  onBusyChange,
}: Props) {
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [result, setResult] = useState<WorkerMemoryCleanupResult | null>(null);

  async function cleanWorkerMemory() {
    if (!window.confirm(
      "Clear retained training models and round data, release idle worker process RAM, and drop filesystem caches on all idle workers? Installed models and datasets will not be deleted; running work will not be interrupted.",
    )) {
      return;
    }

    setBusy(true);
    onBusyChange?.(true);
    setError("");
    setResult(null);
    try {
      const cleanupResult = await cleanupWorkerMemory(token);
      setResult(cleanupResult);
      onComplete?.();
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Could not clean worker RAM.");
    } finally {
      setBusy(false);
      onBusyChange?.(false);
    }
  }

  return (
    <div className="worker-memory-cleanup">
      <button
        aria-label="Clean worker RAM for next experiment"
        className="quiet-button"
        disabled={disabled || busy}
        onClick={() => void cleanWorkerMemory()}
        type="button"
      >
        {busy ? "Cleaning worker RAM…" : "Clean worker RAM for next experiment"}
      </button>
      <p className="memory-cleanup-note">
        Clears idle-worker training RAM and page cache. Preserves installed models and dataset shards; does not interrupt active training or NAS.
      </p>
      {error && <div className="alert inline-alert" role="alert">{error}</div>}
      {result && (
        <section
          aria-label="Worker memory cleanup results"
          className={`memory-cleanup-results ${result.complete ? "complete" : "incomplete"}`}
          role="status"
        >
          <div className="memory-cleanup-results__heading">
            <strong>{result.complete ? "Worker RAM cleanup complete" : "Worker RAM cleanup incomplete"}</strong>
            <span>{result.message}</span>
          </div>
          <div className="memory-cleanup-results__nodes">
            {result.workers.map((worker) => {
              const reserveReady = worker.memory_available_after_mb !== undefined &&
                worker.memory_available_after_mb >= 1024;
              return (
                <div className="memory-cleanup-result" key={worker.node_id}>
                  <strong>{worker.node_id}</strong>
                  {worker.process_memory_cleaned ? (
                    <span>
                      Available RAM {worker.memory_available_before_mb?.toFixed(0)} →{" "}
                      {worker.memory_available_after_mb?.toFixed(0)} MB ({change(
                        worker.memory_available_before_mb,
                        worker.memory_available_after_mb,
                      )} available) · Process RAM {worker.process_rss_before_mb?.toFixed(0)} →{" "}
                      {worker.process_rss_after_mb?.toFixed(0)} MB ({released(
                        worker.process_rss_before_mb,
                        worker.process_rss_after_mb,
                      )} released) · {worker.cleaned_runs} run(s) cleared ·{" "}
                      {worker.os_page_cache_dropped ? "OS cache dropped" : "OS cache not dropped"} ·{" "}
                      {reserveReady ? "NAS 1 GiB reserve ready" : "Below NAS 1 GiB memory reserve"}
                    </span>
                  ) : <span>Skipped</span>}
                  {worker.error && (
                    <span className="memory-cleanup-result__error">{worker.error}</span>
                  )}
                </div>
              );
            })}
          </div>
        </section>
      )}
    </div>
  );
}
