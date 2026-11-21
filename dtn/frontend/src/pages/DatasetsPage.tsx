import { useCallback, useEffect, useRef, useState } from "react";

import {
  dispatchDatasetPreprocessing,
  getAutonomyState,
  getDatasetSamples,
  getLatestPreprocessingReport,
  listDatasets,
} from "../api/client";
import type {
  AutonomyState,
  DatasetMetadata,
  DatasetSample,
  PreprocessingReport,
} from "../api/client";

interface Props {
  token: string;
  active?: boolean;
}

type DatasetState = "ready" | "preprocessed" | "shipped to workers";
type SplitStrategy =
  | "stratified"
  | "dirichlet_a01"
  | "dirichlet_a03"
  | "dirichlet_a05"
  | "dirichlet_a07"
  | "dirichlet_a09";
type Normalization = "standard" | "minmax" | "none";

const OFFICIAL_DATASETS = new Set(["mnist", "cifar10", "fashion_mnist", "femnist"]);
const DATASET_ORDER = ["mnist", "cifar10", "fashion_mnist", "femnist"];
const DATASET_LABELS: Record<string, string> = {
  mnist: "MNIST",
  cifar10: "CIFAR-10",
  fashion_mnist: "Fashion-MNIST",
  femnist: "FEMNIST",
};

function datasetLabel(datasetName: string): string {
  return DATASET_LABELS[datasetName] ?? datasetName;
}

function imageSize(metadata: DatasetMetadata): { height: number; width: number } {
  const counts = new Map<string, number>();
  for (const sizes of Object.values(metadata.image_sizes)) {
    for (const [size, count] of Object.entries(sizes)) {
      counts.set(size, (counts.get(size) ?? 0) + count);
    }
  }
  const mostCommon = [...counts.entries()].sort((left, right) => right[1] - left[1])[0]?.[0];
  if (!mostCommon) return { height: 32, width: 32 };
  const [width, height] = mostCommon.split("x").map(Number);
  return { height, width };
}

function channelCount(metadata: DatasetMetadata): string {
  const firstClassChannels = Object.values(metadata.image_channels ?? {})[0];
  return Object.keys(firstClassChannels ?? {})[0] ?? "—";
}

function datasetState(report: PreprocessingReport | null): DatasetState {
  if (!report) return "ready";
  const expectedWorkers = Object.keys(report.workers ?? {}).length;
  return expectedWorkers > 0
    && Object.keys(report.uploaded_workers ?? {}).length >= expectedWorkers
    ? "shipped to workers"
    : "preprocessed";
}

function actionParameters(
  dataset: DatasetMetadata,
  normalization: Normalization,
  split: SplitStrategy,
  size: { height: number; width: number },
  seed: number,
): Record<string, unknown> {
  return {
    dataset_name: dataset.dataset_name,
    normalization,
    dataset_split: split,
    target_size: size,
    seed,
    replace_existing: true,
  };
}

function errorText(cause: unknown): string {
  return cause instanceof Error ? cause.message : "The coordinator request failed.";
}

export default function DatasetsPage({ token, active = true }: Props) {
  const [datasets, setDatasets] = useState<DatasetMetadata[]>([]);
  const [reports, setReports] = useState<Record<string, PreprocessingReport | null>>({});
  const [samples, setSamples] = useState<DatasetSample[]>([]);
  const [samplesDatasetName, setSamplesDatasetName] = useState("");
  const [samplesError, setSamplesError] = useState("");
  const [selectedName, setSelectedName] = useState("");
  const [autonomy, setAutonomy] = useState<AutonomyState | null>(null);
  const [normalization, setNormalization] = useState<Normalization>("standard");
  const [split, setSplit] = useState<SplitStrategy>("stratified");
  const [targetHeight, setTargetHeight] = useState(32);
  const [targetWidth, setTargetWidth] = useState(32);
  const [seed, setSeed] = useState(42);
  const [busy, setBusy] = useState(false);
  const [progress, setProgress] = useState("");
  const [notice, setNotice] = useState("");
  const [error, setError] = useState("");
  const [warning, setWarning] = useState("");
  const [uncertain, setUncertain] = useState(false);
  const selectedNameRef = useRef("");

  const selected = datasets.find((dataset) => dataset.dataset_name === selectedName) ?? null;
  const selectedReport = selected ? reports[selected.dataset_name] ?? null : null;
  const busyLabel = busy ? progress : "";

  const refresh = useCallback(async () => {
    setError("");
    try {
      const [items, state] = await Promise.all([
        listDatasets(token),
        getAutonomyState(token),
      ]);
      const officialDatasets = items
        .filter((item) => OFFICIAL_DATASETS.has(item.dataset_name))
        .sort((left, right) => DATASET_ORDER.indexOf(left.dataset_name) - DATASET_ORDER.indexOf(right.dataset_name));
      const reportPairs = await Promise.all(
        officialDatasets.map(async (item) => [
          item.dataset_name,
          await getLatestPreprocessingReport(token, item.dataset_name),
        ] as const),
      );
      setDatasets(officialDatasets);
      setReports(Object.fromEntries(reportPairs));
      setAutonomy(state);
      const currentName = selectedNameRef.current;
      const nextName = currentName && officialDatasets.some((item) => item.dataset_name === currentName)
        ? currentName
        : officialDatasets[0]?.dataset_name ?? "";
      if (nextName !== currentName) {
        selectedNameRef.current = nextName;
        setSelectedName(nextName);
        const nextDataset = officialDatasets.find((item) => item.dataset_name === nextName);
        if (nextDataset) {
          const size = imageSize(nextDataset);
          setTargetHeight(size.height);
          setTargetWidth(size.width);
        }
      }
    } catch (cause) {
      setError(errorText(cause));
    }
  }, [token]);

  useEffect(() => {
    if (!active) return;
    // Load the authenticated coordinator's dataset and autonomy state on page entry.
    // eslint-disable-next-line react-hooks/set-state-in-effect
    void refresh();
  }, [active, refresh]);

  useEffect(() => {
    if (!selectedName) return;
    let active = true;
    void getDatasetSamples(token, selectedName)
      .then((items) => {
        if (!active) return;
        setSamples(items);
        setSamplesDatasetName(selectedName);
        setSamplesError("");
      })
      .catch((cause: unknown) => {
        if (!active) return;
        setSamples([]);
        setSamplesDatasetName(selectedName);
        setSamplesError(errorText(cause));
      });
    return () => {
      active = false;
    };
  }, [selectedName, token]);

  function selectDataset(dataset: DatasetMetadata) {
    selectedNameRef.current = dataset.dataset_name;
    setSelectedName(dataset.dataset_name);
    const size = imageSize(dataset);
    setTargetHeight(size.height);
    setTargetWidth(size.width);
  }

  function makeAction(dataset: DatasetMetadata) {
    return {
      action: "preprocess_dataset" as const,
      parameters: actionParameters(
        dataset,
        normalization,
        split,
        { height: targetHeight, width: targetWidth },
        seed,
      ),
    };
  }

  async function submitPreprocessing(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!selected || busy || autonomy?.stop_requested) return;
    setBusy(true);
    setProgress("Preprocessing on the PC backend and shipping to all registered workers…");
    setError("");
    setWarning("");
    setNotice("");
    setUncertain(false);
    const action = makeAction(selected);
    const reason = `Preprocess ${selected.dataset_name} on the PC backend, then replace existing worker shards with the selected split.`;
    const evidence = {
      dataset_name: selected.dataset_name,
      image_count: selected.image_count,
      image_channels: selected.image_channels,
      image_sizes: selected.image_sizes,
      source_split_counts: selected.source_split_counts ?? { train: selected.image_count },
      replace_existing: true,
    };
    try {
      const result = await dispatchDatasetPreprocessing(token, action, reason, evidence);
      setProgress("Preprocessing on the PC backend and shipping to all registered workers…");
      if (!result.result.available) {
        throw new Error(result.result.message ?? "Preprocessing was not available.");
      }
      if (result.result.data) {
        setReports((current) => ({ ...current, [selected.dataset_name]: result.result.data ?? null }));
      }
      setNotice("Preprocessing and shard shipping completed.");
      setWarning(result.history_warning ?? "");
      await refresh();
    } catch (cause) {
      const message = errorText(cause);
      setError(message);
      if (/timed out/i.test(message)) {
        setUncertain(true);
        setNotice("The action may still be running. Check status; it will not be dispatched again.");
      }
    } finally {
      setBusy(false);
      setProgress("");
    }
  }

  async function reconcile() {
    if (busy) return;
    setBusy(true);
    setProgress("Checking dataset preprocessing status…");
    setError("");
    setUncertain(false);
    try {
      const [items, state] = await Promise.all([
        listDatasets(token),
        getAutonomyState(token),
      ]);
      const officialDatasets = items
        .filter((item) => OFFICIAL_DATASETS.has(item.dataset_name))
        .sort((left, right) => DATASET_ORDER.indexOf(left.dataset_name) - DATASET_ORDER.indexOf(right.dataset_name));
      const reportPairs = await Promise.all(
        officialDatasets.map(async (item) => [
          item.dataset_name,
          await getLatestPreprocessingReport(token, item.dataset_name),
        ] as const),
      );
      setDatasets(officialDatasets);
      setReports(Object.fromEntries(reportPairs));
      setAutonomy(state);
      setNotice("Status refreshed. No preprocessing action was repeated.");
    } catch (cause) {
      setError(errorText(cause));
    } finally {
      setBusy(false);
      setProgress("");
    }
  }

  const stopped = autonomy?.stop_requested ?? false;
  return (
    <section className="content datasets-page">
      <div className="page-heading">
        <h1>Datasets</h1>
        <div className="dataset-top-controls">
          {datasets.length > 0 && (
            <label className="dataset-select-label" htmlFor="active-dataset">
              <span>Dataset</span>
              <select
                id="active-dataset"
                onChange={(event) => {
                  const dataset = datasets.find((item) => item.dataset_name === event.target.value);
                  if (dataset) selectDataset(dataset);
                }}
                value={selectedName}
              >
                {datasets.map((dataset) => (
                  <option key={dataset.dataset_name} value={dataset.dataset_name}>
                    {datasetLabel(dataset.dataset_name)}
                  </option>
                ))}
              </select>
            </label>
          )}
          <button className="quiet-button" disabled={busy} onClick={() => void refresh()} type="button">Refresh</button>
        </div>
      </div>
      {stopped && <div className="alert dataset-stop" role="alert">Global STOP is active. Preprocessing is disabled until STOP is reset.</div>}
      {error && <div className="alert inline-alert" role="alert">{error}</div>}
      {warning && <div className="alert inline-alert" role="status">{warning}</div>}
      {notice && <div className="success-note dataset-notice" role="status">{notice}</div>}
      {busy && <div className="dataset-progress" role="status"><span className="spinner" />{busyLabel}</div>}
      {uncertain && <button className="quiet-button dataset-retry" disabled={busy} onClick={() => void reconcile()} type="button">Check status</button>}

      {selected && (
        <article className="panel-card dataset-detail">
          <div className="dataset-overview">
            <div className="dataset-heading">
              <div>
                <h2>{datasetLabel(selected.dataset_name)}</h2>
                <span>{selected.image_count.toLocaleString()} images</span>
              </div>
              <span className={`dataset-state state-${datasetState(selectedReport).replace(" ", "-")}`}>{datasetState(selectedReport)}</span>
            </div>
            <div className="dataset-stats">
              <div><span>Classes</span><strong>{selected.classes.length}</strong></div>
              <div><span>Image size</span><strong>{imageSize(selected).width} × {imageSize(selected).height}</strong></div>
              <div><span>Channels</span><strong>{channelCount(selected)}</strong></div>
            </div>
          </div>

          <section className="dataset-samples" aria-labelledby="dataset-samples-title">
            <div className="dataset-section-heading">
              <h3 id="dataset-samples-title">Samples</h3>
              <span>{samplesDatasetName === selected.dataset_name ? `${samples.length} examples` : "Loading…"}</span>
            </div>
            {samplesError && samplesDatasetName === selected.dataset_name ? (
              <p className="dataset-sample-error" role="status">Samples unavailable.</p>
            ) : samplesDatasetName === selected.dataset_name && samples.length > 0 ? (
              <div className="dataset-sample-grid">
                {samples.map((sample) => (
                  <figure className="dataset-sample" key={sample.label}>
                    <img alt={`${sample.label} sample`} src={sample.image_data} />
                    <figcaption>{sample.label}</figcaption>
                  </figure>
                ))}
              </div>
            ) : null}
          </section>

          {selectedReport && (
            <details className="dataset-report-details">
              <summary>Last preprocessing result</summary>
              <PreprocessingDetails report={selectedReport} />
            </details>
          )}

          <div className="dataset-preprocess">
            <div className="dataset-section-heading">
              <h3>Preprocess and ship</h3>
            </div>
            <form className="dataset-form" onSubmit={(event) => void submitPreprocessing(event)}>
              <label>Normalization
                <select value={normalization} onChange={(event) => setNormalization(event.target.value as Normalization)}>
                  <option value="standard">Standard</option><option value="minmax">Min-max</option><option value="none">None</option>
                </select>
              </label>
              <label>Split strategy
                <select value={split} onChange={(event) => setSplit(event.target.value as SplitStrategy)}>
                  <option value="stratified">Stratified</option>
                  <option value="dirichlet_a01">Dirichlet α=0.1</option>
                  <option value="dirichlet_a03">Dirichlet α=0.3</option>
                  <option value="dirichlet_a05">Dirichlet α=0.5</option>
                  <option value="dirichlet_a07">Dirichlet α=0.7</option>
                  <option value="dirichlet_a09">Dirichlet α=0.9</option>
                </select>
              </label>
              {!selected.source_split_policy && (
                <>
                  <label>Target height
                    <input max={2048} min={1} onChange={(event) => setTargetHeight(Number(event.target.value))} type="number" value={targetHeight} />
                  </label>
                  <label>Target width
                    <input max={2048} min={1} onChange={(event) => setTargetWidth(Number(event.target.value))} type="number" value={targetWidth} />
                  </label>
                </>
              )}
              <label>Seed
                <input max={4294967295} min={0} onChange={(event) => setSeed(Number(event.target.value))} type="number" value={seed} />
              </label>
              <button className="primary-button" disabled={busy || stopped || !autonomy} type="submit">
                {busy ? "Working…" : "Preprocess & ship"}
              </button>
            </form>
          </div>
        </article>
      )}
      {!selected && (
        <div className="dataset-empty">
          {error ? "Dataset list unavailable." : "No datasets are prepared."}
        </div>
      )}
    </section>
  );
}

function PreprocessingDetails({ report }: { report: PreprocessingReport }) {
  const workers = Object.entries(report.workers);
  return (
    <section className="dataset-report">
      <h3>Preprocessing result</h3>
      <p>Split strategy: {report.split_strategy} · normalization: {report.normalization.method} · target: {report.target_size.height}×{report.target_size.width} · {report.channels} channel(s)</p>
      <div className="dataset-report-splits">
        {Object.entries(report.split_sizes).map(([splitName, count]) => (
          <div key={splitName}><strong>{splitName}</strong><span>{count.toLocaleString()} samples</span><small>{Object.entries(report.split_class_counts[splitName] ?? {}).map(([name, amount]) => `${name}: ${amount}`).join(", ")}</small></div>
        ))}
      </div>
      <div className="dataset-table-wrap">
        <table className="dataset-table">
          <thead><tr><th>Worker</th><th>Samples</th><th>Per-class counts</th><th>Shard SHA-256</th><th>PC / Jetson</th></tr></thead>
          <tbody>
            {workers.map(([worker, item]) => {
              const shipped = report.uploaded_workers?.[worker];
              const matches = Boolean(item.pc_shard_sha256 && shipped?.shard_sha256 && item.pc_shard_sha256 === shipped.shard_sha256);
              const verified = Boolean(item.pc_shard_sha256 && shipped?.shard_sha256);
              return <tr key={worker}>
                <td>{worker}</td><td>{item.rows}</td><td>{Object.entries(item.class_counts).map(([name, count]) => `${name}: ${count}`).join(", ")}</td>
                <td className="hash-cell">PC: {item.pc_shard_sha256 ?? "Unavailable"}<br />Jetson: {shipped?.shard_sha256 ?? "Unavailable"}</td>
                <td><span className={matches ? "hash-match" : verified ? "hash-mismatch" : "hash-unavailable"}>{matches ? "Match" : verified ? "Mismatch" : "Unavailable"}</span></td>
              </tr>;
            })}
          </tbody>
        </table>
      </div>
      <p className="dataset-durations">Preprocessing: {report.preprocessing_duration_seconds ?? "—"} s · Shipping: {report.shipping_duration_seconds ?? "—"} s</p>
    </section>
  );
}
