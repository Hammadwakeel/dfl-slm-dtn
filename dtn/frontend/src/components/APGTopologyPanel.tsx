import { useCallback, useEffect, useMemo, useState } from "react";

import {
  getAPGJob,
  getAPGRun,
  getAPGRuns,
  getAutonomyState,
  requestGlobalStop,
  resetGlobalStop,
  startAPGSimulation,
  type APGJob,
  type APGRoundSummary,
  type APGRunDetail,
  type APGRunSummary,
  type APGTestSummary,
  type APGVerdict,
} from "../api/client";

const TOPOLOGIES = ["apg", "ring", "mesh"] as const;
const SETTINGS = ["iid_stratified", "constrained_dirichlet_a01"] as const;
const SEEDS = [17, 42, 2026] as const;
const COLORS: Record<(typeof TOPOLOGIES)[number], string> = {
  apg: "#4ed8a4",
  mesh: "#68a9f2",
  ring: "#e5aa62",
};
const NODES = Array.from({ length: 8 }, (_, index) => `node_${index}`);

type GraphView = "apg" | "mesh" | "ring";
type MetricKey = "mean_validation_accuracy" | "mean_validation_macro_f1" | "mean_validation_loss";

export default function APGTopologyPanel({ token }: { token: string }) {
  const [runs, setRuns] = useState<APGRunSummary[]>([]);
  const [jobs, setJobs] = useState<APGJob[]>([]);
  const [verdicts, setVerdicts] = useState<APGVerdict[]>([]);
  const [detail, setDetail] = useState<APGRunDetail | null>(null);
  const [stopRequested, setStopRequested] = useState(true);
  const [topology, setTopology] = useState<(typeof TOPOLOGIES)[number]>("apg");
  const [dataSetting, setDataSetting] = useState<(typeof SETTINGS)[number]>("iid_stratified");
  const [seed, setSeed] = useState<(typeof SEEDS)[number]>(17);
  const [selectedRunId, setSelectedRunId] = useState("");
  const [selectedRound, setSelectedRound] = useState(1);
  const [graphView, setGraphView] = useState<GraphView>("apg");
  const [confusionView, setConfusionView] = useState<"counts" | "normalized">("counts");
  const [error, setError] = useState("");
  const [starting, setStarting] = useState(false);
  const [stopBusy, setStopBusy] = useState(false);

  const refresh = useCallback(async () => {
    try {
      const [result, autonomy] = await Promise.all([
        getAPGRuns(token),
        getAutonomyState(token),
      ]);
      setRuns(result.runs);
      setJobs(result.jobs);
      setVerdicts(result.verdicts);
      setStopRequested(autonomy.stop_requested);
      setError("");
      setSelectedRunId((current) => {
        if (current && result.runs.some((run) => run.run_id === current)) return current;
        const latestFinished = [...result.runs].reverse().find((run) => run.status === "finished");
        return latestFinished?.run_id ?? "";
      });
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Could not load APG results.");
    }
  }, [token]);

  useEffect(() => {
    // The effect starts asynchronous API refreshes; state updates occur after the requests resolve.
    // eslint-disable-next-line react-hooks/set-state-in-effect
    void refresh();
    const timer = window.setInterval(() => void refresh(), 3000);
    return () => window.clearInterval(timer);
  }, [refresh]);

  useEffect(() => {
    if (!selectedRunId) return;
    let active = true;
    void getAPGRun(token, selectedRunId)
      .then((response) => {
        if (active) {
          setDetail(response);
          setSelectedRound((round) => Math.min(round, response.run.rounds.length || 1));
        }
      })
      .catch((cause: unknown) => {
        if (active) setError(cause instanceof Error ? cause.message : "Could not load APG run.");
      });
    return () => {
      active = false;
    };
  }, [selectedRunId, token]);

  useEffect(() => {
    const activeJobs = jobs.filter((job) => job.status === "queued" || job.status === "running");
    if (!activeJobs.length) return;
    let active = true;
    const timer = window.setInterval(() => {
      void Promise.all(activeJobs.map((job) => getAPGJob(token, job.job_id)))
        .then((updated) => {
          if (!active) return;
          setJobs((current) => current.map((job) => updated.find((item) => item.job_id === job.job_id) ?? job));
          const finished = updated.find((job) => job.status === "finished");
          if (finished) {
            setSelectedRunId(finished.run_id);
            void refresh();
          }
        })
        .catch((cause: unknown) => {
          if (active) setError(cause instanceof Error ? cause.message : "Could not refresh APG job status.");
        });
    }, 1000);
    return () => {
      active = false;
      window.clearInterval(timer);
    };
  }, [jobs, refresh, token]);

  const activeJob = jobs.find((job) => job.status === "queued" || job.status === "running");
  const selectedSummary = runs.find((run) => run.run_id === selectedRunId) ?? null;
  const selectedDetail = detail?.summary.run_id === selectedRunId ? detail : null;
  const rounds = selectedDetail?.run.rounds ?? [];
  const round = rounds.find((item) => item.round === selectedRound) ?? rounds[0] ?? null;
  const sourceLabel = selectedDetail?.summary.source_label ?? activeJob?.source_label ?? "NOT AVAILABLE";
  const peers = useMemo(
    () => runs.filter(
      (run) =>
        run.simulation_mode === selectedSummary?.simulation_mode &&
        run.data_setting === selectedSummary?.data_setting &&
        ["apg", "mesh", "ring"].includes(run.topology),
    ),
    [runs, selectedSummary?.data_setting, selectedSummary?.simulation_mode],
  );
  const latestJob = activeJob ?? jobs[jobs.length - 1];
  const startDisabled = stopRequested || starting || Boolean(activeJob);

  async function startSimulation() {
    setStarting(true);
    setError("");
    try {
      const job = await startAPGSimulation(token, {
        topology,
        data_setting: dataSetting,
        seed,
      });
      setJobs((current) => [...current.filter((item) => item.job_id !== job.job_id), job]);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Could not start APG simulation.");
    } finally {
      setStarting(false);
    }
  }

  async function toggleGlobalStop() {
    setStopBusy(true);
    setError("");
    try {
      const state = stopRequested
        ? await resetGlobalStop(token)
        : await requestGlobalStop(token);
      setStopRequested(state.stop_requested);
      const result = await getAPGRuns(token);
      setJobs(result.jobs);
      setRuns(result.runs);
      setVerdicts(result.verdicts);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Could not update global STOP.");
    } finally {
      setStopBusy(false);
    }
  }

  return (
    <section className="apg-workbench panel-card" aria-labelledby="apg-title">
      <div className="section-heading">
        <div>
          <span className="eyebrow">APG · PC SIMULATION</span>
          <h2 id="apg-title">Topology Learning (APG)</h2>
        </div>
        <span className="apg-source-badge">{sourceLabel}</span>
      </div>

      <div className="apg-launch-controls">
        <button
          aria-label={stopRequested ? "Reset global STOP" : "Activate global STOP"}
          className={stopRequested ? "quiet-button" : "apg-stop-button"}
          disabled={stopBusy}
          onClick={() => void toggleGlobalStop()}
          type="button"
        >
          {stopBusy ? "Updating STOP…" : stopRequested ? "Reset global STOP" : "Global STOP"}
        </button>
        <label>
          Topology
          <select aria-label="APG simulation topology" onChange={(event) => setTopology(event.target.value as typeof topology)} value={topology}>
            {TOPOLOGIES.map((value) => <option key={value} value={value}>{value}</option>)}
          </select>
        </label>
        <label>
          Data setting
          <select aria-label="APG simulation data setting" onChange={(event) => setDataSetting(event.target.value as typeof dataSetting)} value={dataSetting}>
            <option value="iid_stratified">IID</option>
            <option value="constrained_dirichlet_a01">Constrained Dirichlet</option>
          </select>
        </label>
        <label>
          Seed
          <select aria-label="APG simulation seed" onChange={(event) => setSeed(Number(event.target.value) as typeof seed)} value={seed}>
            {SEEDS.map((value) => <option key={value} value={value}>{value}</option>)}
          </select>
        </label>
        <button className="primary-button" disabled={startDisabled} onClick={() => void startSimulation()} type="button">
          {starting ? "Queueing…" : "Run APG simulation (PC only)"}
        </button>
      </div>
      <div className="apg-hybrid-unavailable">
        <button disabled type="button">Run APG on hybrid (3 real Jetsons)</button>
        <span>NOT AVAILABLE · Phase C not implemented or approved</span>
      </div>
      {stopRequested && <div className="alert inline-alert" role="alert">Global STOP is active. APG simulation start is disabled; an active job will stop before TEST evaluation.</div>}
      {latestJob && (
        <div className="apg-job-status" role="status">
          Job {latestJob.job_id} · {latestJob.status} · round {latestJob.rounds_completed}/{latestJob.total_rounds}
          {latestJob.duration_seconds === null ? "" : ` · ${latestJob.duration_seconds.toFixed(1)} PC seconds`}
          {latestJob.error ? ` · ${latestJob.error}` : ""}
        </div>
      )}
      {error && <div className="alert inline-alert" role="alert">{error}</div>}

      <div className="apg-run-header">
        <label>
          Saved run
          <select aria-label="Saved APG result run" onChange={(event) => setSelectedRunId(event.target.value)} value={selectedRunId}>
            <option value="">Select a saved run</option>
            {runs.map((run) => (
              <option key={run.run_id} value={run.run_id}>
                {run.run_id} · {run.status}
              </option>
            ))}
          </select>
        </label>
        <span>run_id: {selectedSummary?.run_id ?? latestJob?.run_id ?? "NOT AVAILABLE"}</span>
        <span>source: {selectedSummary?.source_label ?? "NOT AVAILABLE"}</span>
        <span>data setting: {selectedSummary ? displaySetting(selectedSummary.data_setting) : "NOT AVAILABLE"}</span>
        <span>topology: {selectedSummary?.topology ?? "NOT AVAILABLE"}</span>
        <span>seed: {selectedSummary?.seed ?? "NOT AVAILABLE"}</span>
        <span>status: {selectedSummary?.status ?? latestJob?.status ?? "NOT AVAILABLE"}</span>
        <span>duration: {formatSeconds(selectedSummary?.duration_seconds)} PC · {selectedSummary?.source_label ?? "NOT AVAILABLE"}</span>
      </div>
      <p className="apg-synthetic-note">T synthetic · psi = 0.01</p>

      {selectedDetail && round && (
        <>
          <div className="apg-round-control">
            <label htmlFor="apg-round">Selected round {selectedRound} / {rounds.length} · VALIDATION · {sourceLabel}</label>
            <input
              id="apg-round"
              aria-label="Selected APG round"
              max={Math.max(rounds.length, 1)}
              min={1}
              onChange={(event) => setSelectedRound(Number(event.target.value))}
              type="range"
              value={Math.min(selectedRound, Math.max(rounds.length, 1))}
            />
          </div>
          <div className="apg-graph-mode" role="group" aria-label="Topology graph comparison">
            {(["apg", "mesh", "ring"] as const).map((value) => (
              <button
                aria-pressed={graphView === value}
                key={value}
                onClick={() => setGraphView(value)}
                type="button"
              >
                {value === "apg" ? "APG selected round" : `${value} comparison`}
              </button>
            ))}
          </div>
          <NodeMap edges={graphEdges(graphView, round)} selectedTopology={graphView} source={sourceLabel} />
          <EdgeChangeTable rounds={rounds} source={sourceLabel} />
          <InputHistory rounds={rounds} source={sourceLabel} />

          <div className="apg-chart-grid">
            <ConvergenceChart
              selected={selectedSummary}
              runs={peers}
              metric="mean_validation_accuracy"
              title="VALIDATION accuracy"
              source={sourceLabel}
            />
            <ConvergenceChart
              selected={selectedSummary}
              runs={peers}
              metric="mean_validation_macro_f1"
              title="VALIDATION macro-F1"
              source={sourceLabel}
            />
            <ConvergenceChart
              selected={selectedSummary}
              runs={peers}
              metric="mean_validation_loss"
              title="VALIDATION loss"
              source={sourceLabel}
            />
          </div>
          <PerNodeTable detail={selectedDetail} source={sourceLabel} />
          <TestPanel
            test={selectedDetail.run.final_test?.summary ?? null}
            counts={selectedDetail.run.final_test?.confusion_matrix_counts ?? null}
            normalized={selectedDetail.run.final_test?.confusion_matrix_row_normalized_percent ?? null}
            source={sourceLabel}
            view={confusionView}
            onViewChange={setConfusionView}
          />
        </>
      )}
      <VerdictPanel verdicts={verdicts} />
    </section>
  );
}

function graphEdges(view: GraphView, round: APGRoundSummary): Array<[string, string]> {
  if (view === "apg") return round.edges_used;
  if (view === "mesh") return NODES.flatMap((receiver) => NODES.filter((source) => source !== receiver).map((source) => [receiver, source] as [string, string]));
  return NODES.map((receiver, index) => [receiver, NODES[(index + NODES.length - 1) % NODES.length]]);
}

function NodeMap({
  edges,
  selectedTopology,
  source,
}: {
  edges: Array<[string, string]>;
  selectedTopology: GraphView;
  source: string;
}) {
  const positions = NODES.map((node, index) => {
    const angle = -Math.PI / 2 + (2 * Math.PI * index) / NODES.length;
    return { node, x: 320 + Math.cos(angle) * 230, y: 245 + Math.sin(angle) * 170 };
  });
  const positionsById = new Map(positions.map((point) => [point.node, point]));
  const inDegree = Object.fromEntries(NODES.map((node) => [node, 0])) as Record<string, number>;
  const outDegree = Object.fromEntries(NODES.map((node) => [node, 0])) as Record<string, number>;
  edges.forEach(([receiver, sourceNode]) => {
    inDegree[receiver] += 1;
    outDegree[sourceNode] += 1;
  });
  return (
    <div className="apg-map-card">
      <h3>Node map · {selectedTopology} · receiver selects source · {source}</h3>
      <svg
        className="apg-node-map"
        viewBox="0 0 640 500"
        role="img"
        aria-label={`${selectedTopology} receiver-centric graph source to receiver`}
      >
        <defs>
          <marker id="apg-arrow" markerWidth="8" markerHeight="8" refX="7" refY="3" orient="auto">
            <path d="M0,0 L0,6 L7,3 z" />
          </marker>
        </defs>
        {edges.map(([receiver, sourceNode], index) => {
          const sourcePoint = positionsById.get(sourceNode);
          const receiverPoint = positionsById.get(receiver);
          if (!sourcePoint || !receiverPoint) return null;
          return (
            <line
              data-source={sourceNode}
              data-receiver={receiver}
              key={`${sourceNode}-${receiver}-${index}`}
              markerEnd="url(#apg-arrow)"
              x1={sourcePoint.x}
              x2={receiverPoint.x}
              y1={sourcePoint.y}
              y2={receiverPoint.y}
            />
          );
        })}
        {positions.map(({ node, x, y }) => (
          <g key={node}>
            <circle cx={x} cy={y} r="25" />
            <text x={x} y={y + 4}>{node.slice(-1)}</text>
            <text className="apg-node-label" x={x} y={y + 42}>
              {node} {Number(node.slice(-1)) >= 5 ? `(worker${Number(node.slice(-1)) - 4}, emulated in PC)` : "(simulated)"}
            </text>
            <text className="apg-node-degree" x={x} y={y + 57}>
              in {inDegree[node]} · out {outDegree[node]}
            </text>
          </g>
        ))}
      </svg>
    </div>
  );
}

function EdgeChangeTable({ rounds, source }: { rounds: APGRoundSummary[]; source: string }) {
  return (
    <div className="table-scroll">
      <table className="metrics-table apg-table">
        <caption>APG edge changes · VALIDATION · {source}</caption>
        <thead><tr><th>Round</th><th>SIMULATED edges used</th><th>Changed vs previous round</th><th>Safeguard hits</th><th>g</th><th>APG iterations</th></tr></thead>
        <tbody>
          {rounds.map((item) => (
            <tr key={item.round}>
              <td>{item.round}</td><td>{item.edge_count_used ?? "NOT AVAILABLE"}</td>
              <td>{item.edges_changed_vs_previous_round ?? "NOT AVAILABLE"}</td>
              <td>{item.safeguard_hits ?? "NOT AVAILABLE"}</td>
              <td>{formatNumber(item.g)}</td><td>{item.iterations_used ?? "NOT AVAILABLE"}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function InputHistory({ rounds, source }: { rounds: APGRoundSummary[]; source: string }) {
  return (
    <details className="apg-input-history">
      <summary>APG inputs per round · TRAIN-only Theta and VALIDATION loss · T SIMULATED · {source}</summary>
      <div className="table-scroll">
        <table className="metrics-table apg-table">
          <thead><tr><th>Round</th><th>L normalized range/mean</th><th>T synthetic range/mean</th><th>Theta TRAIN-only range/mean</th><th>Inputs changed vs previous</th></tr></thead>
          <tbody>
            {rounds.map((item) => {
              const inputs = item.inputs;
              const stats = [inputs.loss_stats, inputs.waiting_time_stats, inputs.theta_stats];
              const changed = [
                ["L", inputs.loss_changed],
                ["T", inputs.waiting_time_changed],
                ["Theta", inputs.theta_changed],
              ].filter(([, value]) => value === true).map(([name]) => name).join(", ");
              return (
                <tr key={item.round}>
                  <td>{item.round}</td>
                  {stats.map((value, index) => (
                    <td key={index}>{value ? `${formatNumber(value.min)} / ${formatNumber(value.mean)} / ${formatNumber(value.max)}` : "NOT AVAILABLE"}</td>
                  ))}
                  <td>{item.round === 1 ? "Baseline" : changed || "Constant"}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      {rounds.map((item) => (
        <details key={item.round}>
          <summary>Round {item.round} exact APG input values</summary>
          {item.apg_inputs
            ? <pre>{JSON.stringify(item.apg_inputs, null, 2)}</pre>
            : <p>NOT AVAILABLE</p>}
        </details>
      ))}
    </details>
  );
}

function ConvergenceChart({
  selected,
  runs,
  metric,
  title,
  source,
}: {
  selected: APGRunSummary | null;
  runs: APGRunSummary[];
  metric: MetricKey;
  title: string;
  source: string;
}) {
  const width = 620;
  const height = 220;
  const margin = 35;
  const maxRounds = Math.max(1, ...runs.map((run) => run.rounds.length));
  const values = runs.flatMap((run) =>
    run.rounds.map((round) => round[metric]).filter((value): value is number => typeof value === "number"),
  );
  const minValue = values.length ? Math.min(...values) : 0;
  const maxValue = values.length ? Math.max(...values) : 1;
  const pad = (maxValue - minValue) * 0.1 || 0.05;
  const low = Math.max(0, minValue - pad);
  const high = maxValue + pad;
  const x = (round: number) => margin + ((round - 1) / Math.max(1, maxRounds - 1)) * (width - 2 * margin);
  const y = (value: number) => height - margin - ((value - low) / (high - low)) * (height - 2 * margin);
  const lines = TOPOLOGIES.map((topology) => {
    const sameTopology = runs.filter((run) => run.topology === topology);
    const selectedRun = sameTopology.find((run) => run.run_id === selected?.run_id)
      ?? sameTopology.find((run) => run.seed === selected?.seed);
    const uniqueSeedRuns = new Map<number, APGRunSummary>();
    sameTopology.forEach((run) => {
      if (run.seed === null) return;
      const current = uniqueSeedRuns.get(run.seed);
      if (!current || (current.run_id.startsWith("apg-ui-") && !run.run_id.startsWith("apg-ui-"))) {
        uniqueSeedRuns.set(run.seed, run);
      }
    });
    const points = selectedRun?.rounds
      .map((round) => {
        const value = round[metric];
        return typeof value === "number" ? `${x(round.round)},${y(value)}` : null;
      })
      .filter((point): point is string => point !== null) ?? [];
    const hasThreeSeeds = uniqueSeedRuns.size >= 3;
    const band = hasThreeSeeds
      ? Array.from({ length: maxRounds }, (_, index) => {
          const roundNumber = index + 1;
          const roundValues = [...uniqueSeedRuns.values()]
            .map((run) => run.rounds.find((row) => row.round === roundNumber)?.[metric])
            .filter((value): value is number => typeof value === "number");
          if (roundValues.length < 3) return null;
          const mean = roundValues.reduce((sum, value) => sum + value, 0) / roundValues.length;
          const variance = roundValues.reduce((sum, value) => sum + (value - mean) ** 2, 0) / (roundValues.length - 1);
          const sd = Math.sqrt(variance);
          return { upper: `${x(roundNumber)},${y(mean + sd)}`, lower: `${x(roundNumber)},${y(mean - sd)}` };
        }).filter((point): point is { upper: string; lower: string } => point !== null)
      : [];
    return { topology, points, band };
  });
  return (
    <article className="apg-chart-card">
      <h3>{title} · {source}</h3>
      <p>Seed {selected?.seed ?? "NOT AVAILABLE"} · SD band · 3 saved seeds when available.</p>
      {values.length === 0 ? <p>{title} · {source}: NOT AVAILABLE</p> : (
        <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label={`${title} per round ${source}`}>
          {lines.map(({ topology, points, band }) => (
            <g key={topology}>
              {band.length >= 2 && (
                <polygon
                  className="apg-chart-band"
                  fill={COLORS[topology]}
                          points={[...band.map((point) => point.upper), ...[...band].reverse().map((point) => point.lower)].join(" ")}
                />
              )}
              {points.length > 0 && <polyline fill="none" points={points.join(" ")} stroke={COLORS[topology]} strokeWidth="2.5" />}
            </g>
          ))}
          <text x={margin} y={height - 8}>Round 1</text>
          <text x={width - margin - 35} y={height - 8}>Round {maxRounds}</text>
          <text x={margin} y={margin - 9}>{high.toFixed(3)}</text>
          <text x={margin} y={height - margin + 13}>{low.toFixed(3)}</text>
        </svg>
      )}
      <div className="apg-chart-legend">
        {TOPOLOGIES.map((topology) => <span key={topology} style={{ color: COLORS[topology] }}>{topology}</span>)}
      </div>
    </article>
  );
}

function PerNodeTable({ detail, source }: { detail: APGRunDetail; source: string }) {
  const rounds = detail.run.rounds;
  const nodes = detail.run.nodes ?? [];
  return (
    <div className="table-scroll">
      <table className="metrics-table apg-table">
        <caption>Per-node · VALIDATION per-round metrics · bytes SIMULATED · {source}</caption>
        <thead>
          <tr>
            <th>Node / role</th><th>Shard size (rows)</th><th>TRAIN-only Theta summary</th>
            {rounds.map((round) => <th key={round.round}>VALIDATION accuracy R{round.round}</th>)}
            <th>Bytes SIMULATED</th>
          </tr>
        </thead>
        <tbody>
          {NODES.map((nodeId, index) => {
            const node = nodes.find((item) => item.node_id === nodeId);
            const theta = Array.isArray(node?.theta) ? node.theta : [];
            const thetaLabel = theta.length
              ? `${theta.filter((value) => value > 0).length} classes represented; counts ${Math.min(...theta)}–${Math.max(...theta)}`
              : "NOT AVAILABLE";
            const bytes = rounds.reduce(
              (sum, round) => sum + (round.per_node?.[nodeId]?.bytes_sent_simulated ?? 0),
              0,
            );
            return (
              <tr key={nodeId}>
                <td>{nodeId}{index >= 5 ? ` · worker${index - 4} (emulated in PC)` : " · simulated"}</td>
                <td>{node?.shard_rows ?? "NOT AVAILABLE"}</td>
                <td>{thetaLabel}</td>
                {rounds.map((round) => (
                  <td key={round.round}>{formatPercent(round.per_node?.[nodeId]?.validation_accuracy, "VALIDATION", source)}</td>
                ))}
                <td>{bytes.toLocaleString()} · SIMULATED</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

function TestPanel({
  test,
  counts,
  normalized,
  source,
  view,
  onViewChange,
}: {
  test: APGTestSummary | null;
  counts: number[][] | null;
  normalized: number[][] | null;
  source: string;
  view: "counts" | "normalized";
  onViewChange: (view: "counts" | "normalized") => void;
}) {
  const matrix = view === "counts" ? counts : normalized;
  return (
    <section className="apg-test-panel">
      <h3>Final evaluation · TEST · {source}</h3>
      {!test ? <p>TEST metrics: NOT AVAILABLE.</p> : (
        <>
          <div className="apg-test-metrics">
            {(["accuracy", "macro_f1", "macro_precision", "macro_recall"] as const).map((name) => (
              <div key={name}>
                <span>TEST {name.replace(/_/g, " ")} · {source}</span>
                <strong>{formatNumber(test[name])}</strong>
              </div>
            ))}
          </div>
          <div className="apg-graph-mode" role="group" aria-label="TEST confusion matrix format">
            <button aria-pressed={view === "counts"} onClick={() => onViewChange("counts")} type="button">TEST confusion counts · {source}</button>
            <button aria-pressed={view === "normalized"} onClick={() => onViewChange("normalized")} type="button">TEST row-normalized % · {source}</button>
          </div>
          {!matrix ? <p>TEST confusion matrix: NOT AVAILABLE</p> : (
            <div className="table-scroll">
              <table className="apg-confusion" aria-label={`TEST confusion matrix ${view} ${source}`}>
                <thead><tr><th>TEST true \\ predicted</th>{matrix.map((_, index) => <th key={index}>{index}</th>)}</tr></thead>
                <tbody>
                  {matrix.map((row, rowIndex) => (
                    <tr key={rowIndex}><th>{rowIndex}</th>{row.map((value, columnIndex) => (
                      <td key={columnIndex} style={{ backgroundColor: `rgba(73, 201, 190, ${Math.min(Number(value) / Math.max(1, ...matrix.flat().map(Number)), 1) * 0.65})` }}>
                        {view === "counts" ? Math.round(Number(value)) : `${Number(value).toFixed(1)}%`}
                      </td>
                    ))}</tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </>
      )}
    </section>
  );
}

function VerdictPanel({ verdicts }: { verdicts: APGVerdict[] }) {
  if (verdicts.length === 0) {
    return (
      <section className="apg-verdict-panel">
        <h3>Is topology learning happening?</h3>
        <p>No saved topology runs.</p>
      </section>
    );
  }

  return (
    <section className="apg-verdict-panel">
      <h3>Is topology learning happening?</h3>
      <p>VALIDATION: training · TEST: final comparison only.</p>
      {SETTINGS.map((setting) => (
        <div className="apg-verdict-setting" key={setting}>
          <h4>{displaySetting(setting)}</h4>
          {(["all_simulated", "hybrid_shaped_simulation"] as const).map((mode) => {
            const item = verdicts.find((value) => value.data_setting === setting && value.simulation_mode === mode);
            const rule = item?.rules;
            const hasThree = Number(rule?.test_seed_count ?? 0) >= 3;
            const inputsConstant = rule?.topology_follows_inputs === false;
            return (
              <article key={mode}>
                <strong>{item?.verdict ?? "Not enough data (run more seeds)"}</strong>
                <span>Source: {item?.source_label ?? "NOT AVAILABLE"}</span>
                <p>Topology adapts: {rule?.topology_adapts === true ? "YES" : rule?.topology_adapts === false ? "NO" : "NOT AVAILABLE"}.</p>
                <p>Input/graph alignment: {inputsConstant ? "NO" : rule?.topology_follows_inputs === true ? "YES" : "NOT AVAILABLE"}.</p>
                <p>Validation learning: {rule?.model_learns === true ? "YES" : rule?.model_learns === false ? "NO" : "NOT CONCLUSIVE"}.</p>
                <p>Held-out TEST benefit: {rule?.benefit_over_best_fixed_topology === true ? "YES" : hasThree ? "NO" : "NOT DEMONSTRATED"} · n={String(rule?.test_seed_count ?? 0)}.</p>
                <small>TEST Δ &gt; sample SD · n ≥ 3</small>
              </article>
            );
          })}
        </div>
      ))}
    </section>
  );
}

function displaySetting(value: string): string {
  if (value === "iid_stratified") return "IID";
  if (value === "constrained_dirichlet_a01") return "Constrained Dirichlet";
  return value;
}

function formatNumber(value: number | null | undefined): string {
  return typeof value === "number" && Number.isFinite(value) ? value.toFixed(4) : "NOT AVAILABLE";
}

function formatSeconds(value: number | null | undefined): string {
  return typeof value === "number" && Number.isFinite(value) ? `${value.toFixed(1)} seconds` : "NOT AVAILABLE";
}

function formatPercent(value: number | undefined, split: "VALIDATION" | "TEST", source: string): string {
  return typeof value === "number" && Number.isFinite(value)
    ? `${(value * 100).toFixed(2)}% · ${split} · ${source}`
    : `NOT AVAILABLE · ${split} · ${source}`;
}
