import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import {
  getExperimentGroup,
  getSupervisorRun,
  listSupervisorRuns,
  type ExperimentGroup,
  type ExperimentResults,
  type NASArchitecture,
  type SupervisorRun,
} from "../api/client";
import {
  deriveAgentActions,
  edgeIn,
  edgePairs,
  nodeLabel,
  topologyEdges,
  workerNodes,
  type ActionOutcome,
  type AgentAction,
  type Configuration,
  type Edge,
} from "../lib/agentDecisions";

interface AgentDecisionsPageProps {
  token: string;
  active: boolean;
  results: ExperimentResults | null;
}

const ACTIVE_STATUSES = new Set(["queued", "running", "stopping"]);

const STEP_TITLES: Record<string, string> = {
  optimize_topology: "Topology optimization",
  learn_topology: "Learned topology (APG)",
  run_nas: "Architecture search (NAS)",
  run_hpo: "Hyperparameter tuning",
  optimize_aggregation: "Aggregation optimization",
  optimize_sync: "Synchronization optimization",
  prune_model: "Pruning + fine-tune",
  train_final: "Final DFL training",
  continue_training: "Continued DFL training",
};
const TOPOLOGY_NAMES: Record<string, string> = {
  directed_ring: "Directed ring",
  ring: "Directed ring",
  bidirectional_ring: "Bidirectional ring",
  mesh: "Mesh",
  chain: "Chain",
  star: "Star",
  learned: "Learned (APG)",
};
const AGGREGATION_NAMES: Record<string, string> = {
  fedavg: "FedAvg",
  uniform: "Uniform average",
  metropolis: "Metropolis–Hastings",
  trimmed_mean: "Trimmed mean",
};
const SYNC_NAMES: Record<string, string> = {
  sync: "Synchronous",
  semi_sync: "Semi-synchronous",
  async: "Asynchronous",
};
const SOURCE_NAMES: Record<string, string> = {
  slm: "SLM",
  slm_text: "SLM (text)",
  slm_repaired: "SLM after repair",
  fallback: "Fallback policy",
  rule: "Rule",
};
const OUTCOME_NAMES: Record<ActionOutcome, string> = {
  kept: "Kept",
  reverted: "Reverted",
  not_kept: "Not better — not used",
  applied: "Applied",
  failed: "Failed",
  pending: "Measuring…",
};
const DATASET_NAMES: Record<string, string> = {
  mnist: "MNIST",
  fashion_mnist: "Fashion-MNIST",
  femnist: "FEMNIST",
  cifar10: "CIFAR-10",
};

/** Every action the agent selected, with the configuration before and after it. */
export default function AgentDecisionsPage({ token, active, results }: AgentDecisionsPageProps) {
  const [runs, setRuns] = useState<SupervisorRun[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [run, setRun] = useState<SupervisorRun | null>(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(true);
  const shownId = selectedId ?? runs[0]?.id ?? null;
  const running = run ? ACTIVE_STATUSES.has(run.status) : false;

  const refresh = useCallback(async () => {
    try {
      const nextRuns = await listSupervisorRuns(token);
      setRuns(nextRuns);
      const id = selectedId ?? nextRuns[0]?.id ?? null;
      setRun(id ? await getSupervisorRun(token, id) : null);
      setError("");
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Could not load supervisor runs.");
    } finally {
      setLoading(false);
    }
  }, [token, selectedId]);

  useEffect(() => {
    if (!active) return;
    // Polling an external API; state updates happen after the request resolves.
    // eslint-disable-next-line react-hooks/set-state-in-effect
    void refresh();
    if (!running) return;
    const timer = window.setInterval(() => void refresh(), 15000);
    return () => window.clearInterval(timer);
  }, [active, refresh, running]);

  // The dashboard only keeps the newest master groups; fetch this run's own groups once
  // each (sequentially: the master is on Wi-Fi) for layer lists and sweep candidates.
  const [fetched, setFetched] = useState<Map<string, ExperimentGroup | null>>(new Map());
  const requested = useRef(new Set<string>());
  useEffect(() => {
    if (!active || !run) return;
    const known = new Set((results?.groups ?? []).map((group) => group.group_id));
    const wanted = (run.state?.experiments ?? [])
      .filter((item) => item.status === "completed" || item.step === "baseline_training")
      .map((item) => item.group_id)
      .filter((id) => id && !known.has(id) && !requested.current.has(id));
    if (wanted.length === 0) return;
    wanted.forEach((id) => requested.current.add(id));
    let cancelled = false;
    void (async () => {
      for (const [index, id] of wanted.entries()) {
        if (cancelled) {
          // Let the next run of this effect fetch what this one did not reach.
          wanted.slice(index).forEach((rest) => requested.current.delete(rest));
          return;
        }
        let group: ExperimentGroup | null = null;
        try {
          group = await getExperimentGroup(token, id);
        } catch {
          requested.current.delete(id);
          continue;
        }
        setFetched((current) => new Map(current).set(id, group));
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [active, run, results, token]);
  const groups = useMemo(() => {
    const merged = new Map<string, ExperimentGroup>();
    for (const group of fetched.values()) if (group) merged.set(group.group_id, group);
    for (const group of results?.groups ?? []) merged.set(group.group_id, group);
    return [...merged.values()];
  }, [fetched, results]);

  const actions = useMemo(() => (run ? deriveAgentActions(run, groups) : []), [run, groups]);
  const nodes = useMemo(() => workerNodes(actions, run?.state?.experiments ?? []), [actions, run]);
  const baseline = run?.state?.experiments?.find((item) => item.step === "baseline_training");

  return (
    <section className="content decisions-page">
      <div className="page-heading">
        <div>
          <span className="eyebrow">AGENTIC MODE</span>
          <h1>Agent Decisions</h1>
          <p>
            Each action the agent selected, in order. Every card shows the configuration before the
            step and the one the step proposed, and whether its equal-budget measurement kept or
            reverted the change.
          </p>
        </div>
        {runs.length > 0 && (
          <label className="decisions-run-select">
            Supervisor run
            <select value={shownId ?? ""} onChange={(event) => setSelectedId(event.target.value || null)}>
              {runs.map((item) => (
                <option key={item.id} value={item.id}>
                  {runLabel(item)}
                </option>
              ))}
            </select>
          </label>
        )}
      </div>
      {error && <div className="assistant-error" role="alert">{error}</div>}
      {loading && runs.length === 0 && <p role="status">Loading agent decisions…</p>}
      {!loading && runs.length === 0 && !error && <p className="decisions-empty">No supervisor run yet. Start one on the Agent Supervisor page.</p>}

      {run && (
        <>
          <RunStrip run={run} actions={actions} baselineAccuracy={baseline?.validation_accuracy ?? null} />
          {actions.length === 0 ? (
            <p className="decisions-empty">The agent has not selected an improvement step in this run yet.</p>
          ) : (
            <ol className="decisions-list">
              {actions.map((action) => (
                <li key={`${action.index}-${action.experiment.group_id}`} id={`decision-${action.index}`}>
                  <ActionCard action={action} nodes={nodes} />
                </li>
              ))}
            </ol>
          )}
        </>
      )}
    </section>
  );
}

function runLabel(run: SupervisorRun): string {
  const dataset = DATASET_NAMES[run.goal.dataset] ?? run.goal.dataset;
  const split = run.goal.dataset_split.replace("dirichlet_a0", "α 0.");
  const date = new Date(run.created_at).toLocaleString(undefined, { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" });
  return `${dataset} · ${split} · seed ${run.goal.seed ?? "–"} · ${date} · ${run.status}`;
}

function RunStrip({ run, actions, baselineAccuracy }: { run: SupervisorRun; actions: AgentAction[]; baselineAccuracy: number | null }) {
  const best = run.state?.best?.validation_accuracy ?? null;
  const kept = actions.filter((action) => action.outcome === "kept").length;
  const reverted = actions.filter((action) => action.outcome === "reverted").length;
  return (
    <div className="decisions-strip">
      <div className="decisions-strip-metrics">
        <Metric label="Dataset" value={`${DATASET_NAMES[run.goal.dataset] ?? run.goal.dataset} · ${run.goal.dataset_split}`} />
        <Metric label="Mode" value={run.goal.measure_each_step ? "Fair (each step measured)" : "Limited budget"} />
        <Metric label="Actions selected" value={`${actions.length} (${kept} kept, ${reverted} reverted)`} />
        <Metric label="Baseline → best" value={`${percent(baselineAccuracy)} → ${percent(best)}`} />
      </div>
      <nav className="decisions-flow" aria-label="Action sequence">
        {actions.map((action) => (
          <a key={action.index} href={`#decision-${action.index}`} className={`decisions-chip ${action.outcome}`}>
            <span>{action.index}</span>
            {STEP_TITLES[action.step] ?? action.step}
          </a>
        ))}
      </nav>
    </div>
  );
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div className="decisions-metric">
      <small>{label}</small>
      <strong>{value}</strong>
    </div>
  );
}

function ActionCard({ action, nodes }: { action: AgentAction; nodes: string[] }) {
  return (
    <article className={`decision-card ${action.outcome}`} aria-label={`Action ${action.index}: ${STEP_TITLES[action.step] ?? action.step}`}>
      <header className="decision-card-header">
        <div>
          <span className="decision-index">Action {action.index}</span>
          <h2>{STEP_TITLES[action.step] ?? action.step}</h2>
        </div>
        <div className="decision-badges">
          {action.source && <span className="decision-source">Selected by {SOURCE_NAMES[action.source] ?? action.source}</span>}
          <span className={`decision-outcome ${action.outcome}`}>{OUTCOME_NAMES[action.outcome]}</span>
        </div>
      </header>
      {action.rationale && <blockquote className="decision-rationale">“{action.rationale}”</blockquote>}
      <div className="decision-compare">
        <section className="decision-side before" aria-label="Before">
          <h3>Before</h3>
          <ModuleView action={action} side="before" nodes={nodes} />
        </section>
        <div className="decision-arrow" aria-hidden="true">→</div>
        <section className="decision-side after" aria-label="After">
          <h3>{action.outcome === "reverted" ? "Proposed (reverted)" : "After"}</h3>
          <ModuleView action={action} side="after" nodes={nodes} />
        </section>
      </div>
      {unchangedNote(action) && <p className="decision-unchanged">{unchangedNote(action)}</p>}
      {(action.module === "topology" || action.module === "learned_topology") && <TopologyLegend />}
      <AccuracyCompare action={action} />
      <Candidates group={action.group} step={action.step} />
    </article>
  );
}

function unchangedNote(action: AgentAction): string | null {
  const key = { topology: "topology", aggregation: "aggregation", sync: "sync_mode" }[action.module as string] as keyof Configuration | undefined;
  if (!key || action.outcome === "failed" || action.before[key] !== action.after[key]) return null;
  return "No change proposed: the sweep's best setting is the one already in use, so the measurement only re-trains the same configuration.";
}

function TopologyLegend() {
  return (
    <div className="decision-legend" aria-label="Link legend">
      <span><i className="same" />link in both graphs</span>
      <span><i className="added" />link added by this step</span>
      <span><i className="removed" />link removed by this step</span>
    </div>
  );
}

function ModuleView({ action, side, nodes }: { action: AgentAction; side: "before" | "after"; nodes: string[] }) {
  const config = action[side];
  const other = side === "before" ? action.after : action.before;
  switch (action.module) {
    case "topology":
    case "learned_topology":
      return (
        <>
          <TopologyGraph config={config} other={other} side={side} nodes={nodes} />
          <p className="decision-caption">{TOPOLOGY_NAMES[config.topology] ?? config.topology} · {topologyEdges(config.topology, nodes, config.senders).length} links</p>
        </>
      );
    case "aggregation":
      return <AggregationDiagram rule={config.aggregation} changed={config.aggregation !== other.aggregation} />;
    case "sync":
      return <SyncDiagram mode={config.sync_mode} changed={config.sync_mode !== other.sync_mode} />;
    case "architecture":
      return <ArchitectureView config={config} other={other} />;
    case "hyperparameters":
      return <HyperparameterView config={config} other={other} />;
    case "pruning":
      return <ArchitectureView config={config} other={other} reference={action.before.parameter_count} />;
    default:
      return <TrainingView config={config} />;
  }
}

/* ------------------------------------------------------------------ topology */

function TopologyGraph({ config, other, side, nodes }: { config: Configuration; other: Configuration; side: "before" | "after"; nodes: string[] }) {
  const size = 220;
  const center = size / 2;
  const radius = 82;
  const position = new Map(
    nodes.map((name, index) => {
      const angle = (2 * Math.PI * index) / nodes.length - Math.PI / 2;
      return [name, { x: center + radius * Math.cos(angle), y: center + radius * Math.sin(angle) }];
    }),
  );
  const edges = topologyEdges(config.topology, nodes, config.senders);
  const otherPairs = edgePairs(topologyEdges(other.topology, nodes, other.senders));
  const sameGraph = config.topology === other.topology && config.topology !== "learned";
  const classOf = (edge: Edge) => {
    if (sameGraph || edgeIn(edge, otherPairs)) return "same";
    return side === "after" ? "added" : "removed";
  };
  const markerId = `arrow-${side}-${config.topology}`;
  return (
    <svg className="decision-topology" viewBox={`0 0 ${size} ${size}`} role="img" aria-label={`${TOPOLOGY_NAMES[config.topology] ?? config.topology} topology`}>
      <defs>
        {["same", "added", "removed"].map((kind) => (
          <marker key={kind} id={`${markerId}-${kind}`} viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
            <path d="M0 0 10 5 0 10z" className={`edge-head ${kind}`} />
          </marker>
        ))}
      </defs>
      {edges.map((edge) => {
        const from = position.get(edge.from);
        const to = position.get(edge.to);
        if (!from || !to) return null;
        // Stop short of the node circle so the arrowhead stays visible.
        const dx = to.x - from.x;
        const dy = to.y - from.y;
        const length = Math.hypot(dx, dy) || 1;
        const trim = 13;
        const kind = classOf(edge);
        return (
          <line
            key={`${edge.from}-${edge.to}`}
            className={`edge ${kind}`}
            x1={from.x + (dx / length) * trim}
            y1={from.y + (dy / length) * trim}
            x2={to.x - (dx / length) * trim}
            y2={to.y - (dy / length) * trim}
            markerEnd={edge.directed ? `url(#${markerId}-${kind})` : undefined}
          />
        );
      })}
      {nodes.map((name, index) => {
        const point = position.get(name)!;
        return (
          <g key={name} className="node">
            <circle cx={point.x} cy={point.y} r={11} />
            <text x={point.x} y={point.y + 3.5} textAnchor="middle">{nodeLabel(name, index)}</text>
            <title>{name}</title>
          </g>
        );
      })}
    </svg>
  );
}

/* ------------------------------------------------------------------ aggregation */

const AGGREGATION_WEIGHTS: Record<string, { weights: number[]; trimmed?: number[]; note: string }> = {
  fedavg: { weights: [0.38, 0.27, 0.2, 0.15], note: "Each peer model is weighted by its number of training samples." },
  uniform: { weights: [0.25, 0.25, 0.25, 0.25], note: "Every received model gets the same weight." },
  metropolis: { weights: [0.31, 0.26, 0.23, 0.2], note: "Weights from node degrees (1 / (1 + max degree)); stable on sparse graphs." },
  trimmed_mean: { weights: [0, 0.5, 0.5, 0], trimmed: [0, 3], note: "Per parameter, the highest and lowest values are dropped before averaging." },
};

function AggregationDiagram({ rule, changed }: { rule: string; changed: boolean }) {
  const spec = AGGREGATION_WEIGHTS[rule] ?? { weights: [0.25, 0.25, 0.25, 0.25], note: "" };
  return (
    <div className={`decision-module ${changed ? "changed" : ""}`}>
      <strong className="decision-module-name">{AGGREGATION_NAMES[rule] ?? rule}</strong>
      <svg className="decision-diagram" viewBox="0 0 220 120" role="img" aria-label={`${AGGREGATION_NAMES[rule] ?? rule} weights of four peer models`}>
        {spec.weights.map((weight, index) => {
          const trimmed = spec.trimmed?.includes(index);
          const height = trimmed ? 60 : Math.max(6, weight * 230);
          const x = 22 + index * 48;
          return (
            <g key={index} className={trimmed ? "trimmed" : ""}>
              <rect x={x} y={100 - (trimmed ? 60 : height)} width="30" height={trimmed ? 60 : height} rx="4" className="agg-bar" />
              {trimmed && <path d={`M${x - 2} ${38} L${x + 32} ${102} M${x + 32} ${38} L${x - 2} ${102}`} className="agg-cross" />}
              <text x={x + 15} y={114} textAnchor="middle">peer {index + 1}</text>
              <text x={x + 15} y={100 - (trimmed ? 60 : height) - 4} textAnchor="middle" className="agg-value">
                {trimmed ? "dropped" : weight.toFixed(2)}
              </text>
            </g>
          );
        })}
      </svg>
      <p className="decision-caption">{spec.note} Weights are illustrative.</p>
    </div>
  );
}

/* ------------------------------------------------------------------ synchronization */

const SYNC_LANES = [
  { train: 70, label: "Worker A" },
  { train: 105, label: "Worker B" },
  { train: 150, label: "Worker C (slow)" },
];

function SyncDiagram({ mode, changed }: { mode: string; changed: boolean }) {
  const deadline = 120;
  const slowest = Math.max(...SYNC_LANES.map((lane) => lane.train));
  const notes: Record<string, string> = {
    sync: "Every worker waits for the slowest peer before exchanging models.",
    semi_sync: "Workers wait until a deadline; a peer that is later is skipped for that round.",
    async: "Each worker exchanges as soon as it finishes, with whatever models have arrived.",
  };
  return (
    <div className={`decision-module ${changed ? "changed" : ""}`}>
      <strong className="decision-module-name">{SYNC_NAMES[mode] ?? mode}</strong>
      <svg className="decision-diagram" viewBox="0 0 220 120" role="img" aria-label={`${SYNC_NAMES[mode] ?? mode} round timeline of three workers`}>
        {SYNC_LANES.map((lane, index) => {
          const y = 14 + index * 32;
          const late = mode === "semi_sync" && lane.train > deadline;
          const exchange = mode === "sync" ? slowest : mode === "semi_sync" ? deadline : lane.train;
          const waits = late ? 0 : exchange - lane.train;
          return (
            <g key={lane.label}>
              <text x="2" y={y + 13} className="sync-label">{lane.label.split(" ")[1]}</text>
              <rect x="20" y={y} width={lane.train} height="16" rx="3" className={`sync-train ${late ? "late" : ""}`} />
              {waits > 0 && <rect x={20 + lane.train} y={y} width={waits} height="16" rx="3" className="sync-wait" />}
              {!late && <circle cx={20 + exchange} cy={y + 8} r="4" className="sync-exchange" />}
              {late && <text x={24 + lane.train} y={y + 12} className="sync-label">skipped</text>}
            </g>
          );
        })}
        {mode === "sync" && <line x1={20 + slowest} x2={20 + slowest} y1="6" y2="106" className="sync-barrier" />}
        {mode === "semi_sync" && <line x1={20 + deadline} x2={20 + deadline} y1="6" y2="106" className="sync-barrier" />}
        <rect x="20" y="110" width="10" height="8" rx="2" className="sync-train" />
        <text x="33" y="117" className="sync-label">train</text>
        <rect x="62" y="110" width="10" height="8" rx="2" className="sync-wait" />
        <text x="75" y="117" className="sync-label">wait</text>
        <circle cx="108" cy="114" r="4" className="sync-exchange" />
        <text x="115" y="117" className="sync-label">exchange</text>
        {mode !== "async" && <text x="165" y="117" className="sync-label">| {mode === "sync" ? "barrier" : "deadline"}</text>}
      </svg>
      <p className="decision-caption">{notes[mode] ?? ""} Timeline is illustrative.</p>
    </div>
  );
}

/* ------------------------------------------------------------------ architecture & HPO */

function layerLabel(layer: Record<string, unknown>): string {
  const type = String(layer.type ?? "layer").replace(/_/g, " ").replace("2d", "2D").replace("1d", "1D");
  const detail = layer.filters ?? layer.units ?? layer.embed_dim ?? layer.rate ?? layer.pool_size;
  return detail !== undefined ? `${type} · ${detail}` : type;
}

function layersOf(architecture: NASArchitecture | null): Array<Record<string, unknown>> {
  if (architecture && "layers" in architecture && Array.isArray(architecture.layers)) {
    return architecture.layers as unknown as Array<Record<string, unknown>>;
  }
  return [];
}

function ArchitectureView({ config, other, reference }: { config: Configuration; other: Configuration; reference?: number | null }) {
  const layers = layersOf(config.architecture_detail);
  const otherLabels = new Set(layersOf(other.architecture_detail).map(layerLabel));
  const legacy = config.architecture_detail as { blocks?: number; channels?: number; kernel_size?: number } | null;
  const ratio = reference && config.parameter_count ? reference / config.parameter_count : null;
  return (
    <div className={`decision-module ${config.architecture !== other.architecture ? "changed" : ""}`}>
      <strong className="decision-module-name">{config.architecture}</strong>
      <span className="decision-module-sub">
        {formatCount(config.parameter_count)} parameters{ratio && ratio > 1.001 ? ` · ${ratio.toFixed(2)}× smaller` : ""}
      </span>
      {layers.length > 0 ? (
        <ol className="decision-layers">
          {layers.map((layer, index) => {
            const label = layerLabel(layer);
            return <li key={index} className={otherLabels.size > 0 && !otherLabels.has(label) ? "new" : ""}>{label}</li>;
          })}
        </ol>
      ) : legacy?.blocks ? (
        <p className="decision-caption">{legacy.blocks} conv blocks · {legacy.channels} channels · kernel {legacy.kernel_size}</p>
      ) : (
        <p className="decision-caption">Layer list not reported by the master for this model.</p>
      )}
      <ParameterBar value={config.parameter_count} reference={Math.max(config.parameter_count ?? 0, other.parameter_count ?? 0)} />
    </div>
  );
}

function HyperparameterView({ config, other }: { config: Configuration; other: Configuration }) {
  const rows: Array<[string, keyof NonNullable<Configuration["hyperparameters"]>]> = [
    ["Optimizer", "optimizer"],
    ["Learning rate", "learning_rate"],
    ["Batch size", "batch_size"],
  ];
  return (
    <div className="decision-module">
      <strong className="decision-module-name">{config.architecture}</strong>
      <table className="decision-table">
        <tbody>
          {rows.map(([label, key]) => {
            const value = config.hyperparameters?.[key];
            const changed = value !== undefined && other.hyperparameters?.[key] !== undefined && value !== other.hyperparameters?.[key];
            return (
              <tr key={key} className={changed ? "changed" : ""}>
                <th scope="row">{label}</th>
                <td>{value === undefined ? "baseline default" : String(value)}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

function TrainingView({ config }: { config: Configuration }) {
  return (
    <dl className="decision-config">
      <dt>Topology</dt><dd>{TOPOLOGY_NAMES[config.topology] ?? config.topology}</dd>
      <dt>Aggregation</dt><dd>{AGGREGATION_NAMES[config.aggregation] ?? config.aggregation}</dd>
      <dt>Sync</dt><dd>{SYNC_NAMES[config.sync_mode] ?? config.sync_mode}</dd>
      <dt>Model</dt><dd>{config.architecture} · {formatCount(config.parameter_count)}</dd>
    </dl>
  );
}

function ParameterBar({ value, reference }: { value: number | null; reference: number | null }) {
  if (!value || !reference) return null;
  return (
    <div className="decision-param-bar" title={`${formatCount(value)} of ${formatCount(reference)} parameters`}>
      <span style={{ width: `${Math.min(100, (100 * value) / reference)}%` }} />
    </div>
  );
}

/* ------------------------------------------------------------------ measurement */

function AccuracyCompare({ action }: { action: AgentAction }) {
  const before = action.beforeAccuracy;
  const after = action.afterAccuracy;
  if (before === null && after === null) return null;
  const delta = before !== null && after !== null ? (after - before) * 100 : null;
  const label = action.measurement ? "Equal-budget measurement" : "Validation accuracy of this training";
  return (
    <div className="decision-accuracy">
      <span className="decision-accuracy-title">{label}</span>
      <AccuracyRow name="Best before" value={before} kind="before" />
      <AccuracyRow name={action.measurement ? "With the change" : "This step"} value={after} kind={action.outcome === "reverted" || action.outcome === "not_kept" ? "worse" : "after"} />
      {delta !== null && (
        <span className={`decision-delta ${delta >= 0 ? "up" : "down"}`}>
          {delta >= 0 ? "+" : "−"}{Math.abs(delta).toFixed(1)} points
        </span>
      )}
    </div>
  );
}

function AccuracyRow({ name, value, kind }: { name: string; value: number | null; kind: string }) {
  return (
    <div className="decision-accuracy-row">
      <span>{name}</span>
      <div className="decision-accuracy-track">
        {value !== null && <span className={kind} style={{ width: `${Math.max(0, Math.min(100, value * 100))}%` }} />}
      </div>
      <strong>{percent(value)}</strong>
    </div>
  );
}

function Candidates({ group, step }: { group: ExperimentGroup | null; step: string }) {
  if (!group) return null;
  const items: Array<{ label: string; value: number | null; selected: boolean }> = [];
  if (group.kind === "topology_optimization" && group.candidates?.length) {
    const dimension = (group as ExperimentGroup & { dimension?: string }).dimension ?? "topology";
    const chosen = group.selected_configuration as (Record<string, string> | undefined);
    for (const candidate of group.candidates) {
      const raw = candidate as unknown as Record<string, string>;
      const key = dimension === "aggregation" ? raw.aggregation : dimension === "sync" ? raw.sync_mode : candidate.topology;
      const names = dimension === "aggregation" ? AGGREGATION_NAMES : dimension === "sync" ? SYNC_NAMES : TOPOLOGY_NAMES;
      const chosenKey = dimension === "aggregation" ? chosen?.aggregation : dimension === "sync" ? chosen?.sync_mode : chosen?.topology;
      items.push({ label: names[key] ?? key, value: candidate.mean_final_accuracy ?? null, selected: key === chosenKey });
    }
  } else if (group.kind === "hyperparameter_search" && group.hpo_results?.length) {
    const chosen = group.selected_hyperparameters;
    for (const row of group.hpo_results) {
      const training = row.architecture.training;
      items.push({
        label: training ? `${training.optimizer} · lr ${training.learning_rate} · bs ${training.batch_size}` : row.architecture.id,
        value: row.validation_accuracy,
        selected: Boolean(training && chosen && training.optimizer === chosen.optimizer && training.learning_rate === chosen.learning_rate && training.batch_size === chosen.batch_size),
      });
    }
  } else if (group.kind === "dataset_architecture_search" && group.nas_candidates?.length) {
    const chosen = group.selected_architecture?.id;
    [...group.nas_candidates]
      .sort((left, right) => right.mean_validation_accuracy - left.mean_validation_accuracy)
      .slice(0, 6)
      .forEach((candidate) => items.push({ label: `${candidate.architecture.id} · ${formatCount(candidate.parameter_count)}`, value: candidate.mean_validation_accuracy, selected: candidate.architecture.id === chosen }));
  }
  if (items.length === 0) return null;
  const top = Math.max(...items.map((item) => item.value ?? 0)) || 1;
  return (
    <details className="decision-candidates">
      <summary>What the {step === "run_nas" ? "search" : "sweep"} compared ({items.length} candidates)</summary>
      <ul>
        {items.map((item, index) => (
          <li key={`${item.label}-${index}`} className={item.selected ? "selected" : ""}>
            <span>{item.label}{item.selected ? " ✓ selected" : ""}</span>
            <div className="decision-accuracy-track"><span className={item.selected ? "after" : "before"} style={{ width: `${(100 * (item.value ?? 0)) / top}%` }} /></div>
            <strong>{percent(item.value)}</strong>
          </li>
        ))}
      </ul>
    </details>
  );
}

function percent(value: number | null | undefined): string {
  return typeof value === "number" ? `${(value * 100).toFixed(1)}%` : "–";
}

function formatCount(value: number | null | undefined): string {
  if (typeof value !== "number") return "–";
  if (value >= 1_000_000) return `${(value / 1_000_000).toFixed(2)}M`;
  if (value >= 1_000) return `${(value / 1_000).toFixed(1)}k`;
  return String(value);
}
