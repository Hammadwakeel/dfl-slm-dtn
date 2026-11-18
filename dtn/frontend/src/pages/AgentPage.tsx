import { useCallback, useEffect, useMemo, useState } from "react";

import {
  askSupervisor,
  getSupervisorCluster,
  getSupervisorRun,
  listSupervisorRuns,
  startSupervisorRun,
  stopSupervisorRun,
  type SupervisorCluster,
  type SupervisorEvent,
  type SupervisorGoal,
  type SupervisorPruning,
  type SupervisorRun,
} from "../api/client";

interface AgentPageProps {
  token: string;
  active: boolean;
}

const ACTIVE_STATUSES = new Set(["queued", "running", "stopping"]);
const STAGES = [
  { id: "preflight", label: "Preflight" },
  { id: "prepare_data", label: "Data" },
  { id: "initial_model", label: "Initial model" },
  { id: "baseline_training", label: "Baseline DFL" },
  { id: "improve", label: "Improve" },
  { id: "finished", label: "Finished" },
] as const;
const STEP_LABELS: Record<string, string> = {
  prepare_data: "Prepare data",
  initial_model: "Initial model",
  baseline_training: "Baseline DFL",
  run_nas: "Architecture search",
  run_hpo: "Hyperparameter tuning",
  learn_topology: "Learned topology (APG)",
  optimize_topology: "Topology optimization",
  prune_model: "Pruning + fine-tune",
  train_final: "Final DFL",
  finish_workflow: "Finish",
};
const SOURCE_LABELS: Record<string, string> = {
  slm: "SLM",
  slm_text: "SLM (text)",
  slm_repaired: "SLM after repair",
  fallback: "Fallback policy",
  rule: "Rule",
};
const DEFAULT_GOAL: SupervisorGoal = {
  dataset: "mnist",
  target_accuracy: 0.95,
  max_experiments: 6,
  time_budget_minutes: 480,
  dataset_split: "dirichlet_a03",
  rounds: 10,
  local_epochs: 5,
  topology: "directed_ring",
  nas_budget_seconds: 600,
  aggregation: "fedavg",
  sync_mode: "semi_sync",
  sync_deadline_seconds: 300,
};

/** Agentic mode: a server-side supervisor that monitors and controls the whole cluster. */
export default function AgentPage({ token, active }: AgentPageProps) {
  const [runs, setRuns] = useState<SupervisorRun[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [selected, setSelected] = useState<SupervisorRun | null>(null);
  const [cluster, setCluster] = useState<SupervisorCluster | null>(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(true);
  const activeRun = runs.find((run) => ACTIVE_STATUSES.has(run.status)) ?? null;
  const shownId = selectedId ?? activeRun?.id ?? null;

  const refresh = useCallback(async () => {
    try {
      const [nextRuns, nextCluster] = await Promise.all([
        listSupervisorRuns(token),
        getSupervisorCluster(token),
      ]);
      setRuns(nextRuns);
      setCluster(nextCluster);
      const runningId = nextRuns.find((run) => ACTIVE_STATUSES.has(run.status))?.id;
      const detailId = selectedId ?? runningId ?? null;
      setSelected(detailId ? await getSupervisorRun(token, detailId) : null);
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
    const timer = window.setInterval(() => void refresh(), activeRun ? 3000 : 15000);
    return () => window.clearInterval(timer);
  }, [active, refresh, activeRun]);

  return (
    <section className="content agent-page">
      <div className="page-heading">
        <div>
          <span className="eyebrow">AGENTIC MODE</span>
          <h1>Agent Supervisor</h1>
          <p>
            Give a dataset and a target. The on-device SLM chooses each improvement step; the
            supervisor monitors memory and health, retries transient failures, and records every
            decision. It keeps running if this page is closed.
          </p>
        </div>
        <ClusterPill cluster={cluster} />
      </div>
      {error && <div className="assistant-error" role="alert">{error}</div>}
      {loading && runs.length === 0 && <p role="status">Loading supervisor…</p>}

      {!loading && !activeRun && (
        <GoalForm
          token={token}
          cluster={cluster}
          onStarted={(run) => {
            setSelectedId(null);
            setSelected(run);
            void refresh();
          }}
        />
      )}

      {selected && shownId === selected.id && (
        <RunPanel token={token} run={selected} onChanged={() => void refresh()} />
      )}

      <RunHistory
        runs={runs}
        selectedId={shownId}
        onSelect={(id) => {
          setSelectedId(id === activeRun?.id ? null : id);
        }}
      />
    </section>
  );
}

function ClusterPill({ cluster }: { cluster: SupervisorCluster | null }) {
  if (!cluster) return null;
  const ready = cluster.workers > 0 && cluster.workers_ready === cluster.workers && !cluster.global_stop;
  const detail = cluster.global_stop
    ? "Global STOP active"
    : `${cluster.workers_ready}/${cluster.workers} workers ready · min free ${cluster.min_free_mb ?? "–"} MB`;
  return (
    <span className={`status-pill ${ready ? "online" : "offline"}`} title={cluster.not_ready.join(", ")}>
      <span className="status-dot" />
      {detail}
    </span>
  );
}

function GoalForm({
  token,
  cluster,
  onStarted,
}: {
  token: string;
  cluster: SupervisorCluster | null;
  onStarted: (run: SupervisorRun) => void;
}) {
  const [goal, setGoal] = useState<SupervisorGoal>(DEFAULT_GOAL);
  const [starting, setStarting] = useState(false);
  const [formError, setFormError] = useState("");

  function update<K extends keyof SupervisorGoal>(key: K, value: SupervisorGoal[K]) {
    setGoal((current) => ({ ...current, [key]: value }));
  }

  async function start() {
    setStarting(true);
    setFormError("");
    try {
      onStarted(await startSupervisorRun(token, goal));
    } catch (cause) {
      setFormError(cause instanceof Error ? cause.message : "Could not start the supervisor.");
    } finally {
      setStarting(false);
    }
  }

  return (
    <form
      className="agent-goal-form"
      aria-label="Supervisor goal"
      onSubmit={(event) => {
        event.preventDefault();
        void start();
      }}
    >
      <div className="agent-goal-grid">
        <label>
          Dataset
          <select value={goal.dataset} onChange={(event) => update("dataset", event.target.value as SupervisorGoal["dataset"])}>
            <option value="mnist">MNIST</option>
            <option value="cifar10">CIFAR-10</option>
          </select>
        </label>
        <label>
          Target validation accuracy (%)
          <input
            type="number"
            min={10}
            max={99.9}
            step={0.5}
            value={Math.round(goal.target_accuracy * 1000) / 10}
            onChange={(event) => update("target_accuracy", Number(event.target.value) / 100)}
          />
        </label>
        <label>
          Max experiments
          <input type="number" min={1} max={20} value={goal.max_experiments} onChange={(event) => update("max_experiments", Number(event.target.value))} />
        </label>
        <label>
          Time budget (min)
          <input type="number" min={10} max={2880} value={goal.time_budget_minutes} onChange={(event) => update("time_budget_minutes", Number(event.target.value))} />
        </label>
        <label>
          Data split
          <select value={goal.dataset_split} onChange={(event) => update("dataset_split", event.target.value as SupervisorGoal["dataset_split"])}>
            <option value="stratified">Stratified (IID)</option>
            <option value="dirichlet_a01">Dirichlet α=0.1</option>
            <option value="dirichlet_a03">Dirichlet α=0.3</option>
            <option value="dirichlet_a05">Dirichlet α=0.5</option>
            <option value="dirichlet_a07">Dirichlet α=0.7</option>
          </select>
        </label>
        <label>
          Initial topology
          <select value={goal.topology} onChange={(event) => update("topology", event.target.value as SupervisorGoal["topology"])}>
            <option value="directed_ring">Directed ring</option>
            <option value="bidirectional_ring">Bidirectional ring</option>
            <option value="mesh">Mesh</option>
            <option value="chain">Chain</option>
            <option value="star">Star</option>
          </select>
        </label>
        <label>
          Rounds
          <input type="number" min={1} max={20} value={goal.rounds} onChange={(event) => update("rounds", Number(event.target.value))} />
        </label>
        <label>
          Local epochs
          <input type="number" min={1} max={20} value={goal.local_epochs} onChange={(event) => update("local_epochs", Number(event.target.value))} />
        </label>
        <label>
          Aggregation
          <select value={goal.aggregation} onChange={(event) => update("aggregation", event.target.value as SupervisorGoal["aggregation"])}>
            <option value="fedavg">FedAvg (by sample count)</option>
            <option value="uniform">Uniform average</option>
            <option value="metropolis">Metropolis-Hastings</option>
            <option value="trimmed_mean">Trimmed mean (robust)</option>
          </select>
        </label>
        <label>
          Synchronization
          <select value={goal.sync_mode} onChange={(event) => update("sync_mode", event.target.value as SupervisorGoal["sync_mode"])}>
            <option value="semi_sync">Semi-synchronous (deadline)</option>
            <option value="sync">Synchronous (wait for all)</option>
          </select>
        </label>
        <label>
          Peer deadline (s)
          <input
            type="number"
            min={30}
            max={3600}
            step={30}
            disabled={goal.sync_mode !== "semi_sync"}
            value={goal.sync_deadline_seconds}
            onChange={(event) => update("sync_deadline_seconds", Number(event.target.value))}
          />
        </label>
        <label>
          NAS budget (s)
          <input type="number" min={90} max={600} step={30} value={goal.nas_budget_seconds} onChange={(event) => update("nas_budget_seconds", Number(event.target.value))} />
        </label>
      </div>
      {formError && <div className="assistant-error" role="alert">{formError}</div>}
      <div className="agent-goal-actions">
        <span>
          {cluster?.global_stop
            ? "Global STOP is active; the supervisor will wait until it is reset."
            : "Preflight checks worker readiness and free memory before every step."}
        </span>
        <button className="send-button" type="submit" disabled={starting}>
          {starting ? "Starting…" : "Start supervisor"}
        </button>
      </div>
    </form>
  );
}

function stageIndex(run: SupervisorRun): number {
  if (run.state.stage === "finished" || !ACTIVE_STATUSES.has(run.status)) return STAGES.length - 1;
  const completed = new Set(run.state.completed_steps ?? []);
  if (completed.has("baseline_training")) return 4;
  if (run.state.active?.step === "baseline_training" || run.state.pending_step?.step === "baseline_training") return 3;
  return Math.max(0, STAGES.findIndex((stage) => stage.id === run.state.stage));
}

function RunPanel({ token, run, onChanged }: { token: string; run: SupervisorRun; onChanged: () => void }) {
  const [stopping, setStopping] = useState(false);
  const [panelError, setPanelError] = useState("");
  const isActive = ACTIVE_STATUSES.has(run.status);
  const counters = run.state.counters ?? { slm_decisions: 0, slm_repairs: 0, fallback_decisions: 0, rule_decisions: 0 };
  const best = run.state.best;
  const experimentsUsed = run.state.experiments?.length ?? 0;
  const current = stageIndex(run);
  const lastDecision = useMemo(
    () => [...(run.events ?? [])].reverse().find((event) => event.event_type === "decision"),
    [run.events],
  );
  const minutesLeft = isActive ? run.minutes_left ?? null : null;

  async function stop() {
    if (!window.confirm("Stop the supervisor and its active experiment?")) return;
    setStopping(true);
    setPanelError("");
    try {
      await stopSupervisorRun(token, run.id, "Stopped by the operator from the Agent Supervisor page.");
      onChanged();
    } catch (cause) {
      setPanelError(cause instanceof Error ? cause.message : "Could not stop the run.");
    } finally {
      setStopping(false);
    }
  }

  return (
    <article className="agent-run-panel" aria-label="Supervisor run">
      <header className="agent-run-header">
        <div>
          <span className={`agent-run-status ${run.status}`}>{run.status}</span>
          <h2>{run.task}</h2>
          <small>
            Started {formatDate(run.created_at)} · {run.goal.dataset_split} · {run.goal.rounds} rounds × {run.goal.local_epochs} local epochs
            {run.goal.aggregation ? ` · ${run.goal.aggregation}` : ""}
            {run.goal.sync_mode === "semi_sync" ? ` · semi-sync ${run.goal.sync_deadline_seconds}s` : run.goal.sync_mode ? " · sync" : ""}
          </small>
        </div>
        {isActive && (
          <button className="danger-button" type="button" disabled={stopping || run.status === "stopping"} onClick={() => void stop()}>
            {stopping || run.status === "stopping" ? "Stopping…" : "Stop"}
          </button>
        )}
      </header>
      {panelError && <div className="assistant-error" role="alert">{panelError}</div>}

      <ol className="agent-stage-track" aria-label="Workflow stages">
        {STAGES.map((stage, index) => (
          <li key={stage.id} className={index < current ? "done" : index === current ? "current" : ""}>
            {stage.label}
          </li>
        ))}
      </ol>
      <div className="agent-progress" role="progressbar" aria-valuemin={0} aria-valuemax={100} aria-valuenow={run.progress}>
        <span style={{ width: `${run.progress}%` }} />
      </div>

      <div className="agent-summary-grid">
        <div className="metric">
          <span>Best validation accuracy</span>
          <strong>{best ? `${(best.validation_accuracy * 100).toFixed(2)}%` : "–"}</strong>
          <small>target {(run.goal.target_accuracy * 100).toFixed(1)}%{best?.architecture ? ` · ${best.architecture}` : ""}</small>
        </div>
        <div className="metric">
          <span>Experiments</span>
          <strong>{experimentsUsed} / {run.goal.max_experiments}</strong>
          <small>{minutesLeft !== null ? `${minutesLeft} min left` : run.completed_at ? `ended ${formatDate(run.completed_at)}` : ""}</small>
        </div>
        <div className="metric">
          <span>Decisions</span>
          <strong>{counters.slm_decisions + counters.slm_repairs} SLM</strong>
          <small>{counters.slm_repairs} repaired · {counters.fallback_decisions} fallback · {counters.rule_decisions} rule</small>
        </div>
        <div className="metric">
          <span>Now</span>
          <strong>{STEP_LABELS[run.current_step ?? ""] ?? (run.state.stage === "finished" ? "Finished" : run.stage)}</strong>
          <small>
            {run.state.active?.progress !== undefined
              ? `${Math.round(run.state.active.progress * 100)}% of rounds`
              : run.state.waiting_reason ?? ""}
          </small>
        </div>
      </div>

      {lastDecision && (
        <div className="agent-decision">
          <span className="eyebrow">LATEST DECISION · {SOURCE_LABELS[String(lastDecision.details?.source ?? "")] ?? "Supervisor"}</span>
          <p>{lastDecision.message}</p>
        </div>
      )}
      {run.result && (
        <div className={`agent-outcome ${run.result.outcome}`}>
          <strong>{run.result.outcome.replace(/_/g, " ")}</strong>
          <p>{run.result.summary}</p>
        </div>
      )}

      {run.state.pruning && <PruningPanel pruning={run.state.pruning} />}
      <ExperimentTable run={run} />
      <AskBox token={token} runId={run.id} />
      <EventTimeline events={run.events ?? []} />
    </article>
  );
}

function PruningPanel({ pruning }: { pruning: SupervisorPruning }) {
  return (
    <section className="agent-pruning" aria-label="Pruned model">
      <h3>Pruned model</h3>
      <div className="agent-summary-grid">
        <div className="metric">
          <span>Parameters</span>
          <strong>{pruning.parameters_after.toLocaleString()}</strong>
          <small>from {pruning.parameters_before.toLocaleString()} ({Math.round(pruning.ratio * 100)}% of units removed)</small>
        </div>
        <div className="metric">
          <span>Compression</span>
          <strong>{pruning.compression}×</strong>
          <small>{pruning.architecture_id ?? ""}</small>
        </div>
        <div className="metric">
          <span>Accuracy after fine-tune</span>
          <strong>{pruning.validation_accuracy != null ? formatPercent(pruning.validation_accuracy) : "fine-tuning…"}</strong>
          <small>
            {pruning.reference_accuracy != null
              ? `before pruning ${formatPercent(pruning.reference_accuracy)} (mean of workers)`
              : pruning.source_validation_accuracy != null
                ? `before pruning ${formatPercent(pruning.source_validation_accuracy)} on ${pruning.source_worker ?? "best worker"}`
                : ""}
          </small>
        </div>
        {pruning.accepted != null && (
          <div className="metric">
            <span>Decision</span>
            <strong>{pruning.accepted ? "Pruned model kept" : "Unpruned model kept"}</strong>
            <small>{pruning.accepted ? "accuracy within 2 points of the best" : "pruning cost more than 2 points of accuracy"}</small>
          </div>
        )}
      </div>
    </section>
  );
}

function ExperimentTable({ run }: { run: SupervisorRun }) {
  const experiments = run.state.experiments ?? [];
  if (experiments.length === 0) return null;
  return (
    <div className="agent-experiments">
      <h3>Experiments</h3>
      <div className="table-scroll">
        <table className="metrics-table">
          <thead>
            <tr>
              <th>Step</th>
              <th>Status</th>
              <th>Val. accuracy</th>
              <th>Macro-F1</th>
              <th>Architecture</th>
              <th>Topology</th>
              <th>Aggregation</th>
            </tr>
          </thead>
          <tbody>
            {experiments.map((experiment) => (
              <tr key={experiment.group_id ?? experiment.completed_at}>
                <td>{STEP_LABELS[experiment.step] ?? experiment.step}</td>
                <td title={experiment.error}>{experiment.status}</td>
                <td>
                  {experiment.validation_accuracy != null
                    ? formatPercent(experiment.validation_accuracy)
                    : experiment.search_validation_accuracy != null
                      ? `${formatPercent(experiment.search_validation_accuracy)} (search)`
                      : "–"}
                </td>
                <td>{experiment.macro_f1 != null ? experiment.macro_f1.toFixed(4) : "–"}</td>
                <td>
                  {experiment.architecture ?? "–"}
                  {experiment.parameter_count ? ` (${experiment.parameter_count.toLocaleString()} params)` : ""}
                  {experiment.hyperparameters
                    ? ` · lr ${experiment.hyperparameters.learning_rate} · batch ${experiment.hyperparameters.batch_size}`
                    : ""}
                  {experiment.pruning ? ` · ${experiment.pruning.compression}× smaller` : ""}
                </td>
                <td>{experiment.selected_topology ?? experiment.topology ?? "–"}</td>
                <td title={experiment.partial_rounds ? `${experiment.partial_rounds} round(s) aggregated without every peer` : undefined}>
                  {experiment.aggregation ?? "–"}
                  {experiment.sync_mode === "semi_sync" ? " · semi-sync" : ""}
                  {experiment.partial_rounds ? ` · ${experiment.partial_rounds} partial` : ""}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function AskBox({ token, runId }: { token: string; runId: string }) {
  const [question, setQuestion] = useState("");
  const [answer, setAnswer] = useState("");
  const [asking, setAsking] = useState(false);
  const [askError, setAskError] = useState("");

  async function ask() {
    const text = question.trim();
    if (!text || asking) return;
    setAsking(true);
    setAskError("");
    try {
      setAnswer((await askSupervisor(token, runId, text)).answer);
      setQuestion("");
    } catch (cause) {
      setAskError(cause instanceof Error ? cause.message : "The supervisor could not answer.");
    } finally {
      setAsking(false);
    }
  }

  return (
    <form
      className="agent-ask"
      onSubmit={(event) => {
        event.preventDefault();
        void ask();
      }}
    >
      <label htmlFor={`ask-${runId}`}>Ask the agent about this run</label>
      <div>
        <input
          id={`ask-${runId}`}
          maxLength={2000}
          placeholder="e.g. Why did you choose architecture search?"
          value={question}
          onChange={(event) => setQuestion(event.target.value)}
        />
        <button className="send-button" type="submit" disabled={asking || !question.trim()}>
          {asking ? "Asking…" : "Ask"}
        </button>
      </div>
      {askError && <div className="assistant-error" role="alert">{askError}</div>}
      {answer && <p className="agent-answer">{answer}</p>}
    </form>
  );
}

function EventTimeline({ events }: { events: SupervisorEvent[] }) {
  const recent = [...events].reverse().slice(0, 80);
  return (
    <section className="agent-run-events" aria-label="Supervisor timeline">
      <h3>Timeline</h3>
      <ol>
        {recent.map((event) => (
          <li key={event.id} className={`event-${event.event_type}`}>
            <time dateTime={event.created_at}>{formatTime(event.created_at)}</time>
            <div>
              <strong>{event.event_type.replace(/_/g, " ")}</strong>
              <p>{event.message}</p>
              {event.details && Object.keys(event.details).length > 0 && (
                <details>
                  <summary>Details</summary>
                  <pre>{JSON.stringify(event.details, null, 2)}</pre>
                </details>
              )}
            </div>
          </li>
        ))}
      </ol>
    </section>
  );
}

function RunHistory({
  runs,
  selectedId,
  onSelect,
}: {
  runs: SupervisorRun[];
  selectedId: string | null;
  onSelect: (id: string) => void;
}) {
  if (runs.length === 0) return null;
  return (
    <section className="agent-history" aria-label="Supervisor run history">
      <h3>Run history</h3>
      <ul>
        {runs.map((run) => (
          <li key={run.id}>
            <button
              type="button"
              className={`chat-history-item${run.id === selectedId ? " active" : ""}`}
              aria-current={run.id === selectedId ? "true" : undefined}
              onClick={() => onSelect(run.id)}
            >
              <strong>{run.task}</strong>
              <span>
                <span className={`agent-run-status ${run.status}`}>{run.status}</span>{" "}
                {formatDate(run.created_at)}
                {run.state.best ? ` · best ${(run.state.best.validation_accuracy * 100).toFixed(2)}%` : ""}
              </span>
            </button>
          </li>
        ))}
      </ul>
    </section>
  );
}

function formatPercent(value: number | null | undefined): string {
  return typeof value === "number" ? `${(value * 100).toFixed(2)}%` : "–";
}

function formatDate(value: string): string {
  const date = new Date(value);
  return Number.isNaN(date.getTime())
    ? value
    : date.toLocaleString([], { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" });
}

function formatTime(value: string): string {
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? value : date.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
}
