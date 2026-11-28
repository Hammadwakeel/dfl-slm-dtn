import type {
  ExperimentGroup,
  NASArchitecture,
  SupervisorEvent,
  SupervisorExperiment,
  SupervisorRun,
} from "../api/client";

/** What part of the DFL system an agent step changes. */
export type ModuleKind =
  | "topology"
  | "learned_topology"
  | "architecture"
  | "hyperparameters"
  | "aggregation"
  | "sync"
  | "pruning"
  | "training";

export interface Hyperparameters {
  optimizer: string;
  learning_rate: number;
  batch_size: number;
}

/** The DFL configuration in force before or after one agent step. */
export interface Configuration {
  topology: string;
  aggregation: string;
  sync_mode: string;
  architecture: string;
  parameter_count: number | null;
  architecture_detail: NASArchitecture | null;
  hyperparameters: Hyperparameters | null;
  senders: Record<string, string[]> | null;
}

export type ActionOutcome = "kept" | "reverted" | "not_kept" | "applied" | "failed" | "pending";

export interface AgentAction {
  index: number;
  step: string;
  module: ModuleKind;
  source: string | null;
  rationale: string | null;
  decidedAt: string | null;
  before: Configuration;
  after: Configuration;
  /** Best validation accuracy before the step (what the measurement must beat). */
  beforeAccuracy: number | null;
  /** Validation accuracy of the step's own equal-budget training. */
  afterAccuracy: number | null;
  outcome: ActionOutcome;
  experiment: SupervisorExperiment;
  measurement: SupervisorExperiment | null;
  group: ExperimentGroup | null;
}

export const STEP_MODULES: Record<string, ModuleKind> = {
  optimize_topology: "topology",
  learn_topology: "learned_topology",
  run_nas: "architecture",
  run_hpo: "hyperparameters",
  optimize_aggregation: "aggregation",
  optimize_sync: "sync",
  prune_model: "pruning",
  train_final: "training",
  continue_training: "training",
};

/** Steps whose change is measured afterwards by a separate equal-budget training. */
const MEASURED = new Set(["run_nas", "run_hpo", "optimize_topology", "optimize_aggregation", "optimize_sync"]);
const SKIPPED = new Set(["prepare_data", "initial_model", "measure_step", "ablation_training"]);

function accuracyOf(experiment: SupervisorExperiment | null | undefined): number | null {
  const value = experiment?.validation_accuracy;
  return typeof value === "number" ? value : null;
}

function architectureOf(group: ExperimentGroup | undefined, key: "architecture" | "selected_architecture"): NASArchitecture | null {
  const value = group?.[key];
  return value && typeof value === "object" && "id" in value ? (value as NASArchitecture) : null;
}

function trainingOf(architecture: NASArchitecture | null): Hyperparameters | null {
  if (architecture && "training" in architecture && architecture.training) {
    const { optimizer, learning_rate, batch_size } = architecture.training;
    return { optimizer, learning_rate, batch_size };
  }
  return null;
}

/**
 * Replay a supervisor run into one card per agent action: the configuration before
 * the step, the configuration it proposed, and whether its measurement kept it.
 */
export function deriveAgentActions(run: SupervisorRun, groups: ExperimentGroup[] = []): AgentAction[] {
  const byGroup = new Map(groups.map((group) => [group.group_id, group]));
  const experiments = run.state?.experiments ?? [];
  const decisions = [...(run.events ?? [])]
    .filter((event) => event.event_type === "decision")
    .sort((left, right) => left.id - right.id);
  const used = new Set<number>();
  const takeDecision = (step: string): SupervisorEvent | null => {
    const found = decisions.find((event) => !used.has(event.id) && event.details?.step === step) ?? null;
    if (found) used.add(found.id);
    return found;
  };

  let current: Configuration = {
    topology: run.goal.topology,
    aggregation: run.goal.aggregation,
    sync_mode: run.goal.sync_mode,
    architecture: "baseline",
    parameter_count: null,
    architecture_detail: null,
    hyperparameters: null,
    senders: null,
  };
  let best: number | null = null;
  const actions: AgentAction[] = [];

  experiments.forEach((experiment, index) => {
    const group = byGroup.get(experiment.group_id) ?? null;
    if (experiment.step === "baseline_training") {
      if (experiment.status === "completed") {
        const detail = architectureOf(group ?? undefined, "architecture");
        current = {
          ...current,
          topology: experiment.topology ?? current.topology,
          architecture: experiment.architecture ?? current.architecture,
          parameter_count: experiment.parameter_count ?? current.parameter_count,
          architecture_detail: detail ?? current.architecture_detail,
          hyperparameters: trainingOf(detail) ?? current.hyperparameters,
        };
        best = accuracyOf(experiment);
      }
      return;
    }
    if (SKIPPED.has(experiment.step)) return;
    const module = STEP_MODULES[experiment.step];
    if (!module) return;

    const decision = takeDecision(experiment.step);
    const before = { ...current };
    const after = { ...current };
    if (module === "topology") after.topology = experiment.selected_topology ?? experiment.topology ?? before.topology;
    if (module === "aggregation") after.aggregation = experiment.selected_aggregation ?? before.aggregation;
    if (module === "sync") after.sync_mode = experiment.selected_sync_mode ?? before.sync_mode;
    if (module === "learned_topology") {
      after.topology = "learned";
      after.senders = experiment.topology_learning?.senders ?? null;
    }
    if (module === "architecture" || module === "hyperparameters" || module === "pruning" || module === "training") {
      after.architecture = experiment.architecture ?? before.architecture;
      after.parameter_count = experiment.parameter_count ?? before.parameter_count;
    }
    if (module === "architecture" || module === "hyperparameters") {
      const detail = architectureOf(group ?? undefined, "selected_architecture");
      after.architecture_detail = detail ?? before.architecture_detail;
      after.hyperparameters = experiment.hyperparameters ?? trainingOf(detail) ?? before.hyperparameters;
    }
    if (module === "training") {
      // The master records the requested model; trust it only when it names the trained one.
      const recorded = architectureOf(group ?? undefined, "architecture");
      after.architecture_detail = recorded && recorded.id === experiment.architecture ? recorded : before.architecture_detail;
    }
    if (module === "pruning") {
      const ratio = experiment.pruning?.ratio;
      after.architecture_detail = ratio ? prunedArchitecture(before.architecture_detail, ratio, after.architecture) : null;
    }

    let outcome: ActionOutcome;
    let measurement: SupervisorExperiment | null = null;
    let afterAccuracy: number | null = null;
    const beforeAccuracy = best;
    if (experiment.status !== "completed") {
      outcome = "failed";
    } else if (MEASURED.has(experiment.step)) {
      for (const later of experiments.slice(index + 1)) {
        if (later.step === "measure_step" && later.measures === experiment.step) {
          measurement = later;
          break;
        }
        if (!SKIPPED.has(later.step)) break;
      }
      if (measurement) {
        afterAccuracy = accuracyOf(measurement);
        // The measurement trained the model the workers actually had installed.
        const trained = architectureOf(byGroup.get(measurement.group_id), "architecture");
        if (trained && trained.id === after.architecture && (module === "architecture" || module === "hyperparameters")) {
          after.architecture_detail = trained;
          after.hyperparameters = experiment.hyperparameters ?? trainingOf(trained) ?? after.hyperparameters;
        }
        outcome = measurement.accepted === false ? "reverted" : "kept";
      } else {
        outcome = run.goal.measure_each_step && run.status === "running" ? "pending" : "applied";
      }
    } else {
      afterAccuracy = accuracyOf(experiment);
      if (module === "pruning") outcome = experiment.pruning?.accepted === false ? "reverted" : "kept";
      else if (module === "learned_topology") outcome = afterAccuracy !== null && (best === null || afterAccuracy > best) ? "kept" : "not_kept";
      else outcome = "applied";
    }

    // The learned graph is a probe: the selected topology stays unless the backend adopts it.
    if (outcome === "kept" || outcome === "applied") {
      if (module !== "learned_topology") current = after;
    }
    if (afterAccuracy !== null && outcome !== "reverted" && (best === null || afterAccuracy > best)) best = afterAccuracy;

    actions.push({
      index: actions.length + 1,
      step: experiment.step,
      module,
      source: typeof decision?.details?.source === "string" ? decision.details.source : null,
      rationale: decision ? rationaleOf(decision.message) : null,
      decidedAt: decision?.created_at ?? null,
      before,
      after,
      beforeAccuracy,
      afterAccuracy,
      outcome,
      experiment,
      measurement,
      group,
    });
  });
  return actions;
}

/**
 * The layer list after structured pruning, as common/pruning.py computes it: every
 * conv / dense layer except the classifier keeps round(units * (1 - ratio)), at least 4.
 */
export function prunedArchitecture(architecture: NASArchitecture | null, ratio: number, id: string): NASArchitecture | null {
  if (!architecture || !("layers" in architecture) || !Array.isArray(architecture.layers)) return null;
  const layers = architecture.layers;
  const classifier = layers.map((layer) => layer.type).lastIndexOf("dense");
  return {
    ...architecture,
    id,
    layers: layers.map((layer, index) => {
      const key = layer.type === "dense" ? "units" : layer.type === "conv2d" || layer.type === "separable_conv2d" ? "filters" : null;
      const units = key ? layer[key] : undefined;
      if (!key || index === classifier || typeof units !== "number") return layer;
      return { ...layer, [key]: Math.max(Math.min(4, units), Math.round(units * (1 - ratio))) };
    }),
  };
}

/** "optimize_topology chosen by slm: <reason>" → "<reason>". */
export function rationaleOf(message: string): string {
  const match = /^[a-z_]+ chosen by [a-z_]+:\s*/i.exec(message);
  return match ? message.slice(match[0].length) : message;
}

export interface Edge {
  from: string;
  to: string;
  directed: boolean;
}

/** Peer links of a topology over the given (sorted) workers. */
export function topologyEdges(topology: string, nodes: string[], senders?: Record<string, string[]> | null): Edge[] {
  const n = nodes.length;
  if (topology === "learned") {
    return Object.entries(senders ?? {}).flatMap(([receiver, from]) =>
      from.map((sender) => ({ from: sender, to: receiver, directed: true })),
    );
  }
  const edges: Edge[] = [];
  nodes.forEach((node, index) => {
    const next = nodes[(index + 1) % n];
    if (topology === "mesh") nodes.slice(index + 1).forEach((peer) => edges.push({ from: node, to: peer, directed: false }));
    else if (topology === "bidirectional_ring" && n > 1) edges.push({ from: node, to: next, directed: false });
    else if (topology === "chain" && index + 1 < n) edges.push({ from: node, to: next, directed: false });
    else if (topology === "star" && index > 0) edges.push({ from: nodes[0], to: node, directed: false });
    else if ((topology === "directed_ring" || topology === "ring") && n > 1) edges.push({ from: node, to: next, directed: true });
  });
  return edges;
}

/** Directed message pairs ("a>b") an edge list carries; an undirected link carries both. */
export function edgePairs(edges: Edge[]): Set<string> {
  const pairs = new Set<string>();
  for (const edge of edges) {
    pairs.add(`${edge.from}>${edge.to}`);
    if (!edge.directed) pairs.add(`${edge.to}>${edge.from}`);
  }
  return pairs;
}

/** True when the other graph links the same two workers (in either direction). */
export function edgeIn(edge: Edge, pairs: Set<string>): boolean {
  return pairs.has(`${edge.from}>${edge.to}`) || pairs.has(`${edge.to}>${edge.from}`);
}

/** Worker names to draw: the learned graph's real names when known, else W1…Wn. */
export function workerNodes(actions: AgentAction[], experiments: SupervisorExperiment[]): string[] {
  const names = new Set<string>();
  for (const action of actions) {
    for (const [receiver, senders] of Object.entries(action.after.senders ?? {})) {
      names.add(receiver);
      senders.forEach((sender) => names.add(sender));
    }
  }
  if (names.size > 0) return [...names].sort();
  const count = Math.max(0, ...experiments.map((experiment) => experiment.worker_count ?? 0)) || 7;
  return Array.from({ length: count }, (_, index) => `W${index + 1}`);
}

/** Short node label: trailing number of the name ("dfl-worker-03" → "3"). */
export function nodeLabel(name: string, index: number): string {
  const match = /(\d+)$/.exec(name);
  return match ? String(Number(match[1])) : String(index + 1);
}
