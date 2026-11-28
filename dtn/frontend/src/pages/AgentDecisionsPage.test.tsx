import { cleanup, render, screen, within } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import AgentDecisionsPage from "./AgentDecisionsPage";
import type { ExperimentGroup, SupervisorExperiment, SupervisorRun } from "../api/client";
import { deriveAgentActions, edgeIn, edgePairs, prunedArchitecture, topologyEdges } from "../lib/agentDecisions";

const apiUrl = "http://localhost:8000";

function experiment(step: string, extra: Partial<SupervisorExperiment> = {}): SupervisorExperiment {
  return { step, group_id: `${step}-${Math.random().toString(36).slice(2, 7)}`, kind: "decentralized", status: "completed", ...extra };
}

/** A fair CIFAR-10 run shaped like the recorded 2026-11-25 evidence. */
function runFixture(): SupervisorRun {
  const decision = (id: number, step: string, source = "slm") => ({
    id,
    event_type: "decision",
    message: `${step} chosen by ${source}: reason for ${step}`,
    details: { step, arguments: {}, source, ...(step === "optimize_topology" ? { model: "qwen3.5-dfl:4b" } : {}) },
    created_at: "2026-11-25T10:00:00+00:00",
  });
  return {
    id: "run-1",
    task: "cifar10",
    status: "completed",
    stage: "finished",
    progress: 100,
    current_step: null,
    rationale: null,
    goal: {
      dataset: "cifar10",
      target_accuracy: 0.99,
      max_experiments: 16,
      time_budget_minutes: 720,
      dataset_split: "dirichlet_a01",
      rounds: 10,
      local_epochs: 5,
      topology: "directed_ring",
      nas_budget_seconds: 600,
      aggregation: "fedavg",
      sync_mode: "sync",
      sync_deadline_seconds: 300,
      seed: 42,
      measure_each_step: true,
    },
    state: {
      completed_steps: [],
      experiments: [
        experiment("baseline_training", { group_id: "base", topology: "directed_ring", architecture: "baseline", parameter_count: 324394, validation_accuracy: 0.4246, worker_count: 7 }),
        experiment("optimize_topology", { group_id: "sweep", kind: "topology_optimization", selected_topology: "mesh", topology: "mesh" }),
        experiment("measure_step", { measures: "optimize_topology", topology: "mesh", validation_accuracy: 0.6096, accepted: true }),
        experiment("optimize_sync", { kind: "topology_optimization", selected_sync_mode: "async" }),
        experiment("measure_step", { measures: "optimize_sync", topology: "mesh", validation_accuracy: 0.5946, accepted: false }),
        experiment("prune_model", { architecture: "baseline-pruned30", parameter_count: 161094, validation_accuracy: 0.6578, pruning: { ratio: 0.3, parameters_before: 324394, parameters_after: 161094, compression: 2, accepted: true } }),
      ],
      best: { validation_accuracy: 0.6578, macro_f1: null, group_id: "x", step: "prune_model", architecture: null, topology: "mesh" },
      selected_topology: "mesh",
      nas_architecture: null,
      active: null,
      pending_step: null,
      counters: { slm_decisions: 3, slm_repairs: 0, fallback_decisions: 0, rule_decisions: 0 },
    },
    result: null,
    error: null,
    created_at: "2026-11-25T09:00:00+00:00",
    updated_at: "2026-11-25T15:00:00+00:00",
    completed_at: "2026-11-25T15:00:00+00:00",
    events: [decision(3, "optimize_sync"), decision(2, "optimize_topology"), decision(4, "prune_model")],
  };
}

const sweepGroup = {
  group_id: "sweep",
  kind: "topology_optimization",
  topology: "candidate_sweep",
  status: "completed",
  error: null,
  runs: [],
  selected_configuration: { topology: "mesh", alpha: 0.1, dataset_split: "dirichlet_a01" },
  candidates: [
    { topology: "mesh", alpha: 0.1, mean_final_accuracy: 0.61 },
    { topology: "directed_ring", alpha: 0.1, mean_final_accuracy: 0.42 },
  ],
} as unknown as ExperimentGroup;

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

describe("deriveAgentActions", () => {
  it("pairs every search step with its measurement and the configuration before it", () => {
    const actions = deriveAgentActions(runFixture(), [sweepGroup]);
    expect(actions.map((action) => [action.step, action.outcome])).toEqual([
      ["optimize_topology", "kept"],
      ["optimize_sync", "reverted"],
      ["prune_model", "kept"],
    ]);
    const [topology, sync, prune] = actions;
    expect(topology.before.topology).toBe("directed_ring");
    expect(topology.after.topology).toBe("mesh");
    expect(topology.beforeAccuracy).toBeCloseTo(0.4246);
    expect(topology.afterAccuracy).toBeCloseTo(0.6096);
    expect(topology.rationale).toBe("reason for optimize_topology");
    expect(topology.group?.group_id).toBe("sweep");
    // The reverted sync change is not carried into the next step's "before".
    expect(sync.before.sync_mode).toBe("sync");
    expect(sync.after.sync_mode).toBe("async");
    expect(prune.before.sync_mode).toBe("sync");
    expect(prune.before.topology).toBe("mesh");
    expect(prune.before.parameter_count).toBe(324394);
    expect(prune.after.parameter_count).toBe(161094);
    expect(prune.beforeAccuracy).toBeCloseTo(0.6096);
  });
});

describe("topologyEdges", () => {
  it("draws the standard graphs and diffs them by message direction", () => {
    const nodes = ["a", "b", "c", "d"];
    expect(topologyEdges("mesh", nodes)).toHaveLength(6);
    expect(topologyEdges("directed_ring", nodes)).toHaveLength(4);
    expect(topologyEdges("chain", nodes)).toHaveLength(3);
    expect(topologyEdges("star", nodes)).toHaveLength(3);
    const mesh = edgePairs(topologyEdges("mesh", nodes));
    const ring = edgePairs(topologyEdges("directed_ring", nodes));
    expect(topologyEdges("directed_ring", nodes).every((edge) => edgeIn(edge, mesh))).toBe(true);
    expect(topologyEdges("mesh", nodes).filter((edge) => !edgeIn(edge, ring))).toHaveLength(2);
    expect(topologyEdges("learned", nodes, { b: ["a", "c"] })).toEqual([
      { from: "a", to: "b", directed: true },
      { from: "c", to: "b", directed: true },
    ]);
  });
});

describe("prunedArchitecture", () => {
  it("shrinks every hidden conv and dense layer but keeps the classifier", () => {
    const architecture = {
      id: "nas", blocks: 1, channels: 32, kernel_size: 3, model_type: "custom", channel_schedule: [32], block_type: "conv2d",
      head_units: 64, dropout: 0.3, parameter_count: 1000, training: { optimizer: "adam", learning_rate: 0.001, batch_size: 32 },
      layers: [{ type: "conv2d", filters: 32 }, { type: "batch_normalization" }, { type: "dense", units: 64 }, { type: "dense", units: 10 }],
    } as const;
    const pruned = prunedArchitecture(architecture as never, 0.3, "nas-pruned30");
    expect(pruned?.id).toBe("nas-pruned30");
    expect((pruned as unknown as { layers: unknown[] }).layers).toEqual([
      { type: "conv2d", filters: 22 }, { type: "batch_normalization" }, { type: "dense", units: 45 }, { type: "dense", units: 10 },
    ]);
  });
});

describe("AgentDecisionsPage", () => {
  it("shows each selected action with before and after views", async () => {
    const run = runFixture();
    vi.stubGlobal(
      "fetch",
      vi.fn().mockImplementation((input: RequestInfo | URL) => {
        const url = String(input);
        if (url === `${apiUrl}/api/supervisor/runs`) return Promise.resolve({ ok: true, status: 200, json: async () => [run] });
        if (url === `${apiUrl}/api/supervisor/runs/run-1`) return Promise.resolve({ ok: true, status: 200, json: async () => run });
        return Promise.resolve({ ok: false, status: 404, json: async () => ({ detail: "not found" }) });
      }),
    );
    const results = { configuration: { rounds: 10, epochs: 5, topology: "directed_ring", dataset_split: "dirichlet_a01" }, active_group_id: null, topologies: {}, groups: [sweepGroup] } as const;
    render(<AgentDecisionsPage token="t" active results={results as never} />);

    const topology = await screen.findByRole("article", { name: "Action 1: Topology optimization" });
    expect(within(topology).getByText("Selected by SLM · qwen3.5-dfl:4b")).toBeInTheDocument();
    expect(within(screen.getByRole("article", { name: "Action 3: Pruning + fine-tune" })).getByText("Selected by SLM")).toBeInTheDocument();
    expect(within(topology).getByText("Kept")).toBeInTheDocument();
    expect(within(topology).getByRole("img", { name: "Directed ring topology" })).toBeInTheDocument();
    expect(within(topology).getByRole("img", { name: "Mesh topology" })).toBeInTheDocument();
    expect(within(topology).getByText("+18.5 points")).toBeInTheDocument();
    expect(within(topology).getByText(/2 candidates/)).toBeInTheDocument();

    const sync = screen.getByRole("article", { name: "Action 2: Synchronization optimization" });
    expect(within(sync).getByText("Reverted")).toBeInTheDocument();
    expect(within(sync).getByText("Proposed (reverted)")).toBeInTheDocument();
    expect(within(sync).getByText("Synchronous")).toBeInTheDocument();
    expect(within(sync).getByText("Asynchronous")).toBeInTheDocument();

    const prune = screen.getByRole("article", { name: "Action 3: Pruning + fine-tune" });
    expect(within(prune).getByText(/2\.01× smaller/)).toBeInTheDocument();
  });
});
