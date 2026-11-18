import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import AgentPage from "./AgentPage";
import type { SupervisorRun } from "../api/client";

const apiUrl = "http://localhost:8000";
const cluster = {
  workers: 5,
  workers_ready: 5,
  not_ready: [],
  min_free_mb: 5800,
  low_memory: [],
  min_free_memory_mb_required: 1200,
  global_stop: false,
};

function runFixture(overrides: Partial<SupervisorRun> = {}): SupervisorRun {
  return {
    id: "6f1c1d2e-0000-4000-8000-000000000001",
    task: "Reach 95.00% mean worker validation accuracy on mnist",
    status: "running",
    stage: "monitoring",
    progress: 42,
    current_step: "run_nas",
    rationale: "NAS should find a stronger model.",
    goal: {
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
    },
    state: {
      stage: "monitoring",
      completed_steps: ["prepare_data", "initial_model", "baseline_training"],
      experiments: [
        {
          step: "baseline_training",
          group_id: "directed_ring-1",
          kind: "decentralized",
          status: "completed",
          validation_accuracy: 0.9124,
          macro_f1: 0.9031,
          architecture: "b1_c8_k3",
          topology: "directed_ring",
          local_validation_accuracy: 0.97,
          test_accuracy: 0.8811,
          test: { std_accuracy: 0.012 },
        },
      ],
      best: {
        validation_accuracy: 0.9124,
        macro_f1: 0.9031,
        group_id: "directed_ring-1",
        step: "baseline_training",
        architecture: "b1_c8_k3",
        topology: "directed_ring",
      },
      selected_topology: null,
      nas_architecture: null,
      active: { step: "run_nas", group_id: "nas-1", started_at: "2026-11-17T09:00:00Z" },
      pending_step: null,
      counters: { slm_decisions: 1, slm_repairs: 0, fallback_decisions: 0, rule_decisions: 0 },
      deadline: "2099-01-01T00:00:00Z",
    },
    result: null,
    error: null,
    created_at: "2026-11-17T08:00:00Z",
    updated_at: "2026-11-17T09:00:00Z",
    completed_at: null,
    events: [
      {
        id: 1,
        event_type: "decision",
        message: "run_nas chosen by slm: NAS should find a stronger model.",
        details: { step: "run_nas", source: "slm" },
        created_at: "2026-11-17T09:00:00Z",
      },
    ],
    ...overrides,
  };
}

function mockApi(runs: SupervisorRun[], extra?: (url: string, init?: RequestInit) => unknown) {
  const fetchMock = vi.fn().mockImplementation((input: RequestInfo | URL, init?: RequestInit) => {
    const url = String(input);
    const handled = extra?.(url, init);
    if (handled !== undefined) return Promise.resolve({ ok: true, status: 200, json: async () => handled });
    if (url === `${apiUrl}/api/supervisor/runs` && (!init?.method || init.method === "GET")) {
      return Promise.resolve({ ok: true, status: 200, json: async () => runs });
    }
    if (url === `${apiUrl}/api/supervisor/cluster`) {
      return Promise.resolve({ ok: true, status: 200, json: async () => cluster });
    }
    const run = runs.find((item) => url === `${apiUrl}/api/supervisor/runs/${item.id}`);
    if (run) return Promise.resolve({ ok: true, status: 200, json: async () => run });
    return Promise.resolve({ ok: false, status: 404, json: async () => ({ detail: "not found" }) });
  });
  vi.stubGlobal("fetch", fetchMock);
  return fetchMock;
}

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

describe("Agent Supervisor page", () => {
  it("shows the goal form and starts a supervisor run with the chosen goal", async () => {
    const started = runFixture({ status: "running", stage: "preflight", progress: 0, events: [] });
    const fetchMock = mockApi([], (url, init) =>
      url === `${apiUrl}/api/supervisor/runs` && init?.method === "POST" ? started : undefined,
    );
    render(<AgentPage token="dtn-token" active />);

    expect(await screen.findByRole("form", { name: "Supervisor goal" })).toBeInTheDocument();
    expect(screen.getByText(/5\/5 workers ready/)).toBeInTheDocument();
    fireEvent.change(screen.getByLabelText("Dataset"), { target: { value: "cifar10" } });
    fireEvent.change(screen.getByLabelText("Target validation accuracy (%)"), { target: { value: "70" } });
    fireEvent.change(screen.getByLabelText("Aggregation"), { target: { value: "metropolis" } });
    fireEvent.click(screen.getByRole("button", { name: "Start supervisor" }));

    await waitFor(() => {
      const post = fetchMock.mock.calls.find(
        ([url, init]) => String(url) === `${apiUrl}/api/supervisor/runs` && (init as RequestInit)?.method === "POST",
      );
      expect(post).toBeDefined();
      const body = JSON.parse(String((post?.[1] as RequestInit).body));
      expect(body.dataset).toBe("cifar10");
      expect(body.target_accuracy).toBeCloseTo(0.7);
      expect(body.rounds).toBe(10);
      expect(body.local_epochs).toBe(5);
      expect(body.aggregation).toBe("metropolis");
      expect(body.sync_mode).toBe("semi_sync");
      expect(body.sync_deadline_seconds).toBe(300);
    });
  });

  it("shows the active run with its stage, best result, decision source, and experiments", async () => {
    const run = runFixture();
    mockApi([run]);
    render(<AgentPage token="dtn-token" active />);

    expect(await screen.findByRole("article", { name: "Supervisor run" })).toBeInTheDocument();
    expect(screen.queryByRole("form", { name: "Supervisor goal" })).not.toBeInTheDocument();
    expect(screen.getAllByText("91.24%").length).toBeGreaterThan(0);
    expect(screen.getByText(/LATEST DECISION · SLM/)).toBeInTheDocument();
    expect(screen.getByText("Architecture search")).toBeInTheDocument();
    expect(screen.getByRole("progressbar")).toHaveAttribute("aria-valuenow", "42");
    expect(screen.getByText("Baseline DFL", { selector: "td" })).toBeInTheDocument();
    // Global test accuracy and the inflated worker-reported accuracy are both shown.
    expect(screen.getByText("88.11% ± 1.20%")).toBeInTheDocument();
    expect(screen.getByText("97.00%")).toBeInTheDocument();
  });

  it("explains a warm-started final training", async () => {
    const run = runFixture();
    run.state.warm_start = {
      from_step: "learn_topology",
      from_validation_accuracy: 0.4188,
      source_worker: "worker4",
    };
    mockApi([run]);
    render(<AgentPage token="dtn-token" active />);

    const note = await screen.findByLabelText("Warm start");
    expect(note).toHaveTextContent("41.88% validation");
    expect(note).toHaveTextContent("worker4");
  });

  it("lists a rejected prune and its gentler retry", async () => {
    const run = runFixture();
    const attempt = {
      parameters_before: 8531,
      parameters_after: 6800,
      compression: 1.25,
      architecture_id: "hpo-pruned15",
      source_worker: "worker6",
    };
    run.state.pruning_attempts = [
      { ...attempt, ratio: 0.3, validation_accuracy: 0.81, accepted: false },
      { ...attempt, ratio: 0.15, validation_accuracy: 0.935, accepted: true },
    ];
    run.state.pruning = run.state.pruning_attempts[1];
    mockApi([run]);
    render(<AgentPage token="dtn-token" active />);

    const attempts = await screen.findByLabelText("Pruning attempts");
    expect(attempts).toHaveTextContent("30% pruned: 81.00% · rejected, unpruned model restored");
    expect(attempts).toHaveTextContent("15% pruned: 93.50% · kept");
  });

  it("stops an active run after confirmation", async () => {
    vi.stubGlobal("confirm", vi.fn(() => true));
    const run = runFixture();
    const fetchMock = mockApi([run], (url, init) =>
      url.endsWith(`/api/supervisor/runs/${run.id}/stop`) && init?.method === "POST"
        ? { ...run, status: "stopping" }
        : undefined,
    );
    render(<AgentPage token="dtn-token" active />);

    fireEvent.click(await screen.findByRole("button", { name: "Stop" }));

    await waitFor(() =>
      expect(
        fetchMock.mock.calls.some(([url]) => String(url).endsWith(`/api/supervisor/runs/${run.id}/stop`)),
      ).toBe(true),
    );
  });

  it("asks the agent a question about the run", async () => {
    const run = runFixture();
    mockApi([run], (url, init) =>
      url.endsWith(`/api/supervisor/runs/${run.id}/ask`) && init?.method === "POST"
        ? { run_id: run.id, answer: "NAS is running because the baseline was below target." }
        : undefined,
    );
    render(<AgentPage token="dtn-token" active />);

    fireEvent.change(await screen.findByLabelText("Ask the agent about this run"), {
      target: { value: "NAS kyun chal raha hai?" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Ask" }));

    expect(await screen.findByText("NAS is running because the baseline was below target.")).toBeInTheDocument();
  });

  it("shows tuned hyperparameters and the pruned model", async () => {
    const base = runFixture();
    const run = runFixture({
      state: {
        ...base.state,
        experiments: [
          ...base.state.experiments,
          {
            step: "run_hpo",
            group_id: "hpo-1",
            kind: "hyperparameter_search",
            status: "completed",
            architecture: "hpo-adam-lr3e-3-bs64",
            search_validation_accuracy: 0.93,
            hyperparameters: { optimizer: "adam", learning_rate: 0.003, batch_size: 64 },
          },
        ],
        pruning: {
          ratio: 0.5,
          parameters_before: 24010,
          parameters_after: 6402,
          compression: 3.75,
          architecture_id: "nas-cnn-pruned50",
          source_worker: "worker3",
          source_validation_accuracy: 0.96,
          validation_accuracy: 0.955,
        },
      },
    });
    mockApi([run]);
    render(<AgentPage token="dtn-token" active />);

    expect(await screen.findByRole("region", { name: "Pruned model" })).toBeInTheDocument();
    expect(screen.getByText("3.75×")).toBeInTheDocument();
    expect(screen.getByText("6,402")).toBeInTheDocument();
    expect(screen.getByText("95.50%")).toBeInTheDocument();
    expect(screen.getByText("Hyperparameter tuning", { selector: "td" })).toBeInTheDocument();
    expect(screen.getByText("93.00% (search)")).toBeInTheDocument();
    expect(screen.getByText(/lr 0.003 · batch 64/)).toBeInTheDocument();
  });

  it("does not poll while the page is hidden", () => {
    const fetchMock = mockApi([]);
    render(<AgentPage token="dtn-token" active={false} />);
    expect(fetchMock).not.toHaveBeenCalled();
  });
});
