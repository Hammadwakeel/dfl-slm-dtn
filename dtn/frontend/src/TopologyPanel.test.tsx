import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import type { ExperimentResults } from "./api/client";
import { TrainingTopologyPanel } from "./App";

const { dispatchAutonomyAction, getExperimentResults } = vi.hoisted(() => ({
  dispatchAutonomyAction: vi.fn(),
  getExperimentResults: vi.fn(),
}));

vi.mock("./api/client", () => ({
  dispatchAutonomyAction,
  getExperimentResults,
}));

const results: ExperimentResults = {
  configuration: {
    rounds: 10,
    epochs: 5,
    topology: "chain",
    dataset_split: "dirichlet_a03",
  },
  active_group_id: null,
  topologies: {},
  groups: [{
    group_id: "optimizer-real-sweep",
    kind: "topology_optimization",
    dataset_name: "mnist",
    reward_metric_source: "validation_accuracy",
    reward_config: {
      weights: {
        accuracy_gain: 0.5,
        convergence_speed: 0.2,
        communication_efficiency: 0.15,
        peer_wait_efficiency: 0.15,
      },
    },
    topology: "candidate_sweep",
    rounds: 10,
    epochs: 5,
    seeds: [17, 42, 2026],
    status: "completed",
    error: null,
    runs: [],
    selection: {
      method: "weighted_measured_reward",
      selected_candidate: { topology: "chain", alpha: 0.3, mean_reward: 0.73 },
      baselines: {
        random_seeded: { topology: "star", alpha: 0.5, mean_reward: 0.6 },
        best_observed: { topology: "mesh", alpha: 0.5, mean_final_accuracy: 0.58 },
      },
      candidate_count: 10,
    },
    candidates: [{
      topology: "chain",
      alpha: 0.3,
      completed_runs: 3,
      mean_reward: 0.73,
      mean_final_accuracy: 0.56,
      mean_components: {
        accuracy_gain: 0.8,
        convergence_speed: 0.7,
        communication_efficiency: 0.6,
        peer_wait_efficiency: 0.9,
      },
    }],
  }],
};

afterEach(() => {
  cleanup();
  sessionStorage.clear();
  vi.clearAllMocks();
  dispatchAutonomyAction.mockResolvedValue({
    status: "executed",
    result: { available: true, data: { group_id: "optimization-42" } },
  });
  getExperimentResults.mockResolvedValue(results);
});

describe("Topology optimizer panel", () => {
  it("shows reward v2 components, the weakest worker, and the quality gate", () => {
    const v2Results: ExperimentResults = {
      ...results,
      groups: [{
        ...results.groups[0],
        group_id: "optimizer-v2",
        seeds: [17],
        reward_config: {
          version: 2,
          weights: { quality: 0.5, learning_speed: 0.2, bandwidth_efficiency: 0.15, wait_efficiency: 0.15 },
          worst_worker_share: 0.5,
          quality_gate_min_worker_accuracy: 0.9,
        },
        selection: {
          method: "quality_gated_reward_v2",
          quality_gate_met: false,
          selected_candidate: { topology: "mesh", alpha: 0.3, mean_reward: 0.682 },
          baselines: {
            random_seeded: { topology: "chain", alpha: 0.3, mean_reward: 0.6057 },
            best_observed: { topology: "mesh", alpha: 0.3, mean_final_accuracy: 0.983 },
          },
          candidate_count: 2,
        },
        candidates: [
          {
            topology: "mesh",
            alpha: 0.3,
            completed_runs: 1,
            reward_version: 2,
            mean_reward: 0.682,
            std_reward: 0,
            mean_final_accuracy: 0.983,
            mean_min_worker_accuracy: 0.97,
            passes_quality_gate: true,
            quality_source: "global_validation",
            mean_components: { quality: 0.8757, learning_speed: 0.6617, bandwidth_efficiency: 0.0059, wait_efficiency: 0.7393 },
          },
          {
            topology: "chain",
            alpha: 0.3,
            completed_runs: 1,
            reward_version: 2,
            mean_reward: 0.6057,
            std_reward: 0,
            mean_final_accuracy: 0.834,
            mean_min_worker_accuracy: 0.505,
            passes_quality_gate: false,
            quality_source: "global_validation",
            mean_components: { quality: 0.5785, learning_speed: 0.434, bandwidth_efficiency: 0.7514, wait_efficiency: 0.7792 },
          },
        ],
      }],
    };
    render(<TrainingTopologyPanel experiments={v2Results} token="test-token" />);

    expect(screen.getByText(/quality_gated_reward_v2/)).toBeInTheDocument();
    expect(screen.getByRole("columnheader", { name: "Weakest worker" })).toBeInTheDocument();
    expect(screen.getByRole("columnheader", { name: "Quality (Q)" })).toBeInTheDocument();
    expect(screen.queryByRole("columnheader", { name: "Accuracy gain" })).not.toBeInTheDocument();
    expect(screen.getByText("50.50%")).toBeInTheDocument();
    expect(screen.getByText("pass")).toBeInTheDocument();
    expect(screen.getByText("fail")).toBeInTheDocument();
    expect(screen.getAllByText("shared balanced split")).toHaveLength(2);
    expect(screen.getByText(/No topology brought every worker to 90.00% accuracy/)).toBeInTheDocument();
    expect(screen.getByText(/Reward v2 = 0.50·Q/)).toBeInTheDocument();
    const rows = screen.getByRole("table").querySelectorAll("tbody tr");
    expect(rows[0]).toHaveClass("best-result-row");
  });

  it("shows the active real topology, candidate reward breakdown, and baselines", () => {
    render(<TrainingTopologyPanel experiments={results} token="test-token" />);

    expect(screen.getByText("Active: chain")).toBeInTheDocument();
    expect(screen.getByRole("img", { name: "chain worker communication graph" })).toBeInTheDocument();
    expect(screen.getByText("dfl-worker1 ↔ dfl-worker2")).toBeInTheDocument();
    expect(screen.getByText(/weighted_measured_reward/)).toBeInTheDocument();
    expect(screen.getByText("0.7300")).toBeInTheDocument();
    expect(screen.getByText("0.800")).toBeInTheDocument();
    expect(screen.getByText("BEST TOPOLOGY")).toBeInTheDocument();
    expect(screen.getByRole("table").querySelector("tbody tr")).toHaveClass("best-result-row");
    expect(screen.getByText("BEST", { selector: "span" })).toBeInTheDocument();
    expect(screen.getByText(/Best-observed baseline: mesh/)).toBeInTheDocument();
    expect(screen.getByText(/Ranking: validation accuracy, convergence, communication, and peer wait/)).toBeInTheDocument();
    expect(screen.getByRole("columnheader", { name: "Validation accuracy" })).toBeInTheDocument();
    expect(screen.getByRole("columnheader", { name: "Peer communication" })).toBeInTheDocument();
    expect(screen.getByRole("columnheader", { name: "Peer wait efficiency" })).toBeInTheDocument();
  });

  it("draws the selected topology when it differs from the active configuration", () => {
    const selectedTopologyResults = structuredClone(results);
    selectedTopologyResults.configuration.topology = "star";

    render(<TrainingTopologyPanel experiments={selectedTopologyResults} token="test-token" />);

    expect(screen.getByText("Active: star")).toBeInTheDocument();
    expect(screen.getByText("Selected: chain")).toBeInTheDocument();
    expect(screen.getByRole("img", { name: "chain worker communication graph" })).toBeInTheDocument();
    expect(screen.getByText("dfl-worker1 ↔ dfl-worker2")).toBeInTheDocument();
    expect(screen.queryByText("dfl-worker1 ↔ dfl-worker3")).not.toBeInTheDocument();
  });

  it("renders the selected topology using all workers and coordinator-provided edges", () => {
    const fiveWorkerResults = structuredClone(results);
    fiveWorkerResults.configuration.topology = "star";
    fiveWorkerResults.topologies.chain = [
      ["worker1", "worker2"],
      ["worker2", "worker3"],
      ["worker3", "worker4"],
      ["worker4", "worker5"],
    ];

    render(<TrainingTopologyPanel experiments={fiveWorkerResults} token="test-token" />);

    expect(screen.getByRole("img", { name: "chain worker communication graph" }))
      .toBeInTheDocument();
    expect(screen.getByText("dfl-worker5")).toBeInTheDocument();
    expect(screen.getByText("dfl-worker4 ↔ dfl-worker5")).toBeInTheDocument();
    expect(screen.getByRole("img", { name: "chain worker communication graph" })
      .querySelectorAll("circle")).toHaveLength(5);
  });

  it("shows arrows in both directions for bidirectional topologies", () => {
    render(<TrainingTopologyPanel experiments={results} token="test-token" />);

    const graph = screen.getByRole("img", { name: "chain worker communication graph" });
    const edges = graph.querySelectorAll("line");

    expect(edges.length).toBeGreaterThan(0);
    for (const edge of edges) {
      expect(edge).toHaveAttribute("marker-start", "url(#topology-arrow)");
      expect(edge).toHaveAttribute("marker-end", "url(#topology-arrow)");
    }
  });

  it("shows only forward arrows for a directed ring and keeps arrowheads outside worker nodes", () => {
    const directedRingResults = structuredClone(results);
    directedRingResults.groups[0].selection = undefined;
    directedRingResults.groups[0].selected_configuration = {
      topology: "directed_ring",
      alpha: 0.3,
      dataset_split: "dirichlet_a03",
    };
    directedRingResults.configuration.topology = "directed_ring";
    directedRingResults.topologies.directed_ring = [
      "worker1",
      "worker2",
      "worker3",
      "worker1",
    ];

    render(<TrainingTopologyPanel experiments={directedRingResults} token="test-token" />);

    const graph = screen.getByRole("img", {
      name: "directed_ring worker communication graph",
    });
    const firstEdge = graph.querySelector("line");

    expect(firstEdge).toHaveAttribute("marker-end", "url(#topology-arrow)");
    expect(firstEdge).not.toHaveAttribute("marker-start");
    expect(Number(firstEdge?.getAttribute("x1"))).toBeGreaterThan(300);
  });

  it("warns when displaying a legacy sweep without a recorded validation metric source", () => {
    const legacyResults = structuredClone(results);
    delete legacyResults.groups[0].reward_metric_source;

    render(<TrainingTopologyPanel experiments={legacyResults} token="test-token" />);

    expect(screen.getByText(/Legacy reward may include held-out TEST scores/)).toBeInTheDocument();
    expect(screen.getByRole("columnheader", { name: "Legacy reward" })).toBeInTheDocument();
  });

  it("uses peer wait as the current criterion and labels old runtime scores as legacy", () => {
    const legacyResults = structuredClone(results);
    delete legacyResults.groups[0].reward_config;
    legacyResults.groups[0].candidates![0].mean_components!.runtime_efficiency = 0.72;

    render(<TrainingTopologyPanel experiments={legacyResults} token="test-token" />);

    expect(screen.getByRole("columnheader", { name: "Runtime efficiency (legacy)" })).toBeInTheDocument();
    expect(screen.queryByRole("columnheader", { name: "Peer wait efficiency" })).not.toBeInTheDocument();
  });

  it("renders a placeholder when older topology results omit the mean reward", () => {
    const incompleteResults = structuredClone(results);
    delete incompleteResults.groups[0].candidates?.[0].mean_reward;
    delete incompleteResults.groups[0].candidates?.[0].mean_components;

    render(<TrainingTopologyPanel experiments={incompleteResults} token="test-token" />);

    const row = screen.getByRole("table").querySelector("tbody tr");
    expect(row?.textContent?.match(/—/g)).toHaveLength(11);
  });

  it("uses completed run metrics when saved topology aggregates are missing", () => {
    const runResults = structuredClone(results);
    const group = runResults.groups[0];
    group.candidates = undefined;
    group.runs = [
      {
        run_id: "chain-seed-17",
        seed: 17,
        topology: "chain",
        alpha: 0.3,
        status: "completed",
        rounds: [{
          round: 10,
          topology: "chain",
          workers: {
            worker1: {
              node_id: "worker1",
              run_id: "chain-seed-17",
              round: 10,
              topology: "chain",
              peak_process_rss_mb: 100,
              validation_accuracy: 0.96,
              validation_macro_precision: 0.88,
              validation_macro_recall: 0.89,
              validation_macro_f1: 0.90,
              round_seconds: 5,
              bytes_sent: 1000,
              bytes_received: 500,
              peer_wait_seconds: 0.2,
            },
            worker2: {
              node_id: "worker2",
              run_id: "chain-seed-17",
              round: 10,
              topology: "chain",
              peak_process_rss_mb: 100,
              validation_accuracy: 0.94,
              validation_macro_precision: 0.90,
              validation_macro_recall: 0.91,
              validation_macro_f1: 0.92,
              round_seconds: 7,
              bytes_sent: 2000,
              bytes_received: 500,
              peer_wait_seconds: 0.4,
            },
          },
        }],
      },
      {
        run_id: "chain-seed-42",
        seed: 42,
        topology: "chain",
        alpha: 0.3,
        status: "completed",
        rounds: [{
          round: 10,
          topology: "chain",
          workers: {
            worker1: {
              node_id: "worker1",
              run_id: "chain-seed-42",
              round: 10,
              topology: "chain",
              peak_process_rss_mb: 100,
              validation_accuracy: 0.98,
              validation_macro_precision: 0.95,
              validation_macro_recall: 0.97,
              validation_macro_f1: 0.96,
              round_seconds: 9,
              bytes_sent: 1000,
              bytes_received: 500,
              peer_wait_seconds: 0.6,
            },
          },
        }],
      },
    ];

    render(<TrainingTopologyPanel experiments={runResults} token="test-token" />);

    const row = screen.getByRole("table").querySelector("tbody tr");
    expect(row).toHaveTextContent("chain");
    expect(row).toHaveTextContent("2/3");
    expect(row).toHaveTextContent("96.50%");
    expect(row).toHaveTextContent("92.00%");
    expect(row).toHaveTextContent("93.50%");
    expect(row).toHaveTextContent("93.50%");
    expect(row).toHaveTextContent("7.50 s");
    expect(row).toHaveTextContent("2,750 B");
    expect(row).toHaveTextContent("0.45 s");
  });

  it("does not report zero accuracy gain when a one-round run has no initial baseline", () => {
    const oneRoundResults = structuredClone(results);
    const group = oneRoundResults.groups[0];
    group.rounds = 1;
    group.seeds = [17];
    group.runs = [{
      run_id: "chain-one-round",
      seed: 17,
      topology: "chain",
      alpha: 0.3,
      status: "completed",
      rounds: [{
        round: 1,
        topology: "chain",
        workers: {
          worker1: {
            node_id: "worker1",
            run_id: "chain-one-round",
            round: 1,
            topology: "chain",
            peak_process_rss_mb: 100,
            validation_accuracy: 0.53,
          },
        },
      }],
    }];
    group.candidates![0].mean_components!.accuracy_gain = 0;

    render(<TrainingTopologyPanel experiments={oneRoundResults} token="test-token" />);

    const row = screen.getByRole("table").querySelector("tbody tr");
    const accuracyGainCell = row?.querySelectorAll("td")[8];
    expect(accuracyGainCell).toHaveTextContent("—");
    expect(accuracyGainCell).toHaveAttribute(
      "title",
      "Unavailable: this single-round run has no pre-training validation baseline.",
    );
  });

  it("shows candidate and worker progress while a topology sweep is running", () => {
    const runningResults = structuredClone(results);
    const runningGroup = runningResults.groups[0];
    runningGroup.status = "running";
    runningGroup.selection = undefined;
    runningGroup.candidates = ([
      "directed_ring",
      "bidirectional_ring",
      "mesh",
      "chain",
      "star",
    ] as const).map((topology) => ({
      topology,
      alpha: 0.3,
    }));
    runningGroup.runs = Array.from({ length: 5 }, (_, index) => ({
      run_id: `completed-${index}`,
      seed: 17,
      topology: "directed_ring" as const,
      alpha: 0.3,
      status: "completed" as const,
      rounds: [],
    }));
    runningGroup.runs.push({
      run_id: "directed-ring-a03-seed-17",
      seed: 17,
      topology: "chain",
      alpha: 0.3,
      status: "running",
      rounds: Array.from({ length: 3 }, (_, index) => ({
        round: index + 1,
        topology: "chain",
        workers: {},
      })),
      progress: {
        round: 4,
        workers: {
          worker1: { epoch: 4, epochs: 5, phase: "local_training" },
        },
      },
    });

    render(<TrainingTopologyPanel experiments={runningResults} token="test-token" />);

    const progressCard = screen.getByRole("status", { name: "Topology sweep progress" });
    expect(progressCard).toHaveTextContent("5 of 15 runs complete · 10 remaining");
    expect(progressCard).toHaveTextContent(/chain.*alpha 0\.3.*seed 17.*round 4\/10/);
    expect(screen.getByRole("status")).toHaveTextContent("worker1 epoch 4/5");
    expect(screen.getByRole("progressbar", { name: "Overall sweep progress" })).toHaveAttribute("aria-valuenow", "5");
    expect(screen.getByRole("progressbar", { name: "Current candidate run progress" })).toHaveAttribute("aria-valuenow", "4");
    expect(progressCard).toHaveTextContent("33% complete");
    expect(screen.getByRole("button", { name: "Sweep in progress…" })).toBeDisabled();
  });

  it("starts the validated real-device topology sweep", async () => {
    dispatchAutonomyAction.mockResolvedValue({
      status: "executed",
      result: { available: true, data: { group_id: "optimization-42" } },
    });
    render(<TrainingTopologyPanel experiments={results} token="test-token" />);

    fireEvent.click(screen.getByRole("button", { name: "Run 5-topology sweep" }));

    await waitFor(() => expect(dispatchAutonomyAction).toHaveBeenCalledWith("test-token", {
      action: "optimize_topology",
      parameters: { rounds: 10, epochs: 5, seeds: [17, 42, 2026], alphas: [0.3] },
    }, "Run a validation-scored MNIST topology sweep on the real DFL workers.", {
      dataset_name: "mnist",
      topologies: ["directed_ring", "bidirectional_ring", "mesh", "chain", "star"],
      alphas: [0.3],
      rounds: 10,
      epochs: 5,
      seeds: [17, 42, 2026],
    }));
    expect(screen.getByRole("status")).toHaveTextContent("optimization-42");
  });

  it("starts a bounded topology smoke test with one round, epoch, and seed", async () => {
    dispatchAutonomyAction.mockResolvedValue({
      status: "executed",
      result: { available: true, data: { group_id: "smoke-test-42" } },
    });
    render(<TrainingTopologyPanel experiments={results} token="test-token" />);

    fireEvent.click(screen.getByRole("button", { name: "Run topology smoke test" }));

    await waitFor(() => expect(dispatchAutonomyAction).toHaveBeenCalledWith("test-token", {
      action: "optimize_topology",
      parameters: { rounds: 1, epochs: 1, seeds: [17], alphas: [0.3] },
    }, "Run a one-round, one-epoch MNIST topology smoke test on the real DFL workers.", {
      dataset_name: "mnist",
      topologies: ["directed_ring", "bidirectional_ring", "mesh", "chain", "star"],
      alphas: [0.3],
      rounds: 1,
      epochs: 1,
      seeds: [17],
    }));
    expect(screen.getByText("Topology smoke test smoke-test-42 started.")).toBeInTheDocument();
    expect(sessionStorage.getItem("dtn-topology-smoke-group-id")).toBe("smoke-test-42");

    cleanup();
    const runningSmokeTest = structuredClone(results);
    runningSmokeTest.active_group_id = "smoke-test-42";
    runningSmokeTest.groups.unshift({
      ...runningSmokeTest.groups[0],
      group_id: "smoke-test-42",
      rounds: 1,
      epochs: 1,
      seeds: [17],
      status: "running",
      runs: [{
        run_id: "directed-ring-smoke",
        seed: 17,
        topology: "directed_ring",
        alpha: 0.3,
        status: "running",
        rounds: [],
      }],
    });
    render(<TrainingTopologyPanel experiments={runningSmokeTest} token="test-token" />);

    expect(screen.getByRole("status", { name: "Saved topology smoke test" }))
      .toHaveTextContent("smoke-test-42 · running · 0/5 topologies complete");
    expect(screen.getByRole("button", { name: "Run topology smoke test" })).toBeDisabled();
  });

  it("starts the requested sweep immediately", async () => {
    dispatchAutonomyAction.mockResolvedValue({
      status: "executed",
      result: { available: true, data: { group_id: "optimization-auto" } },
    });
    render(<TrainingTopologyPanel experiments={results} token="test-token" />);

    fireEvent.click(screen.getByRole("button", { name: "Run 5-topology sweep" }));

    expect(await screen.findByRole("status")).toHaveTextContent("optimization-auto");
  });
});
