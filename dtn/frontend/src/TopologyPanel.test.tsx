import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import type { ExperimentResults } from "./api/client";
import { TrainingTopologyPanel } from "./App";

const { sendSlmCommand } = vi.hoisted(() => ({ sendSlmCommand: vi.fn() }));

vi.mock("./api/client", () => ({
  sendSlmCommand,
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
        runtime_efficiency: 0.9,
      },
    }],
  }],
};

afterEach(() => {
  cleanup();
  vi.clearAllMocks();
});

describe("Topology optimizer panel", () => {
  it("shows the active real topology, candidate reward breakdown, and baselines", () => {
    render(<TrainingTopologyPanel experiments={results} token="test-token" />);

    expect(screen.getByText("Active: chain")).toBeInTheDocument();
    expect(screen.getByRole("img", { name: "chain worker communication graph" })).toBeInTheDocument();
    expect(screen.getByText("dfl-worker1 ↔ dfl-worker2")).toBeInTheDocument();
    expect(screen.getByText(/weighted_measured_reward/)).toBeInTheDocument();
    expect(screen.getByText("0.7300")).toBeInTheDocument();
    expect(screen.getByText("0.800")).toBeInTheDocument();
    expect(screen.getByText(/Best-observed baseline: mesh/)).toBeInTheDocument();
  });

  it("starts the validated real-device topology sweep", async () => {
    sendSlmCommand.mockResolvedValue({
      result: { available: true, data: { group_id: "optimization-42" } },
    });
    render(<TrainingTopologyPanel experiments={results} token="test-token" />);

    fireEvent.click(screen.getByRole("button", { name: "Run 5-topology sweep" }));

    await waitFor(() => expect(sendSlmCommand).toHaveBeenCalledWith("test-token", {
      action: "optimize_topology",
      parameters: { rounds: 10, epochs: 5, seeds: [17, 42, 2026], alphas: [0.3, 0.5] },
    }));
    expect(screen.getByRole("status")).toHaveTextContent("optimization-42");
  });
});
