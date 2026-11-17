import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import type { APGRunDetail, APGRunsResponse, APGVerdict } from "../api/client";
import APGTopologyPanel from "./APGTopologyPanel";

const api = vi.hoisted(() => ({
  getAPGJob: vi.fn(),
  getAPGRun: vi.fn(),
  getAPGRuns: vi.fn(),
  getAutonomyState: vi.fn(),
  startAPGSimulation: vi.fn(),
}));

vi.mock("../api/client", () => api);

const round = {
  round: 1,
  split: "VALIDATION" as const,
  source_label: "SIMULATED" as const,
  mean_validation_accuracy: 0.6,
  mean_validation_macro_f1: 0.5,
  mean_validation_loss: 0.8,
  edges_used: [["node_1", "node_0"]] as Array<[string, string]>,
  edge_count_used: 1,
  edges_changed_vs_previous_round: null,
  safeguard_hits: 2,
  g: 0.12,
  iterations_used: 2,
  inputs: {
    round: 1,
    loss_changed: null,
    waiting_time_changed: null,
    theta_changed: null,
    loss_stats: { min: 0.1, mean: 0.3, max: 0.9 },
    waiting_time_stats: { min: 0.1, mean: 0.3, max: 0.9 },
    theta_stats: { min: 0, mean: 4, max: 10 },
    inputs_available: true,
  },
  apg_inputs: {
    loss_matrix_normalized: [[null, 0.1]],
    wait_matrix_normalized: [[null, 0.2]],
    theta_train_only: [[1, 2]],
  },
  per_node: {
    node_0: { validation_accuracy: 0.6, validation_macro_f1: 0.5, validation_loss: 0.8, bytes_sent_simulated: 12 },
  },
  bytes_sent_simulated: 12,
};

const summary = {
  run_id: "apg-ui-test",
  simulation_mode: "all_simulated",
  source_label: "SIMULATED" as const,
  data_setting: "iid_stratified",
  topology: "apg",
  seed: 17,
  status: "finished",
  split_labels: { rounds: "VALIDATION" as const, final_metrics: "TEST" as const },
  duration_seconds: 12,
  rounds: [round],
  final_test_summary: { accuracy: 0.7, macro_f1: 0.6, macro_precision: 0.6, macro_recall: 0.7 },
};

const detail: APGRunDetail = {
  summary,
  run: {
    run_id: summary.run_id,
    simulation_mode: summary.simulation_mode,
    source_label: summary.source_label,
    data_setting: summary.data_setting,
    topology: summary.topology,
    seed: 17,
    pc_simulation_wall_seconds: 12,
    rounds: [round],
    nodes: Array.from({ length: 8 }, (_, index) => ({
      node_id: `node_${index}`,
      shard_rows: 300,
      theta: [1, 2, 0],
      node_role: index < 5 ? "simulated" : "emulated_worker",
    })),
    final_test: {
      split: "TEST",
      source: "SIMULATED",
      summary: summary.final_test_summary,
      confusion_matrix_counts: Array.from({ length: 10 }, () => Array(10).fill(0)),
      confusion_matrix_row_normalized_percent: Array.from({ length: 10 }, () => Array(10).fill(0)),
    },
  },
};

function verdict(text: APGVerdict["verdict"]): APGVerdict {
  return {
    data_setting: "iid_stratified",
    simulation_mode: "all_simulated",
    source_label: "SIMULATED",
    verdict: text,
    rules: {
      topology_adapts: true,
      topology_follows_inputs: false,
      model_learns: null,
      validation_mean_gain: null,
      validation_seed_sd: null,
      validation_seed_count: 1,
      benefit_over_best_fixed_topology: null,
      paired_test_mean_delta: null,
      paired_test_sample_sd: null,
      test_seed_count: 1,
      test_split: "TEST",
      validation_split: "VALIDATION",
      benefit_rule: "paired test rule",
    },
    runs: ["apg-ui-test"],
  };
}

function response(jobs: APGRunsResponse["jobs"] = [], verdictText: APGVerdict["verdict"] = "Not enough data (run more seeds)"): APGRunsResponse {
  return {
    runs: [summary],
    jobs,
    verdicts: [
      verdict(verdictText),
      { ...verdict(verdictText), data_setting: "constrained_dirichlet_a01" },
      { ...verdict(verdictText), simulation_mode: "hybrid_shaped_simulation", source_label: "HYBRID-SHAPED" },
      { ...verdict(verdictText), data_setting: "constrained_dirichlet_a01", simulation_mode: "hybrid_shaped_simulation", source_label: "HYBRID-SHAPED" },
    ],
  };
}

function emptyResponse(): APGRunsResponse {
  return { runs: [], jobs: [], verdicts: [] };
}

afterEach(() => {
  cleanup();
  vi.clearAllMocks();
});

describe("APG topology learning UI", () => {
  it("shows a clean empty state when no saved topology runs remain", async () => {
    api.getAPGRuns.mockResolvedValue(emptyResponse());
    api.getAutonomyState.mockResolvedValue({ stop_requested: false });
    render(<APGTopologyPanel token="test-token" />);

    expect(await screen.findByText("No saved topology runs.")).toBeInTheDocument();
    expect(screen.getByRole("combobox", { name: "Saved APG result run" })).toHaveValue("");
    expect(screen.queryByText("Not enough data (run more seeds)")).not.toBeInTheDocument();
  });

  it("shows labeled simulation controls and the permanently unavailable hybrid button", async () => {
    api.getAPGRuns.mockResolvedValue(response());
    api.getAutonomyState.mockResolvedValue({ stop_requested: false });
    api.getAPGRun.mockResolvedValue(detail);
    render(<APGTopologyPanel token="test-token" />);

    expect(await screen.findByText("SIMULATED")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Run APG simulation (PC only)" })).toBeEnabled();
    expect(screen.getByRole("button", { name: "Run APG on hybrid (3 real Jetsons)" })).toBeDisabled();
    expect(screen.getByText("NOT AVAILABLE · Phase C not implemented or approved")).toBeInTheDocument();
    expect(screen.getAllByText(/VALIDATION accuracy/).length).toBeGreaterThan(0);
    expect(screen.getAllByText(/TEST accuracy/).length).toBeGreaterThan(0);
  });

  it("disables PC simulation on global STOP or while another APG job is active", async () => {
    api.getAPGRuns.mockResolvedValue(response());
    api.getAutonomyState.mockResolvedValue({ stop_requested: true });
    const { rerender } = render(<APGTopologyPanel token="test-token" />);
    expect(await screen.findByRole("button", { name: "Run APG simulation (PC only)" })).toBeDisabled();

    api.getAutonomyState.mockResolvedValue({ stop_requested: false });
    api.getAPGRuns.mockResolvedValue(response([{
      job_id: "active-job",
      run_id: "active-job",
      topology: "apg",
      data_setting: "iid_stratified",
      seed: 17,
      simulation_mode: "all_simulated",
      source_label: "SIMULATED",
      status: "running",
      rounds_completed: 4,
      total_rounds: 30,
      created_at: "now",
      started_at: "now",
      completed_at: null,
      duration_seconds: null,
      result_file: null,
      error: null,
    }]));
    rerender(<APGTopologyPanel token="test-token" />);
    await waitFor(() => expect(screen.getByRole("button", { name: "Run APG simulation (PC only)" })).toBeDisabled());
  });

  it("renders one-seed verdict as insufficient and draws source-to-receiver arrows", async () => {
    api.getAPGRuns.mockResolvedValue(response([], "Not enough data (run more seeds)"));
    api.getAutonomyState.mockResolvedValue({ stop_requested: false });
    api.getAPGRun.mockResolvedValue(detail);
    render(<APGTopologyPanel token="test-token" />);

    expect((await screen.findAllByText("Not enough data (run more seeds)")).length).toBeGreaterThan(0);
    expect(await screen.findByRole("img", { name: "apg receiver-centric graph source to receiver" })).toBeInTheDocument();
    const edge = document.querySelector('line[data-source="node_0"][data-receiver="node_1"]');
    expect(edge).toBeInTheDocument();
    expect(edge?.getAttribute("marker-end")).toBe("url(#apg-arrow)");
    expect(screen.getByText(/node_5 · worker1 \(emulated in PC\)/)).toBeInTheDocument();
    expect(screen.getByText(/TRAIN-only Theta summary/)).toBeInTheDocument();
  });

  it("posts one PC simulation request with the selected run configuration", async () => {
    api.getAPGRuns.mockResolvedValue(response());
    api.getAutonomyState.mockResolvedValue({ stop_requested: false });
    api.getAPGRun.mockResolvedValue(detail);
    api.startAPGSimulation.mockResolvedValue({
      job_id: "queued-1",
      run_id: "queued-1",
      topology: "ring",
      data_setting: "constrained_dirichlet_a01",
      seed: 42,
      simulation_mode: "all_simulated",
      source_label: "SIMULATED",
      status: "queued",
      rounds_completed: 0,
      total_rounds: 30,
      created_at: "now",
      started_at: null,
      completed_at: null,
      duration_seconds: null,
      result_file: null,
      error: null,
    });
    render(<APGTopologyPanel token="test-token" />);
    await screen.findByText("SIMULATED");
    fireEvent.change(screen.getByRole("combobox", { name: "APG simulation topology" }), { target: { value: "ring" } });
    fireEvent.change(screen.getByRole("combobox", { name: "APG simulation data setting" }), { target: { value: "constrained_dirichlet_a01" } });
    fireEvent.change(screen.getByRole("combobox", { name: "APG simulation seed" }), { target: { value: "42" } });
    fireEvent.click(screen.getByRole("button", { name: "Run APG simulation (PC only)" }));

    await waitFor(() => expect(api.startAPGSimulation).toHaveBeenCalledWith("test-token", {
      topology: "ring",
      data_setting: "constrained_dirichlet_a01",
      seed: 42,
    }));
  });
});
