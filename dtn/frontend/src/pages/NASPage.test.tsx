import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import type { ExperimentResults } from "../api/client";
import NASPage from "./NASPage";

vi.mock("../api/client", () => ({
  sendSlmCommand: vi.fn().mockResolvedValue({
    result: { available: true, data: { group_id: "nas-pilot" } },
  }),
}));

const results: ExperimentResults = {
  configuration: { rounds: 10, epochs: 5, topology: "bidirectional_ring", dataset_split: "stratified" },
  active_group_id: null,
  topologies: {},
  groups: [{
    group_id: "nas-pilot",
    kind: "neural_architecture_search",
    topology: "bidirectional_ring",
    rounds: 1,
    epochs: 1,
    seeds: [17],
    status: "completed",
    error: null,
    runs: [],
    architecture_space: [],
    selected_architecture: { id: "b1_c8_k3", blocks: 1, channels: 8, kernel_size: 3 },
    nas_candidates: [{
      architecture: { id: "b1_c8_k3", blocks: 1, channels: 8, kernel_size: 3 },
      parameter_count: 5354,
      mean_validation_accuracy: 0.45,
      worker_accuracies: [0.4, 0.5, 0.45],
      mean_elapsed_seconds: 7,
      max_peak_rss_mb: 800,
    }],
    comparison: {
      fixed: {
        completed_runs: 1, mean_accuracy: 0.4, mean_macro_f1: 0.3,
        mean_training_runtime_seconds: 10, mean_runtime_including_search_seconds: 10,
        total_runtime_including_search_seconds: 10,
        search_seconds: 0, peak_rss_mb: 800, mean_communication_bytes: 3000,
      },
      nas: {
        completed_runs: 1, mean_accuracy: 0.45, mean_macro_f1: 0.35,
        mean_training_runtime_seconds: 11, mean_runtime_including_search_seconds: 21,
        total_runtime_including_search_seconds: 21,
        search_seconds: 10, peak_rss_mb: 900, mean_communication_bytes: 4000,
      },
      centralized_fixed: {
        completed_runs: 1, mean_accuracy: 0.5, mean_macro_f1: 0.4,
        mean_training_runtime_seconds: 9, mean_runtime_including_search_seconds: 9,
        total_runtime_including_search_seconds: 9,
        search_seconds: 0, peak_rss_mb: 700, mean_communication_bytes: 0,
      },
    },
    nas_beats_fixed: true,
  }],
};

describe("NASPage", () => {
  afterEach(() => cleanup());

  it("shows candidates, the common model choice, and the experiment comparison", () => {
    render(<NASPage token="token" results={results} error="" onRefresh={vi.fn()} />);
    expect(screen.getByText(/Selected common architecture/)).toBeTruthy();
    expect(screen.getAllByText("45.00%")).toHaveLength(2);
    expect(screen.getByText("NAS-selected model outperformed the fixed DFL model on mean held-out accuracy.")).toBeTruthy();
    expect(screen.getByText("Centralized fixed")).toBeTruthy();
  });

  it("starts the small search and one-round pilot", async () => {
    const { sendSlmCommand } = await import("../api/client");
    render(<NASPage token="token" results={results} error="" onRefresh={vi.fn()} />);
    fireEvent.click(screen.getByRole("button", { name: "Run search + 1-round pilot" }));
    await waitFor(() => expect(sendSlmCommand).toHaveBeenCalledWith("token", {
      action: "enable_nas",
      parameters: {
        enabled: true,
        rounds: 1,
        epochs: 1,
        seeds: [17],
        search_time_limit_seconds: 300,
      },
    }));
  });
});
