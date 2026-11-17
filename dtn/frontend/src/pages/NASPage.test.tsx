import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import type { ExperimentResults } from "../api/client";
import NASPage from "./NASPage";

vi.mock("../api/client", () => ({
  cleanupWorkerMemory: vi.fn().mockResolvedValue({
    complete: true,
    phase: "cleanup",
    message: "Memory cleanup completed on all workers.",
    workers: [{
      node_id: "worker1",
      success: true,
      cleaned_runs: 2,
      process_memory_cleaned: true,
      memory_available_before_mb: 1200,
      memory_available_after_mb: 1600,
      process_rss_before_mb: 700,
      process_rss_after_mb: 500,
      os_page_cache_dropped: true,
    }],
  }),
  deleteExperimentGroup: vi.fn().mockResolvedValue(undefined),
  dispatchAutonomyAction: vi.fn().mockResolvedValue({
    status: "executed",
    result: { available: true, data: { group_id: "dataset-nas-pilot" } },
  }),
  sendSlmCommand: vi.fn().mockResolvedValue({
    status: "executed",
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
  afterEach(() => {
    cleanup();
    vi.unstubAllGlobals();
  });

  it("cleans worker process RAM and reports whether the NAS memory reserve is ready", async () => {
    const { cleanupWorkerMemory } = await import("../api/client");
    const onRefresh = vi.fn();
    vi.stubGlobal("confirm", vi.fn(() => true));
    render(<NASPage token="token" results={results} error="" onRefresh={onRefresh} onUseSelectedArchitecture={vi.fn()} />);

    fireEvent.click(screen.getByRole("button", { name: "Clean worker RAM for next experiment" }));

    await waitFor(() => expect(cleanupWorkerMemory).toHaveBeenCalledWith("token"));
    expect(await screen.findByText("Worker RAM cleanup complete")).toBeInTheDocument();
    expect(screen.getByText(/Process RAM 700 → 500 MB \(\+200 MB released\)/)).toBeInTheDocument();
    expect(screen.getByText(/NAS 1 GiB reserve ready/)).toBeInTheDocument();
    expect(onRefresh).toHaveBeenCalledOnce();
  });

  it("shows candidates, the common model choice, and the experiment comparison", () => {
    render(<NASPage token="token" results={results} error="" onRefresh={vi.fn()} onUseSelectedArchitecture={vi.fn()} />);
    expect(screen.getByText("BEST MODEL · VALIDATION SELECTED")).toBeInTheDocument();
    expect(screen.getByText("BEST MODEL · VALIDATION SELECTED")).toBeInTheDocument();
    expect(screen.getByRole("img", { name: /Selected architecture · b1_c8_k3/ })).toBeInTheDocument();
    expect(screen.getAllByText("45.00%")).toHaveLength(2);
    expect(screen.getByText("NAS vs fixed · held-out accuracy: NAS higher.")).toBeTruthy();
    expect(screen.getByText("Centralized fixed")).toBeTruthy();
    const tables = screen.getAllByRole("table");
    expect(tables[0].querySelector("tbody tr")).toHaveClass("best-result-row");
    expect(tables[0].querySelector("tbody tr")).toHaveTextContent("BEST MODEL");
    const bestComparison = [...tables[1].querySelectorAll("tbody tr")]
      .find((row) => row.classList.contains("best-result-row"));
    expect(bestComparison).toHaveTextContent("BEST HELD-OUT ACCURACY");
    expect(bestComparison).toHaveTextContent("Centralized fixed");
  });

  it("starts the small search and one-round pilot", async () => {
    const { dispatchAutonomyAction } = await import("../api/client");
    render(<NASPage token="token" results={results} error="" onRefresh={vi.fn()} onUseSelectedArchitecture={vi.fn()} />);
    fireEvent.click(screen.getByRole("button", { name: "Search architectures" }));
    await waitFor(() => expect(dispatchAutonomyAction).toHaveBeenCalledWith("token", {
      action: "enable_nas",
      parameters: {
        enabled: true,
        dataset_name: "mnist",
        search_time_limit_seconds: 600,
      },
    }, "Run the full 10-minute MNIST Pareto NAS search on all registered workers using validation data.", {
      dataset_name: "mnist",
      search_time_limit_seconds: 600,
    }));
    expect(await screen.findByRole("status")).toHaveTextContent("dataset-nas-pilot");
  });

  it("starts a bounded 90-second MNIST NAS smoke test", async () => {
    const { dispatchAutonomyAction } = await import("../api/client");
    vi.mocked(dispatchAutonomyAction).mockClear();
    render(<NASPage token="token" results={results} error="" onRefresh={vi.fn()} onUseSelectedArchitecture={vi.fn()} />);

    fireEvent.click(screen.getByRole("button", {
      name: "Run 90-second smoke test",
    }));

    await waitFor(() => expect(dispatchAutonomyAction).toHaveBeenCalledWith("token", {
      action: "enable_nas",
      parameters: {
        enabled: true,
        dataset_name: "mnist",
        search_time_limit_seconds: 90,
        smoke_test: true,
      },
    }, "Run a 90-second MNIST NAS smoke test using one lightweight candidate for one epoch on all registered workers.", {
      dataset_name: "mnist",
      search_time_limit_seconds: 90,
      smoke_test: true,
    }));
    expect(await screen.findByRole("status")).toHaveTextContent("MNIST NAS smoke test dataset-nas-pilot started");
  });

  it("renders the real MNIST dataset search result group", () => {
    const datasetResults = structuredClone(results);
    datasetResults.groups[0].group_id = "dataset-nas-pilot";
    datasetResults.groups[0].kind = "dataset_architecture_search";
    datasetResults.groups[0].dataset_name = "mnist";
    datasetResults.groups[0].seed = 2026;
    delete datasetResults.groups[0].rounds;
    delete datasetResults.groups[0].epochs;
    const datasetCandidate = datasetResults.groups[0].nas_candidates?.[0];
    if (!datasetCandidate) throw new Error("Expected the NAS fixture to include a candidate.");
    delete datasetCandidate.mean_elapsed_seconds;
    delete datasetCandidate.max_peak_rss_mb;

    const onUseSelectedArchitecture = vi.fn();
    const onViewCandidate = vi.fn();
    render(<NASPage token="token" results={datasetResults} error="" onRefresh={vi.fn()} onUseSelectedArchitecture={onUseSelectedArchitecture} onViewCandidate={onViewCandidate} />);

    expect(screen.getByText("BEST MODEL · VALIDATION SELECTED")).toBeInTheDocument();
    expect(screen.getByText((_, element) =>
      Boolean(element?.tagName === "P" && element.textContent?.includes("dataset-nas-pilot · mnist · completed · bidirectional_ring topology · seed 2026")),
    )).toBeInTheDocument();
    expect(screen.getAllByText("—")).toHaveLength(6);
    fireEvent.click(screen.getByRole("button", { name: "Use selected architecture in Model Studio" }));
    expect(onUseSelectedArchitecture).toHaveBeenCalledWith(datasetResults.groups[0].selected_architecture);
    expect(screen.getByRole("img", { name: /Selected architecture · b1_c8_k3/ })).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Show architecture for b1_c8_k3" }));
    expect(screen.getByRole("img", { name: /Candidate architecture · b1_c8_k3/ })).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "View details for b1_c8_k3" }));
    expect(onViewCandidate).toHaveBeenCalledWith(
      datasetCandidate,
      datasetResults.groups[0],
    );
  });

  it("shows candidate results as each worker finishes before aggregate selection", () => {
    const activeResults = structuredClone(results);
    const activeGroup = activeResults.groups[0];
    activeGroup.status = "running";
    delete activeGroup.selected_architecture;
    activeGroup.nas_candidates = [];
    activeGroup.worker_search = [{
      node_id: "worker1",
      elapsed_seconds: 4.2,
      peak_rss_mb: 950,
      candidates: [{
        architecture: { id: "b1_c8_k3", blocks: 1, channels: 8, kernel_size: 3 },
        parameter_count: 5354,
        validation_accuracy: 0.45,
        validation_samples: 1000,
        elapsed_seconds: 2.1,
        peak_rss_mb: 950,
      }],
    }];
    activeGroup.worker_search_progress = {
      worker1: {
        status: "completed",
        started_at: 100,
        completed_at: 104.2,
        candidate_count: 1,
        elapsed_seconds: 4.2,
      },
      worker2: { status: "searching", started_at: 100 },
      worker3: { status: "searching", started_at: 100 },
    };

    render(<NASPage token="token" results={activeResults} error="" onRefresh={vi.fn()} onUseSelectedArchitecture={vi.fn()} />);

    expect(screen.getByText("3/3 workers reported · 1 results received.")).toBeInTheDocument();
    expect(screen.getByRole("group", { name: "Worker status details" })).toHaveTextContent("worker1");
    expect(screen.getByRole("group", { name: "Worker status details" })).toHaveTextContent("Results received");
    expect(screen.getByRole("group", { name: "Worker status details" })).toHaveTextContent("worker2");
    expect(screen.getByRole("group", { name: "Worker status details" })).toHaveTextContent("Searching candidates");
    expect(screen.getByRole("heading", { name: "Worker search results" })).toBeInTheDocument();
    expect(screen.getAllByText("worker1")).toHaveLength(2);
    expect(screen.getAllByText("45.00%")).toHaveLength(2);
    expect(screen.getByRole("progressbar", { name: "Worker results received" })).toHaveAttribute("aria-valuenow", "1");
  });

  it("explains that candidate metrics are pending while a running search has no worker results yet", () => {
    const activeResults = structuredClone(results);
    activeResults.active_group_id = "nas-pilot";
    activeResults.groups[0].status = "running";
    activeResults.groups[0].nas_candidates = [];
    delete activeResults.groups[0].selected_architecture;
    activeResults.groups[0].worker_search = [];

    render(<NASPage token="token" results={activeResults} error="" onRefresh={vi.fn()} onUseSelectedArchitecture={vi.fn()} />);

    expect(screen.getByRole("heading", { name: "Architecture search is in progress" })).toBeInTheDocument();
    expect(screen.getByText("Run accepted · waiting for worker updates.")).toBeInTheDocument();
    expect(screen.getByText("LIVE")).toBeInTheDocument();
    expect(screen.getByRole("progressbar", { name: "Worker results received" })).toHaveAttribute("aria-valuenow", "0");
    expect(screen.getByText("Progress: worker results received, not training percentage.")).toBeInTheDocument();
    expect(screen.getByRole("group", { name: "Worker status details" })).toHaveTextContent("No update received");
    expect(screen.getByText("No combined candidate scores yet")).toBeInTheDocument();
  });

  it("keeps earlier NAS results visible while a newer search has no results yet", () => {
    const mixedResults = structuredClone(results);
    const previousGroup = structuredClone(mixedResults.groups[0]);
    previousGroup.group_id = "previous-nas-result";
    const activeGroup = structuredClone(mixedResults.groups[0]);
    activeGroup.group_id = "latest-nas-search";
    activeGroup.status = "running";
    delete activeGroup.selected_architecture;
    activeGroup.nas_candidates = [];
    delete activeGroup.comparison;
    mixedResults.active_group_id = activeGroup.group_id;
    mixedResults.groups = [previousGroup, activeGroup];

    render(<NASPage token="token" results={mixedResults} error="" onRefresh={vi.fn()} onUseSelectedArchitecture={vi.fn()} />);

    expect(screen.getByText(/latest-nas-search · MNIST · running/)).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "Previous NAS results" })).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "previous-nas-result" })).toBeInTheDocument();
    expect(screen.getByText("45.00%")).toBeInTheDocument();
  });

  it("starts NAS without an approval round-trip", async () => {
    const { dispatchAutonomyAction } = await import("../api/client");
    render(
      <NASPage token="token" results={results} error="" onRefresh={vi.fn()} onUseSelectedArchitecture={vi.fn()} />,
    );
    fireEvent.click(screen.getByRole("button", { name: "Search architectures" }));
    expect(await screen.findByText(/MNIST architecture search dataset-nas-pilot started/)).toBeInTheDocument();
    expect(dispatchAutonomyAction).toHaveBeenCalledOnce();
  });

  it("deletes a completed NAS result after confirmation", async () => {
    const { deleteExperimentGroup } = await import("../api/client");
    vi.stubGlobal("confirm", vi.fn(() => true));
    render(<NASPage token="token" results={results} error="" onRefresh={vi.fn()} onUseSelectedArchitecture={vi.fn()} />);

    fireEvent.click(screen.getByRole("button", { name: "Delete NAS result" }));

    await waitFor(() => expect(deleteExperimentGroup).toHaveBeenCalledWith("token", "nas-pilot"));
  });
});
