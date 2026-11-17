import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import type { ExperimentResults } from "../api/client";
import ExperimentsPage from "./ExperimentsPage";

const results: ExperimentResults = {
  configuration: { rounds: 10, epochs: 5, topology: "ring", dataset_split: "stratified" },
  active_group_id: null,
  topologies: {
    ring: ["worker1", "worker2", "worker3"],
    mesh: [["worker1", "worker2"], ["worker1", "worker3"], ["worker2", "worker3"]],
  },
  groups: [{
    group_id: "ring-seed-test",
    dataset_name: "mnist",
    kind: "decentralized",
    topology: "ring",
    rounds: 10,
    epochs: 5,
    seeds: [17, 42, 2026],
    status: "completed",
    error: null,
    runs: [{
      run_id: "ring-seed-17",
      seed: 17,
      topology: "ring",
      status: "completed",
      rounds: [{
        round: 1,
        topology: "ring",
        workers: {
          worker1: {
            node_id: "worker1",
            run_id: "ring-seed-17",
            round: 1,
            topology: "ring",
            accuracy: 0.75,
            loss: 0.8,
            macro_f1: 0.73,
            macro_precision: 0.72,
            macro_recall: 0.74,
            validation_accuracy: 0.75,
            validation_macro_f1: 0.73,
            validation_macro_precision: 0.72,
            validation_macro_recall: 0.74,
            validation_sample_count: 20,
            training_loss: 0.8,
            peak_process_rss_mb: 820,
            evaluation_seconds: 1.25,
            bytes_sent: 4096,
            bytes_received: 4096,
            exchange_seconds: 0.12,
            peer_wait_seconds: 0.04,
            validation_confusion_matrix_counts: [[9, 1], [0, 10]],
            validation_confusion_matrix_row_normalized_percent: [[90, 10], [0, 100]],
            validation_per_class_precision: [1, 10 / 11],
            validation_per_class_recall: [0.9, 1],
            validation_per_class_f1: [18 / 19, 20 / 21],
          },
          worker2: {
            node_id: "worker2",
            run_id: "ring-seed-17",
            round: 1,
            topology: "ring",
            accuracy: 0.8,
            validation_accuracy: 0.8,
            validation_macro_precision: 0.8,
            validation_macro_recall: 0.8,
            validation_macro_f1: 0.8,
            validation_sample_count: 20,
            training_loss: 0.9,
            peak_process_rss_mb: 800,
            evaluation_seconds: 1.5,
            validation_confusion_matrix_counts: [[8, 2], [2, 8]],
            validation_confusion_matrix_row_normalized_percent: [[80, 20], [20, 80]],
            validation_per_class_precision: [0.8, 0.8],
            validation_per_class_recall: [0.8, 0.8],
            validation_per_class_f1: [0.8, 0.8],
          },
        },
      }],
    }],
  }],
};

afterEach(() => {
  cleanup();
});

describe("Experiments page", () => {
  it("renders real worker test metrics, peer exchange, and topology", () => {
    render(<ExperimentsPage token="token" results={results} error="" onRefresh={vi.fn()} onOpenModelStudio={vi.fn()} />);

    expect(screen.getByRole("button", { name: "View experiment details for ring-seed-test" })).toBeInTheDocument();
    expect(screen.getByText("Best experiment")).toBeInTheDocument();
    expect(screen.queryByText("worker1 → worker2")).not.toBeInTheDocument();
    expect(screen.queryByRole("link", { name: "Download graph" })).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "View experiment details for ring-seed-test" }));

    expect(screen.getByText("ring-seed-test")).toBeInTheDocument();
    expect(screen.getByText("worker1 → worker2")).toBeInTheDocument();
    expect(screen.getAllByText("75.00%").length).toBeGreaterThan(0);
    expect(screen.getAllByText("72.00%").length).toBeGreaterThan(0);
    expect(screen.getAllByText("74.00%").length).toBeGreaterThan(0);
    expect(screen.getAllByText("0.8000").length).toBeGreaterThan(0);
    expect(screen.getAllByText("820.0 MB").length).toBeGreaterThan(0);
    expect(screen.getByText("8192")).toBeInTheDocument();
    expect(screen.getByText("0.120 s")).toBeInTheDocument();
    expect(screen.getByText("0.040 s")).toBeInTheDocument();
    expect(screen.getByRole("link", { name: "Download graph" })).toHaveAttribute(
      "download",
      "ring-seed-17-accuracy.svg",
    );
    expect(screen.getByRole("region", { name: "Combined final validation results" })).toBeInTheDocument();
    expect(screen.getByText("Final validation results · all workers together")).toBeInTheDocument();
    expect(screen.getAllByText("87.50%").length).toBeGreaterThan(0);
    expect(screen.getByText("87.59%")).toBeInTheDocument();
    expect(screen.getByText("40")).toBeInTheDocument();
    expect(screen.getByText("17")).toBeInTheDocument();
    expect(screen.getByText("Combined per-class validation metrics")).toBeInTheDocument();
    expect(screen.getByText("Final-round worker metrics · individual breakdown")).toBeInTheDocument();
    expect(screen.getAllByText("worker1").length).toBeGreaterThan(0);
    expect(screen.getAllByText("worker2").length).toBeGreaterThan(0);
    expect(screen.getByText("Training loss")).toBeInTheDocument();
    expect(screen.getAllByText("Macro F1").length).toBeGreaterThan(0);
    expect(screen.getByText("Samples")).toBeInTheDocument();
    expect(screen.getByText("Combined validation samples")).toBeInTheDocument();
    expect(screen.getByText("1.25 s")).toBeInTheDocument();
    expect(screen.getAllByText("Actual / Predicted").length).toBeGreaterThan(0);
    fireEvent.click(screen.getByRole("button", { name: "Back to experiments" }));
    expect(screen.getByRole("button", { name: "View experiment details for ring-seed-test" })).toBeInTheDocument();
    expect(screen.queryByText("worker1 → worker2")).not.toBeInTheDocument();
  });

  it("renders dataset validation metrics when legacy metric names are absent", () => {
    const datasetResults = structuredClone(results);
    const metrics = datasetResults.groups[0].runs[0].rounds[0].workers.worker1 as unknown as Record<string, unknown>;
    delete metrics.accuracy;
    delete metrics.loss;
    delete metrics.macro_f1;
    metrics.validation_accuracy = 0.62;
    metrics.training_loss = 0.9;
    metrics.validation_macro_f1 = 0.58;

    render(<ExperimentsPage token="token" results={datasetResults} error="" onRefresh={vi.fn()} onOpenModelStudio={vi.fn()} />);
    fireEvent.click(screen.getByRole("button", { name: "View experiment details for ring-seed-test" }));

    expect(screen.getAllByText("62.00%").length).toBeGreaterThan(0);
    expect(screen.getAllByText("0.9000").length).toBeGreaterThan(0);
    expect(screen.getAllByText("58.00%").length).toBeGreaterThan(0);
  });

  it("shows legacy saved groups with their dataset from the installed architecture", () => {
    const legacyResults = structuredClone(results);
    const group = legacyResults.groups[0] as unknown as Record<string, unknown>;
    delete group.dataset_name;
    group.architecture = { dataset_name: "mnist" };

    render(<ExperimentsPage token="token" results={legacyResults} error="" onRefresh={vi.fn()} onOpenModelStudio={vi.fn()} />);

    fireEvent.click(screen.getByRole("button", { name: "View experiment details for ring-seed-test" }));
    expect(screen.getByText("ring-seed-test")).toBeInTheDocument();
    expect(screen.getByText((_, element) =>
      Boolean(element?.tagName === "P" && element.textContent?.includes("mnist · 10 rounds · 5 local epochs")),
    )).toBeInTheDocument();
  });

  it("renders dataset NAS groups with a singular seed and no round configuration", () => {
    const datasetResults = structuredClone(results);
    const group = datasetResults.groups[0] as unknown as Record<string, unknown>;
    delete group.seeds;
    delete group.rounds;
    delete group.epochs;
    group.kind = "dataset_architecture_search";
    group.seed = 42;

    render(<ExperimentsPage token="token" results={datasetResults} error="" onRefresh={vi.fn()} onOpenModelStudio={vi.fn()} />);

    fireEvent.click(screen.getByRole("button", { name: "View experiment details for ring-seed-test" }));
    expect(screen.getByText("dataset architecture search")).toBeInTheDocument();
    expect(screen.getByText(/seed 42/)).toBeInTheDocument();
  });

  it("routes new training through Model Studio", () => {
    const onOpenModelStudio = vi.fn();
    render(<ExperimentsPage token="token" results={results} error="" onRefresh={vi.fn()} onOpenModelStudio={onOpenModelStudio} />);
    fireEvent.click(screen.getByRole("button", { name: "View experiment details for ring-seed-test" }));
    fireEvent.click(screen.getByRole("button", { name: "Open Model Studio" }));
    expect(onOpenModelStudio).toHaveBeenCalledOnce();
  });

  it("deletes a saved experiment group after confirmation", async () => {
    const fetchMock = vi.fn().mockResolvedValue({ ok: true, status: 204 });
    const onRefresh = vi.fn();
    vi.stubGlobal("fetch", fetchMock);
    vi.stubGlobal("confirm", vi.fn(() => true));
    render(<ExperimentsPage token="token" results={results} error="" onRefresh={onRefresh} onOpenModelStudio={vi.fn()} />);

    fireEvent.click(screen.getByRole("button", { name: "View experiment details for ring-seed-test" }));
    fireEvent.click(screen.getByRole("button", { name: "Delete experiment" }));

    await waitFor(() => {
      expect(fetchMock).toHaveBeenCalledWith(
        "http://localhost:8000/api/slm/results/ring-seed-test",
        expect.objectContaining({ method: "DELETE" }),
      );
    });
    expect(onRefresh).toHaveBeenCalledOnce();
    expect(screen.queryByRole("button", { name: "Delete experiment" })).not.toBeInTheDocument();
  });

  it("deletes multiple selected saved experiments", async () => {
    const secondGroup = structuredClone(results.groups[0]);
    secondGroup.group_id = "second-experiment";
    const multiResults = { ...structuredClone(results), groups: [...results.groups, secondGroup] };
    const fetchMock = vi.fn().mockResolvedValue({ ok: true, status: 204 });
    const onRefresh = vi.fn();
    vi.stubGlobal("fetch", fetchMock);
    vi.stubGlobal("confirm", vi.fn(() => true));
    render(<ExperimentsPage token="token" results={multiResults} error="" onRefresh={onRefresh} onOpenModelStudio={vi.fn()} />);

    fireEvent.click(screen.getByRole("checkbox", { name: "Select all experiments" }));
    expect(screen.getByText("2 selected")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Delete selected" }));

    await waitFor(() => {
      expect(fetchMock).toHaveBeenCalledTimes(2);
      expect(onRefresh).toHaveBeenCalledOnce();
    });
    expect(fetchMock.mock.calls.map(([url]) => String(url)).sort()).toEqual([
      "http://localhost:8000/api/slm/results/ring-seed-test",
      "http://localhost:8000/api/slm/results/second-experiment",
    ]);
  });

  it("marks the completed experiment with the highest final validation accuracy as best", () => {
    const comparedResults = structuredClone(results);
    const lowerAccuracyGroup = structuredClone(comparedResults.groups[0]);
    lowerAccuracyGroup.group_id = "lower-accuracy";
    lowerAccuracyGroup.runs[0].rounds[0].workers.worker1.validation_accuracy = 0.6;
    lowerAccuracyGroup.runs[0].rounds[0].workers.worker2.validation_accuracy = 0.7;
    comparedResults.groups.push(lowerAccuracyGroup);

    render(<ExperimentsPage token="token" results={comparedResults} error="" onRefresh={vi.fn()} onOpenModelStudio={vi.fn()} />);

    const bestExperiment = screen.getByRole("button", { name: "View experiment details for ring-seed-test" });
    const lowerExperiment = screen.getByRole("button", { name: "View experiment details for lower-accuracy" });
    expect(bestExperiment).toHaveTextContent("Best experiment");
    expect(lowerExperiment).not.toHaveTextContent("Best experiment");
  });
});
