import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import type { ExperimentResults, SlmCommandResult } from "../api/client";
import ExperimentsPage from "./ExperimentsPage";

const { sendSlmCommand, startCentralizedBaseline } = vi.hoisted(() => ({
  sendSlmCommand: vi.fn(),
  startCentralizedBaseline: vi.fn(),
}));

vi.mock("../api/client", () => ({
  sendSlmCommand,
  startCentralizedBaseline,
}));

const results: ExperimentResults = {
  configuration: { rounds: 10, epochs: 5, topology: "ring", dataset_split: "stratified" },
  active_group_id: null,
  topologies: {
    ring: ["worker1", "worker2", "worker3"],
    mesh: [["worker1", "worker2"], ["worker1", "worker3"], ["worker2", "worker3"]],
  },
  groups: [{
    group_id: "ring-seed-test",
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
            peak_process_rss_mb: 820,
            bytes_sent: 4096,
            bytes_received: 4096,
            exchange_seconds: 0.12,
          },
        },
      }],
    }],
  }],
};

afterEach(() => {
  cleanup();
  vi.clearAllMocks();
});

describe("Experiments page", () => {
  it("renders real worker test metrics, peer exchange, and topology", () => {
    render(<ExperimentsPage token="test-token" results={results} error="" onRefresh={vi.fn()} />);

    expect(screen.getByText("ring-seed-test")).toBeInTheDocument();
    expect(screen.getByText("worker1 → worker2")).toBeInTheDocument();
    expect(screen.getByText("75.00%")).toBeInTheDocument();
    expect(screen.getByText("0.8000")).toBeInTheDocument();
    expect(screen.getByText("820.0 MB")).toBeInTheDocument();
    expect(screen.getByText("8192")).toBeInTheDocument();
    expect(screen.getByText("0.120 s")).toBeInTheDocument();
  });

  it("configures training through the authenticated SLM action route", async () => {
    sendSlmCommand.mockResolvedValue({
      result: { available: true, data: { group_id: "new-ring-run" } },
    } satisfies Partial<SlmCommandResult>);
    render(<ExperimentsPage token="test-token" results={results} error="" onRefresh={vi.fn()} />);

    fireEvent.click(screen.getByRole("button", { name: "Start distributed" }));

    await waitFor(() => expect(sendSlmCommand).toHaveBeenCalledTimes(4));
    expect(sendSlmCommand).toHaveBeenNthCalledWith(1, "test-token", {
      action: "set_topology",
      parameters: { topology: "ring" },
    });
    expect(sendSlmCommand).toHaveBeenNthCalledWith(4, "test-token", {
      action: "start_training",
      parameters: { rounds: 10, epochs: 5, dataset_split: "dirichlet_a03" },
    });
    expect(startCentralizedBaseline).not.toHaveBeenCalled();
  });
});
