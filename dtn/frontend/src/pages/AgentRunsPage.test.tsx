import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import AgentRunsPage from "./AgentRunsPage";

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

describe("AgentRunsPage", () => {
  it("shows the running workflow loop and omits the standalone task launcher", async () => {
    const fetchMock = vi.fn().mockImplementation((input: RequestInfo | URL) => {
      const url = String(input);
      if (url.endsWith("/api/datasets")) {
        return Promise.resolve({ ok: true, json: async () => [] });
      }
      if (url.endsWith("/api/autonomy/state")) {
        return Promise.resolve({
          ok: true,
          json: async () => ({
            stop_requested: false,
            loops: [{
              id: "active-loop-42",
              workflow_id: "44444444-4444-4444-8444-444444444444",
              status: "RUNNING",
              iteration: 2,
              max_iterations: 8,
              time_limit_seconds: 3600,
              stop_rule: "target_metric",
              target_metric: 0.9,
              best_metric: 0.73,
              no_improvement_count: 0,
              failure_reason: null,
              created_at: "2026-11-09T08:00:00Z",
              deadline_at: "2026-11-09T09:00:00Z",
              updated_at: "2026-11-09T08:10:00Z",
            }],
          }),
        });
      }
      if (url.endsWith("/api/slm/results")) {
        return Promise.resolve({
          ok: true,
          json: async () => ({
            configuration: { rounds: 1, epochs: 1, topology: "ring", dataset_split: "stratified" },
            active_group_id: null,
            topologies: {},
            groups: [],
          }),
        });
      }
      if (url.includes("/api/autonomy/history") || url.endsWith("/api/slm/conversations")) {
        return Promise.resolve({ ok: true, json: async () => [] });
      }
      return Promise.resolve({ ok: true, json: async () => ({}) });
    });
    vi.stubGlobal("fetch", fetchMock);

    render(<AgentRunsPage token="test-token" active />);

    expect(await screen.findByText("Loop active-loop-42 is running")).toBeInTheDocument();
    expect(screen.getByText(/Active loop: active-loop-42/)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Kill loop active-loop-42" })).toBeEnabled();
    expect(screen.queryByLabelText("Task for the Agent")).not.toBeInTheDocument();
    expect(screen.queryByRole("heading", { name: "Run history" })).not.toBeInTheDocument();
    expect(screen.queryByText(/Checkpoints and logs are stored in the coordinator database/)).not.toBeInTheDocument();
    expect(fetchMock.mock.calls.some(([url]) => String(url).endsWith("/api/slm/agent/runs"))).toBe(false);
  });

  it("does not claim that no loop is running when coordinator status is unavailable", async () => {
    vi.stubGlobal("fetch", vi.fn().mockRejectedValue(
      new Error("Coordinator request timed out. Verify the current loop status before retrying."),
    ));

    render(<AgentRunsPage token="test-token" active />);

    expect(await screen.findByText(/Could not refresh saved workflow details:/)).toBeInTheDocument();
    expect(screen.getByText("Coordinator unavailable.")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Start Agent" })).toBeDisabled();
    expect(screen.getAllByRole("alert")).toHaveLength(1);
  });
});
