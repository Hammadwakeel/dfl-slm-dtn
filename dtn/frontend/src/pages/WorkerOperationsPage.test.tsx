import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import WorkerOperationsPage from "./WorkerOperationsPage";
import type { Device } from "../types";

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

const worker: Device = {
  node_id: "worker1",
  role: "worker",
  hostname: "dfl-worker1",
  ip: "192.168.11.123",
  port: 5001,
  online: true,
  last_seen: null,
  last_error: null,
  latest_status: {
    node_id: "worker1",
    role: "worker",
    hostname: "dfl-worker1",
    sampled_at: new Date().toISOString(),
    cpu_utilization_percent: 5,
    gpu_utilization_percent: 0,
    memory: { total_bytes: 8_000_000_000, available_bytes: 1_000_000_000, used_bytes: 7_000_000_000 },
    temperatures_celsius: {},
  },
};

const snapshot = {
  node_id: "worker1",
  sampled_at: new Date().toISOString(),
  memory_available_mb: 950,
  nas_active: false,
  training: {
    active_run_id: "run-1",
    scheduled_run_id: "run-1",
    active_round: 2,
    active_epoch: 1,
    total_epochs: 3,
    active_phase: "local_training",
  },
  processes: [
    { pid: 123, ppid: 1, user: "hammad", cpu_percent: 6, memory_percent: 34, rss_mb: 2048, name: "python" },
  ],
};

describe("WorkerOperationsPage", () => {
  it("shows process usage and requests DFL training cancellation", async () => {
    const fetchMock = vi.fn().mockImplementation((input: RequestInfo | URL) => {
      const url = String(input);
      const data = url.endsWith("/api/devices/workers/processes")
        ? { workers: [snapshot] }
        : { accepted: true, run_id: "run-1" };
      return Promise.resolve({ ok: true, status: 200, json: async () => data });
    });
    vi.stubGlobal("fetch", fetchMock);
    vi.stubGlobal("confirm", vi.fn(() => true));

    render(<WorkerOperationsPage token="test-token" devices={[worker]} />);

    expect(await screen.findByText("2048 MB")).toBeInTheDocument();
    expect(screen.getByText("python")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Stop DFL training" }));

    expect(await screen.findByText("Cancellation requested for run-1.")).toBeInTheDocument();
    expect(fetchMock).toHaveBeenCalledWith(
      "http://localhost:8000/api/devices/worker1/stop-training",
      expect.objectContaining({ method: "POST" }),
    );
    await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(3));
  });

  it("does not allow reboot during an active NAS search", async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      status: 200,
      json: async () => ({
        workers: [{ ...snapshot, nas_active: true, training: { ...snapshot.training, active_run_id: null, scheduled_run_id: null } }],
      }),
    });
    vi.stubGlobal("fetch", fetchMock);

    render(<WorkerOperationsPage token="test-token" devices={[worker]} />);

    expect(await screen.findByText("Running")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Reboot Jetson" })).toBeDisabled();
  });

  it("requests cancellation of an active NAS search", async () => {
    const fetchMock = vi.fn().mockImplementation((input: RequestInfo | URL) => {
      const url = String(input);
      return Promise.resolve({
        ok: true,
        status: 200,
        json: async () => url.endsWith("/api/devices/workers/processes")
          ? {
              workers: [{
                ...snapshot,
                nas_active: true,
                training: { ...snapshot.training, active_run_id: null, scheduled_run_id: null },
              }],
            }
          : { accepted: true, message: "NAS search cancellation requested." },
      });
    });
    vi.stubGlobal("fetch", fetchMock);
    vi.stubGlobal("confirm", vi.fn(() => true));

    render(<WorkerOperationsPage token="test-token" devices={[worker]} />);
    fireEvent.click(await screen.findByRole("button", { name: "Stop NAS search" }));

    expect(await screen.findByText("NAS search cancellation requested.")).toBeInTheDocument();
    expect(fetchMock).toHaveBeenCalledWith(
      "http://localhost:8000/api/devices/worker1/stop-nas",
      expect.objectContaining({ method: "POST" }),
    );
  });

  it("shows individual worker API failures without crashing the page", async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      status: 200,
      json: async () => ({
        workers: [{ node_id: "worker1", error: "worker1 returned HTTP 404: process endpoint missing" }],
      }),
    });
    vi.stubGlobal("fetch", fetchMock);

    render(<WorkerOperationsPage token="test-token" devices={[worker]} />);

    expect(await screen.findByText("worker1 returned HTTP 404: process endpoint missing")).toBeInTheDocument();
    expect(screen.getByText("Unavailable")).toBeInTheDocument();
  });

  it("explains when the coordinator has not loaded the worker operations API", async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      ok: false,
      status: 404,
      json: async () => ({ detail: "Not Found" }),
    });
    vi.stubGlobal("fetch", fetchMock);

    render(<WorkerOperationsPage token="test-token" devices={[worker]} />);

    expect(await screen.findByText(
      "The coordinator returned 404 for worker process monitoring. Install the updated backend and refresh this page.",
    )).toBeInTheDocument();
    await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(1));
  });
});
