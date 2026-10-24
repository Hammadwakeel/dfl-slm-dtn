import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import App from "./App";
import type { Device } from "./types";

const nodes: Device[] = [
  ["master", "master", "dfl-master", "192.168.11.15", 5002],
  ["worker1", "worker", "dfl-worker1", "192.168.11.123", 5001],
  ["worker2", "worker", "dfl-worker2", "192.168.11.14", 5001],
  ["worker3", "worker", "dfl-worker3", "192.168.11.13", 5001],
].map(([node_id, role, hostname, ip, port]) => ({
  node_id: String(node_id),
  role: role as Device["role"],
  hostname: String(hostname),
  ip: String(ip),
  port: Number(port),
  online: true,
  last_seen: "2026-10-24T12:00:00Z",
  last_error: null,
  latest_status: {
    node_id: String(node_id),
    role: role as Device["role"],
    hostname: String(hostname),
    sampled_at: "2026-10-24T12:00:00Z",
    cpu_utilization_percent: 10,
    gpu_utilization_percent: 2,
    memory: { total_bytes: 1000, available_bytes: 500, used_bytes: 500 },
    temperatures_celsius: { cpu: 45 },
  },
}));

afterEach(() => {
  cleanup();
  sessionStorage.clear();
  vi.unstubAllGlobals();
});

describe("Dashboard", () => {
  it("loads all four devices from the coordinator API", async () => {
    sessionStorage.setItem("dtn-api-token", "test-session-token");
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => nodes,
    });
    class MockWebSocket extends EventTarget {
      constructor() {
        super();
        window.setTimeout(() => this.dispatchEvent(new Event("open")), 0);
      }
      send() {}
      close() {}
    }
    vi.stubGlobal("fetch", fetchMock);
    vi.stubGlobal("WebSocket", MockWebSocket);

    render(<App />);

    expect(await screen.findByText("dfl-master")).toBeInTheDocument();
    expect(screen.getByText("dfl-worker1")).toBeInTheDocument();
    expect(screen.getByText("dfl-worker2")).toBeInTheDocument();
    expect(screen.getByText("dfl-worker3")).toBeInTheDocument();
    expect(fetchMock).toHaveBeenCalledWith(
      "http://localhost:8000/api/devices",
      expect.objectContaining({
        headers: { Authorization: "Bearer test-session-token" },
      }),
    );
  });
});
