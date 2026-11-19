import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { StrictMode } from "react";

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
  it("preserves dataset controls and selection when navigating between pages", async () => {
    sessionStorage.setItem("dtn-api-token", "test-session-token");
    const datasets = [
      {
        dataset_name: "mnist",
        classes: ["0", "1"],
        class_counts: { "0": 1, "1": 1 },
        image_sizes: { "0": { "28x28": 1 }, "1": { "28x28": 1 } },
        image_channels: { "0": { "1": 1 }, "1": { "1": 1 } },
        image_count: 2,
      },
      {
        dataset_name: "cifar10",
        classes: ["cat", "dog"],
        class_counts: { cat: 1, dog: 1 },
        image_sizes: { cat: { "32x32": 1 }, dog: { "32x32": 1 } },
        image_channels: { cat: { "3": 1 }, dog: { "3": 1 } },
        image_count: 2,
      },
    ];
    const fetchMock = vi.fn().mockImplementation((input: RequestInfo | URL) => {
      const url = String(input);
      let data: unknown;
      let status = 200;
      if (url.endsWith("/api/datasets")) data = datasets;
      else if (url.endsWith("/api/autonomy/state")) data = { stop_requested: false, loops: [] };
      else if (url.includes("/api/slm/results?limit=")) {
        data = {
          configuration: { rounds: 1, epochs: 1, topology: "ring", dataset_split: "iid" },
          active_group_id: null,
          topologies: {},
          groups: [],
        };
      } else if (url.endsWith("/api/apg/runs")) data = { runs: [], jobs: [], verdicts: [] };
      else if (url.endsWith("/api/devices")) data = nodes;
      else if (url.includes("/preprocessing-report")) {
        status = 404;
        data = { detail: "No report" };
      } else if (url.includes("/samples")) data = [];
      else data = { detail: "Unexpected test request" };
      return Promise.resolve({
        ok: status >= 200 && status < 300,
        status,
        json: async () => data,
      });
    });
    vi.stubGlobal("fetch", fetchMock);
    vi.stubGlobal("WebSocket", class extends EventTarget {
      send() {}
      close() {}
    });

    render(<App />);
    fireEvent.click(await screen.findByRole("button", { name: "Datasets" }));
    const datasetSelect = await screen.findByRole("combobox", { name: "Dataset" });
    fireEvent.change(datasetSelect, { target: { value: "cifar10" } });
    const splitSelect = screen.getByRole("combobox", { name: "Split strategy" });
    fireEvent.change(splitSelect, { target: { value: "dirichlet_a07" } });

    fireEvent.click(screen.getByRole("button", { name: "Devices" }));
    expect(screen.getByRole("button", { name: "Devices" })).toHaveAttribute("aria-current", "page");
    fireEvent.change(screen.getByLabelText("Sequential worker ID"), {
      target: { value: "worker5" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Datasets" }));

    expect(screen.getByRole("combobox", { name: "Dataset" })).toHaveValue("cifar10");
    expect(screen.getByRole("combobox", { name: "Split strategy" })).toHaveValue("dirichlet_a07");
    fireEvent.click(screen.getByRole("button", { name: "Devices" }));
    expect(screen.getByLabelText("Sequential worker ID")).toHaveValue("worker5");
  });

  it("loads all four devices from the coordinator API", async () => {
    sessionStorage.setItem("dtn-api-token", "test-session-token");
    const additionalWorker: Device = {
      ...nodes[1],
      node_id: "lab-probe",
      hostname: "Lab probe",
      ip: "127.0.0.1",
      port: 5101,
      is_managed: true,
    };
    const allNodes = [...nodes, additionalWorker];
    const fetchMock = vi.fn().mockImplementation((input: RequestInfo | URL) => {
      const url = String(input);
      const data = url.includes("/api/slm/results?limit=")
        ? {
            configuration: { rounds: 1, epochs: 1, topology: "ring", dataset_split: "iid" },
            active_group_id: null,
            topologies: {},
            groups: [],
          }
        : url.endsWith("/api/apg/runs")
          ? { runs: [], jobs: [], verdicts: [] }
          : url.endsWith("/api/autonomy/state")
            ? { stop_requested: false }
            : allNodes;
      return Promise.resolve({ ok: true, json: async () => data });
    });
    const sockets: Array<{ closedBeforeOpen: boolean }> = [];
    class MockWebSocket extends EventTarget {
      closedBeforeOpen = false;
      isOpen = false;

      constructor() {
        super();
        sockets.push(this);
        window.setTimeout(() => {
          this.isOpen = true;
          this.dispatchEvent(new Event("open"));
        }, 0);
      }
      send() {}
      close() {
        this.closedBeforeOpen = !this.isOpen;
      }
    }
    vi.stubGlobal("fetch", fetchMock);
    vi.stubGlobal("WebSocket", MockWebSocket);

    render(<StrictMode><App /></StrictMode>);

    const expandSidebar = screen.getByRole("button", { name: "Expand sidebar" });
    expect(expandSidebar).toHaveAttribute("aria-expanded", "false");
    expect(expandSidebar.parentElement).toHaveClass("sidebar-bottom");
    expect(expandSidebar).toBe(expandSidebar.parentElement?.lastElementChild);
    expect(document.querySelectorAll(".nav-item .nav-icon")).toHaveLength(10);
    fireEvent.click(expandSidebar);
    expect(screen.getByRole("button", { name: "Collapse sidebar" })).toHaveAttribute("aria-expanded", "true");
    fireEvent.click(screen.getByRole("button", { name: "Collapse sidebar" }));
    expect(screen.getByRole("button", { name: "Expand sidebar" })).toHaveAttribute("aria-expanded", "false");
    expect(screen.queryByText("DFL-SLM")).not.toBeInTheDocument();
    expect(screen.queryByText(/Workspace\s*\//)).not.toBeInTheDocument();
    expect(screen.queryByText("Live updates")).not.toBeInTheDocument();
    expect(screen.queryByLabelText("Signed in user")).not.toBeInTheDocument();
    expect(screen.queryByText(/LATEST TRAINING ROUND/)).not.toBeInTheDocument();
    expect(screen.queryByText(/directed_ring-1793946977-ddbf61/)).not.toBeInTheDocument();

    expect(await screen.findByText("dfl-master")).toBeInTheDocument();
    expect(screen.getByText("dfl-worker1")).toBeInTheDocument();
    expect(screen.getByText("dfl-worker2")).toBeInTheDocument();
    expect(screen.getByText("dfl-worker3")).toBeInTheDocument();
    expect(screen.getByText("Lab probe")).toBeInTheDocument();
    expect(fetchMock).toHaveBeenCalledWith(
      "http://localhost:8000/api/devices",
      expect.objectContaining({
        headers: { Authorization: "Bearer test-session-token" },
      }),
    );
    expect(sockets).toHaveLength(1);
    expect(sockets[0].closedBeforeOpen).toBe(false);

    fireEvent.click(screen.getByRole("button", { name: /Topology/ }));
    expect(screen.getByRole("button", { name: "Topology" })).toHaveAttribute("aria-current", "page");
    expect(
      screen.getByRole("img", { name: "Master connected to 4 worker devices" }),
    ).toBeInTheDocument();
    expect(screen.getByText(/Added in dashboard/)).toBeInTheDocument();
  });

  it("shows measured per-worker memory cleanup results", async () => {
    sessionStorage.setItem("dtn-api-token", "test-session-token");
    const cleanupReport = {
      complete: true,
      phase: "cleanup",
      message: "Memory cleanup completed on all workers.",
      workers: [
        {
          node_id: "worker1",
          success: true,
          cleaned_runs: 2,
          process_memory_cleaned: true,
          memory_available_before_mb: 100,
          memory_available_after_mb: 150,
          process_rss_before_mb: 75,
          process_rss_after_mb: 40,
          os_page_cache_dropped: true,
        },
      ],
    };
    const fetchMock = vi.fn().mockImplementation((input: RequestInfo | URL) => {
      const url = String(input);
      const data = url.endsWith("/api/devices/memory/cleanup")
        ? cleanupReport
        : url.includes("/api/slm/results?limit=")
          ? {
              configuration: { rounds: 1, epochs: 1, topology: "ring", dataset_split: "iid" },
              active_group_id: null,
              topologies: {},
              groups: [],
            }
          : url.endsWith("/api/apg/runs")
            ? { runs: [], jobs: [], verdicts: [] }
            : url.endsWith("/api/autonomy/state")
              ? { stop_requested: false }
              : nodes;
      return Promise.resolve({ ok: true, json: async () => data });
    });
    vi.stubGlobal("fetch", fetchMock);
    vi.stubGlobal("WebSocket", class extends EventTarget {
      send() {}
      close() {}
    });
    vi.stubGlobal("confirm", vi.fn(() => true));

    render(<App />);
    fireEvent.click(await screen.findByRole("button", { name: "Clean worker RAM for next experiment" }));

    expect(await screen.findByText("Worker RAM cleanup complete")).toBeInTheDocument();
    expect(screen.getByText(/Available RAM 100 → 150 MB \(\+50 MB available\)/)).toBeInTheDocument();
    expect(screen.getByText(/Process RAM 75 → 40 MB \(\+35 MB released\)/)).toBeInTheDocument();
    const cleanupRequest = fetchMock.mock.calls.find(
      ([url]) => String(url).endsWith("/api/devices/memory/cleanup"),
    );
    expect(cleanupRequest?.[1]?.method).toBe("POST");
  });

  it("opens a saved NAS candidate in its model-details page", async () => {
    sessionStorage.setItem("dtn-api-token", "test-session-token");
    const architecture = {
      id: "mnist-attention",
      model_type: "custom" as const,
      blocks: 2,
      channels: 64,
      channel_schedule: [32, 64],
      kernel_size: 3,
      block_type: "conv2d" as const,
      head_units: 64,
      dropout: 0.2,
      attention: true,
      parameter_count: 42_000,
      training: { optimizer: "adam" as const, learning_rate: 0.001, batch_size: 32 },
      layers: [
        { type: "conv2d" as const, filters: 32, kernel_size: 3, activation: "relu" },
        { type: "patch_embedding" as const, patch_size: 1, embed_dim: 32 },
        { type: "transformer_encoder" as const, num_heads: 4, ff_dim: 64, activation: "gelu" },
        { type: "global_average_pooling1d" as const },
        { type: "dense" as const, units: 10, activation: "linear" },
      ],
    };
    const candidate = {
      architecture,
      parameter_count: 42_000,
      mean_validation_accuracy: 0.91,
      worker_accuracies: [0.9, 0.92, 0.91],
      mean_validation_macro_precision: 0.9,
      mean_validation_macro_recall: 0.89,
      mean_validation_macro_f1: 0.895,
      mean_elapsed_seconds: 14.5,
      max_peak_rss_mb: 1800,
      model_summary: "Model: custom_mnist_model\nTotal params: 42,000",
      validation_confusion_matrix: [[9, 1], [0, 10]],
    };
    const experimentResults = {
      configuration: { rounds: 1, epochs: 1, topology: "ring", dataset_split: "stratified" },
      active_group_id: null,
      topologies: {},
      groups: [{
        group_id: "mnist-search-1",
        dataset_name: "mnist",
        kind: "dataset_architecture_search" as const,
        topology: "ring" as const,
        status: "completed" as const,
        error: null,
        runs: [],
        selected_architecture: architecture,
        nas_candidates: [candidate],
        dataset_classes: ["0", "1"],
        dataset_image_shape: [28, 28, 1],
      }],
    };
    const fetchMock = vi.fn().mockImplementation((input: RequestInfo | URL) => {
      const url = String(input);
      const data = url.includes("/api/slm/results?limit=")
        ? experimentResults
        : url.endsWith("/api/apg/runs")
          ? { runs: [], jobs: [], verdicts: [] }
          : url.endsWith("/api/autonomy/state")
            ? { stop_requested: false }
            : nodes;
      return Promise.resolve({ ok: true, json: async () => data });
    });
    vi.stubGlobal("fetch", fetchMock);
    vi.stubGlobal("WebSocket", class extends EventTarget {
      send() {}
      close() {}
    });

    render(<App />);
    fireEvent.click(await screen.findByRole("button", { name: /Architecture Search/ }));
    fireEvent.click(await screen.findByRole("button", {
      name: "View details for mnist-attention",
    }));

    expect(await screen.findByRole("heading", { name: "Model details" })).toBeInTheDocument();
    expect(document.querySelector(".keras-model-summary")).toHaveTextContent("Total params: 42,000");
    expect(screen.getByText("Summed worker-validation confusion matrix · rows are true classes, columns are predicted")).toBeInTheDocument();
  });
});
