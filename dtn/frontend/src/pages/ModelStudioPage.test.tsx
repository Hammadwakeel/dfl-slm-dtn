import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import ModelStudioPage from "./ModelStudioPage";

const WORKER_IDS = ["worker1", "worker2", "worker3", "worker4", "worker5"];

const metadata = {
  dataset_name: "mnist",
  classes: Array.from({ length: 10 }, (_, index) => String(index)),
  class_counts: Object.fromEntries(Array.from({ length: 10 }, (_, index) => [String(index), 7_000])),
  image_sizes: {},
  image_channels: {},
  image_count: 70_000,
  expanded_image_bytes: 54_880_000,
};

const preprocessingReport = {
  dataset_name: "mnist",
  image_count: 70_000,
  target_size: { height: 28, width: 28 },
  channels: 1,
  image_shape: [28, 28, 1],
  normalization: { method: "standard" },
  split_strategy: "stratified",
  split_sizes: { train: 54_000, validation: 6_000, test: 10_000 },
  split_class_counts: {},
  workers: {
    worker1: { rows: 10_800, class_counts: {}, shard_path: "", pc_shard_sha256: "one" },
    worker2: { rows: 10_800, class_counts: {}, shard_path: "", pc_shard_sha256: "two" },
    worker3: { rows: 10_800, class_counts: {}, shard_path: "", pc_shard_sha256: "three" },
    worker4: { rows: 10_800, class_counts: {}, shard_path: "", pc_shard_sha256: "four" },
    worker5: { rows: 10_800, class_counts: {}, shard_path: "", pc_shard_sha256: "five" },
  },
  uploaded_workers: {
    worker1: { rows: 10_800, class_counts: {}, shard_sha256: "one" },
    worker2: { rows: 10_800, class_counts: {}, shard_sha256: "two" },
    worker3: { rows: 10_800, class_counts: {}, shard_sha256: "three" },
    worker4: { rows: 10_800, class_counts: {}, shard_sha256: "four" },
    worker5: { rows: 10_800, class_counts: {}, shard_sha256: "five" },
  },
};

const initialModelReport = {
  dataset_name: "mnist",
  architecture: {
    model_type: "custom",
    layers: [
      { type: "conv2d", filters: 8, kernel_size: 3, strides: 1, padding: "same", activation: "relu" },
      { type: "global_average_pooling2d" },
      { type: "dense", units: 10, activation: "linear" },
    ],
    training: { optimizer: "adam", learning_rate: 0.001, batch_size: 32 },
    dataset_name: "mnist",
    input_shape: [28, 28, 1],
    classes: Array.from({ length: 10 }, (_, index) => String(index)),
    num_classes: 10,
    parameter_count: 170,
    initialization_seed: 42,
    normalization: { method: "standard", mean: [0.13], standard_deviation: [0.3] },
  },
  parameter_count: 170,
  architecture_sha256: "architecture-hash",
  weights_sha256: "weights-hash",
};

function response(body: unknown, status = 200): Response {
  return {
    ok: status >= 200 && status < 300,
    status,
    json: async () => body,
  } as Response;
}

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

describe("ModelStudioPage", () => {
  it("builds a custom model, then starts training with its installed manifest", async () => {
    let modelInstalled = false;
    const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      if (url.endsWith("/api/datasets/mnist/initial-model") && init?.method === "DELETE") {
        modelInstalled = false;
        return response(undefined, 204);
      }
      if (url.endsWith("/api/datasets")) return response([metadata]);
      if (url.endsWith("/api/autonomy/state")) {
        return response({ stop_requested: false, loops: [] });
      }
      if (url.endsWith("/api/datasets/mnist/preprocessing-report")) return response(preprocessingReport);
      if (url.endsWith("/api/datasets/mnist/initial-model")) {
        return modelInstalled ? response(initialModelReport) : response({ detail: "Not found" }, 404);
      }
      if (url.endsWith("/api/autonomy/dispatch")) {
        const body = JSON.parse(String(init?.body));
        if (body.action.action === "create_initial_model") {
          modelInstalled = true;
          return response({
            status: "executed",
            result: {
              available: true,
              data: {
                ...initialModelReport,
                workers: Object.fromEntries(WORKER_IDS.map((worker) => [
                  worker,
                  {
                    architecture_sha256: initialModelReport.architecture_sha256,
                    weights_sha256: initialModelReport.weights_sha256,
                    parameter_count: initialModelReport.parameter_count,
                  },
                ])),
              },
            },
          });
        }
        return response({ status: "executed", result: { available: true, data: { group_id: "mnist-run-1" } } });
      }
      return response({ detail: "Unexpected test request" }, 500);
    });
    const onRefresh = vi.fn();
    vi.stubGlobal("fetch", fetchMock);
    vi.stubGlobal("confirm", vi.fn(() => true));
    render(<ModelStudioPage token="session-token" results={null} error="" onRefresh={onRefresh} />);

    expect(await screen.findByRole("button", { name: "Build and install model" })).toBeEnabled();
    fireEvent.change(screen.getByLabelText("Layer 1 filters"), { target: { value: "16" } });
    expect(screen.getByText("Custom 9-layer model")).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "Build and install model" }));
    expect(await screen.findByText(/Build and install requested for Custom 9-layer model on MNIST/)).toBeInTheDocument();
    expect(await screen.findByText(/Model Custom 9-layer model was initialized/)).toBeInTheDocument();
    expect(screen.getByRole("img", { name: /Custom 3-layer model architecture/ })).toBeInTheDocument();
    expect(screen.getByText(/worker1: model installed; weight hash matches coordinator; 170 parameters/)).toBeInTheDocument();
    expect(screen.getByLabelText("Training dataset")).toHaveValue("mnist");
    expect(await screen.findByRole("button", { name: "Start training" })).toBeEnabled();

    fireEvent.click(screen.getByRole("button", { name: "Start training" }));
    expect(await screen.findByText(/Training group mnist-run-1 started/)).toBeInTheDocument();
    expect(onRefresh).toHaveBeenCalled();

    const dispatchedActions = fetchMock.mock.calls
      .filter(([url]) => String(url).endsWith("/api/autonomy/dispatch"))
      .map(([, init]) => JSON.parse(String(init?.body)).action);
    expect(dispatchedActions).toEqual([
      {
        action: "create_initial_model",
        parameters: {
          model_name: "mnist",
          architecture: expect.objectContaining({
            model_type: "custom",
            training: { optimizer: "adam", learning_rate: 0.001, batch_size: 32 },
          }),
          seed: 42,
        },
      },
      {
        action: "start_training",
        parameters: {
          dataset_name: "mnist",
          rounds: 10,
          epochs: 5,
          architecture: initialModelReport.architecture,
        },
      },
    ]);
    const installedLayers = dispatchedActions[0].parameters.architecture.layers as Array<Record<string, unknown>>;
    expect(installedLayers[0].filters).toBe(16);

    fireEvent.click(screen.getByRole("button", { name: "Delete installed model" }));
    expect(await screen.findByText("Installed MNIST model deleted from the coordinator and all configured workers.")).toBeInTheDocument();
    expect(fetchMock).toHaveBeenCalledWith(
      "http://localhost:8000/api/datasets/mnist/initial-model",
      expect.objectContaining({ method: "DELETE" }),
    );
  });

  it("builds the selected architecture from NAS in Model Studio", async () => {
    let modelInstalled = false;
    const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      void init;
      const url = String(input);
      if (url.endsWith("/api/datasets")) return response([metadata]);
      if (url.endsWith("/api/autonomy/state")) {
        return response({ stop_requested: false, loops: [] });
      }
      if (url.endsWith("/api/datasets/mnist/preprocessing-report")) return response(preprocessingReport);
      if (url.endsWith("/api/datasets/mnist/initial-model")) {
        return modelInstalled ? response(initialModelReport) : response({ detail: "Not found" }, 404);
      }
      if (url.endsWith("/api/autonomy/dispatch")) {
        modelInstalled = true;
        return response({ status: "executed", result: { available: true, data: initialModelReport } });
      }
      return response({ detail: "Unexpected test request" }, 500);
    });
    vi.stubGlobal("fetch", fetchMock);
    render(
      <ModelStudioPage
        token="session-token"
        results={null}
        error=""
        onRefresh={vi.fn()}
        nasArchitecture={{
          id: "nas-d3-w16x16x32-k5-conv-h64-drop20-attn",
          model_type: "custom",
          blocks: 3,
          channels: 32,
          channel_schedule: [16, 16, 32],
          kernel_size: 5,
          block_type: "conv2d",
          head_units: 64,
          dropout: 0.2,
          attention: true,
          parameter_count: 32794,
          layers: [
            { type: "patch_embedding", patch_size: 1, embed_dim: 32 },
            { type: "transformer_encoder", embed_dim: 32, num_heads: 4, ff_dim: 64, dropout: 0.1, activation: "gelu" },
            { type: "global_average_pooling1d" },
            { type: "dense", units: 64, activation: "gelu" },
            { type: "dropout", rate: 0.2 },
            { type: "dense", units: 10, activation: "linear" },
          ],
          training: { optimizer: "adam", learning_rate: 0.001, batch_size: 32 },
        }}
        onClearNasArchitecture={vi.fn()}
      />,
    );

    expect(await screen.findByText("NAS-selected MNIST architecture")).toBeInTheDocument();
    expect(screen.getByText(/nas-d3-w16x16x32-k5-conv-h64-drop20-attn · 3 convolution blocks · 16\/16\/32 channels · 5×5 kernels · 32,794 parameters/)).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Build and install model" }));
    expect(await screen.findByText(/Model Custom 6-layer model was initialized/)).toBeInTheDocument();

    const buildAction = fetchMock.mock.calls
      .filter(([url]) => String(url).endsWith("/api/autonomy/dispatch"))
      .map(([, init]) => JSON.parse(String(init?.body)).action)[0];
    expect(buildAction.parameters.architecture).toEqual({
      model_type: "custom",
      layers: [
        { type: "patch_embedding", patch_size: 1, embed_dim: 32 },
        { type: "transformer_encoder", embed_dim: 32, num_heads: 4, ff_dim: 64, dropout: 0.1, activation: "gelu" },
        { type: "global_average_pooling1d" },
        { type: "dense", units: 64, activation: "gelu" },
        { type: "dropout", rate: 0.2 },
        { type: "dense", units: 10, activation: "linear" },
      ],
      training: { optimizer: "adam", learning_rate: 0.001, batch_size: 32 },
    });
  });

  it("shows persisted worker results as grouped round metric cards", async () => {
    vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input);
      if (url.endsWith("/api/datasets")) return response([metadata]);
      if (url.endsWith("/api/autonomy/state")) {
        return response({ mode: "ASK_HUMAN", stop_requested: false, pending_decisions: 0, loops: [] });
      }
      if (url.endsWith("/preprocessing-report")) return response(preprocessingReport);
      if (url.endsWith("/initial-model")) return response(initialModelReport);
      return response({ detail: "Unexpected test request" }, 500);
    }));
    render(
      <ModelStudioPage
        token="session-token"
        results={{
          configuration: { rounds: 2, epochs: 1, topology: "ring", dataset_split: "mnist" },
          active_group_id: "mnist-run-live",
          topologies: {},
          groups: [{
            group_id: "mnist-run-live",
            dataset_name: "mnist",
            kind: "decentralized",
            topology: "ring",
            status: "running",
            rounds: 2,
            epochs: 1,
            error: null,
            runs: [{
              run_id: "run-live",
              seed: 42,
              topology: "ring",
              status: "running",
              rounds: [{
                round: 1,
                topology: "ring",
                workers: {
                  worker1: {
                    node_id: "worker1",
                    run_id: "run-live",
                    status: "completed",
                    round: 1,
                    topology: "ring",
                    peak_process_rss_mb: 200,
                    loss: 0.25,
                    validation_accuracy: 0.92,
                    validation_macro_f1: 0.91,
                    validation_macro_precision: 0.92,
                    validation_macro_recall: 0.9,
                    round_seconds: 2,
                    peer_wait_seconds: 0.2,
                    bytes_sent: 100,
                    bytes_received: 200,
                  },
                },
              }],
            }],
          }],
        }}
        error=""
        onRefresh={vi.fn()}
      />,
    );

    expect(await screen.findByText((_, element) =>
      Boolean(element?.tagName === "P" &&
        element.textContent?.includes("Training group mnist-run-live is running")),
    )).toBeInTheDocument();
    expect(screen.getByText("Round-by-round results")).toBeInTheDocument();
    expect(screen.getByText("1 runs · 1 saved rounds · worker metrics")).toBeInTheDocument();
    expect(screen.getByText("Round 1")).toBeInTheDocument();
    expect(screen.getByText("Validation accuracy")).toBeInTheDocument();
    expect(screen.getByText("92.00%")).toBeInTheDocument();
    expect(screen.getByText("0.9100")).toBeInTheDocument();
    expect(screen.getByText("0.9200")).toBeInTheDocument();
    expect(screen.getByText("0.9000")).toBeInTheDocument();
    expect(screen.getByText("Runtime 2.00 s")).toBeInTheDocument();
    expect(screen.getByText("Peer wait 0.20 s")).toBeInTheDocument();
    expect(screen.getByText("Sent 100 B")).toBeInTheDocument();
    expect(screen.getByText("Received 200 B")).toBeInTheDocument();
    expect(screen.queryByText(/worker1: round 1/)).not.toBeInTheDocument();
  });

  it("shows an explicit live wait state while the first worker round is still running", async () => {
    vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input);
      if (url.endsWith("/api/datasets")) return response([metadata]);
      if (url.endsWith("/api/autonomy/state")) {
        return response({ mode: "ASK_HUMAN", stop_requested: false, pending_decisions: 0, loops: [] });
      }
      if (url.endsWith("/preprocessing-report")) return response(preprocessingReport);
      if (url.endsWith("/initial-model")) return response(initialModelReport);
      return response({ detail: "Unexpected test request" }, 500);
    }));
    render(
      <ModelStudioPage
        token="session-token"
        results={{
          configuration: { rounds: 30, epochs: 10, topology: "directed_ring", dataset_split: "mnist" },
          active_group_id: "mnist-run-waiting",
          topologies: {},
          groups: [{
            group_id: "mnist-run-waiting",
            dataset_name: "mnist",
            kind: "decentralized",
            topology: "directed_ring",
            status: "running",
            rounds: 30,
            epochs: 10,
            started_at: Date.now() / 1000 - 120,
            error: null,
            runs: [{
              run_id: "mnist-run-waiting-seed",
              seed: 2026,
              topology: "directed_ring",
              status: "running",
              rounds: [],
            }],
          }],
        }}
        error=""
        onRefresh={vi.fn()}
      />,
    );

    expect(await screen.findByRole("status")).toHaveTextContent(
      /Waiting for round 1 of 30 results · 2m \d+s elapsed/,
    );
    expect(screen.getByText("No round metrics yet.")).toBeInTheDocument();
  });

  it("shows persisted per-worker epoch progress and keeps coordinator errors visible", async () => {
    vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input);
      if (url.endsWith("/api/datasets")) {
        throw new Error("The user-confirmed action did not complete.");
      }
      if (url.endsWith("/api/autonomy/state")) {
        return response({ mode: "ASK_HUMAN", stop_requested: false, pending_decisions: 0, loops: [] });
      }
      if (url.endsWith("/preprocessing-report")) return response(preprocessingReport);
      if (url.endsWith("/initial-model")) return response(initialModelReport);
      return response({ detail: "Unexpected test request" }, 500);
    }));
    render(
      <ModelStudioPage
        token="session-token"
        results={{
          configuration: { rounds: 30, epochs: 10, topology: "directed_ring", dataset_split: "mnist" },
          active_group_id: "mnist-run-progress",
          topologies: {},
          groups: [{
            group_id: "mnist-run-progress",
            dataset_name: "mnist",
            kind: "decentralized",
            topology: "directed_ring",
            status: "running",
            rounds: 30,
            epochs: 10,
            error: null,
            runs: [{
              run_id: "mnist-run-progress-seed",
              seed: 2026,
              topology: "directed_ring",
              status: "running",
              progress: {
                round: 1,
                workers: {
                  worker1: { epoch: 3, epochs: 10, phase: "local_training" },
                  worker2: { epoch: 10, epochs: 10, phase: "peer_exchange" },
                },
              },
              rounds: [],
            }],
          }],
        }}
        error=""
        onRefresh={vi.fn()}
      />,
    );

    expect(await screen.findByText("worker1: local epoch 3 of 10")).toBeInTheDocument();
    expect(screen.getByText("worker2: local epochs complete · exchanging weights")).toBeInTheDocument();
    expect(await screen.findByText("The user-confirmed action did not complete.")).toBeInTheDocument();
  });

  it("shows the recorded round timeout instead of the stale generic approval error", async () => {
    vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input);
      if (url.endsWith("/api/datasets")) return response([metadata]);
      if (url.endsWith("/api/autonomy/state")) {
        return response({ mode: "ASK_HUMAN", stop_requested: false, pending_decisions: 0, loops: [] });
      }
      if (url.endsWith("/preprocessing-report")) return response(preprocessingReport);
      if (url.endsWith("/initial-model")) return response(initialModelReport);
      return response({ detail: "Unexpected test request" }, 500);
    }));
    render(
      <ModelStudioPage
        token="session-token"
        results={{
          configuration: { rounds: 30, epochs: 10, topology: "directed_ring", dataset_split: "mnist" },
          active_group_id: null,
          topologies: {},
          groups: [{
            group_id: "directed_ring-timed-out",
            dataset_name: "mnist",
            kind: "decentralized",
            topology: "directed_ring",
            status: "failed",
            rounds: 30,
            epochs: 10,
            error: "TrainingError: Timed out waiting for all workers to complete round 1.",
            runs: [{
              run_id: "directed-ring-timed-out-seed",
              seed: 2026,
              topology: "directed_ring",
              status: "failed",
              error: "TrainingError: Timed out waiting for all workers to complete round 1.",
              rounds: [],
            }],
          }],
        }}
        error=""
        onRefresh={vi.fn()}
      />,
    );

    expect(await screen.findByText(
      "TrainingError: Timed out waiting for all workers to complete round 1.",
    )).toBeInTheDocument();
    expect(screen.queryByText("The user-confirmed action did not complete.")).not.toBeInTheDocument();
  });

  it("shows older coordinator groups when dataset_name is nested in the model manifest", async () => {
    vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input);
      if (url.endsWith("/api/datasets")) return response([metadata]);
      if (url.endsWith("/api/autonomy/state")) {
        return response({ mode: "ASK_HUMAN", stop_requested: false, pending_decisions: 0, loops: [] });
      }
      if (url.endsWith("/preprocessing-report")) return response(preprocessingReport);
      if (url.endsWith("/initial-model")) return response(initialModelReport);
      return response({ detail: "Unexpected test request" }, 500);
    }));
    render(
      <ModelStudioPage
        token="session-token"
        results={{
          configuration: { rounds: 1, epochs: 1, topology: "ring", dataset_split: "mnist" },
          active_group_id: null,
          topologies: {},
          groups: [{
            group_id: "legacy-mnist-run",
            architecture: { dataset_name: "mnist" },
            kind: "decentralized",
            topology: "ring",
            status: "completed",
            rounds: 1,
            epochs: 1,
            error: null,
            runs: [{
              run_id: "legacy-run",
              seed: 42,
              topology: "ring",
              status: "completed",
              rounds: [{
                round: 1,
                topology: "ring",
                workers: {
                  worker1: {
                    node_id: "worker1",
                    run_id: "legacy-run",
                    round: 1,
                    topology: "ring",
                    peak_process_rss_mb: 200,
                    validation_accuracy: 0.9,
                  },
                },
              }],
            }],
          }],
        }}
        error=""
        onRefresh={vi.fn()}
      />,
    );

    expect(await screen.findByText("legacy-mnist-run")).toBeInTheDocument();
    expect(screen.getByText((_, element) =>
      Boolean(element?.tagName === "P" &&
        element.textContent?.includes("legacy-mnist-run is completed") &&
        element.textContent?.includes("Dataset: mnist · ring")),
    )).toBeInTheDocument();
  });

  it("shows the saved training group count when no group is active", async () => {
    vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input);
      if (url.endsWith("/api/datasets")) return response([metadata]);
      if (url.endsWith("/api/autonomy/state")) {
        return response({ mode: "ASK_HUMAN", stop_requested: false, pending_decisions: 0, loops: [] });
      }
      if (url.endsWith("/preprocessing-report")) return response(preprocessingReport);
      if (url.endsWith("/initial-model")) return response({ detail: "Not found" }, 404);
      return response({ detail: "Unexpected test request" }, 500);
    }));
    render(
      <ModelStudioPage
        token="session-token"
        results={{
          configuration: { rounds: 1, epochs: 1, topology: "ring", dataset_split: "mnist" },
          active_group_id: null,
          topologies: {},
          groups: [],
        }}
        error=""
        onRefresh={vi.fn()}
      />,
    );

    expect(await screen.findByText("Saved training groups: 0.")).toBeInTheDocument();
  });

  it("offers the curated pretrained DeiT design", async () => {
    vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input);
      if (url.endsWith("/api/datasets")) return response([metadata]);
      if (url.endsWith("/api/autonomy/state")) {
        return response({ mode: "ASK_HUMAN", stop_requested: false, pending_decisions: 0, loops: [] });
      }
      if (url.endsWith("/preprocessing-report")) return response(preprocessingReport);
      if (url.endsWith("/initial-model")) return response({ detail: "Not found" }, 404);
      return response({ detail: "Unexpected test request" }, 500);
    }));
    render(<ModelStudioPage token="session-token" results={null} error="" onRefresh={vi.fn()} />);

    fireEvent.change(await screen.findByLabelText("Model family"), {
      target: { value: "pretrained_deit" },
    });

    expect(screen.getByText("facebook/deit-tiny-patch16-224")).toBeInTheDocument();
    expect(screen.getByText(/about 5.5M parameters/)).toBeInTheDocument();
    expect(screen.getByText("5,526,346")).toBeInTheDocument();
  });

  it("keeps model creation unavailable until worker MNIST shards are ready", async () => {
    vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input);
      if (url.endsWith("/api/datasets")) return response([metadata]);
      if (url.endsWith("/api/autonomy/state")) {
        return response({ mode: "ASK_HUMAN", stop_requested: false, pending_decisions: 0, loops: [] });
      }
      if (url.endsWith("/preprocessing-report")) return response({ ...preprocessingReport, uploaded_workers: {} });
      if (url.endsWith("/initial-model")) return response({ detail: "Not found" }, 404);
      return response({ detail: "Unexpected test request" }, 500);
    }));
    render(<ModelStudioPage token="session-token" results={null} error="" onRefresh={vi.fn()} />);
    await waitFor(() => expect(screen.getByRole("button", { name: "Build and install model" })).toBeDisabled());
  });

  it("requires the shard hash for every configured worker", async () => {
    const incompleteReport = {
      ...preprocessingReport,
      uploaded_workers: {
        ...preprocessingReport.uploaded_workers,
        worker5: { ...preprocessingReport.uploaded_workers.worker5, shard_sha256: "wrong" },
      },
    };
    vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input);
      if (url.endsWith("/api/datasets")) return response([metadata]);
      if (url.endsWith("/api/autonomy/state")) {
        return response({ mode: "ASK_HUMAN", stop_requested: false, pending_decisions: 0, loops: [] });
      }
      if (url.endsWith("/preprocessing-report")) return response(incompleteReport);
      if (url.endsWith("/initial-model")) return response({ detail: "Not found" }, 404);
      return response({ detail: "Unexpected test request" }, 500);
    }));
    render(<ModelStudioPage token="session-token" results={null} error="" onRefresh={vi.fn()} />);
    await waitFor(() => expect(screen.getByRole("button", { name: "Build and install model" })).toBeDisabled());
  });
});
