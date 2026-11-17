import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import type { ExperimentGroup, NASCandidate } from "../api/client";
import ModelDetailsPage from "./ModelDetailsPage";

afterEach(cleanup);

const candidate: NASCandidate = {
  architecture: {
    id: "mnist-attention",
    model_type: "custom",
    blocks: 2,
    channels: 64,
    channel_schedule: [32, 64],
    kernel_size: 3,
    block_type: "conv2d",
    head_units: 64,
    dropout: 0.2,
    attention: true,
    parameter_count: 42_000,
    training: { optimizer: "adam", learning_rate: 0.001, batch_size: 32 },
    layers: [
      { type: "conv2d", filters: 32, kernel_size: 3, activation: "relu" },
      { type: "batch_normalization" },
      { type: "max_pooling2d", pool_size: 2 },
      { type: "patch_embedding", patch_size: 1, embed_dim: 32 },
      { type: "transformer_encoder", num_heads: 4, ff_dim: 64, activation: "gelu" },
      { type: "global_average_pooling1d" },
      { type: "dense", units: 10, activation: "linear" },
    ],
  },
  parameter_count: 42_000,
  mean_validation_accuracy: 0.91,
  worker_accuracies: [0.9, 0.92, 0.91],
  mean_validation_macro_precision: 0.9,
  mean_validation_macro_recall: 0.89,
  mean_validation_macro_f1: 0.895,
  mean_elapsed_seconds: 14.5,
  max_peak_rss_mb: 1800,
  generation: 2,
  fidelity_epochs: 3,
  model_summary: "Model: custom_mnist_model\nTotal params: 42,000",
  validation_confusion_matrix: [[9, 1], [0, 10]],
};

const group: ExperimentGroup = {
  group_id: "mnist-search-1",
  kind: "dataset_architecture_search",
  topology: "ring",
  status: "completed",
  error: null,
  runs: [],
  dataset_image_shape: [28, 28, 1],
  dataset_classes: ["0", "1"],
};

describe("ModelDetailsPage", () => {
  it("shows the saved architecture, Keras summary, validation metrics, and confusion matrix", () => {
    render(
      <ModelDetailsPage
        candidate={candidate}
        group={group}
        onBack={vi.fn()}
        onUseInModelStudio={vi.fn()}
      />,
    );

    expect(screen.getByRole("heading", { name: "Model details" })).toBeInTheDocument();
    expect(screen.getByText(/32 filters/)).toBeInTheDocument();
    expect(screen.getByText(/4-head self-attention/)).toBeInTheDocument();
    expect(document.querySelector(".keras-model-summary")).toHaveTextContent("Total params: 42,000");
    expect(screen.getByText("91.00%")).toBeInTheDocument();
    expect(screen.getByText("0.8950")).toBeInTheDocument();
    expect(screen.getByText("Summed worker-validation confusion matrix · rows are true classes, columns are predicted")).toBeInTheDocument();
    expect(screen.getAllByText("10").length).toBeGreaterThan(0);
    expect(screen.getByText(/held-out test excluded/)).toBeInTheDocument();
  });
});
