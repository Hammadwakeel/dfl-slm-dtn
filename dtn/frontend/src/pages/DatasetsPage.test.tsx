import { cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import DatasetsPage from "./DatasetsPage";

const report = {
  dataset_name: "mnist",
  image_count: 70_000,
  target_size: { height: 28, width: 28 },
  channels: 1,
  image_shape: [28, 28, 1],
  normalization: { method: "standard" },
  split_strategy: "stratified",
  split_method: "official_mnist_train_validation_test",
  official_test_set_preserved: true,
  split_sizes: { train: 54_000, validation: 6_000, test: 10_000 },
  split_class_counts: {},
  workers: {
    worker1: { rows: 18_000, class_counts: {}, shard_path: "", pc_shard_sha256: "same-hash" },
    worker2: { rows: 18_000, class_counts: {}, shard_path: "", pc_shard_sha256: "same-hash" },
    worker3: { rows: 18_000, class_counts: {}, shard_path: "", pc_shard_sha256: "same-hash" },
  },
  uploaded_workers: {
    worker1: { rows: 18_000, class_counts: {}, shard_sha256: "same-hash" },
    worker2: { rows: 18_000, class_counts: {}, shard_sha256: "same-hash" },
    worker3: { rows: 18_000, class_counts: {}, shard_sha256: "same-hash" },
  },
  preprocessing_duration_seconds: 1.2,
  shipping_duration_seconds: 2.3,
};

const mnist = {
  dataset_name: "mnist",
  classes: Array.from({ length: 10 }, (_, index) => String(index)),
  class_counts: Object.fromEntries(Array.from({ length: 10 }, (_, index) => [String(index), 7_000])),
  image_sizes: Object.fromEntries(Array.from({ length: 10 }, (_, index) => [String(index), { "28x28": 7_000 }])),
  image_channels: Object.fromEntries(Array.from({ length: 10 }, (_, index) => [String(index), { "1": 7_000 }])),
  image_count: 70_000,
  expanded_image_bytes: 54_880_000,
  source_split_counts: { train: 60_000, test: 10_000 },
  source_split_policy: { type: "official_mnist_npz" },
};

const cifar10 = {
  dataset_name: "cifar10",
  classes: ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"],
  class_counts: Object.fromEntries(["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"].map((name) => [name, 6_000])),
  image_sizes: Object.fromEntries(["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"].map((name) => [name, { "32x32": 6_000 }])),
  image_channels: Object.fromEntries(["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"].map((name) => [name, { "3": 6_000 }])),
  image_count: 60_000,
  expanded_image_bytes: 184_320_000,
  source_split_counts: { train: 50_000, test: 10_000 },
  source_split_policy: { type: "official_cifar10_npz" },
};

function response(body: unknown, status = 200): Response {
  return {
    ok: status >= 200 && status < 300,
    status,
    json: async () => body,
  } as Response;
}

function baseFetch(options: {
  datasets?: Record<string, unknown>[];
  stop?: boolean;
  reportByName?: Record<string, unknown>;
  dispatchReport?: Record<string, unknown>;
  historyWarning?: string;
}) {
  const items = options.datasets ?? [mnist];
  return vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
    const url = String(input);
    if (url.endsWith("/api/datasets")) return response(items);
    if (url.endsWith("/api/autonomy/state")) {
      return response({ stop_requested: options.stop ?? false, loops: [] });
    }
    if (url.includes("/samples")) {
      return response([{ label: "sample class", image_data: "data:image/png;base64,test" }]);
    }
    if (url.includes("/preprocessing-report")) {
      const name = decodeURIComponent(url.split("/api/datasets/")[1].split("/")[0]);
      return options.reportByName?.[name] ? response(options.reportByName[name]) : response({ detail: "Not found" }, 404);
    }
    if (url.endsWith("/api/autonomy/dispatch") && init?.method === "POST") {
      return response({
        status: "executed",
        result: { available: true, data: options.dispatchReport ?? report },
        ...(options.historyWarning ? { history_warning: options.historyWarning } : {}),
      });
    }
    return response({ detail: "Unexpected test request" }, 500);
  });
}

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

describe("DatasetsPage", () => {
  it("offers official MNIST and CIFAR-10 in the dataset dropdown and has no upload controls", async () => {
    const fetchMock = baseFetch({
      datasets: [mnist, cifar10, { ...mnist, dataset_name: "another-dataset" }],
    });
    vi.stubGlobal("fetch", fetchMock);
    render(<DatasetsPage token="session-token" />);

    const datasetSelect = await screen.findByRole("combobox", { name: "Dataset" });
    expect(within(datasetSelect).getByRole("option", { name: /MNIST/ })).toBeInTheDocument();
    expect(within(datasetSelect).getByRole("option", { name: /CIFAR-10/ })).toBeInTheDocument();
    expect(within(datasetSelect).queryByRole("option", { name: /another-dataset/ })).not.toBeInTheDocument();
    expect(await screen.findByRole("img", { name: "sample class sample" })).toBeInTheDocument();
    expect(screen.getByText("28 × 28")).toBeInTheDocument();
    fireEvent.change(datasetSelect, { target: { value: "cifar10" } });
    expect(screen.getByRole("heading", { name: "CIFAR-10" })).toBeInTheDocument();
    expect(screen.getByText("32 × 32")).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /upload/i })).not.toBeInTheDocument();
    expect(screen.queryByLabelText(/ZIP archive|dataset name/i)).not.toBeInTheDocument();
  });

  it("shows setup instructions when both official datasets are missing", async () => {
    vi.stubGlobal("fetch", baseFetch({ datasets: [] }));
    render(<DatasetsPage token="session-token" />);
    expect(await screen.findByText("No datasets are prepared.")).toBeInTheDocument();
  });

  it("dispatches preprocessing immediately without approval controls", async () => {
    const fetchMock = baseFetch({ reportByName: { mnist: report } });
    vi.stubGlobal("fetch", fetchMock);
    render(<DatasetsPage token="session-token" />);
    await screen.findByRole("combobox", { name: "Dataset" });
    fireEvent.click(screen.getByRole("button", { name: "Preprocess & ship" }));
    expect(await screen.findByText("Preprocessing and shard shipping completed.")).toBeInTheDocument();
    expect(screen.queryByText(/approval/i)).not.toBeInTheDocument();

    const dispatchCall = fetchMock.mock.calls.find(([url]) => String(url).endsWith("/api/autonomy/dispatch"));
    const dispatchBody = JSON.parse(String(dispatchCall?.[1]?.body));
    expect(dispatchBody.action.parameters).toEqual({
      dataset_name: "mnist",
      normalization: "standard",
      dataset_split: "stratified",
      target_size: { height: 28, width: 28 },
      seed: 42,
      replace_existing: true,
    });
    expect(dispatchBody.evidence.source_split_counts).toEqual({ train: 60_000, test: 10_000 });
    expect(dispatchBody.evidence.replace_existing).toBe(true);

    await waitFor(() => expect(fetchMock.mock.calls.filter(([url]) => String(url).endsWith("/api/autonomy/dispatch"))).toHaveLength(1));
    expect(screen.queryByRole("button", { name: /approve|reject/i })).not.toBeInTheDocument();
    expect(screen.getByText("Last preprocessing result")).toBeInTheDocument();
  });

  it("dispatches the selected CIFAR-10 dataset with its official split and image dimensions", async () => {
    const cifarReport = {
      ...report,
      dataset_name: "cifar10",
      image_count: 60_000,
      target_size: { height: 32, width: 32 },
      channels: 3,
      image_shape: [32, 32, 3],
      split_sizes: { train: 45_000, validation: 5_000, test: 10_000 },
    };
    const fetchMock = baseFetch({
      datasets: [mnist, cifar10],
      reportByName: { cifar10: cifarReport },
      dispatchReport: cifarReport,
    });
    vi.stubGlobal("fetch", fetchMock);
    render(<DatasetsPage token="session-token" />);

    const datasetSelect = await screen.findByRole("combobox", { name: "Dataset" });
    fireEvent.change(datasetSelect, { target: { value: "cifar10" } });
    fireEvent.click(screen.getByRole("button", { name: "Preprocess & ship" }));
    expect(await screen.findByText("Preprocessing and shard shipping completed.")).toBeInTheDocument();

    const dispatchCall = fetchMock.mock.calls.find(([url]) => String(url).endsWith("/api/autonomy/dispatch"));
    const dispatchBody = JSON.parse(String(dispatchCall?.[1]?.body));
    expect(dispatchBody.action.parameters).toEqual({
      dataset_name: "cifar10",
      normalization: "standard",
      dataset_split: "stratified",
      target_size: { height: 32, width: 32 },
      seed: 42,
      replace_existing: true,
    });
    expect(dispatchBody.evidence.source_split_counts).toEqual({ train: 50_000, test: 10_000 });
    expect(dispatchBody.evidence.replace_existing).toBe(true);
  });

  it("reports CIFAR-10 completion separately from a failed master history sync", async () => {
    const historyWarning =
      "The action completed, but the master did not save its history record (HTTP 422). Do not repeat the action; update the master service.";
    vi.stubGlobal(
      "fetch",
      baseFetch({
        datasets: [mnist, cifar10],
        reportByName: { cifar10: report },
        dispatchReport: report,
        historyWarning,
      }),
    );
    render(<DatasetsPage token="session-token" />);

    const datasetSelect = await screen.findByRole("combobox", { name: "Dataset" });
    fireEvent.change(datasetSelect, { target: { value: "cifar10" } });
    fireEvent.click(screen.getByRole("button", { name: "Preprocess & ship" }));

    expect(await screen.findByText("Preprocessing and shard shipping completed.")).toBeInTheDocument();
    expect(await screen.findByText(historyWarning)).toHaveAttribute("role", "status");
  });

  it("offers and dispatches Dirichlet alpha 0.1 and 0.7 split strategies", async () => {
    const fetchMock = baseFetch({});
    vi.stubGlobal("fetch", fetchMock);
    render(<DatasetsPage token="session-token" />);

    const splitSelect = await screen.findByRole("combobox", { name: "Split strategy" });
    expect(within(splitSelect).getByRole("option", { name: "Dirichlet α=0.1" })).toBeInTheDocument();
    expect(within(splitSelect).getByRole("option", { name: "Dirichlet α=0.7" })).toBeInTheDocument();

    const dispatchCalls = () => fetchMock.mock.calls.filter(
      ([url]) => String(url).endsWith("/api/autonomy/dispatch"),
    );
    for (const [index, strategy] of ["dirichlet_a01", "dirichlet_a07"].entries()) {
      fireEvent.change(splitSelect, { target: { value: strategy } });
      fireEvent.click(screen.getByRole("button", { name: "Preprocess & ship" }));
      await waitFor(() => expect(dispatchCalls()).toHaveLength(index + 1));

      const dispatchBody = JSON.parse(String(dispatchCalls()[index][1]?.body));
      expect(dispatchBody.action.parameters.dataset_split).toBe(strategy);
      await waitFor(() => {
        expect(screen.getByRole("button", { name: "Preprocess & ship" })).toBeEnabled();
      });
    }
  });

  it("disables preprocessing while global STOP is active", async () => {
    vi.stubGlobal("fetch", baseFetch({ stop: true }));
    render(<DatasetsPage token="session-token" />);
    expect(await screen.findByRole("alert")).toHaveTextContent("Global STOP is active");
    expect(screen.getByRole("button", { name: "Preprocess & ship" })).toBeDisabled();
  });
});
