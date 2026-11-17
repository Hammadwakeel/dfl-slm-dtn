import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import DevicesPage from "./DevicesPage";
import type { Device } from "../types";

const registeredDevice: Device = {
  node_id: "lab-worker-1",
  role: "worker",
  hostname: "Lab worker 1",
  ip: "127.0.0.1",
  port: 5001,
  training_port: 5004,
  training_ready: true,
  training_error: null,
  is_managed: true,
  online: true,
  last_seen: "2026-10-24T12:00:00Z",
  last_error: null,
  latest_status: null,
};

const configuredDevice: Device = {
  ...registeredDevice,
  node_id: "worker1",
  hostname: "Configured worker",
  ip: "192.168.11.21",
  is_managed: false,
};

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

describe("DevicesPage", () => {
  it("registers a device and reports its immediate reachability", async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      status: 201,
      json: async () => registeredDevice,
    });
    const onRefresh = vi.fn();
    vi.stubGlobal("fetch", fetchMock);

    render(<DevicesPage token="coordinator-token" devices={[]} onRefresh={onRefresh} />);
    fireEvent.change(screen.getByLabelText("Sequential worker ID"), {
      target: { value: "LAB-WORKER-1" },
    });
    fireEvent.change(screen.getByLabelText("Display name"), { target: { value: "Lab worker 1" } });
    fireEvent.change(screen.getByLabelText("Private IP address"), { target: { value: "127.0.0.1" } });
    fireEvent.change(screen.getByLabelText("Status port"), { target: { value: "5001" } });
    fireEvent.change(screen.getByLabelText("Training API port"), { target: { value: "5004" } });
    fireEvent.change(screen.getByLabelText("Device API token"), {
      target: { value: "device-token-that-is-at-least-thirty-two-chars" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Register and test connection" }));

    expect(
      await screen.findByText(
        "Lab worker 1 registered; status API reachable; training peers synchronized.",
      ),
    ).toBeInTheDocument();
    expect(fetchMock).toHaveBeenCalledWith(
      "http://localhost:8000/api/devices",
      expect.objectContaining({
        method: "POST",
        body: JSON.stringify({
          node_id: "lab-worker-1",
          hostname: "Lab worker 1",
          ip: "127.0.0.1",
          port: 5001,
          training_port: 5004,
          api_token: "device-token-that-is-at-least-thirty-two-chars",
        }),
      }),
    );
    expect(onRefresh).toHaveBeenCalledOnce();
  });

  it("removes managed devices and refreshes the registry", async () => {
    const fetchMock = vi.fn().mockResolvedValue({ ok: true, status: 204 });
    const confirmMock = vi.fn(() => true);
    const onRefresh = vi.fn();
    vi.stubGlobal("fetch", fetchMock);
    vi.stubGlobal("confirm", confirmMock);

    render(
      <DevicesPage
        token="coordinator-token"
        devices={[registeredDevice]}
        onRefresh={onRefresh}
      />,
    );
    fireEvent.click(screen.getByRole("button", { name: "Remove" }));

    await waitFor(() => {
      expect(fetchMock).toHaveBeenCalledWith(
        "http://localhost:8000/api/devices/lab-worker-1",
        expect.objectContaining({ method: "DELETE" }),
      );
    });
    expect(confirmMock).toHaveBeenCalledOnce();
    expect(
      await screen.findByText("Lab worker 1 removed from the device registry."),
    ).toBeInTheDocument();
    expect(onRefresh).toHaveBeenCalledOnce();
  });

  it("displays configured cluster devices without offering unsafe registry actions", () => {
    render(
      <DevicesPage
        token="coordinator-token"
        devices={[configuredDevice, registeredDevice]}
        onRefresh={vi.fn()}
      />,
    );

    expect(screen.getByText("2 devices · 1 dashboard-managed")).toBeInTheDocument();
    expect(screen.getByText("Configured worker")).toBeInTheDocument();
    expect(screen.getByText("Cluster-configured")).toBeInTheDocument();
    expect(screen.getByText(/Deployment-managed · cannot edit or remove here/)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Edit" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Remove" })).toBeInTheDocument();
  });

  it("edits a managed device and leaves its token unchanged when no replacement is entered", async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      status: 200,
      json: async () => ({ ...registeredDevice, hostname: "Updated worker" }),
    });
    const onRefresh = vi.fn();
    vi.stubGlobal("fetch", fetchMock);
    render(<DevicesPage token="coordinator-token" devices={[registeredDevice]} onRefresh={onRefresh} />);

    fireEvent.click(screen.getByRole("button", { name: "Edit" }));
    fireEvent.change(screen.getByLabelText("New display name"), {
      target: { value: "Updated worker" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Save changes" }));

    expect(await screen.findByText("Updated worker updated and reachable.")).toBeInTheDocument();
    expect(fetchMock).toHaveBeenCalledWith(
      "http://localhost:8000/api/devices/lab-worker-1",
      expect.objectContaining({
        method: "PATCH",
        body: JSON.stringify({
          hostname: "Updated worker",
          ip: "127.0.0.1",
          port: 5001,
          training_port: 5004,
        }),
      }),
    );
    expect(onRefresh).toHaveBeenCalledOnce();
  });
});
