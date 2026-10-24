import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import DeviceCard from "./DeviceCard";
import type { Device } from "../types";

const onlineDevice: Device = {
  node_id: "master",
  role: "master",
  hostname: "dfl-master",
  ip: "192.168.11.15",
  port: 5002,
  online: true,
  last_seen: "2026-10-24T12:00:00Z",
  last_error: null,
  latest_status: {
    node_id: "master",
    role: "master",
    hostname: "dfl-master",
    sampled_at: "2026-10-24T12:00:00Z",
    cpu_utilization_percent: 38,
    gpu_utilization_percent: 17,
    memory: { total_bytes: 1000, available_bytes: 400, used_bytes: 600 },
    temperatures_celsius: { cpu: 51.2, gpu: 48.9 },
  },
};

describe("DeviceCard", () => {
  it("shows real device metrics and online state", () => {
    render(<DeviceCard device={onlineDevice} />);
    expect(screen.getByText("dfl-master")).toBeInTheDocument();
    expect(screen.getByText("Online")).toBeInTheDocument();
    expect(screen.getByText("38%")).toBeInTheDocument();
    expect(screen.getByText("17%")).toBeInTheDocument();
    expect(screen.getByText("60%")).toBeInTheDocument();
    expect(screen.getByText("51°C")).toBeInTheDocument();
  });

  it("shows unavailable readings for an offline device without samples", () => {
    render(
      <DeviceCard
        device={{
          ...onlineDevice,
          online: false,
          last_seen: null,
          latest_status: null,
        }}
      />,
    );
    expect(screen.getByText("Offline")).toBeInTheDocument();
    expect(screen.getAllByText("--")).toHaveLength(4);
    expect(screen.getByText("Never")).toBeInTheDocument();
  });
});
