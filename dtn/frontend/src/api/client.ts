import type { Device } from "../types";

export type DeviceEvent =
  | { type: "device_snapshot"; devices: Device[] }
  | { type: "device_status"; device: Device };

export interface SlmChatResponse {
  reply: string;
  model_reply: string;
  action: {
    action: string;
    parameters: Record<string, unknown>;
  };
  result: {
    available: boolean;
    message?: string;
    data?: unknown;
  };
  metrics: {
    latency_ms: number;
    model_latency_ms: number;
    system_ram_total_mb: number;
    system_ram_used_before_mb: number;
    system_ram_used_after_mb: number;
    system_ram_used_delta_mb: number;
    system_ram_available_after_mb: number;
  };
}

export interface WorkerRoundMetrics {
  node_id: string;
  run_id: string;
  round: number;
  topology: "ring" | "mesh" | "centralized";
  local_epochs?: number;
  sample_count?: number;
  training_loss?: number;
  training_accuracy?: number;
  accuracy: number;
  loss: number;
  macro_f1: number;
  training_seconds?: number;
  runtime_seconds?: number;
  round_seconds?: number;
  evaluation_seconds?: number;
  rss_before_mb?: number;
  rss_after_mb?: number;
  peak_process_rss_mb: number;
  system_memory_available_delta_mb?: number;
  bytes_sent?: number;
  bytes_received?: number;
  peer_messages_sent?: number;
  peer_messages_received?: number;
  exchange_seconds?: number;
  neighbours?: string[];
  weight_bytes?: number;
}

export interface ExperimentRound {
  round: number;
  topology: "ring" | "mesh" | "centralized";
  completed_at?: number;
  workers: Record<string, WorkerRoundMetrics>;
}

export interface ExperimentRun {
  run_id: string;
  seed: number;
  topology: "ring" | "mesh" | "centralized";
  status: "running" | "completed" | "failed";
  rounds: ExperimentRound[];
}

export interface ExperimentGroup {
  group_id: string;
  kind: "decentralized" | "centralized_baseline";
  topology: "ring" | "mesh" | "centralized";
  rounds: number;
  epochs: number;
  seeds: number[];
  status: "running" | "completed" | "failed" | "stopped";
  started_at?: number;
  completed_at?: number | null;
  error: string | null;
  runs: ExperimentRun[];
}

export interface ExperimentResults {
  configuration: { rounds: number; epochs: number; topology: "ring" | "mesh" };
  active_group_id: string | null;
  topologies: {
    ring: string[];
    mesh: string[][];
  };
  groups: ExperimentGroup[];
}

export interface SlmCommandResult {
  action: { action: string; parameters: Record<string, unknown> };
  result: { available: boolean; message?: string; data?: Record<string, unknown> };
}

const apiBase =
  import.meta.env.VITE_API_BASE_URL ??
  `${window.location.protocol}//${window.location.hostname}:8000`;

export async function checkToken(token: string): Promise<void> {
  const response = await fetch(`${apiBase}/api/auth/check`, {
    headers: { Authorization: `Bearer ${token}` },
  });
  if (!response.ok) {
    throw new Error(response.status === 401 ? "Token was not accepted." : `API error (${response.status}).`);
  }
}

export async function getDevices(token: string): Promise<Device[]> {
  const response = await fetch(`${apiBase}/api/devices`, {
    headers: { Authorization: `Bearer ${token}` },
  });
  if (!response.ok) throw new Error(`Could not load devices (HTTP ${response.status}).`);
  return response.json() as Promise<Device[]>;
}

export async function sendSlmMessage(
  token: string,
  message: string,
): Promise<SlmChatResponse> {
  const response = await fetch(`${apiBase}/api/slm/chat`, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${token}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ message }),
  });
  if (!response.ok) {
    const body = (await response.json().catch(() => ({}))) as { detail?: string; error?: string };
    throw new Error(body.detail ?? body.error ?? `SLM request failed (HTTP ${response.status}).`);
  }
  return response.json() as Promise<SlmChatResponse>;
}

export async function sendSlmCommand(
  token: string,
  action: SlmCommandResult["action"],
): Promise<SlmCommandResult> {
  const response = await fetch(`${apiBase}/api/slm/command`, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${token}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify(action),
  });
  if (!response.ok) throw new Error(`Training command failed (HTTP ${response.status}).`);
  return response.json() as Promise<SlmCommandResult>;
}

export async function getExperimentResults(token: string): Promise<ExperimentResults> {
  const response = await fetch(`${apiBase}/api/slm/results`, {
    headers: { Authorization: `Bearer ${token}` },
  });
  if (!response.ok) throw new Error(`Could not load experiment results (HTTP ${response.status}).`);
  return response.json() as Promise<ExperimentResults>;
}

export async function startCentralizedBaseline(
  token: string,
  rounds: number,
  epochs: number,
): Promise<void> {
  const response = await fetch(`${apiBase}/api/slm/baseline`, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${token}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ rounds, epochs }),
  });
  if (!response.ok) throw new Error(`Could not start centralized baseline (HTTP ${response.status}).`);
}

export function openDeviceEvents(
  token: string,
  onMessage: (message: DeviceEvent) => void,
  onOpen?: () => void,
  onClose?: () => void,
): WebSocket {
  const wsBase = apiBase.replace(/^http/, "ws");
  const socket = new WebSocket(`${wsBase}/ws/devices`);
  socket.addEventListener("open", () => {
    socket.send(JSON.stringify({ type: "auth", token }));
    onOpen?.();
  });
  socket.addEventListener("message", (event) => {
    try {
      const message = JSON.parse(String(event.data)) as DeviceEvent;
      onMessage(message);
    } catch {
      // Ignore malformed frames; the REST API remains authoritative.
    }
  });
  socket.addEventListener("close", () => onClose?.());
  return socket;
}
