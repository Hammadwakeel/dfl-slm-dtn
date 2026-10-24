import type { Device } from "../types";

export type DeviceEvent =
  | { type: "device_snapshot"; devices: Device[] }
  | { type: "device_status"; device: Device };

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
