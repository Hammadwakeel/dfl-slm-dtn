import type { Device } from "../types";

interface DeviceCardProps {
  device: Device;
}

function percentage(value: number | null | undefined): string {
  return value == null ? "--" : `${Math.round(value)}%`;
}

function memory(device: Device): string {
  const status = device.latest_status?.memory;
  if (!status || status.total_bytes <= 0) return "--";
  return `${Math.round((status.used_bytes / status.total_bytes) * 100)}%`;
}

function temperature(device: Device): string {
  const readings = Object.values(device.latest_status?.temperatures_celsius ?? {});
  return readings.length ? `${Math.round(Math.max(...readings))}°C` : "--";
}

function relativeTime(value: string | null): string {
  if (!value) return "Never";
  const seconds = Math.max(0, Math.floor((Date.now() - new Date(value).getTime()) / 1000));
  if (seconds < 60) return `${seconds}s ago`;
  if (seconds < 3600) return `${Math.floor(seconds / 60)}m ago`;
  return `${Math.floor(seconds / 3600)}h ago`;
}

export default function DeviceCard({ device }: DeviceCardProps) {
  const status = device.latest_status;
  return (
    <article className={`device-card ${device.online ? "is-online" : "is-offline"}`}>
      <header className="device-card__header">
        <div className="device-card__identity">
          <div className={`device-avatar ${device.role}`}>
            {device.role === "master" ? "M" : "W"}
          </div>
          <div>
            <h3>{device.hostname}</h3>
            <p>{device.role === "master" ? "Coordinator" : "Worker node"} · {device.ip}</p>
          </div>
        </div>
        <span className={`status-pill ${device.online ? "online" : "offline"}`}>
          <span className="status-dot" />
          {device.online ? "Online" : "Offline"}
        </span>
      </header>

      <div className="metric-grid">
        <Metric label="CPU" value={percentage(status?.cpu_utilization_percent)} />
        <Metric label="GPU" value={percentage(status?.gpu_utilization_percent)} />
        <Metric label="Memory" value={memory(device)} />
        <Metric label="Temperature" value={temperature(device)} />
      </div>

      <footer className="device-card__footer">
        <span>Last seen</span>
        <strong>{relativeTime(device.last_seen)}</strong>
      </footer>
    </article>
  );
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div className="metric">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}
