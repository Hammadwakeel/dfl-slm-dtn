export interface DeviceStatus {
  node_id: string;
  role: "master" | "worker";
  hostname: string;
  sampled_at: string;
  cpu_utilization_percent: number | null;
  gpu_utilization_percent: number | null;
  memory: {
    total_bytes: number;
    available_bytes: number;
    used_bytes: number;
  };
  temperatures_celsius: Record<string, number>;
}

export interface Device {
  node_id: string;
  role: "master" | "worker";
  hostname: string;
  ip: string;
  port: number;
  training_port?: number | null;
  is_managed?: boolean;
  training_ready?: boolean;
  training_error?: string | null;
  online: boolean;
  last_seen: string | null;
  last_error: string | null;
  latest_status: DeviceStatus | null;
}
