import { useCallback, useEffect, useState } from "react";

import { checkToken, getDevices, openDeviceEvents } from "./api/client";
import type { DeviceEvent } from "./api/client";
import type { Device } from "./types";
import DeviceCard from "./components/DeviceCard";

const navigation = [
  { id: "dashboard", label: "Dashboard", icon: "▦" },
  { id: "topology", label: "Topology", icon: "⌘" },
  { id: "experiments", label: "Experiments", icon: "◷" },
  { id: "models", label: "Model Studio", icon: "◈" },
  { id: "search", label: "Architecture Search", icon: "⌕" },
  { id: "assistant", label: "SLM Assistant", icon: "✦" },
];

export default function App() {
  const [token, setToken] = useState(() => sessionStorage.getItem("dtn-api-token") ?? "");
  const [page, setPage] = useState("dashboard");
  const [devices, setDevices] = useState<Device[]>([]);
  const [error, setError] = useState("");
  const [connected, setConnected] = useState(false);

  const refresh = useCallback(async (activeToken: string) => {
    try {
      setDevices(await getDevices(activeToken));
      setError("");
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Could not load device status.");
    }
  }, []);

  useEffect(() => {
    if (!token) return;
    void refresh(token);
    let reconnectTimer: number | undefined;
    let socket: WebSocket | undefined;
    let disposed = false;

    const connect = () => {
      if (disposed) return;
      socket = openDeviceEvents(
        token,
        (message: DeviceEvent) => {
          if (message.type === "device_snapshot") setDevices(message.devices);
          if (message.type === "device_status") {
            setDevices((current) => {
              const found = current.some((device) => device.node_id === message.device.node_id);
              return found
                ? current.map((device) =>
                    device.node_id === message.device.node_id ? message.device : device,
                  )
                : [...current, message.device];
            });
          }
        },
        () => setConnected(true),
        () => {
          setConnected(false);
          if (!disposed) reconnectTimer = window.setTimeout(connect, 2500);
        },
      );
    };
    connect();
    const fallbackRefresh = window.setInterval(() => void refresh(token), 15000);
    return () => {
      disposed = true;
      window.clearInterval(fallbackRefresh);
      if (reconnectTimer !== undefined) window.clearTimeout(reconnectTimer);
      socket?.close();
    };
  }, [token, refresh]);

  async function submitToken(value: string) {
    setError("");
    try {
      await checkToken(value);
      sessionStorage.setItem("dtn-api-token", value);
      setToken(value);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Authentication failed.");
    }
  }

  function signOut() {
    sessionStorage.removeItem("dtn-api-token");
    setToken("");
    setDevices([]);
    setConnected(false);
  }

  if (!token) {
    return <SignIn onSubmit={submitToken} error={error} />;
  }

  const onlineCount = devices.filter((device) => device.online).length;
  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-mark">D</div>
          <div>
            <strong>DFL-SLM</strong>
            <span>NETWORK TWIN</span>
          </div>
        </div>
        <div className="workspace-label">WORKSPACE</div>
        <nav aria-label="Main navigation">
          {navigation.map((item) => (
            <button
              className={`nav-item ${page === item.id ? "active" : ""}`}
              key={item.id}
              onClick={() => setPage(item.id)}
              type="button"
            >
              <span className="nav-icon">{item.icon}</span>
              {item.label}
            </button>
          ))}
        </nav>
        <div className="sidebar-bottom">
          <div className="cluster-health">
            <span className={`health-indicator ${onlineCount === devices.length && devices.length ? "healthy" : ""}`} />
            <div>
              <strong>Cluster status</strong>
              <span>{onlineCount}/{devices.length || 4} nodes online</span>
            </div>
          </div>
          <button className="sign-out" type="button" onClick={signOut}>Sign out</button>
        </div>
      </aside>

      <main className="main-area">
        <header className="topbar">
          <div className="breadcrumb">Workspace <span>/</span> {navigation.find((item) => item.id === page)?.label}</div>
          <div className="topbar-right">
            <span className={`connection-status ${connected ? "connected" : ""}`}>
              <span className="status-dot" />
              {connected ? "Live updates" : "Reconnecting"}
            </span>
            <div className="user-avatar" aria-label="Signed in user">H</div>
          </div>
        </header>

        {page === "dashboard" && (
          <Dashboard devices={devices} error={error} connected={connected} onRetry={() => void refresh(token)} />
        )}
        {page === "topology" && <Topology devices={devices} />}
        {page !== "dashboard" && page !== "topology" && (
          <Placeholder page={page} />
        )}
      </main>
    </div>
  );
}

function SignIn({ onSubmit, error }: { onSubmit: (token: string) => void; error: string }) {
  const [value, setValue] = useState("");
  return (
    <main className="login-screen">
      <form
        className="login-card"
        onSubmit={(event) => {
          event.preventDefault();
          void onSubmit(value.trim());
        }}
      >
        <div className="brand-mark login-mark">D</div>
        <span className="eyebrow">DIGITAL TWIN NETWORK</span>
        <h1>Sign in to your cluster</h1>
        <p>Enter the coordinator API token to view authenticated, live Jetson status.</p>
        <label htmlFor="api-token">API token</label>
        <input
          autoComplete="off"
          id="api-token"
          onChange={(event) => setValue(event.target.value)}
          placeholder="Paste coordinator token"
          type="password"
          value={value}
        />
        {error && <div role="alert" className="alert">{error}</div>}
        <button className="primary-button" disabled={!value.trim()} type="submit">Connect to network</button>
        <small>The token is kept in this browser tab session only.</small>
      </form>
    </main>
  );
}

function Dashboard({
  devices,
  error,
  connected,
  onRetry,
}: {
  devices: Device[];
  error: string;
  connected: boolean;
  onRetry: () => void;
}) {
  const online = devices.filter((device) => device.online).length;
  return (
    <section className="content">
      <div className="page-heading">
        <div>
          <span className="eyebrow">NETWORK OVERVIEW</span>
          <h1>Infrastructure dashboard</h1>
          <p>Live health and resource utilization across your distributed cluster.</p>
        </div>
        <div className={`live-badge ${connected ? "active" : ""}`}>
          <span className="status-dot" /> {connected ? "LIVE" : "CONNECTING"}
        </div>
      </div>

      <div className="summary-grid">
        <Summary label="Cluster nodes" value={devices.length || 4} detail="Registered devices" mark="◇" />
        <Summary label="Online nodes" value={`${online}/${devices.length || 4}`} detail="Reporting status" mark="↗" />
        <Summary label="Network" value={connected ? "Connected" : "Waiting"} detail="Live status stream" mark="⌁" />
      </div>

      <div className="section-heading">
        <div>
          <h2>Compute nodes</h2>
          <p>Status updates are polled from each device and streamed in real time.</p>
        </div>
        <button className="quiet-button" onClick={onRetry} type="button">Refresh</button>
      </div>

      {error && (
        <div className="alert inline-alert" role="alert">
          {error} <button type="button" onClick={onRetry}>Retry</button>
        </div>
      )}

      {devices.length === 0 && !error ? (
        <div className="device-grid">
          {Array.from({ length: 4 }, (_, index) => <div className="device-skeleton" key={index} />)}
        </div>
      ) : (
        <div className="device-grid">
          {devices.map((device) => <DeviceCard device={device} key={device.node_id} />)}
        </div>
      )}
      <div className="data-note"><span>i</span> Metrics shown are collected directly from the physical Jetson devices.</div>
    </section>
  );
}

function Summary({ label, value, detail, mark }: { label: string; value: string | number; detail: string; mark: string }) {
  return (
    <article className="summary-card">
      <div className="summary-card__top"><span>{label}</span><span className="summary-mark">{mark}</span></div>
      <strong>{value}</strong>
      <small>{detail}</small>
    </article>
  );
}

function Topology({ devices }: { devices: Device[] }) {
  const master = devices.find((device) => device.role === "master");
  const workers = devices.filter((device) => device.role === "worker");
  return (
    <section className="content">
      <div className="page-heading">
        <div>
          <span className="eyebrow">CLUSTER MAP</span>
          <h1>Network topology</h1>
          <p>Coordinator and worker layout for the distributed learning cluster.</p>
        </div>
      </div>
      <div className="topology-panel">
        <div className="topology-legend">
          <span><i className="legend-master" /> Master coordinator</span>
          <span><i className="legend-worker" /> Worker node</span>
          <span><i className="legend-line" /> Network link</span>
        </div>
        <svg className="topology-lines" viewBox="0 0 800 440" role="img" aria-label="Master connected to three worker nodes">
          <line x1="400" y1="150" x2="145" y2="310" />
          <line x1="400" y1="150" x2="400" y2="310" />
          <line x1="400" y1="150" x2="655" y2="310" />
          <circle cx="400" cy="150" r="5" /><circle cx="145" cy="310" r="5" />
          <circle cx="400" cy="310" r="5" /><circle cx="655" cy="310" r="5" />
        </svg>
        <div className="topology-nodes">
          <TopologyNode device={master} fallback="dfl-master" role="master" />
          <div className="worker-nodes">
            {[0, 1, 2].map((index) => (
              <TopologyNode
                device={workers[index]}
                fallback={`dfl-worker${index + 1}`}
                key={workers[index]?.node_id ?? index}
                role="worker"
              />
            ))}
          </div>
        </div>
      </div>
    </section>
  );
}

function TopologyNode({ device, fallback, role }: { device?: Device; fallback: string; role: "master" | "worker" }) {
  return (
    <article className={`topology-node ${role} ${device?.online ? "node-online" : "node-offline"}`}>
      <div className="node-orb">{role === "master" ? "M" : "W"}</div>
      <div>
        <strong>{device?.hostname ?? fallback}</strong>
        <span>{device ? `${device.online ? "Online" : "Offline"} · ${device.ip}` : "Waiting for registry"}</span>
      </div>
      <span className={`node-state ${device?.online ? "online" : "offline"}`} />
    </article>
  );
}

function Placeholder({ page }: { page: string }) {
  const item = navigation.find((candidate) => candidate.id === page);
  return (
    <section className="content">
      <div className="page-heading">
        <div>
          <span className="eyebrow">WORKSPACE</span>
          <h1>{item?.label}</h1>
          <p>This workspace is prepared for the next phase of the DFL-SLM platform.</p>
        </div>
      </div>
      <div className="coming-soon">
        <div className="coming-icon">{item?.icon}</div>
        <span className="eyebrow">PHASE 3+</span>
        <h2>{item?.label} workspace</h2>
        <p>Device connectivity and live monitoring are active. This feature will be built in a later phase.</p>
        <span className="planned-tag">Planned</span>
      </div>
    </section>
  );
}
