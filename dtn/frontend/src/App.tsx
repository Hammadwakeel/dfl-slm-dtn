import { useCallback, useEffect, useState } from "react";

import { checkToken, getDevices, getExperimentResults, openDeviceEvents } from "./api/client";
import { sendSlmCommand } from "./api/client";
import type { DeviceEvent, ExperimentGroup, ExperimentResults, TrainingTopology } from "./api/client";
import type { Device } from "./types";
import DeviceCard from "./components/DeviceCard";
import AssistantPage from "./pages/AssistantPage";
import ExperimentsPage from "./pages/ExperimentsPage";

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
  const [experimentResults, setExperimentResults] = useState<ExperimentResults | null>(null);
  const [experimentError, setExperimentError] = useState("");
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

  const refreshExperiments = useCallback(async (activeToken: string) => {
    try {
      setExperimentResults(await getExperimentResults(activeToken));
      setExperimentError("");
    } catch (cause) {
      setExperimentError(cause instanceof Error ? cause.message : "Could not load experiment results.");
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

  useEffect(() => {
    if (!token) return;
    void refreshExperiments(token);
    const timer = window.setInterval(() => void refreshExperiments(token), 3000);
    return () => window.clearInterval(timer);
  }, [token, refreshExperiments]);

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
    setExperimentResults(null);
    setExperimentError("");
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
          <Dashboard
            devices={devices}
            error={error}
            connected={connected}
            experiments={experimentResults}
            onRetry={() => void refresh(token)}
          />
        )}
        {page === "topology" && <Topology devices={devices} experiments={experimentResults} token={token} />}
        {page === "experiments" && (
          <ExperimentsPage
            token={token}
            results={experimentResults}
            error={experimentError}
            onRefresh={() => void refreshExperiments(token)}
          />
        )}
        {page === "assistant" && <AssistantPage token={token} />}
        {page !== "dashboard" && page !== "topology" && (
          page !== "assistant" && page !== "experiments" && <Placeholder page={page} />
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
  experiments,
  onRetry,
}: {
  devices: Device[];
  error: string;
  connected: boolean;
  experiments: ExperimentResults | null;
  onRetry: () => void;
}) {
  const online = devices.filter((device) => device.online).length;
  const latest = latestExperiment(experiments?.groups ?? []);
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

      {latest && (
        <section className="latest-experiment panel-card">
          <div>
            <span className="eyebrow">LATEST TRAINING ROUND · {latest.group.topology.toUpperCase()}</span>
            <h2>{latest.group.group_id}</h2>
            <p>Seed {latest.run.seed} · Round {latest.round.round} · {latest.group.status}</p>
          </div>
          <div className="latest-experiment__metrics">
            {Object.entries(latest.round.workers).map(([node, metrics]) => (
              <div key={node}>
                <span>{node}</span>
                <strong>{(metrics.accuracy * 100).toFixed(2)}%</strong>
                <small>test accuracy · {metrics.peak_process_rss_mb.toFixed(0)} MB peak RAM</small>
              </div>
            ))}
          </div>
        </section>
      )}

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

function Topology({
  devices,
  experiments,
  token,
}: {
  devices: Device[];
  experiments: ExperimentResults | null;
  token: string;
}) {
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
      <TrainingTopologyPanel experiments={experiments} token={token} />
    </section>
  );
}

export function TrainingTopologyPanel({
  experiments,
  token,
}: {
  experiments: ExperimentResults | null;
  token: string;
}) {
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState("");
  const [error, setError] = useState("");
  const topology = experiments?.configuration.topology ?? "directed_ring";
  const latestGroup = [...(experiments?.groups ?? [])]
    .reverse()
    .find((group) => group.kind === "topology_optimization");
  const links = topologyLinks(topology);

  async function optimize() {
    setBusy(true);
    setMessage("");
    setError("");
    try {
      const response = await sendSlmCommand(token, {
        action: "optimize_topology",
        parameters: { rounds: 10, epochs: 5, seeds: [17, 42, 2026], alphas: [0.3, 0.5] },
      });
      if (!response.result.available) throw new Error(response.result.message ?? "Optimizer unavailable.");
      const groupId = response.result.data?.group_id;
      setMessage(groupId ? `Real-device topology sweep ${String(groupId)} started.` : "Topology sweep started.");
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Could not start topology optimization.");
    } finally {
      setBusy(false);
    }
  }

  return (
    <section className="training-map panel-card">
      <div className="section-heading">
        <div>
          <h2>Training topology optimizer</h2>
          <p>Candidate runs train on real workers using Dirichlet alpha 0.3/0.5 shards.</p>
        </div>
        <div className="topology-actions">
          <span className="run-status completed">Active: {topology}</span>
          <button className="primary-button" disabled={busy || Boolean(experiments?.active_group_id)} onClick={() => void optimize()} type="button">
            {busy ? "Starting…" : "Run 5-topology sweep"}
          </button>
        </div>
      </div>
      <p className="active-topology-note">
        Active configuration: <strong>{topology}</strong> · {experiments?.configuration.dataset_split ?? "stratified"}
      </p>
      <div className="training-map__nodes">
        {["dfl-worker1", "dfl-worker2", "dfl-worker3"].map((worker) => <span key={worker}>{worker}</span>)}
      </div>
      <svg
        className="training-graph"
        viewBox="0 0 600 210"
        role="img"
        aria-label={`${topology} worker communication graph`}
      >
        <defs>
          <marker id="topology-arrow" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto">
            <path d="M0,0 L0,6 L6,3 z" />
          </marker>
        </defs>
        {topologyEdges(topology).map(([from, to]) => {
          const positions = [{ x: 300, y: 45 }, { x: 145, y: 155 }, { x: 455, y: 155 }];
          const start = positions[from];
          const end = positions[to];
          return (
            <line
              key={`${from}-${to}`}
              x1={start.x}
              y1={start.y}
              x2={end.x}
              y2={end.y}
              markerEnd={topology === "directed_ring" || topology === "ring" ? "url(#topology-arrow)" : undefined}
            />
          );
        })}
        {[{ x: 300, y: 45 }, { x: 145, y: 155 }, { x: 455, y: 155 }].map((point, index) => (
          <g key={index}>
            <circle cx={point.x} cy={point.y} r="23" />
            <text x={point.x} y={point.y + 4}>{`W${index + 1}`}</text>
          </g>
        ))}
      </svg>
      <div className="training-map__links">{links.map((link) => <span key={link}>{link}</span>)}</div>
      {message && <div className="success-note" role="status">{message}</div>}
      {error && <div className="alert inline-alert" role="alert">{error}</div>}
      {latestGroup?.selection && (
        <div className="optimizer-selection">
          <strong>Measured selector: {latestGroup.selection.method}</strong>
          <span>
            Selected {latestGroup.selection.selected_candidate.topology} at alpha {latestGroup.selection.selected_candidate.alpha}
            {" "}· reward {latestGroup.selection.selected_candidate.mean_reward.toFixed(4)}
          </span>
          <span>Random baseline: {latestGroup.selection.baselines.random_seeded.topology} · reward {latestGroup.selection.baselines.random_seeded.mean_reward?.toFixed(4) ?? "—"}</span>
          <span>Best-observed baseline: {latestGroup.selection.baselines.best_observed.topology} · accuracy {formatAccuracy(latestGroup.selection.baselines.best_observed.mean_final_accuracy)}</span>
        </div>
      )}
      {latestGroup?.candidates?.length ? (
        <div className="table-scroll">
          <table className="metrics-table topology-reward-table">
            <thead>
              <tr><th>Topology</th><th>Alpha</th><th>Runs</th><th>Reward</th><th>Final accuracy</th><th>Gain</th><th>Convergence</th><th>Communication</th><th>Runtime</th></tr>
            </thead>
            <tbody>
              {latestGroup.candidates.map((candidate) => (
                <tr key={`${candidate.topology}-${candidate.alpha}`}>
                  <td>{candidate.topology}</td>
                  <td>{candidate.alpha}</td>
                  <td>{candidate.completed_runs}</td>
                  <td>{candidate.mean_reward.toFixed(4)}</td>
                  <td>{formatAccuracy(candidate.mean_final_accuracy)}</td>
                  <td>{candidate.mean_components.accuracy_gain?.toFixed(3) ?? "—"}</td>
                  <td>{candidate.mean_components.convergence_speed?.toFixed(3) ?? "—"}</td>
                  <td>{candidate.mean_components.communication_efficiency?.toFixed(3) ?? "—"}</td>
                  <td>{candidate.mean_components.runtime_efficiency?.toFixed(3) ?? "—"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : (
        <p className="active-topology-note">No measured topology-optimization results yet.</p>
      )}
      {latestGroup && <small className="active-topology-note">Latest sweep: {latestGroup.group_id} · {latestGroup.status} · {latestGroup.rounds} rounds × {latestGroup.epochs} epochs · seeds {latestGroup.seeds.join(", ")}</small>}
    </section>
  );
}

function topologyLinks(topology: TrainingTopology): string[] {
  if (topology === "star") return ["dfl-worker1 ↔ dfl-worker2", "dfl-worker1 ↔ dfl-worker3"];
  if (topology === "chain") return ["dfl-worker1 ↔ dfl-worker2", "dfl-worker2 ↔ dfl-worker3"];
  if (topology === "directed_ring" || topology === "ring") {
    return ["dfl-worker1 → dfl-worker2", "dfl-worker2 → dfl-worker3", "dfl-worker3 → dfl-worker1"];
  }
  return ["dfl-worker1 ↔ dfl-worker2", "dfl-worker2 ↔ dfl-worker3", "dfl-worker3 ↔ dfl-worker1"];
}

function topologyEdges(topology: TrainingTopology): Array<[number, number]> {
  if (topology === "star") return [[0, 1], [0, 2]];
  if (topology === "chain") return [[0, 1], [1, 2]];
  if (topology === "directed_ring" || topology === "ring") return [[0, 1], [1, 2], [2, 0]];
  return [[0, 1], [1, 2], [2, 0]];
}

function formatAccuracy(value?: number): string {
  return value === undefined ? "—" : `${(value * 100).toFixed(2)}%`;
}

function latestExperiment(groups: ExperimentGroup[]) {
  const candidates = groups
    .filter((group) => group.kind === "decentralized" || group.kind === "centralized_baseline")
    .flatMap((group) => group.runs.flatMap((run) => {
      const round = run.rounds[run.rounds.length - 1];
      return round ? [{ group, run, round }] : [];
    }));
  return candidates[candidates.length - 1];
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
