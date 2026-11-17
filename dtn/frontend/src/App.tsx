import { useCallback, useEffect, useState } from "react";
import type { ReactNode } from "react";

import {
  checkToken,
  dispatchAutonomyAction,
  getDevices,
  getExperimentResults,
  openDeviceEvents,
} from "./api/client";
import type { DeviceEvent, ExperimentGroup, ExperimentResults, NASArchitecture, NASCandidate, NASCustomArchitecture, TopologyCandidate, TrainingTopology, WorkerRoundMetrics } from "./api/client";
import type { Device } from "./types";
import DeviceCard from "./components/DeviceCard";
import DatabaseStatusBot from "./components/DatabaseStatusBot";
import WorkerMemoryCleanupControl from "./components/WorkerMemoryCleanupControl";
import AssistantPage from "./pages/AssistantPage";
import ExperimentsPage from "./pages/ExperimentsPage";
import NASPage from "./pages/NASPage";
import DatasetsPage from "./pages/DatasetsPage";
import ModelStudioPage from "./pages/ModelStudioPage";
import ModelDetailsPage from "./pages/ModelDetailsPage";
import DevicesPage from "./pages/DevicesPage";
import WorkerOperationsPage from "./pages/WorkerOperationsPage";
import AgentPage from "./pages/AgentPage";

const navigation = [
  { id: "dashboard", label: "Dashboard" },
  { id: "topology", label: "Topology" },
  { id: "devices", label: "Devices" },
  { id: "operations", label: "Worker Operations" },
  { id: "experiments", label: "Experiments" },
  { id: "datasets", label: "Datasets" },
  { id: "models", label: "Model Studio" },
  { id: "search", label: "Architecture Search" },
  { id: "agent", label: "Agent Supervisor" },
  { id: "assistant", label: "SLM Assistant" },
] as const;
type NavigationIconName = (typeof navigation)[number]["id"] | "activity" | "network" | "nodes" | "expand" | "collapse" | "signOut";
const TOPOLOGY_SMOKE_GROUP_KEY = "dtn-topology-smoke-group-id";
const PAGE_STORAGE_KEY = "dtn-active-page";
const NAVIGATION_PAGE_IDS = new Set<string>(navigation.map((item) => item.id));

function storedPage(): string {
  const savedPage = sessionStorage.getItem(PAGE_STORAGE_KEY);
  return savedPage && NAVIGATION_PAGE_IDS.has(savedPage) ? savedPage : "dashboard";
}

function NavigationIcon({ name }: { name: NavigationIconName }) {
  const paths: Record<NavigationIconName, ReactNode> = {
    dashboard: <><rect x="3" y="3" width="7" height="7" rx="1.5" /><rect x="14" y="3" width="7" height="7" rx="1.5" /><rect x="3" y="14" width="7" height="7" rx="1.5" /><rect x="14" y="14" width="7" height="7" rx="1.5" /></>,
    topology: <><circle cx="12" cy="5" r="2.5" /><circle cx="6" cy="18" r="2.5" /><circle cx="18" cy="18" r="2.5" /><path d="M10.8 7.2 7.2 15.8m6-8.6 3.6 8.6M8.5 18h7" /></>,
    devices: <><rect x="3" y="4" width="18" height="7" rx="2" /><rect x="3" y="14" width="18" height="7" rx="2" /><path d="M7 7.5h.01M7 17.5h.01M11 7.5h6m-6 10h6" /></>,
    operations: <><path d="M12 3v3m0 12v3M3 12h3m12 0h3" /><circle cx="12" cy="12" r="6" /><path d="m12 9 2 3-2 3-2-3 2-3Z" /></>,
    experiments: <><circle cx="12" cy="12" r="9" /><path d="M12 7v5l3 2" /></>,
    datasets: <><ellipse cx="12" cy="5" rx="8" ry="3" /><path d="M4 5v7c0 1.7 3.6 3 8 3s8-1.3 8-3V5m-16 7v7c0 1.7 3.6 3 8 3s8-1.3 8-3v-7" /></>,
    models: <><path d="m12 3 8 4.5v9L12 21l-8-4.5v-9L12 3Z" /><path d="m4.5 7.7 7.5 4.4 7.5-4.4M12 12v9" /></>,
    search: <><circle cx="10.8" cy="10.8" r="6.8" /><path d="m16 16 4.5 4.5M10.8 7.5v6.6m-3.3-3.3h6.6" /></>,
    assistant: <><path d="m12 3 1.8 5.2L19 10l-5.2 1.8L12 17l-1.8-5.2L5 10l5.2-1.8L12 3Z" /><path d="m19 14 .9 2.1L22 17l-2.1.9L19 20l-.9-2.1L16 17l2.1-.9L19 14ZM5 3l.7 1.8L7.5 5.5l-1.8.7L5 8l-.7-1.8-1.8-.7 1.8-.7L5 3Z" /></>,
    agent: <><path d="M5 4h14v4H5zM5 10h14v4H5zM5 16h9v4H5z" /><path d="m17 18 2 2 3-4" /></>,
    nodes: <><circle cx="6" cy="12" r="3" /><circle cx="18" cy="6" r="3" /><circle cx="18" cy="18" r="3" /><path d="m8.6 10.7 6.8-3.4m-6.8 6 6.8 3.4" /></>,
    activity: <><path d="M3 12h4l3-8 4 16 3-8h4" /></>,
    network: <><path d="M12 3v5m0 8v5M3 12h5m8 0h5" /><circle cx="12" cy="12" r="4" /><circle cx="12" cy="3" r="1" /><circle cx="12" cy="21" r="1" /><circle cx="3" cy="12" r="1" /><circle cx="21" cy="12" r="1" /></>,
    expand: <><rect x="3" y="3" width="18" height="18" rx="3" /><path d="M9 3v18m6-12 3 3-3 3" /></>,
    collapse: <><rect x="3" y="3" width="18" height="18" rx="3" /><path d="M9 3v18m9-12-3 3 3 3" /></>,
    signOut: <><path d="M10 17l5-5-5-5m5 5H3" /><path d="M12 3h6a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2h-6" /></>,
  };
  return <svg aria-hidden="true" className="nav-icon" fill="none" stroke="currentColor" strokeLinecap="round" strokeLinejoin="round" strokeWidth="1.7" viewBox="0 0 24 24">{paths[name]}</svg>;
}

export default function App() {
  const [token, setToken] = useState(() => sessionStorage.getItem("dtn-api-token") ?? "");
  const [page, setActivePage] = useState(storedPage);
  const [visitedPages, setVisitedPages] = useState<Set<string>>(
    () => new Set(["dashboard", storedPage()]),
  );
  const [sidebarCollapsed, setSidebarCollapsed] = useState(true);
  const [devices, setDevices] = useState<Device[]>([]);
  const [experimentResults, setExperimentResults] = useState<ExperimentResults | null>(null);
  const [selectedNasArchitecture, setSelectedNasArchitecture] = useState<
    NASCustomArchitecture | NASArchitecture | null
  >(null);
  const [selectedNasCandidate, setSelectedNasCandidate] = useState<{
    candidate: NASCandidate;
    group: ExperimentGroup;
  } | null>(null);
  const [experimentError, setExperimentError] = useState("");
  const [error, setError] = useState("");
  const [connected, setConnected] = useState(false);

  function navigateTo(nextPage: string) {
    setActivePage(nextPage);
    setVisitedPages((current) => new Set(current).add(nextPage));
    if (NAVIGATION_PAGE_IDS.has(nextPage)) {
      sessionStorage.setItem(PAGE_STORAGE_KEY, nextPage);
    }
  }

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
    // This starts an asynchronous external API refresh, not a synchronous derived-state update.
    // eslint-disable-next-line react-hooks/set-state-in-effect
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
                  device.node_id === message.device.node_id
                    ? { ...device, ...message.device }
                    : device,
                )
                : [...current, message.device];
            });
          }
          if (message.type === "device_removed") {
            setDevices((current) =>
              current.filter((device) => device.node_id !== message.node_id),
            );
          }
        },
        () => setConnected(true),
        () => {
          setConnected(false);
          if (!disposed) reconnectTimer = window.setTimeout(connect, 2500);
        },
      );
    };
    const initialConnectTimer = window.setTimeout(connect, 0);
    const fallbackRefresh = window.setInterval(() => void refresh(token), 15000);
    return () => {
      disposed = true;
      if (initialConnectTimer !== undefined) window.clearTimeout(initialConnectTimer);
      window.clearInterval(fallbackRefresh);
      if (reconnectTimer !== undefined) window.clearTimeout(reconnectTimer);
      socket?.close();
    };
  }, [token, refresh]);

  useEffect(() => {
    if (!token) return;
    // This starts an asynchronous external API refresh, not a synchronous derived-state update.
    // eslint-disable-next-line react-hooks/set-state-in-effect
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
    sessionStorage.removeItem(TOPOLOGY_SMOKE_GROUP_KEY);
    sessionStorage.removeItem(PAGE_STORAGE_KEY);
    setToken("");
    setActivePage("dashboard");
    setVisitedPages(new Set(["dashboard"]));
    setDevices([]);
    setExperimentResults(null);
    setSelectedNasArchitecture(null);
    setSelectedNasCandidate(null);
    setExperimentError("");
    setConnected(false);
  }

  if (!token) {
    return <SignIn onSubmit={submitToken} error={error} />;
  }

  const onlineCount = devices.filter((device) => device.online).length;
  return (
    <div className={`app-shell ${sidebarCollapsed ? "sidebar-collapsed" : ""}`}>
      <aside className={`sidebar ${sidebarCollapsed ? "collapsed" : ""}`}>
        <div className="workspace-label">WORKSPACE</div>
        <nav aria-label="Main navigation">
          {navigation.map((item) => (
            <button
              className={`nav-item ${page === item.id ? "active" : ""}`}
              aria-label={item.label}
              aria-current={page === item.id ? "page" : undefined}
              data-tooltip={item.label}
              key={item.id}
              onClick={() => navigateTo(item.id)}
              title={item.label}
              type="button"
            >
              <NavigationIcon name={item.id} />
              <span className="nav-label">{item.label}</span>
            </button>
          ))}
        </nav>
        <div className="sidebar-bottom">
          <div className="cluster-health" title={`${onlineCount}/${devices.length || 4} nodes online`}>
            <span className={`health-indicator ${onlineCount === devices.length && devices.length ? "healthy" : ""}`} />
            <div className="cluster-health-label">
              <strong>Cluster status</strong>
              <span>{onlineCount}/{devices.length || 4} nodes online</span>
            </div>
          </div>
          <button
            aria-label="Sign out"
            className="sign-out"
            data-tooltip="Sign out"
            onClick={signOut}
            title="Sign out"
            type="button"
          >
            <NavigationIcon name="signOut" />
            <span className="sign-out-label">Sign out</span>
          </button>
          <button
            aria-expanded={!sidebarCollapsed}
            aria-label={sidebarCollapsed ? "Expand sidebar" : "Collapse sidebar"}
            className="sidebar-toggle"
            onClick={() => setSidebarCollapsed((collapsed) => !collapsed)}
            title={sidebarCollapsed ? "Expand sidebar" : "Collapse sidebar"}
            type="button"
          >
            <NavigationIcon name={sidebarCollapsed ? "expand" : "collapse"} />
          </button>
        </div>
      </aside>

      <main className="main-area">
        {visitedPages.has("dashboard") && (
          <div hidden={page !== "dashboard"}>
            <Dashboard
              devices={devices}
              token={token}
              error={error}
              connected={connected}
              onRetry={() => void refresh(token)}
            />
          </div>
        )}
        {visitedPages.has("topology") && (
          <div hidden={page !== "topology"}>
            <Topology devices={devices} experiments={experimentResults} token={token} />
          </div>
        )}
        {visitedPages.has("devices") && (
          <div hidden={page !== "devices"}>
            <DevicesPage
              token={token}
              devices={devices}
              onRefresh={() => void refresh(token)}
            />
          </div>
        )}
        {visitedPages.has("operations") && (
          <div hidden={page !== "operations"}>
            <WorkerOperationsPage token={token} devices={devices} />
          </div>
        )}
        {visitedPages.has("experiments") && (
          <div hidden={page !== "experiments"}>
            <ExperimentsPage
              token={token}
              results={experimentResults}
              error={experimentError}
              onRefresh={() => void refreshExperiments(token)}
              onOpenModelStudio={() => navigateTo("models")}
            />
          </div>
        )}
        {visitedPages.has("search") && (
          <div hidden={page !== "search"}>
            <NASPage
              token={token}
              results={experimentResults}
              error={experimentError}
              onRefresh={() => {
                void refreshExperiments(token);
                void refresh(token);
              }}
              onUseSelectedArchitecture={(architecture) => {
                setSelectedNasArchitecture(architecture);
                navigateTo("models");
              }}
              onViewCandidate={(candidate, group) => {
                setSelectedNasCandidate({ candidate, group });
                navigateTo("model-details");
              }}
            />
          </div>
        )}
        {visitedPages.has("assistant") && (
          <div hidden={page !== "assistant"}>
            <AssistantPage token={token} onOpenAgent={() => navigateTo("agent")} />
          </div>
        )}
        {visitedPages.has("agent") && (
          <div hidden={page !== "agent"}>
            <AgentPage token={token} active={page === "agent"} />
          </div>
        )}
        {visitedPages.has("datasets") && (
          <div hidden={page !== "datasets"}>
            <DatasetsPage token={token} active={page === "datasets"} />
          </div>
        )}
        {visitedPages.has("models") && (
          <div hidden={page !== "models"}>
            <ModelStudioPage
              token={token}
              results={experimentResults}
              error={experimentError}
              onRefresh={() => void refreshExperiments(token)}
              nasArchitecture={selectedNasArchitecture}
              onClearNasArchitecture={() => setSelectedNasArchitecture(null)}
            />
          </div>
        )}
        {visitedPages.has("model-details") && selectedNasCandidate && (
          <div hidden={page !== "model-details"}>
            <ModelDetailsPage
              candidate={selectedNasCandidate.candidate}
              group={selectedNasCandidate.group}
              onBack={() => navigateTo("search")}
              onUseInModelStudio={() => {
                setSelectedNasArchitecture(selectedNasCandidate.candidate.architecture);
                navigateTo("models");
              }}
            />
          </div>
        )}
        {page !== "dashboard" && page !== "topology" && (
          page !== "assistant" && page !== "agent-runs" && page !== "datasets" && page !== "models" && page !== "experiments" && page !== "search" && page !== "devices" && page !== "operations" && page !== "model-details" && <Placeholder page={page} />
        )}
      </main>
      <DatabaseStatusBot token={token} />
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
        <label htmlFor="api-token">API token</label>
        <input
          autoComplete="off"
          id="api-token"
          onChange={(event) => setValue(event.target.value)}
          placeholder="Coordinator token"
          type="password"
          value={value}
        />
        {error && <div role="alert" className="alert">{error}</div>}
        <button className="primary-button" disabled={!value.trim()} type="submit">Connect to network</button>
        <small>Token stored in this browser tab only.</small>
      </form>
    </main>
  );
}

function Dashboard({
  devices,
  token,
  error,
  connected,
  onRetry,
}: {
  devices: Device[];
  token: string;
  error: string;
  connected: boolean;
  onRetry: () => void;
}) {
  const online = devices.filter((device) => device.online).length;
  const onlineWorkers = devices.filter((device) => device.role === "worker" && device.online).length;

  return (
    <section className="content">
      <div className="page-heading">
        <div>
          <h1>Infrastructure dashboard</h1>
        </div>
        <div className={`live-badge ${connected ? "active" : ""}`}>
          <span className="status-dot" /> {connected ? "LIVE" : "CONNECTING"}
        </div>
      </div>

      <div className="summary-grid">
        <Summary label="Cluster nodes" value={devices.length || 4} detail="Registered devices" icon="nodes" />
        <Summary label="Online nodes" value={`${online}/${devices.length || 4}`} detail="Reporting status" icon="activity" />
        <Summary label="Network" value={connected ? "Connected" : "Waiting"} detail="Live status stream" icon="network" />
      </div>

      <div className="section-heading">
        <div>
          <h2>Compute nodes</h2>
        </div>
        <div className="dashboard-actions">
          <button className="quiet-button" onClick={onRetry} type="button">Refresh</button>
        </div>
      </div>

      <WorkerMemoryCleanupControl
        token={token}
        disabled={onlineWorkers === 0}
        onComplete={onRetry}
      />

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
    </section>
  );
}

function Summary({ label, value, detail, icon }: { label: string; value: string | number; detail: string; icon: NavigationIconName }) {
  return (
    <article className="summary-card">
      <div className="summary-card__top"><span>{label}</span><span className="summary-mark"><NavigationIcon name={icon} /></span></div>
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
  const topologyWidth = 1000;
  const masterX = topologyWidth / 2;
  const workerSpacing = workers.length > 0 ? topologyWidth / workers.length : 0;
  return (
    <section className="content">
      <div className="page-heading">
        <div>
          <h1>Network topology</h1>
        </div>
      </div>
      <div className="topology-panel">
        <div className="topology-legend">
          <span><i className="legend-master" /> Master coordinator</span>
          <span><i className="legend-worker" /> Worker node</span>
          <span><i className="legend-line" /> Network link</span>
        </div>
        <svg
          className="topology-lines"
          viewBox={`0 0 ${topologyWidth} 440`}
          role="img"
          aria-label={`Master connected to ${workers.length} worker devices`}
        >
          {workers.map((device, index) => {
            const workerX = workerSpacing * (index + 0.5);
            return (
              <g key={device.node_id}>
                <line x1={masterX} y1="150" x2={workerX} y2="310" />
                <circle cx={workerX} cy="310" r="5" />
              </g>
            );
          })}
          <circle cx={masterX} cy="150" r="5" />
        </svg>
        <div className="topology-nodes">
          <TopologyNode device={master} role="master" />
          {workers.length === 0 ? (
            <p className="topology-empty">No worker devices are registered.</p>
          ) : (
            <div
              className="worker-nodes"
              style={{ gridTemplateColumns: `repeat(${workers.length}, minmax(0, 1fr))` }}
            >
              {workers.map((device) => (
              <TopologyNode
                device={device}
                key={device.node_id}
                role="worker"
              />
            ))}
            </div>
          )}
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
  const [smokeTestGroupId, setSmokeTestGroupId] = useState(
    () => sessionStorage.getItem(TOPOLOGY_SMOKE_GROUP_KEY) ?? "",
  );
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState("");
  const [error, setError] = useState("");
  const topology = experiments?.configuration.topology ?? "directed_ring";
  const smokeTestGroup = experiments?.groups.find(
    (group) => group.group_id === smokeTestGroupId,
  );
  const latestGroup = [...(experiments?.groups ?? [])]
    .reverse()
    .find((group) => group.kind === "topology_optimization");
  const displayCandidates = latestGroup?.candidates?.length
    ? latestGroup.candidates
    : topologyCandidatesFromRuns(latestGroup);
  const topologySelection = latestGroup?.selection &&
    "baselines" in latestGroup.selection
    ? latestGroup.selection
    : undefined;
  const selectedTopology = topologySelection?.selected_candidate
    ?? (latestGroup?.selected_configuration
      ? {
          topology: latestGroup.selected_configuration.topology,
          alpha: latestGroup.selected_configuration.alpha,
        }
      : undefined);
  const diagramTopology = latestGroup?.status === "completed" && selectedTopology
    ? selectedTopology.topology
    : topology;
  const topologyGraph = topologyGraphFor(
    experiments?.topologies ?? {},
    diagramTopology,
  );
  const graphPositions = topologyGraph.workerIds.map((_, index) => {
    const angle = -Math.PI / 2 + (index * 2 * Math.PI) / topologyGraph.workerIds.length;
    return {
      x: 300 + Math.cos(angle) * 170,
      y: 120 + Math.sin(angle) * 78,
    };
  });
  const validationReward = latestGroup?.reward_metric_source === "validation_accuracy";
  const peerWaitIsRewardCriterion =
    latestGroup?.reward_config?.weights?.peer_wait_efficiency !== undefined;
  const latestGroupSeeds = latestGroup?.seeds ?? (latestGroup?.seed === undefined ? [] : [latestGroup.seed]);
  const expectedTopologyRuns = displayCandidates.length > 0
    ? displayCandidates.length * latestGroupSeeds.length
    : undefined;
  const completedTopologyRuns = latestGroup?.runs.filter((run) => run.status === "completed").length ?? 0;
  const remainingTopologyRuns = expectedTopologyRuns === undefined
    ? undefined
    : Math.max(expectedTopologyRuns - completedTopologyRuns, 0);
  const overallSweepPercent = expectedTopologyRuns
    ? Math.min((completedTopologyRuns / expectedTopologyRuns) * 100, 100)
    : 0;
  const activeTopologyRun = latestGroup?.runs.find((run) => run.status === "running");
  const activeTopologyRound = activeTopologyRun
    ? Math.max(activeTopologyRun.rounds.length, activeTopologyRun.progress?.round ?? 0)
    : 0;
  const activeRunRounds = latestGroup?.rounds ?? 0;
  const activeRunPercent = activeRunRounds > 0
    ? Math.min((activeTopologyRound / activeRunRounds) * 100, 100)
    : 0;
  const activeWorkerProgress = Object.entries(activeTopologyRun?.progress?.workers ?? {})
    .map(([worker, progress]) => (
      progress.epoch !== null && progress.epochs !== null
        ? `${worker} epoch ${progress.epoch}/${progress.epochs}`
        : `${worker} ${progress.phase?.replace(/_/g, " ") ?? "working"}`
    ));
  const latestGroupDetails = latestGroup
    ? [
        typeof latestGroup.rounds === "number" ? `${latestGroup.rounds} rounds` : undefined,
        typeof latestGroup.epochs === "number" ? `${latestGroup.epochs} epochs` : undefined,
        latestGroupSeeds.length > 0 ? `seeds ${latestGroupSeeds.join(", ")}` : undefined,
      ].filter((detail): detail is string => detail !== undefined).join(" · ")
    : "";
  async function optimize(smokeTest = false) {
    const rounds = smokeTest ? 1 : 10;
    const epochs = smokeTest ? 1 : 5;
    const seeds = smokeTest ? [17] : [17, 42, 2026];
    const description = smokeTest
      ? "Run a one-round, one-epoch MNIST topology smoke test on the real DFL workers."
      : "Run a validation-scored MNIST topology sweep on the real DFL workers.";
    setBusy(true);
    setMessage("");
    setError("");
    try {
      const dispatched = await dispatchAutonomyAction(token, {
        action: "optimize_topology",
        parameters: { rounds, epochs, seeds, alphas: [0.3] },
      }, description, {
        dataset_name: "mnist",
        topologies: ["directed_ring", "bidirectional_ring", "mesh", "chain", "star"],
        alphas: [0.3],
        rounds,
        epochs,
        seeds,
      });
      if (!dispatched.result.available) throw new Error(dispatched.result.message ?? "Optimizer unavailable.");
      const data = dispatched.result.data;
      const groupId = data?.group_id;
      const runDescription = smokeTest ? "Topology smoke test" : "Real-device topology sweep";
      setMessage(groupId ? `${runDescription} ${String(groupId)} started.` : `${runDescription} started.`);
      if (smokeTest && typeof groupId === "string") {
        sessionStorage.setItem(TOPOLOGY_SMOKE_GROUP_KEY, groupId);
        setSmokeTestGroupId(groupId);
      }
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Could not start topology optimization.");
    } finally {
      setBusy(false);
    }
  }

  return (
    <section className="training-map panel-card">
      <div className="section-heading">
        <h2>Training topology optimizer</h2>
        <div className="topology-actions">
          <span className="run-status completed">Active: {topology}</span>
          {diagramTopology !== topology && (
            <span className="run-status completed">Selected: {diagramTopology}</span>
          )}
          <button
            className="secondary-button"
            disabled={busy || Boolean(experiments?.active_group_id) || latestGroup?.status === "running"}
            onClick={() => void optimize(true)}
            type="button"
          >
            {busy ? "Starting smoke test…" : latestGroup?.status === "running" ? "Topology run in progress…" : "Run topology smoke test"}
          </button>
          <button
            className="primary-button"
            disabled={busy || Boolean(experiments?.active_group_id) || latestGroup?.status === "running"}
            onClick={() => void optimize()}
            type="button"
          >
            {busy ? "Starting…" : latestGroup?.status === "running" ? "Sweep in progress…" : "Run 5-topology sweep"}
          </button>
        </div>
      </div>
      <p className="active-topology-note">
        Active configuration: <strong>{topology}</strong> · {experiments?.configuration.dataset_split ?? "stratified"}
      </p>
      <p className="active-topology-note">
        {validationReward || !latestGroup
          ? "Ranking: validation accuracy, convergence, communication, and peer wait · TEST reserved for final evaluation."
          : "Legacy reward may include held-out TEST scores · not an independent final evaluation."}
      </p>
      <div className="training-map__nodes">
        {topologyGraph.workerIds.map((worker) => <span key={worker}>{topologyWorkerLabel(worker)}</span>)}
      </div>
      <svg
        className="training-graph"
        viewBox="0 0 600 210"
        role="img"
        aria-label={`${diagramTopology} worker communication graph`}
      >
        <defs>
          <marker
            id="topology-arrow"
            viewBox="0 0 8 8"
            markerWidth="12"
            markerHeight="12"
            refX="7"
            refY="4"
            orient="auto-start-reverse"
            markerUnits="userSpaceOnUse"
          >
            <path d="M0,0 L8,4 L0,8 z" />
          </marker>
        </defs>
        {topologyGraph.edges.map(([from, to]) => {
          const startIndex = topologyGraph.workerIds.indexOf(from);
          const endIndex = topologyGraph.workerIds.indexOf(to);
          const start = graphPositions[startIndex];
          const end = graphPositions[endIndex];
          const dx = end.x - start.x;
          const dy = end.y - start.y;
          const distance = Math.hypot(dx, dy);
          const nodeInset = 28;
          const offsetX = (dx / distance) * nodeInset;
          const offsetY = (dy / distance) * nodeInset;
          return (
            <line
              key={`${from}-${to}`}
              x1={start.x + offsetX}
              y1={start.y + offsetY}
              x2={end.x - offsetX}
              y2={end.y - offsetY}
              markerEnd="url(#topology-arrow)"
              markerStart={topologyGraph.bidirectional ? "url(#topology-arrow)" : undefined}
            />
          );
        })}
        {graphPositions.map((point, index) => (
          <g key={index}>
            <circle cx={point.x} cy={point.y} r="23" />
            <text x={point.x} y={point.y + 4}>{`W${index + 1}`}</text>
          </g>
        ))}
      </svg>
      <div className="training-map__links">
        {topologyGraph.edges.map(([from, to]) => (
          <span key={`${from}-${to}`}>
            {topologyWorkerLabel(from)} {topologyGraph.bidirectional ? "↔" : "→"} {topologyWorkerLabel(to)}
          </span>
        ))}
      </div>
      {message && <div className="success-note" role="status">{message}</div>}
      {error && <div className="alert inline-alert" role="alert">{error}</div>}
      {smokeTestGroupId && (
        <div className="topology-smoke-state" role="status" aria-label="Saved topology smoke test">
          <strong>Latest topology smoke test</strong>
          {smokeTestGroup ? (
            <>
              <span>
                {smokeTestGroup.group_id} · {smokeTestGroup.status}
                {` · ${smokeTestGroup.runs.filter((run) => run.status === "completed").length}/5 topologies complete`}
              </span>
              {smokeTestGroup.error && <span>{smokeTestGroup.error}</span>}
            </>
          ) : (
            <span>{smokeTestGroupId} · waiting for saved coordinator results</span>
          )}
        </div>
      )}
      {latestGroup?.status === "running" && (
        <div className="topology-progress" role="status" aria-label="Topology sweep progress">
          <div className="topology-progress__heading">
            <div>
              <span className="topology-progress__eyebrow">LIVE EXPERIMENT</span>
              <h3>Topology sweep in progress</h3>
            </div>
            <span className="topology-progress__badge">RUNNING</span>
          </div>

          <div className="topology-progress__section">
            <div className="topology-progress__labels">
              <strong>Overall sweep</strong>
              <span>
                {completedTopologyRuns} of {expectedTopologyRuns ?? "—"} runs complete
                {remainingTopologyRuns !== undefined && ` · ${remainingTopologyRuns} remaining`}
              </span>
            </div>
            <div
              className="topology-progress__track"
              role="progressbar"
              aria-label="Overall sweep progress"
              aria-valuemin={0}
              aria-valuemax={expectedTopologyRuns ?? 0}
              aria-valuenow={completedTopologyRuns}
              aria-valuetext={`${completedTopologyRuns} of ${expectedTopologyRuns ?? "unknown"} candidate runs complete`}
            >
              <span style={{ width: `${overallSweepPercent}%` }} />
            </div>
            <div className="topology-progress__scale">
              <span>{overallSweepPercent.toFixed(0)}% complete</span>
              <span>{expectedTopologyRuns ?? "—"} total candidate runs</span>
            </div>
          </div>

          {activeTopologyRun && (
            <div className="topology-progress__active">
              <div className="topology-progress__labels">
                <strong>Current run</strong>
                <span>
                  {activeTopologyRun.topology} <i>·</i> alpha {activeTopologyRun.alpha ?? "—"}
                  <i>·</i> seed {activeTopologyRun.seed}
                  <i>·</i> round {activeTopologyRound}/{activeRunRounds || "—"}
                </span>
              </div>
              <div
                className="topology-progress__track topology-progress__track--current"
                role="progressbar"
                aria-label="Current candidate run progress"
                aria-valuemin={0}
                aria-valuemax={activeRunRounds}
                aria-valuenow={activeTopologyRound}
                aria-valuetext={`Round ${activeTopologyRound} of ${activeRunRounds || "unknown"}`}
              >
                <span style={{ width: `${activeRunPercent}%` }} />
              </div>
            </div>
          )}

          <div className="topology-progress__workers">
            <span className="topology-progress__workers-label">Worker progress</span>
            {activeWorkerProgress.length > 0
              ? activeWorkerProgress.map((progress) => <span className="topology-progress__worker" key={progress}>{progress}</span>)
              : <span className="topology-progress__worker">Waiting for workers to report their current epoch</span>}
          </div>

        </div>
      )}
      {topologySelection && (
        <div className="optimizer-selection">
          <strong>Measured selector: {topologySelection.method}</strong>
          <span>
            <span className="best-result-badge">BEST TOPOLOGY</span>{" "}
            {topologySelection.selected_candidate.topology} at alpha {topologySelection.selected_candidate.alpha}
            {" "}· {validationReward ? "validation-derived reward" : "legacy reward"} {topologySelection.selected_candidate.mean_reward?.toFixed(4) ?? "—"} · real workers
          </span>
          <span>Random baseline: {topologySelection.baselines.random_seeded.topology} · {validationReward ? "validation-derived reward" : "legacy reward"} {topologySelection.baselines.random_seeded.mean_reward?.toFixed(4) ?? "—"}</span>
          <span>Best-observed baseline: {topologySelection.baselines.best_observed.topology} · {validationReward ? "validation accuracy" : "legacy accuracy"} {formatAccuracy(topologySelection.baselines.best_observed.mean_final_accuracy)}</span>
        </div>
      )}
      {latestGroup && displayCandidates.length > 0 ? (
        <div className="table-scroll">
          <table className="metrics-table topology-reward-table">
            <thead>
              <tr>
                <th>Topology</th><th>Alpha</th><th>Runs</th>
                <th>{validationReward ? "Validation-derived reward" : "Legacy reward"}</th>
                <th>{validationReward ? "Validation accuracy" : "Legacy accuracy"}</th>
                <th>Macro precision</th><th>Macro recall</th><th>Macro F1</th>
                <th>Accuracy gain</th><th>Convergence</th>
                <th>Peer communication</th>
                <th>{peerWaitIsRewardCriterion ? "Peer wait efficiency" : "Runtime efficiency (legacy)"}</th>
                <th>Mean runtime / worker</th>
                <th>Peer transfer / run</th><th>Peer wait / worker</th>
              </tr>
            </thead>
            <tbody>
              {displayCandidates.map((candidate) => {
                const observed = candidateRunMetrics(latestGroup, candidate);
                const completedRuns = observed.completedRuns || candidate.completed_runs || 0;
                const isSelected = selectedTopology?.topology === candidate.topology
                  && selectedTopology.alpha === candidate.alpha;
                return (
                  <tr className={isSelected ? "best-result-row" : undefined} key={`${candidate.topology}-${candidate.alpha}`}>
                    <td>
                      {candidate.topology}
                      {isSelected && <span className="best-result-badge">BEST</span>}
                    </td>
                    <td>{candidate.alpha}</td>
                    <td>{completedRuns}/{latestGroupSeeds.length}</td>
                    <td>{candidate.mean_reward?.toFixed(4) ?? "—"}</td>
                    <td>{formatAccuracy(candidate.mean_final_accuracy ?? observed.meanAccuracy)}</td>
                    <td>{formatAccuracy(observed.meanPrecision)}</td>
                    <td>{formatAccuracy(observed.meanRecall)}</td>
                    <td>{formatAccuracy(observed.meanMacroF1)}</td>
                    <td
                      title={
                        (candidate.mean_accuracy_gain_available ?? observed.accuracyGainAvailable) === false
                          ? "Unavailable: this single-round run has no pre-training validation baseline."
                          : undefined
                      }
                    >
                      {(candidate.mean_accuracy_gain_available ?? observed.accuracyGainAvailable) === false
                        ? "—"
                        : candidate.mean_components?.accuracy_gain?.toFixed(3) ?? "—"}
                    </td>
                    <td title="Zero means the configured validation-accuracy target was not reached.">
                      {candidate.mean_components?.convergence_speed?.toFixed(3) ?? "—"}
                    </td>
                    <td title="Zero means the sweep's saved communication normalization cap was reached.">
                      {candidate.mean_components?.communication_efficiency?.toFixed(3) ?? "—"}
                    </td>
                    <td title={peerWaitIsRewardCriterion
                      ? "Zero means total worker peer wait reached the saved wait-time normalization cap."
                      : "Legacy reward component: zero means its saved runtime normalization cap was reached."}>
                      {(peerWaitIsRewardCriterion
                        ? candidate.mean_components?.peer_wait_efficiency
                        : candidate.mean_components?.runtime_efficiency)?.toFixed(3) ?? "—"}
                    </td>
                    <td>{observed.meanRuntime?.toFixed(2) ?? "—"} s</td>
                    <td>{observed.meanCommunication?.toLocaleString() ?? "—"} B</td>
                    <td>{observed.meanPeerWait?.toFixed(2) ?? "—"} s</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      ) : (
        <p className="active-topology-note">No measured results.</p>
      )}
      {latestGroup && (
        <p className="active-topology-note">
          {peerWaitIsRewardCriterion
            ? "Zero convergence: 50% validation target not reached · zero cost efficiency: saved cap reached."
            : "Legacy runtime reward · zero convergence: 50% validation target not reached · zero cost efficiency: saved cap reached."}
          {" "}Historical criteria retained.
        </p>
      )}
      {latestGroup && <small className="active-topology-note">Latest sweep: {latestGroup.group_id} · {latestGroup.status}{latestGroupDetails ? ` · ${latestGroupDetails}` : ""}</small>}
    </section>
  );
}

function topologyWorkerLabel(workerId: string): string {
  return workerId.startsWith("dfl-") ? workerId : `dfl-${workerId}`;
}

function topologyGraphFor(
  topologies: ExperimentResults["topologies"],
  topology: TrainingTopology,
): { workerIds: string[]; edges: Array<[string, string]>; bidirectional: boolean } {
  const savedTopology = topologies[topology];
  if (Array.isArray(savedTopology) && savedTopology.length > 0) {
    if (savedTopology.every((worker): worker is string => typeof worker === "string")) {
      const workers = [...new Set(savedTopology)];
      const edges = savedTopology.slice(0, -1).map(
        (worker, index) => [worker, savedTopology[index + 1]] as [string, string],
      );
      return { workerIds: workers, edges, bidirectional: false };
    }

    const edges = savedTopology.flatMap((edge): Array<[string, string]> => (
      Array.isArray(edge) && edge.length === 2
        && typeof edge[0] === "string"
        && typeof edge[1] === "string"
        ? [[edge[0], edge[1]]]
        : []
    ));
    if (edges.length > 0) {
      return {
        workerIds: [...new Set(edges.flatMap(([from, to]) => [from, to]))],
        edges,
        bidirectional: true,
      };
    }
  }

  const workerIds = ["worker1", "worker2", "worker3"];
  const edges: Array<[string, string]> = topology === "star"
    ? [[workerIds[0], workerIds[1]], [workerIds[0], workerIds[2]]]
    : topology === "chain"
      ? [[workerIds[0], workerIds[1]], [workerIds[1], workerIds[2]]]
      : [[workerIds[0], workerIds[1]], [workerIds[1], workerIds[2]], [workerIds[2], workerIds[0]]];
  return {
    workerIds,
    edges,
    bidirectional: topology !== "ring" && topology !== "directed_ring",
  };
}

function formatAccuracy(value?: number): string {
  return value === undefined ? "—" : `${(value * 100).toFixed(2)}%`;
}

function average(values: number[]): number | undefined {
  return values.length
    ? values.reduce((total, value) => total + value, 0) / values.length
    : undefined;
}

function averageWorkerMetric(
  workers: WorkerRoundMetrics[],
  select: (metrics: WorkerRoundMetrics) => number | undefined,
): number | undefined {
  return average(
    workers
      .map(select)
      .filter((value): value is number => value !== undefined && Number.isFinite(value)),
  );
}

function candidateRunMetrics(
  group: ExperimentGroup,
  candidate: TopologyCandidate,
) {
  const runs = group.runs.filter(
    (run) =>
      run.status === "completed" &&
      run.topology === candidate.topology &&
      run.alpha === candidate.alpha,
  );
  const runMetrics = runs.flatMap((run) => {
    const finalRound = [...run.rounds]
      .reverse()
      .find((round) => Object.keys(round.workers).length > 0);
    if (!finalRound) return [];
    const workers = Object.values(finalRound.workers);
    const accuracy = averageWorkerMetric(
      workers,
      (metrics) => metrics.validation_accuracy ?? metrics.accuracy,
    );
    const macroF1 = averageWorkerMetric(
      workers,
      (metrics) => metrics.validation_macro_f1 ?? metrics.macro_f1,
    );
    const precision = averageWorkerMetric(
      workers,
      (metrics) => metrics.validation_macro_precision ?? metrics.macro_precision,
    );
    const recall = averageWorkerMetric(
      workers,
      (metrics) => metrics.validation_macro_recall ?? metrics.macro_recall,
    );
    const runtime = averageWorkerMetric(
      workers,
      (metrics) => metrics.round_seconds ?? metrics.runtime_seconds,
    );
    const peerWait = averageWorkerMetric(workers, (metrics) => metrics.peer_wait_seconds);
    const communication = workers.reduce(
      (total, metrics) => total + (metrics.bytes_sent ?? 0) + (metrics.bytes_received ?? 0),
      0,
    );
    return [{ accuracy, macroF1, precision, recall, runtime, peerWait, communication }];
  });
  return {
    completedRuns: runs.length,
    accuracyGainAvailable: runs.length
      ? runs.every((run) => {
        if (run.rounds.length > 1) return true;
        const firstRoundWorkers = Object.values(run.rounds[0]?.workers ?? {});
        return firstRoundWorkers.length > 0
          && firstRoundWorkers.every(
            (metrics) => typeof metrics.initial_validation_accuracy === "number",
          );
      })
      : undefined,
    meanAccuracy: average(runMetrics.flatMap((metrics) => metrics.accuracy === undefined ? [] : [metrics.accuracy])),
    meanMacroF1: average(runMetrics.flatMap((metrics) => metrics.macroF1 === undefined ? [] : [metrics.macroF1])),
    meanPrecision: average(runMetrics.flatMap((metrics) => metrics.precision === undefined ? [] : [metrics.precision])),
    meanRecall: average(runMetrics.flatMap((metrics) => metrics.recall === undefined ? [] : [metrics.recall])),
    meanRuntime: average(runMetrics.flatMap((metrics) => metrics.runtime === undefined ? [] : [metrics.runtime])),
    meanPeerWait: average(runMetrics.flatMap((metrics) => metrics.peerWait === undefined ? [] : [metrics.peerWait])),
    meanCommunication: average(runMetrics.map((metrics) => metrics.communication)),
  };
}

function topologyCandidatesFromRuns(group: ExperimentGroup | undefined): TopologyCandidate[] {
  const candidates = new Map<string, TopologyCandidate>();
  for (const run of group?.runs ?? []) {
    if (run.topology === "centralized" || typeof run.alpha !== "number") continue;
    const key = `${run.topology}:${run.alpha}`;
    if (!candidates.has(key)) {
      candidates.set(key, { topology: run.topology, alpha: run.alpha });
    }
  }
  return [...candidates.values()];
}

function TopologyNode({ device, role }: { device?: Device; role: "master" | "worker" }) {
  return (
    <article className={`topology-node ${role} ${device?.online ? "node-online" : "node-offline"}`}>
      <div className="node-orb">{role === "master" ? "M" : "W"}</div>
      <div>
        <strong>{device?.hostname ?? "Master device"}</strong>
        <span>
          {device ? `${device.ip}:${device.port}` : "Not registered"}
          {" · "}
          {device?.online ? "Reachable" : "Offline"}
          {device?.is_managed ? " · Added in dashboard" : ""}
        </span>
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
        </div>
      </div>
      <div className="coming-soon">
        <div className="coming-icon"><NavigationIcon name={item?.id ?? "dashboard"} /></div>
        <span className="eyebrow">PHASE 3+</span>
        <h2>{item?.label} workspace</h2>
        <span className="planned-tag">Planned</span>
      </div>
    </section>
  );
}
