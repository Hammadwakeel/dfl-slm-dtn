import type { Device } from "../types";

export type DeviceEvent =
  | { type: "device_snapshot"; devices: Device[] }
  | { type: "device_status"; device: Device }
  | { type: "device_removed"; node_id: string };

export type TrainingTopology =
  | "ring"
  | "directed_ring"
  | "bidirectional_ring"
  | "mesh"
  | "chain"
  | "star"
  | "learned";

export interface NASLegacyArchitecture {
  id: string;
  blocks: number;
  channels: number;
  kernel_size: number;
}

export type NASLayerType =
  | "conv2d" | "separable_conv2d" | "max_pooling2d" | "average_pooling2d"
  | "batch_normalization" | "dropout" | "flatten"
  | "global_average_pooling2d" | "dense" | "patch_embedding"
  | "transformer_encoder" | "global_average_pooling1d";

export interface NASArchitectureLayer {
  type: NASLayerType;
  filters?: number;
  kernel_size?: number;
  strides?: number;
  padding?: "same" | "valid";
  activation?: string;
  pool_size?: number;
  rate?: number;
  units?: number;
  patch_size?: number;
  embed_dim?: number;
  num_heads?: number;
  ff_dim?: number;
  dropout?: number;
}

export interface NASArchitectureTraining {
  optimizer: "adam" | "adamw" | "sgd" | "rmsprop";
  learning_rate: number;
  batch_size: number;
}

export interface NASCustomArchitecture extends NASLegacyArchitecture {
  model_type: "custom";
  channel_schedule: number[];
  block_type: "conv2d" | "separable_conv2d";
  head_units: number;
  dropout: number;
  attention?: boolean;
  parameter_count: number;
  layers: NASArchitectureLayer[];
  training: NASArchitectureTraining;
}

export type NASArchitecture = NASLegacyArchitecture | NASCustomArchitecture;

export interface NASCandidate {
  architecture: NASArchitecture;
  parameter_count: number;
  mean_validation_accuracy: number;
  worker_accuracies: number[];
  model_summary?: string;
  validation_confusion_matrix?: number[][];
  mean_elapsed_seconds?: number;
  max_peak_rss_mb?: number;
  mean_validation_macro_precision?: number;
  mean_validation_macro_recall?: number;
  mean_validation_macro_f1?: number;
  generation?: number;
  fidelity_epochs?: number;
  is_pareto_optimal?: boolean;
}

export interface NASWorkerSearch {
  node_id: string;
  dataset_classes?: string[];
  candidates: Array<{
    architecture: NASArchitecture;
    parameter_count: number;
    validation_accuracy: number;
    validation_samples: number;
    elapsed_seconds: number;
    peak_rss_mb: number;
    validation_macro_precision?: number;
    validation_macro_recall?: number;
    validation_macro_f1?: number;
    validation_loss?: number;
    validation_confusion_matrix?: number[][];
    model_summary?: string;
    fidelity_epochs?: number;
  }>;
  elapsed_seconds: number;
  peak_rss_mb: number;
  fidelity_epochs?: number;
  search_method?: string;
  gpu?: string[];
}

export interface NASWorkerSearchProgress {
  status: "searching" | "completed" | "failed" | "stopped";
  started_at: number;
  completed_at?: number;
  candidate_count?: number;
  generation?: number;
  fidelity_epochs?: number;
  elapsed_seconds?: number;
  error?: string;
}

export interface NASGeneration {
  generation: number;
  fidelity_epochs: number;
  candidates: NASCandidate[];
}

export interface SlmChatResponse {
  conversation_id: string;
  reply: string;
}

export interface SlmConversationSummary {
  id: string;
  title: string;
  updated_at: string;
  message_count: number;
}

export interface SlmConversationMessage {
  id: number;
  role: "user" | "assistant";
  content: string;
  created_at: string;
}

export interface SlmConversation extends SlmConversationSummary {
  created_at: string;
  messages: SlmConversationMessage[];
}

export interface DatabaseAssistantSource {
  record_type: string;
  record_id: string;
  recorded_at: string;
}

export interface DatabaseAssistantReply {
  answer: string;
  sources: DatabaseAssistantSource[];
}

export interface WorkerRoundMetrics {
  node_id: string;
  run_id: string;
  status?: string;
  round: number;
  topology: TrainingTopology | "centralized";
  dataset_split?: string;
  aggregation?: string;
  sync_mode?: string;
  peers_used?: string[];
  missing_peers?: string[];
  local_epochs?: number;
  sample_count?: number;
  training_loss?: number;
  training_accuracy?: number;
  initial_validation_accuracy?: number;
  accuracy?: number;
  validation_accuracy?: number;
  loss?: number;
  validation_loss?: number;
  macro_f1?: number;
  validation_macro_f1?: number;
  validation_sample_count?: number;
  macro_precision?: number;
  validation_macro_precision?: number;
  macro_recall?: number;
  validation_macro_recall?: number;
  validation_confusion_matrix_counts?: number[][];
  validation_confusion_matrix_row_normalized_percent?: number[][];
  validation_per_class_precision?: number[];
  validation_per_class_recall?: number[];
  validation_per_class_f1?: number[];
  training_seconds?: number;
  runtime_seconds?: number;
  round_seconds?: number;
  peer_wait_seconds?: number;
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
  topology: TrainingTopology | "centralized";
  completed_at?: number;
  workers: Record<string, WorkerRoundMetrics>;
}

export interface ExperimentRun {
  run_id: string;
  seed: number;
  topology: TrainingTopology | "centralized";
  dataset_split?: string;
  alpha?: number;
  status: "running" | "completed" | "failed" | "stopped";
  error?: string | null;
  progress?: {
    round: number;
    workers: Record<string, {
      epoch: number | null;
      epochs: number | null;
      phase: "local_training" | "peer_exchange" | null;
    }>;
  };
  rounds: ExperimentRound[];
}

export interface ExperimentGroup {
  group_id: string;
  dataset_name?: string;
  reward_config?: { weights?: Record<string, number> };
  reward_metric_source?: "validation_accuracy" | "accuracy";
  architecture?: { dataset_name?: string; [key: string]: unknown };
  kind:
    | "decentralized"
    | "centralized_baseline"
    | "topology_optimization"
    | "neural_architecture_search"
    | "dataset_architecture_search"
    | "hyperparameter_search";
  hpo_results?: Array<{
    architecture: { id: string; training?: { optimizer: string; learning_rate: number; batch_size: number } };
    validation_accuracy: number;
    validation_macro_f1: number;
    elapsed_seconds: number;
  }>;
  selected_hyperparameters?: { optimizer: string; learning_rate: number; batch_size: number };
  learned_senders?: Record<string, string[]>;
  topology_learning?: {
    edges: number;
    mesh_edges: number;
    iterations: number;
    objective: number;
    ring_added_for_connectivity: boolean;
    mean_peer_loss: number;
    mean_wait_seconds: number;
  };
  topology: TrainingTopology | "centralized" | "candidate_sweep";
  dataset_split?: string;
  aggregation?: "fedavg" | "uniform" | "metropolis" | "trimmed_mean";
  sync_mode?: "sync" | "semi_sync";
  sync_deadline_seconds?: number | null;
  alphas?: number[];
  candidates?: TopologyCandidate[];
  nas_candidates?: NASCandidate[];
  worker_search?: NASWorkerSearch[];
  worker_search_progress?: Record<string, NASWorkerSearchProgress>;
  search_method?: string;
  nas_generations?: NASGeneration[];
  dataset_classes?: string[];
  dataset_image_shape?: number[];
  search_time_limit_seconds?: number;
  smoke_test?: boolean;
  architecture_space?: NASArchitecture[];
  fixed_architecture?: NASArchitecture;
  selected_architecture?: NASArchitecture;
  nas_beats_fixed?: boolean;
  search_elapsed_seconds?: number;
  search_peak_rss_mb?: number;
  comparison?: Record<
    "fixed" | "nas" | "centralized_fixed",
    {
      completed_runs: number;
      mean_accuracy: number;
      mean_macro_f1: number;
      mean_training_runtime_seconds: number;
      mean_runtime_including_search_seconds: number;
      total_runtime_including_search_seconds: number;
      search_seconds: number;
      peak_rss_mb: number;
      mean_communication_bytes: number;
    }
  >;
  selection?: {
    method: string;
    selected_candidate: { topology: TrainingTopology; alpha: number; mean_reward?: number };
    baselines: Record<string, { topology: TrainingTopology; alpha: number; mean_reward?: number; mean_final_accuracy?: number }>;
    candidate_count: number;
  } | {
    method: string;
    fidelity_epochs: number;
    selected_candidate: NASCandidate;
    pareto_front: NASCandidate[];
  };
  selected_configuration?: { topology: TrainingTopology; alpha: number; dataset_split: string };
  rounds?: number;
  epochs?: number;
  seed?: number;
  seeds?: number[] | null;
  status: "running" | "completed" | "failed" | "stopped";
  started_at?: number;
  completed_at?: number | null;
  error: string | null;
  runs: ExperimentRun[];
}

export interface TopologyCandidate {
  topology: TrainingTopology;
  alpha: number;
  completed_runs?: number;
  mean_reward?: number;
  mean_final_accuracy?: number;
  mean_accuracy_gain_available?: boolean;
  mean_components?: Record<string, number>;
}

export interface ExperimentResults {
  configuration: {
    rounds: number;
    epochs: number;
    topology: TrainingTopology;
    dataset_split: string;
  };
  active_group_id: string | null;
  topologies: Record<string, string[] | string[][]>;
  groups: ExperimentGroup[];
}

export interface APGRoundSummary {
  round: number;
  split: "VALIDATION";
  source_label: "SIMULATED" | "HYBRID-SHAPED" | "HARDWARE" | "NOT AVAILABLE";
  mean_validation_accuracy: number | null;
  mean_validation_macro_f1: number | null;
  mean_validation_loss: number | null;
  edges_used: Array<[string, string]>;
  edge_count_used: number | null;
  edges_changed_vs_previous_round: number | null;
  safeguard_hits: number | null;
  g: number | null;
  iterations_used: number | null;
  inputs: {
    round: number;
    loss_changed: boolean | null;
    waiting_time_changed: boolean | null;
    theta_changed: boolean | null;
    loss_stats: { min: number | null; mean: number | null; max: number | null } | null;
    waiting_time_stats: { min: number | null; mean: number | null; max: number | null } | null;
    theta_stats: { min: number | null; mean: number | null; max: number | null } | null;
    inputs_available: boolean;
  };
  apg_inputs?: {
    loss_matrix_normalized?: Array<Array<number | null>>;
    wait_matrix_normalized?: Array<Array<number | null>>;
    wait_matrix_simulated_seconds?: Array<Array<number | null>>;
    theta_train_only?: number[][];
  } | null;
  per_node?: Record<string, {
    node_role?: string;
    emulated_worker_id?: string | null;
    shard_rows?: number;
    train_rows?: number;
    validation_rows?: number;
    validation_accuracy?: number;
    validation_macro_f1?: number;
    validation_loss?: number;
    bytes_sent_simulated?: number;
    bytes_received_simulated?: number;
  }>;
  bytes_sent_simulated: number | null;
}

export interface APGTestSummary {
  accuracy?: number;
  macro_f1?: number;
  macro_precision?: number;
  macro_recall?: number;
}

export interface APGRunSummary {
  run_id: string;
  simulation_mode: string;
  source_label: "SIMULATED" | "HYBRID-SHAPED" | "HARDWARE" | "NOT AVAILABLE";
  data_setting: string;
  topology: string;
  seed: number | null;
  status: string;
  split_labels: { rounds: "VALIDATION"; final_metrics: "TEST" };
  duration_seconds: number | null;
  rounds: APGRoundSummary[];
  final_test_summary: APGTestSummary | null;
}

export interface APGNodeDetails {
  node_id?: string;
  node_role?: string;
  emulated_worker_id?: string | null;
  shard_rows?: number;
  train_rows?: number;
  theta?: number[];
}

export interface APGRunDetail {
  summary: APGRunSummary;
  run: {
    run_id: string;
    simulation_mode: string;
    source_label: string;
    data_setting: string;
    topology: string;
    seed: number;
    pc_simulation_wall_seconds?: number;
    rounds: APGRoundSummary[];
    nodes?: APGNodeDetails[];
    final_test?: {
      split: "TEST";
      source: string;
      summary: APGTestSummary;
      confusion_matrix_counts: number[][];
      confusion_matrix_row_normalized_percent: number[][];
    };
  };
}

export interface APGVerdict {
  data_setting: string;
  simulation_mode: string;
  source_label: "SIMULATED" | "HYBRID-SHAPED";
  verdict:
    | "Topology is adapting and improves over ring/mesh (3 seeds)"
    | "Topology is adapting, no demonstrated benefit over ring/mesh"
    | "Topology is not adapting (same graph every round)"
    | "Not enough data (run more seeds)";
  rules: Record<string, boolean | number | string | null>;
  runs: Array<string | null>;
}

export interface APGJob {
  job_id: string;
  run_id: string;
  topology: "apg" | "ring" | "mesh";
  data_setting: string;
  seed: number;
  simulation_mode: "all_simulated";
  source_label: "SIMULATED";
  status: "queued" | "running" | "finished" | "failed" | "stopped";
  rounds_completed: number;
  total_rounds: number;
  created_at: string;
  started_at: string | null;
  completed_at: string | null;
  duration_seconds: number | null;
  result_file: string | null;
  error: string | null;
}

export interface APGRunsResponse {
  runs: APGRunSummary[];
  jobs: APGJob[];
  verdicts: APGVerdict[];
}

export interface SlmAction {
  action: string;
  parameters: Record<string, unknown>;
}

export type SlmCommandResult =
  {
    status: "executed";
    result: { available: boolean; message?: string; data?: Record<string, unknown> };
  };

export interface DatasetMetadata {
  dataset_name: string;
  classes: string[];
  class_counts: Record<string, number>;
  image_sizes: Record<string, Record<string, number>>;
  image_channels?: Record<string, Record<string, number>> | null;
  image_count: number;
  expanded_image_bytes: number;
  source_split_counts?: Record<string, number>;
  source_split_policy?: Record<string, string>;
}

export interface DatasetSample {
  label: string;
  image_data: string;
}

export interface InitialModelReport {
  dataset_name: string;
  architecture: {
    design?: NASArchitecture;
    input_shape: number[];
    classes: string[];
    num_classes: number;
    parameter_count: number;
    [key: string]: unknown;
  };
  parameter_count: number;
  architecture_sha256: string;
  weights_sha256: string;
  workers?: Record<string, {
    architecture_sha256?: string;
    weights_sha256?: string;
    parameter_count?: number;
  }>;
}

export interface PreprocessingReport {
  dataset_name: string;
  image_count: number;
  target_size: { height: number; width: number };
  channels: number;
  image_shape: number[];
  normalization: { method: string; [key: string]: unknown };
  split_strategy: string;
  split_sizes: Record<string, number>;
  split_class_counts: Record<string, Record<string, number>>;
  workers: Record<string, { rows: number; class_counts: Record<string, number>; shard_path: string; pc_shard_sha256?: string }>;
  uploaded_workers?: Record<string, { rows: number; class_counts: Record<string, number>; shard_sha256: string }>;
  preprocessing_duration_seconds?: number;
  shipping_duration_seconds?: number;
  [key: string]: unknown;
}

export interface AutonomyState {
  stop_requested: boolean;
}

export interface ExecutedDispatch {
  status: "executed";
  result: { available: boolean; message?: string; data?: PreprocessingReport };
  history_warning?: string;
}

export type DatasetDispatch = ExecutedDispatch;
export type ActionDispatch = {
  status: "executed";
  result: { available: boolean; message?: string; data?: Record<string, unknown> };
  history_warning?: string;
};

const apiBase =
  import.meta.env.VITE_API_BASE_URL ??
  `${window.location.protocol}//${window.location.hostname}:8000`;
const masterSlmBase =
  import.meta.env.VITE_MASTER_SLM_BASE_URL ?? "http://192.168.11.15:5003";

type ActionDispatchResponse = {
  status: "executed";
  result: unknown;
  history_warning?: string;
};

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function isActionDispatchResult(value: unknown): value is ActionDispatch["result"] {
  return isRecord(value)
    && typeof value.available === "boolean"
    && (value.message === undefined || typeof value.message === "string")
    && (value.data === undefined || isRecord(value.data));
}

export class CoordinatorApiError extends Error {
  constructor(
    message: string,
    readonly status?: number,
  ) {
    super(message);
    this.name = "CoordinatorApiError";
  }
}

async function coordinatorRequest<T>(
  token: string,
  path: string,
  init: RequestInit = {},
  timeoutMs = 30_000,
): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`${apiBase}${path}`, {
      ...init,
      signal: AbortSignal.timeout(timeoutMs),
      headers: {
        Authorization: `Bearer ${token}`,
        ...init.headers,
      },
    });
  } catch (cause) {
    if (cause instanceof DOMException && cause.name === "TimeoutError") {
      throw new CoordinatorApiError("Coordinator request timed out. Verify the current loop status before retrying.");
    }
    throw new CoordinatorApiError(cause instanceof Error ? cause.message : "Could not reach the coordinator API.");
  }
  if (!response.ok) {
    let detail = "";
    let body: unknown;
    try {
      body = await response.json();
    } catch (cause) {
      if (!(cause instanceof SyntaxError)) throw cause;
    }
    if (typeof body === "object" && body !== null) {
      const errorBody = body as { detail?: unknown; error?: unknown };
      detail = typeof errorBody.detail === "string"
        ? errorBody.detail
        : typeof errorBody.error === "string"
          ? errorBody.error
          : "";
    }
    const messages: Record<number, string> = {
      401: "Authentication failed. Sign in again with a valid coordinator token.",
      409: detail || "A dataset with this name already exists.",
      422: detail || "The dataset name, ZIP, or image content was rejected.",
    };
    throw new CoordinatorApiError(messages[response.status] ?? (detail || `Coordinator API error (HTTP ${response.status}).`), response.status);
  }
  if (response.status === 204) return undefined as T;
  return response.json() as Promise<T>;
}

async function masterSlmRequest<T>(
  token: string,
  path: string,
  init: RequestInit = {},
): Promise<T> {
  const response = await fetch(`${masterSlmBase}${path}`, {
    ...init,
    signal: AbortSignal.timeout(150_000),
    headers: {
      Authorization: "Bearer " + token,
      ...(init.body ? { "Content-Type": "application/json" } : {}),
      ...init.headers,
    },
  });
  if (!response.ok) {
    let detail = "";
    try {
      const body: unknown = await response.json();
      if (typeof body === "object" && body !== null) {
        const errorBody = body as { error?: unknown; detail?: unknown };
        detail = typeof errorBody.error === "string"
          ? errorBody.error
          : typeof errorBody.detail === "string"
            ? errorBody.detail
            : "";
      }
    } catch (cause) {
      if (!(cause instanceof SyntaxError)) throw cause;
    }
    throw new Error(detail || `SLM master error (HTTP ${response.status}).`);
  }
  if (response.status === 204) return undefined as T;
  return response.json() as Promise<T>;
}

export async function listDatasets(token: string): Promise<DatasetMetadata[]> {
  return coordinatorRequest(token, "/api/datasets");
}

export async function getDatasetSamples(
  token: string,
  datasetName: string,
): Promise<DatasetSample[]> {
  return coordinatorRequest(token, `/api/datasets/${encodeURIComponent(datasetName)}/samples`);
}

export async function getLatestPreprocessingReport(
  token: string,
  datasetName: string,
): Promise<PreprocessingReport | null> {
  try {
    return await coordinatorRequest(
      token,
      `/api/datasets/${encodeURIComponent(datasetName)}/preprocessing-report`,
    );
  } catch (cause) {
    if (cause instanceof CoordinatorApiError && cause.status === 404) return null;
    throw cause;
  }
}

export async function getInitialModelReport(
  token: string,
  datasetName: string,
): Promise<InitialModelReport | null> {
  try {
    return await coordinatorRequest(
      token,
      `/api/datasets/${encodeURIComponent(datasetName)}/initial-model`,
    );
  } catch (cause) {
    if (cause instanceof CoordinatorApiError && cause.status === 404) return null;
    throw cause;
  }
}

export async function getAutonomyState(token: string): Promise<AutonomyState> {
  return coordinatorRequest(token, "/api/autonomy/state");
}

export async function requestGlobalStop(token: string): Promise<AutonomyState> {
  return coordinatorRequest(token, "/api/autonomy/stop", { method: "POST" });
}

export async function resetGlobalStop(token: string): Promise<AutonomyState> {
  return coordinatorRequest(token, "/api/autonomy/reset-stop", { method: "POST" });
}

export async function dispatchDatasetPreprocessing(
  token: string,
  action: { action: "preprocess_dataset"; parameters: Record<string, unknown> },
  reason: string,
  evidence: Record<string, unknown>,
): Promise<DatasetDispatch> {
  return coordinatorRequest(
    token,
    "/api/autonomy/dispatch",
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ action, reason, evidence }),
    },
    20 * 60_000,
  );
}

export async function dispatchAutonomyAction(
  token: string,
  action: SlmAction,
  reason: string,
  evidence: Record<string, unknown>,
): Promise<ActionDispatch> {
  const response = await coordinatorRequest<ActionDispatchResponse>(
    token,
    "/api/autonomy/dispatch",
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ action, reason, evidence }),
    },
    20 * 60_000,
  );
  const result = isActionDispatchResult(response.result)
    ? response.result
    : isRecord(response.result) && isActionDispatchResult(response.result.result)
      ? response.result.result
      : null;
  if (!result) {
    throw new CoordinatorApiError("Coordinator returned an invalid action dispatch result.");
  }
  return { ...response, result };
}

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

export interface WorkerMemoryCleanupResult {
  complete: boolean;
  phase: "preflight" | "cleanup";
  message: string;
  workers: Array<{
    node_id: string;
    success: boolean;
    error?: string;
    cleaned_runs?: number;
    process_memory_cleaned?: boolean;
    memory_available_before_mb?: number;
    memory_available_after_mb?: number;
    process_rss_before_mb?: number;
    process_rss_after_mb?: number;
    os_page_cache_dropped?: boolean;
  }>;
}

export interface WorkerProcess {
  pid: number;
  ppid: number;
  user: string;
  cpu_percent: number;
  memory_percent: number;
  rss_mb: number;
  name: string;
}

export interface WorkerProcessSnapshotData {
  node_id: string;
  sampled_at: string;
  memory_available_mb: number;
  nas_active: boolean;
  training: {
    active_run_id: string | null;
    scheduled_run_id: string | null;
    active_round: number | null;
    active_epoch: number | null;
    total_epochs: number | null;
    active_phase: string | null;
  };
  processes: WorkerProcess[];
}

export interface WorkerProcessSnapshotError {
  node_id: string;
  error: string;
}

export type WorkerProcessSnapshot =
  | WorkerProcessSnapshotData
  | WorkerProcessSnapshotError;

export async function getWorkerProcesses(token: string): Promise<WorkerProcessSnapshot[]> {
  const result = await coordinatorRequest<{ workers: WorkerProcessSnapshot[] }>(
    token,
    "/api/devices/workers/processes",
  );
  return result.workers;
}

export async function stopWorkerTraining(
  token: string,
  nodeId: string,
): Promise<{ accepted: boolean; message?: string; run_id?: string }> {
  return coordinatorRequest(
    token,
    `/api/devices/${encodeURIComponent(nodeId)}/stop-training`,
    { method: "POST" },
  );
}

export async function stopWorkerNasSearch(
  token: string,
  nodeId: string,
): Promise<{ accepted: boolean; message?: string; run_id?: string }> {
  return coordinatorRequest(
    token,
    `/api/devices/${encodeURIComponent(nodeId)}/stop-nas`,
    { method: "POST" },
  );
}

export async function rebootWorker(
  token: string,
  nodeId: string,
): Promise<{ status: string; action: string }> {
  return coordinatorRequest(
    token,
    `/api/devices/${encodeURIComponent(nodeId)}/reboot`,
    { method: "POST" },
  );
}

export async function cleanupWorkerMemory(
  token: string,
): Promise<WorkerMemoryCleanupResult> {
  return coordinatorRequest(
    token,
    "/api/devices/memory/cleanup",
    { method: "POST" },
    60_000,
  );
}

export interface RegisterDeviceRequest {
  node_id: string;
  hostname: string;
  ip: string;
  port: number;
  training_port: number;
  api_token: string;
}

export interface UpdateDeviceRequest {
  hostname?: string;
  ip?: string;
  port?: number;
  training_port?: number;
  api_token?: string;
}

export async function registerDevice(
  token: string,
  device: RegisterDeviceRequest,
): Promise<Device> {
  return coordinatorRequest(token, "/api/devices", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(device),
  });
}

export async function updateDevice(
  token: string,
  nodeId: string,
  device: UpdateDeviceRequest,
): Promise<Device> {
  return coordinatorRequest(
    token,
    `/api/devices/${encodeURIComponent(nodeId)}`,
    {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(device),
    },
  );
}

export async function removeDevice(token: string, nodeId: string): Promise<void> {
  await coordinatorRequest<void>(
    token,
    `/api/devices/${encodeURIComponent(nodeId)}`,
    { method: "DELETE" },
  );
}

export async function sendSlmMessage(
  token: string,
  message: string,
  conversationId?: string,
): Promise<SlmChatResponse> {
  const response = await fetch(`${masterSlmBase}/rag/chat`, {
    method: "POST",
    signal: AbortSignal.timeout(150_000),
    headers: {
      Authorization: `Bearer ${token}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      message,
      ...(conversationId ? { conversation_id: conversationId } : {}),
    }),
  });
  if (!response.ok) {
    const body = (await response.json().catch(() => ({}))) as { detail?: string; error?: string };
    throw new Error(body.detail ?? body.error ?? `SLM request failed (HTTP ${response.status}).`);
  }
  return response.json() as Promise<SlmChatResponse>;
}

export async function getSlmConversations(token: string): Promise<SlmConversationSummary[]> {
  return masterSlmRequest(token, "/rag/conversations");
}

export async function getSlmConversation(token: string, conversationId: string): Promise<SlmConversation> {
  return masterSlmRequest(token, `/rag/conversations/${encodeURIComponent(conversationId)}`);
}

export async function deleteSlmConversation(token: string, conversationId: string): Promise<void> {
  return masterSlmRequest(
    token,
    `/rag/conversations/${encodeURIComponent(conversationId)}`,
    { method: "DELETE" },
  );
}

export type SupervisorRunStatus = "queued" | "running" | "stopping" | "completed" | "failed" | "stopped";

export interface SupervisorGoal {
  dataset: "mnist" | "cifar10";
  target_accuracy: number;
  max_experiments: number;
  time_budget_minutes: number;
  dataset_split: "stratified" | "dirichlet_a01" | "dirichlet_a03" | "dirichlet_a05" | "dirichlet_a07";
  rounds: number;
  local_epochs: number;
  topology: "directed_ring" | "bidirectional_ring" | "mesh" | "chain" | "star";
  nas_budget_seconds: number;
  aggregation: "fedavg" | "uniform" | "metropolis" | "trimmed_mean";
  sync_mode: "sync" | "semi_sync";
  sync_deadline_seconds: number;
  seed?: number;
}

export interface SupervisorExperiment {
  step: string;
  group_id: string;
  kind: string;
  status: string;
  validation_accuracy?: number | null;
  macro_f1?: number | null;
  architecture?: string | null;
  parameter_count?: number | null;
  topology?: string | null;
  selected_topology?: string | null;
  aggregation?: string;
  sync_mode?: string;
  partial_rounds?: number;
  hyperparameters?: { optimizer: string; learning_rate: number; batch_size: number };
  search_validation_accuracy?: number;
  pruning?: SupervisorPruning;
  error?: string;
  completed_at?: string;
}

export interface SupervisorPruning {
  ratio: number;
  parameters_before: number;
  parameters_after: number;
  compression: number;
  architecture_id?: string;
  source_worker?: string;
  source_validation_accuracy?: number | null;
  validation_accuracy?: number | null;
  macro_f1?: number | null;
}

export interface SupervisorBest {
  validation_accuracy: number;
  macro_f1: number | null;
  group_id: string;
  step: string;
  architecture: string | null;
  topology: string | null;
}

export interface SupervisorCounters {
  slm_decisions: number;
  slm_repairs: number;
  fallback_decisions: number;
  rule_decisions: number;
}

export interface SupervisorState {
  stage?: string;
  completed_steps: string[];
  experiments: SupervisorExperiment[];
  best: SupervisorBest | null;
  selected_topology: string | null;
  nas_architecture: Record<string, unknown> | null;
  pruning?: SupervisorPruning | null;
  active: { step: string; group_id: string; started_at: string; progress?: number } | null;
  pending_step: { step: string; rationale?: string; source?: string } | null;
  counters: SupervisorCounters;
  waiting_reason?: string;
  deadline?: string;
}

export interface SupervisorEvent {
  id: number;
  event_type: string;
  message: string;
  details: Record<string, unknown> | null;
  created_at: string;
}

export interface SupervisorRun {
  id: string;
  task: string;
  status: SupervisorRunStatus;
  stage: string;
  progress: number;
  current_step: string | null;
  rationale: string | null;
  goal: SupervisorGoal;
  state: SupervisorState;
  result: {
    outcome: string;
    summary: string;
    decided_by?: string;
    best?: SupervisorBest | null;
    counters?: SupervisorCounters;
  } | null;
  error: string | null;
  created_at: string;
  updated_at: string;
  completed_at: string | null;
  minutes_left?: number | null;
  events?: SupervisorEvent[];
}

export interface SupervisorCluster {
  workers: number;
  workers_ready: number;
  not_ready: string[];
  min_free_mb: number | null;
  low_memory: string[];
  min_free_memory_mb_required: number;
  global_stop: boolean;
}

export async function listSupervisorRuns(token: string): Promise<SupervisorRun[]> {
  return coordinatorRequest(token, "/api/supervisor/runs");
}

export async function getSupervisorRun(token: string, runId: string): Promise<SupervisorRun> {
  return coordinatorRequest(token, `/api/supervisor/runs/${encodeURIComponent(runId)}`);
}

export async function startSupervisorRun(token: string, goal: SupervisorGoal): Promise<SupervisorRun> {
  return coordinatorRequest(token, "/api/supervisor/runs", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(goal),
  });
}

export async function stopSupervisorRun(token: string, runId: string, reason: string): Promise<SupervisorRun> {
  return coordinatorRequest(token, `/api/supervisor/runs/${encodeURIComponent(runId)}/stop`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ reason }),
  });
}

export async function askSupervisor(
  token: string,
  runId: string,
  question: string,
): Promise<{ run_id: string; answer: string }> {
  return coordinatorRequest(
    token,
    `/api/supervisor/runs/${encodeURIComponent(runId)}/ask`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    },
    150_000,
  );
}

export async function getSupervisorCluster(token: string): Promise<SupervisorCluster> {
  return coordinatorRequest(token, "/api/supervisor/cluster");
}

export async function askDatabaseAssistant(
  token: string,
  question: string,
): Promise<DatabaseAssistantReply> {
  return coordinatorRequest<DatabaseAssistantReply>(
    token,
    "/api/database-assistant/ask",
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    },
  );
}

export async function sendSlmCommand(
  token: string,
  action: SlmAction,
): Promise<SlmCommandResult> {
  const response = await fetch(`${apiBase}/api/slm/command`, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${token}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ action }),
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

export async function deleteExperimentGroup(
  token: string,
  groupId: string,
): Promise<void> {
  await coordinatorRequest<void>(
    token,
    `/api/slm/results/${encodeURIComponent(groupId)}`,
    { method: "DELETE" },
  );
}

export async function deleteInitialModel(
  token: string,
  datasetName: string,
): Promise<void> {
  await coordinatorRequest<void>(
    token,
    `/api/datasets/${encodeURIComponent(datasetName)}/initial-model`,
    { method: "DELETE" },
  );
}

export async function getAPGRuns(token: string): Promise<APGRunsResponse> {
  return coordinatorRequest(token, "/api/apg/runs");
}

export async function getAPGRun(token: string, runId: string): Promise<APGRunDetail> {
  return coordinatorRequest(token, `/api/apg/runs/${encodeURIComponent(runId)}`);
}

export async function startAPGSimulation(
  token: string,
  request: { topology: "apg" | "ring" | "mesh"; data_setting: string; seed: number },
): Promise<APGJob> {
  return coordinatorRequest(
    token,
    "/api/apg/jobs",
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(request),
    },
    30_000,
  );
}

export async function getAPGJob(token: string, jobId: string): Promise<APGJob> {
  return coordinatorRequest(token, `/api/apg/jobs/${encodeURIComponent(jobId)}`);
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
