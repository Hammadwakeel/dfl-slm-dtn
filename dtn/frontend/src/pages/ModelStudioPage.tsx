import { useCallback, useEffect, useState } from "react";

import {
  deleteInitialModel,
  dispatchAutonomyAction,
  getAutonomyState,
  getInitialModelReport,
  getLatestPreprocessingReport,
  listDatasets,
} from "../api/client";
import LayeredArchitectureView from "../components/LayeredArchitectureView";
import { layersFromNASArchitecture } from "../components/architecture";
import type {
  AutonomyState,
  ExperimentGroup,
  ExperimentRun,
  ExperimentResults,
  InitialModelReport,
  NASArchitecture,
  NASArchitectureLayer,
  NASLayerType,
  NASArchitectureTraining,
  PreprocessingReport,
} from "../api/client";

interface Props {
  token: string;
  results: ExperimentResults | null;
  error: string;
  onRefresh: () => void;
  nasArchitecture?: NASArchitecture | null;
  onClearNasArchitecture?: () => void;
}

type ModelLayer = NASArchitectureLayer;
type LayerType = NASLayerType;
type ModelDesign =
  | NASArchitecture
  | { model_type: "custom"; layers: ModelLayer[]; training: TrainingSettings }
  | { model_type: "pretrained_deit"; model_id: string; training: TrainingSettings };
type TrainingSettings = NASArchitectureTraining;

const ACTIVATIONS = ["relu", "gelu", "silu", "tanh", "sigmoid", "linear", "softmax", "elu", "leaky_relu"];
const MAX_MODEL_PARAMETERS = 20_000_000;
const DEFAULT_LAYERS: ModelLayer[] = [
  { type: "conv2d", filters: 32, kernel_size: 3, strides: 1, padding: "same", activation: "relu" },
  { type: "batch_normalization" },
  { type: "max_pooling2d", pool_size: 2 },
  { type: "conv2d", filters: 64, kernel_size: 3, strides: 1, padding: "same", activation: "gelu" },
  { type: "max_pooling2d", pool_size: 2 },
  { type: "flatten" },
  { type: "dense", units: 256, activation: "gelu" },
  { type: "dropout", rate: 0.25 },
  { type: "dense", units: 10, activation: "linear" },
];

function isLegacyArchitecture(value: unknown): value is NASArchitecture {
  return typeof value === "object" && value !== null && "id" in value &&
    typeof value.id === "string" && "blocks" in value && "channels" in value && "kernel_size" in value;
}

function isModelDesign(value: unknown): value is ModelDesign {
  if (typeof value !== "object" || value === null) return false;
  const design = value as Record<string, unknown>;
  return (design.model_type === "custom" && Array.isArray(design.layers)) ||
    (design.model_type === "pretrained_deit" && design.model_id === "facebook/deit-tiny-patch16-224") ||
    isLegacyArchitecture(value);
}

function installedDesign(model: InitialModelReport | null): ModelDesign | null {
  if (!model) return null;
  if (isModelDesign(model.architecture)) return model.architecture as ModelDesign;
  if (isLegacyArchitecture(model.architecture.design)) return model.architecture.design;
  const layers = model.architecture.layers;
  if (!Array.isArray(layers)) return null;
  const convolutions = layers.filter(
    (layer): layer is Record<string, unknown> =>
      typeof layer === "object" && layer !== null && "type" in layer && layer.type === "conv2d",
  );
  const first = convolutions[0];
  if (!first || typeof first.filters !== "number" || !Array.isArray(first.kernel_size)) return null;
  const filters = first.filters;
  const kernelSize = first.kernel_size;
  if (
    typeof kernelSize[0] !== "number" ||
    typeof kernelSize[1] !== "number" ||
    kernelSize[0] !== kernelSize[1] ||
    convolutions.some((layer) => {
      const layerKernel = layer.kernel_size;
      return layer.filters !== filters ||
        !Array.isArray(layerKernel) ||
        layerKernel[0] !== kernelSize[0] ||
        layerKernel[1] !== kernelSize[1];
    })
  ) return null;
  return {
    id: `b${convolutions.length}_c${filters}_k${kernelSize[0]}`,
    blocks: convolutions.length,
    channels: filters,
    kernel_size: kernelSize[0],
  };
}

function estimatedParameters(
  design: ModelDesign,
  imageShape: number[] = [28, 28, 1],
  classes = 10,
): number {
  if ("model_type" in design && design.model_type === "pretrained_deit") {
    return 5_717_416 - (1000 * 192 + 1000) + classes * 192 + classes;
  }
  if ("layers" in design && "parameter_count" in design) {
    return design.parameter_count;
  }
  if ("layers" in design) {
    let shape = imageShape.slice();
    let parameters = 0;
    for (const layer of design.layers) {
      if ((layer.type === "conv2d" || layer.type === "separable_conv2d") && shape.length === 3) {
        const filters = layer.filters ?? 32;
        const kernel = layer.kernel_size ?? 3;
        parameters += layer.type === "separable_conv2d"
          ? kernel ** 2 * shape[2] + shape[2] * filters + filters
          : kernel ** 2 * shape[2] * filters + filters;
        shape = [
          Math.ceil(shape[0] / (layer.strides ?? 1)),
          Math.ceil(shape[1] / (layer.strides ?? 1)),
          filters,
        ];
      } else if (["max_pooling2d", "average_pooling2d"].includes(layer.type) && shape.length === 3) {
        const pool = layer.pool_size ?? 2;
        shape = [Math.ceil(shape[0] / pool), Math.ceil(shape[1] / pool), shape[2]];
      } else if (layer.type === "batch_normalization" && shape.length) {
        parameters += shape[shape.length - 1] * 4;
      } else if (layer.type === "flatten") {
        shape = [shape.reduce((product, dimension) => product * dimension, 1)];
      } else if (layer.type === "global_average_pooling2d" && shape.length === 3) {
        shape = [shape[2]];
      } else if (layer.type === "dense" && shape.length) {
        const units = layer.units ?? classes;
        parameters += shape[shape.length - 1] * units + units;
        shape = [...shape.slice(0, -1), units];
      } else if (layer.type === "patch_embedding" && shape.length === 3) {
        const patch = layer.patch_size ?? 7;
        const embedding = layer.embed_dim ?? 64;
        parameters += patch ** 2 * shape[2] * embedding + embedding +
          (shape[0] / patch) * (shape[1] / patch) * embedding;
        shape = [(shape[0] / patch) * (shape[1] / patch), embedding];
      } else if (layer.type === "transformer_encoder" && shape.length === 2) {
        const embed = shape[1];
        const ff = layer.ff_dim ?? embed * 2;
        parameters += 4 * embed * embed + 4 * embed + 2 * embed * ff + ff + embed + 4 * embed;
      } else if (layer.type === "global_average_pooling1d" && shape.length === 2) {
        shape = [shape[1]];
      }
    }
    return parameters;
  }
  if ("blocks" in design) {
    return design.blocks * (design.kernel_size ** 2 * design.channels ** 2 + design.channels) +
      design.kernel_size ** 2 * imageShape[2] * design.channels + design.channels * classes + classes;
  }
  return 0;
}

function defaultLayer(type: LayerType): ModelLayer {
  switch (type) {
    case "conv2d":
    case "separable_conv2d":
      return { type, filters: 32, kernel_size: 3, strides: 1, padding: "same", activation: "relu" };
    case "dense": return { type, units: 128, activation: "relu" };
    case "max_pooling2d":
    case "average_pooling2d": return { type, pool_size: 2 };
    case "dropout": return { type, rate: 0.25 };
    case "patch_embedding": return { type, patch_size: 7, embed_dim: 64 };
    case "transformer_encoder": return { type, num_heads: 4, ff_dim: 128, dropout: 0.1, activation: "gelu" };
    default: return { type };
  }
}

function latestMnistGroup(results: ExperimentResults | null): ExperimentGroup | null {
  const groups = [...(results?.groups ?? [])].reverse();
  return groups.find((group) =>
    group.dataset_name === "mnist" || group.architecture?.dataset_name === "mnist",
  ) ?? groups.find((group) =>
    !group.dataset_name && !group.architecture?.dataset_name,
  ) ?? groups[0] ?? null;
}

function groupDatasetName(group: ExperimentGroup): string {
  return group.dataset_name ?? group.architecture?.dataset_name ?? "not reported by coordinator";
}

function errorText(cause: unknown): string {
  return cause instanceof Error ? cause.message : "The coordinator request failed.";
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}

function activityEntry(message: string): string {
  return `${new Date().toLocaleTimeString()} ${message}`;
}

function formatElapsedTime(seconds: number): string {
  const totalSeconds = Math.max(0, Math.floor(seconds));
  const hours = Math.floor(totalSeconds / 3600);
  const minutes = Math.floor((totalSeconds % 3600) / 60);
  const remainingSeconds = totalSeconds % 60;
  if (hours > 0) return `${hours}h ${minutes}m`;
  if (minutes > 0) return `${minutes}m ${remainingSeconds}s`;
  return `${remainingSeconds}s`;
}

function formatMetric(value: number | undefined, digits = 4): string {
  return value === undefined || !Number.isFinite(value) ? "—" : value.toFixed(digits);
}

function TrainingRunHistory({
  run,
  targetRounds,
  isActive,
  groupError,
}: {
  run: ExperimentRun;
  targetRounds: number;
  isActive: boolean;
  groupError: string | null;
}) {
  const rounds = [...run.rounds].reverse();
  const latestRound = rounds[0]?.round;
  const stateLabel = run.status === "completed"
    ? "Completed"
    : run.status === "running"
      ? "Running"
      : run.status === "failed"
        ? "Failed"
        : "Stopped";

  return (
    <details className="model-run-history" open={isActive}>
      <summary>
        <span className="model-run-history__identity">
          <strong>{run.topology}</strong>
          {run.alpha !== undefined && <span>alpha {run.alpha}</span>}
          <span>seed {run.seed}</span>
        </span>
        <span className={`model-run-history__status model-run-history__status--${run.status}`}>
          {stateLabel}
        </span>
        <span className="model-run-history__round-count">
          {run.rounds.length}/{targetRounds} rounds
        </span>
      </summary>
      {run.error && run.error !== groupError && <p className="model-run-history__error">{run.error}</p>}
      {rounds.length > 0 ? (
        <div className="model-round-history">
          {rounds.map((round) => (
            <details
              className="model-round-history__item"
              key={round.round}
              open={isActive && round.round === latestRound}
            >
              <summary>
                <strong>Round {round.round}</strong>
                <span>{Object.keys(round.workers).length} worker results</span>
              </summary>
              <div className="model-worker-round-grid">
                {Object.entries(round.workers).map(([worker, metrics]) => {
                  const accuracy = metrics.validation_accuracy ?? metrics.accuracy;
                  return (
                    <article className="model-worker-round" key={worker}>
                      <div className="model-worker-round__heading">
                        <strong>{worker}</strong>
                        <span>{metrics.status ?? "Result saved"}</span>
                      </div>
                      <div className="model-worker-round__metrics">
                        <div>
                          <span>Validation accuracy</span>
                          <strong>{accuracy === undefined ? "—" : `${(accuracy * 100).toFixed(2)}%`}</strong>
                        </div>
                        <div><span>Macro F1</span><strong>{formatMetric(metrics.validation_macro_f1 ?? metrics.macro_f1)}</strong></div>
                        <div><span>Precision</span><strong>{formatMetric(metrics.validation_macro_precision ?? metrics.macro_precision)}</strong></div>
                        <div><span>Recall</span><strong>{formatMetric(metrics.validation_macro_recall ?? metrics.macro_recall)}</strong></div>
                      </div>
                      <div className="model-worker-round__foot">
                        <span>Runtime {formatMetric(metrics.round_seconds ?? metrics.runtime_seconds, 2)} s</span>
                        <span>Peer wait {formatMetric(metrics.peer_wait_seconds, 2)} s</span>
                        <span>Sent {metrics.bytes_sent?.toLocaleString() ?? "—"} B</span>
                        <span>Received {metrics.bytes_received?.toLocaleString() ?? "—"} B</span>
                      </div>
                    </article>
                  );
                })}
              </div>
            </details>
          ))}
        </div>
      ) : (
        <p className="model-run-history__empty">No round metrics saved for this run yet.</p>
      )}
    </details>
  );
}

export default function ModelStudioPage({
  token,
  results,
  error: resultsError,
  onRefresh,
  nasArchitecture = null,
  onClearNasArchitecture,
}: Props) {
  const [datasetAvailable, setDatasetAvailable] = useState(false);
  const [preprocessing, setPreprocessing] = useState<PreprocessingReport | null>(null);
  const [model, setModel] = useState<InitialModelReport | null>(null);
  const [autonomy, setAutonomy] = useState<AutonomyState | null>(null);
  const [modelMode, setModelMode] = useState<"custom" | "pretrained_deit">("custom");
  const [layers, setLayers] = useState<ModelLayer[]>(DEFAULT_LAYERS.map((layer) => ({ ...layer })));
  const [trainingSettings, setTrainingSettings] = useState<TrainingSettings>({
    optimizer: "adam",
    learning_rate: 0.001,
    batch_size: 32,
  });
  const [rounds, setRounds] = useState(10);
  const [epochs, setEpochs] = useState(5);
  const [busy, setBusy] = useState(false);
  const [notice, setNotice] = useState("");
  const [error, setError] = useState("");
  const [activityLog, setActivityLog] = useState<string[]>([]);
  const [clock, setClock] = useState(() => Date.now());

  const refresh = useCallback(async () => {
    setError("");
    try {
      const [datasets, state] = await Promise.all([
        listDatasets(token),
        getAutonomyState(token),
      ]);
      const hasMnist = datasets.some((dataset) => dataset.dataset_name === "mnist");
      setDatasetAvailable(hasMnist);
      const [nextPreprocessing, nextModel] = hasMnist
        ? await Promise.all([
          getLatestPreprocessingReport(token, "mnist"),
          getInitialModelReport(token, "mnist"),
        ])
        : [null, null];
      setPreprocessing(nextPreprocessing);
      setModel(nextModel);
      setAutonomy(state);
      const design = installedDesign(nextModel);
      if (design && "model_type" in design) {
        setModelMode(design.model_type);
        if (design.model_type === "custom") setLayers(design.layers);
        setTrainingSettings(design.training ?? { optimizer: "adam", learning_rate: 0.001, batch_size: 32 });
      } else if (design) {
        setModelMode("custom");
      }
    } catch (cause) {
      setError(errorText(cause));
    }
  }, [token]);

  useEffect(() => {
    // Load persisted MNIST readiness and the installed model when this page opens.
    // eslint-disable-next-line react-hooks/set-state-in-effect
    void refresh();
  }, [refresh]);

  const installedArchitecture = installedDesign(model);
  const selectedArchitecture: ModelDesign = nasArchitecture ?? (modelMode === "pretrained_deit"
    ? {
        model_type: "pretrained_deit",
        model_id: "facebook/deit-tiny-patch16-224",
        training: trainingSettings,
      }
    : { model_type: "custom", layers, training: trainingSettings });
  const activeArchitecture = nasArchitecture ?? installedArchitecture ?? selectedArchitecture;
  const isPretrained = "model_type" in activeArchitecture &&
    activeArchitecture.model_type === "pretrained_deit";
  const customArchitecture = "model_type" in activeArchitecture &&
    activeArchitecture.model_type === "custom"
    ? activeArchitecture
    : null;
  const legacyArchitecture = !("model_type" in activeArchitecture) &&
    "id" in activeArchitecture
    ? activeArchitecture
    : null;
  const activeModelType = isPretrained ? "pretrained_deit" : customArchitecture ? "custom" : "legacy_cnn";
  const architectureName = isPretrained
    ? "DeiT-Tiny pretrained ViT"
    : legacyArchitecture
      ? legacyArchitecture.id
      : `Custom ${customArchitecture?.layers.length ?? 0}-layer model`;
  const architectureLayers = customArchitecture
    ? customArchitecture.layers
    : legacyArchitecture
      ? layersFromNASArchitecture(legacyArchitecture)
      : isPretrained
        ? [
            { type: "patch_embedding" as const, patch_size: 16, embed_dim: 192 },
            { type: "transformer_encoder" as const, num_heads: 3, ff_dim: 768 },
            { type: "global_average_pooling1d" as const },
            { type: "dense" as const, units: 10, activation: "linear" },
          ]
        : [];
  const architectureLayerLabels = isPretrained
    ? ["Patch embedding", "Transformer × 12", "Global token pooling", "MNIST classifier"]
    : undefined;
  const modelParameters = nasArchitecture
    ? estimatedParameters(nasArchitecture, preprocessing?.image_shape ?? [28, 28, 1])
    : model?.parameter_count ??
      estimatedParameters(activeArchitecture, preprocessing?.image_shape ?? [28, 28, 1]);
  const outputLayer = customArchitecture?.layers[customArchitecture.layers.length - 1];
  const canBuildDesign = modelParameters > 0 &&
    modelParameters <= MAX_MODEL_PARAMETERS &&
    (activeModelType !== "custom" ||
      (customArchitecture !== null &&
        customArchitecture.layers.length >= 2 &&
        outputLayer?.type === "dense" &&
        outputLayer.units === 10 &&
        outputLayer.activation === "linear"));
  const workerShardsReady = Boolean(
    preprocessing &&
    Object.keys(preprocessing.workers).length > 0 &&
    Object.entries(preprocessing.workers).every(([worker, item]) => {
      const localHash = item.pc_shard_sha256;
      const remoteHash = preprocessing.uploaded_workers?.[worker]?.shard_sha256;
      return Boolean(localHash && remoteHash && localHash === remoteHash);
    }),
  );
  const datasetReady = datasetAvailable && Boolean(preprocessing) && workerShardsReady;
  const stopped = autonomy?.stop_requested ?? false;
  const activeGroup = results?.groups.find((group) => group.group_id === results.active_group_id) ?? null;
  const activeGroupId = activeGroup?.group_id ?? null;
  const latestGroup = latestMnistGroup(results);
  const displayedGroup = activeGroup ?? latestGroup;
  const activeRun = displayedGroup?.runs.find((run) => run.status === "running");
  const completedRuns = displayedGroup?.runs.filter((run) => run.rounds.length > 0) ?? [];
  const lastRun = completedRuns[completedRuns.length - 1];
  const lastRound = lastRun?.rounds[lastRun.rounds.length - 1];
  const activeRunRound = activeRun?.rounds.length ?? 0;
  const currentRound = activeRun?.progress?.round ??
    Math.min(displayedGroup?.rounds ?? 0, activeRunRound + 1);
  const elapsedSeconds = activeGroup?.started_at === undefined
    ? null
    : (clock / 1000) - activeGroup.started_at;
  const visibleActionError = error;
  const savedGroupCount = results?.groups.length ?? 0;
  const savedRoundCount = displayedGroup?.runs.reduce(
    (total, run) => total + run.rounds.length,
    0,
  ) ?? 0;

  useEffect(() => {
    if (!activeGroupId) return undefined;
    const timer = window.setInterval(() => setClock(Date.now()), 1000);
    return () => window.clearInterval(timer);
  }, [activeGroupId]);

  async function runAction(action: {
    action: "create_initial_model" | "start_training" | "stop_training";
    parameters: Record<string, unknown>;
  }, reason: string, evidence: Record<string, unknown>, onProgress?: (message: string) => void) {
    onProgress?.("Submitting action; waiting for coordinator response and worker verification.");
    const dispatched = await dispatchAutonomyAction(token, action, reason, evidence);
    if (!dispatched.result.available) {
      throw new Error(dispatched.result.message ?? "The action is unavailable.");
    }
    return dispatched.result.data ?? {};
  }

  function appendActivity(message: string) {
    setActivityLog((current) => [...current, activityEntry(message)].slice(-100));
  }

  async function createModel() {
    if (!preprocessing || !workerShardsReady || busy || stopped || model) return;
    setBusy(true);
    setError("");
    setNotice("");
    setActivityLog([activityEntry(`Build and install requested for ${architectureName} on MNIST.`)]);
    try {
      const modelArchitecture = nasArchitecture && "layers" in nasArchitecture
        ? {
            model_type: "custom" as const,
            layers: nasArchitecture.layers,
            training: nasArchitecture.training,
          }
        : activeArchitecture;
      const report = await runAction(
        {
          action: "create_initial_model",
          parameters: {
            model_name: "mnist",
            architecture: modelArchitecture,
            seed: 42,
          },
        },
        `Build and install ${architectureName} for official MNIST.`,
        {
          dataset_name: "mnist",
          architecture: activeArchitecture,
          parameter_count: modelParameters,
        },
        appendActivity,
      );
      appendActivity("Coordinator returned success after completing model installation and verification.");
      if (typeof report.parameter_count === "number") {
        appendActivity(`Installed model parameter count: ${report.parameter_count.toLocaleString()}.`);
      }
      if (typeof report.architecture_sha256 === "string") {
        appendActivity(`Architecture SHA-256 verified: ${report.architecture_sha256}.`);
      }
      if (typeof report.weights_sha256 === "string") {
        appendActivity(`Weights SHA-256 verified: ${report.weights_sha256}.`);
      }
      if (typeof report.workers === "object" && report.workers !== null) {
        for (const [worker, result] of Object.entries(report.workers)) {
          if (!isRecord(result)) continue;
          const hashesMatch = typeof result.weights_sha256 === "string" &&
            result.weights_sha256 === report.weights_sha256;
          const parameterCount = typeof result.parameter_count === "number"
            ? `; ${result.parameter_count.toLocaleString()} parameters`
            : "";
          appendActivity(
            `${worker}: model installed${hashesMatch ? "; weight hash matches coordinator" : ""}${parameterCount}.`,
          );
        }
      }
      await refresh();
      setNotice(`Model ${architectureName} was initialized and verified on all configured workers.`);
      if (nasArchitecture) onClearNasArchitecture?.();
    } catch (cause) {
      setError(errorText(cause));
      appendActivity(`Build and install failed: ${errorText(cause)}`);
    } finally {
      setBusy(false);
    }
  }

  async function deleteModel() {
    if (!model || !window.confirm("Delete the installed MNIST model from this coordinator and all configured workers? This cannot be undone.")) return;
    setBusy(true);
    setError("");
    setNotice("");
    appendActivity("Model deletion requested; checking worker state.");
    try {
      await deleteInitialModel(token, "mnist");
      appendActivity("Installed MNIST model removed from the coordinator and all workers.");
      await refresh();
      onRefresh();
      setNotice("Installed MNIST model deleted from the coordinator and all configured workers.");
    } catch (cause) {
      setError(errorText(cause));
      appendActivity(`Model deletion failed: ${errorText(cause)}`);
    } finally {
      setBusy(false);
    }
  }

  async function startTraining() {
    if (!model || !installedArchitecture || busy || stopped || activeGroup) return;
    setBusy(true);
    setError("");
    setNotice("");
    try {
      const data = await runAction(
        {
          action: "start_training",
          parameters: {
            dataset_name: "mnist",
            rounds,
            epochs,
            architecture: installedArchitecture,
          },
        },
        `Start ${rounds} rounds of ${epochs}-epoch distributed training for the installed MNIST model.`,
        {
          dataset_name: "mnist",
          architecture: installedArchitecture,
          rounds,
          epochs,
        },
        appendActivity,
      );
      const groupId = typeof data.group_id === "string" ? data.group_id : "";
      setNotice(groupId ? `Training group ${groupId} started on the workers.` : "MNIST training started on the workers.");
      appendActivity(groupId ? `Training group ${groupId} accepted by the coordinator.` : "Training accepted by the coordinator.");
      onRefresh();
    } catch (cause) {
      setError(errorText(cause));
      appendActivity(`Training start failed: ${errorText(cause)}`);
    } finally {
      setBusy(false);
    }
  }

  async function stopTraining() {
    if (!activeGroup || busy || stopped) return;
    setBusy(true);
    setError("");
    setNotice("");
    try {
      await runAction(
        { action: "stop_training", parameters: {} },
        `Stop active MNIST training group ${activeGroup.group_id}.`,
        { dataset_name: "mnist", group_id: activeGroup.group_id },
        appendActivity,
      );
      setNotice(`Stop requested for training group ${activeGroup.group_id}.`);
      appendActivity(`Stop requested for training group ${activeGroup.group_id}.`);
      onRefresh();
    } catch (cause) {
      setError(errorText(cause));
      appendActivity(`Stop request failed: ${errorText(cause)}`);
    } finally {
      setBusy(false);
    }
  }

  const modelCanBeCreated = Boolean(preprocessing && workerShardsReady && !model && !stopped && !busy);
  const trainingSettingsValid = Number.isInteger(rounds) && rounds >= 1 && rounds <= 100 &&
    Number.isInteger(epochs) && epochs >= 1 && epochs <= 50;
  const trainingCanStart = Boolean(
    model && installedArchitecture && !nasArchitecture && datasetReady &&
      !activeGroup && !stopped && !busy && trainingSettingsValid,
  );

  function updateLayer(index: number, patch: Partial<ModelLayer>) {
    setLayers((current) =>
      current.map((layer, layerIndex) =>
        layerIndex === index ? { ...layer, ...patch } : layer,
      ),
    );
  }

  function updateLayerNumber(index: number, field: keyof ModelLayer, value: string) {
    const numberValue = Number(value);
    if (Number.isFinite(numberValue)) updateLayer(index, { [field]: numberValue });
  }

  return (
    <section className="content model-studio">
      <div className="page-heading">
        <h1>Model Studio</h1>
        <button className="quiet-button" disabled={busy} onClick={() => void refresh()} type="button">Refresh status</button>
      </div>

      {stopped && <div className="alert dataset-stop" role="alert">Global STOP is active. Model creation and training are disabled.</div>}
      {visibleActionError && <div className="alert inline-alert" role="alert">{visibleActionError}</div>}
      {notice && <div className="success-note dataset-notice" role="status">{notice}</div>}
      {resultsError && <div className="alert inline-alert" role="alert">{resultsError}</div>}
      {busy && <div className="dataset-progress" role="status"><span className="spinner" />Processing the confirmed action…</div>}

      <div className="model-studio-grid">
        <article className="panel-card model-design-card">
          <div className="section-heading">
          <h2>Architecture designer</h2>
            <span className={`dataset-state ${model ? "state-shipped-to-workers" : workerShardsReady ? "state-preprocessed" : "state-ready"}`}>
              {model ? "Installed" : workerShardsReady ? "MNIST ready" : "Needs preprocessing"}
            </span>
          </div>

          <label className="model-dataset-selection">
            Training dataset
            <select aria-label="Training dataset" disabled={!datasetAvailable || busy || Boolean(activeGroup)} value={datasetAvailable ? "mnist" : ""}>
              {datasetAvailable
                ? <option value="mnist">Official MNIST · 70,000 images</option>
                : <option value="">MNIST dataset unavailable</option>}
            </select>
          </label>

          {!model && !nasArchitecture && (
            <div className="model-builder-controls">
              <label>
                Model family
                <select
                  disabled={busy}
                  onChange={(event) => setModelMode(event.target.value as "custom" | "pretrained_deit")}
                  value={modelMode}
                >
                  <option value="custom">Custom Keras architecture</option>
                  <option value="pretrained_deit">Pretrained Vision Transformer — DeiT-Tiny</option>
                </select>
              </label>
              {modelMode === "custom" ? (
                <div className="model-layer-editor">
                  <div className="model-layer-editor-heading">
                    <strong>Layer stack</strong>
                    <button
                      className="quiet-button"
                      disabled={busy || layers.length >= 32}
                      onClick={() => setLayers((current) => [...current, defaultLayer("dense")])}
                      type="button"
                    >
                      Add layer
                    </button>
                  </div>
                  {layers.map((layer, index) => (
                    <div className="model-layer-row" key={`layer-${index}`}>
                      <strong>{index + 1}</strong>
                      <select
                        aria-label={`Layer ${index + 1} type`}
                        disabled={busy}
                        onChange={(event) => {
                          const type = event.target.value as LayerType;
                          updateLayer(index, defaultLayer(type));
                        }}
                        value={layer.type}
                      >
                        <option value="conv2d">Convolution 2D</option>
                        <option value="separable_conv2d">Separable convolution 2D</option>
                        <option value="max_pooling2d">Max pooling</option>
                        <option value="average_pooling2d">Average pooling</option>
                        <option value="batch_normalization">Batch normalization</option>
                        <option value="dropout">Dropout</option>
                        <option value="flatten">Flatten</option>
                        <option value="global_average_pooling2d">Global average pool 2D</option>
                        <option value="dense">Dense</option>
                        <option value="patch_embedding">Vision patch embedding</option>
                        <option value="transformer_encoder">Transformer / self-attention block</option>
                        <option value="global_average_pooling1d">Global token pooling</option>
                      </select>
                      {(layer.type === "conv2d" || layer.type === "separable_conv2d") && (
                        <>
                          <label>Filters<input aria-label={`Layer ${index + 1} filters`} max={1024} min={1} onChange={(event) => updateLayerNumber(index, "filters", event.target.value)} type="number" value={layer.filters ?? 32} /></label>
                          <label>Kernel<input aria-label={`Layer ${index + 1} kernel`} max={15} min={1} onChange={(event) => updateLayerNumber(index, "kernel_size", event.target.value)} type="number" value={layer.kernel_size ?? 3} /></label>
                          <label>Stride<input aria-label={`Layer ${index + 1} stride`} max={8} min={1} onChange={(event) => updateLayerNumber(index, "strides", event.target.value)} type="number" value={layer.strides ?? 1} /></label>
                          <label>Activation<select aria-label={`Layer ${index + 1} activation`} onChange={(event) => updateLayer(index, { activation: event.target.value })} value={layer.activation ?? "relu"}>{ACTIVATIONS.map((activation) => <option key={activation} value={activation}>{activation}</option>)}</select></label>
                        </>
                      )}
                      {(layer.type === "max_pooling2d" || layer.type === "average_pooling2d") && (
                        <label>Pool size<input aria-label={`Layer ${index + 1} pool size`} max={8} min={2} onChange={(event) => updateLayerNumber(index, "pool_size", event.target.value)} type="number" value={layer.pool_size ?? 2} /></label>
                      )}
                      {layer.type === "dropout" && (
                        <label>Dropout rate<input aria-label={`Layer ${index + 1} dropout rate`} max={0.95} min={0} onChange={(event) => updateLayerNumber(index, "rate", event.target.value)} step={0.05} type="number" value={layer.rate ?? 0.25} /></label>
                      )}
                      {layer.type === "dense" && (
                        <>
                          <label>Units<input aria-label={`Layer ${index + 1} units`} max={8192} min={1} onChange={(event) => updateLayerNumber(index, "units", event.target.value)} type="number" value={layer.units ?? 128} /></label>
                          <label>Activation<select aria-label={`Layer ${index + 1} activation`} onChange={(event) => updateLayer(index, { activation: event.target.value })} value={layer.activation ?? "relu"}>{ACTIVATIONS.map((activation) => <option key={activation} value={activation}>{activation}</option>)}</select></label>
                        </>
                      )}
                      {layer.type === "patch_embedding" && (
                        <>
                          <label>Patch size<input aria-label={`Layer ${index + 1} patch size`} max={14} min={1} onChange={(event) => updateLayerNumber(index, "patch_size", event.target.value)} type="number" value={layer.patch_size ?? 7} /></label>
                          <label>Embedding dim<input aria-label={`Layer ${index + 1} embedding size`} max={1024} min={8} onChange={(event) => updateLayerNumber(index, "embed_dim", event.target.value)} type="number" value={layer.embed_dim ?? 64} /></label>
                        </>
                      )}
                      {layer.type === "transformer_encoder" && (
                        <>
                          <label>Heads<input aria-label={`Layer ${index + 1} attention heads`} max={32} min={1} onChange={(event) => updateLayerNumber(index, "num_heads", event.target.value)} type="number" value={layer.num_heads ?? 4} /></label>
                          <label>FFN width<input aria-label={`Layer ${index + 1} feed-forward width`} max={4096} min={8} onChange={(event) => updateLayerNumber(index, "ff_dim", event.target.value)} type="number" value={layer.ff_dim ?? 128} /></label>
                          <label>Dropout<input aria-label={`Layer ${index + 1} attention dropout`} max={0.95} min={0} onChange={(event) => updateLayerNumber(index, "dropout", event.target.value)} step={0.05} type="number" value={layer.dropout ?? 0.1} /></label>
                        </>
                      )}
                      <button
                        aria-label={`Remove layer ${index + 1}`}
                        className="remove-device-button"
                        disabled={busy || layers.length <= 2}
                        onClick={() => setLayers((current) => current.filter((_, layerIndex) => layerIndex !== index))}
                        type="button"
                      >
                        Remove
                      </button>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="pretrained-model-callout">
                  <strong>facebook/deit-tiny-patch16-224</strong>
                  <span>Pretrained DeiT-Tiny vision transformer · ImageNet weights · about 5.5M parameters</span>
                  <span>MNIST images are resized to 224 × 224 and converted to RGB on each worker.</span>
                </div>
              )}
              <div className="model-training-options">
                <label>Optimizer
                  <select value={trainingSettings.optimizer} onChange={(event) => setTrainingSettings((current) => ({ ...current, optimizer: event.target.value as TrainingSettings["optimizer"] }))}>
                    <option value="adam">Adam</option><option value="adamw">AdamW</option><option value="sgd">SGD</option><option value="rmsprop">RMSprop</option>
                  </select>
                </label>
                <label>Learning rate
                  <input aria-label="Learning rate" max={0.1} min={0.000001} onChange={(event) => setTrainingSettings((current) => ({ ...current, learning_rate: Number(event.target.value) }))} step="any" type="number" value={trainingSettings.learning_rate} />
                </label>
                <label>Batch size
                  <input aria-label="Batch size" max={256} min={8} onChange={(event) => setTrainingSettings((current) => ({ ...current, batch_size: Number(event.target.value) }))} step={8} type="number" value={trainingSettings.batch_size} />
                </label>
              </div>
            </div>
          )}
          {!model && nasArchitecture && (
            <div className="pretrained-model-callout">
              <strong>NAS-selected MNIST architecture</strong>
              <span>
                {nasArchitecture.id} · {nasArchitecture.blocks} convolution blocks ·{" "}
                {"channel_schedule" in nasArchitecture
                  ? `${nasArchitecture.channel_schedule.join("/")} channels`
                  : `${nasArchitecture.channels} channels`} ·{" "}
                {nasArchitecture.kernel_size}×{nasArchitecture.kernel_size} kernels
                {"parameter_count" in nasArchitecture
                  ? ` · ${nasArchitecture.parameter_count.toLocaleString()} parameters`
                  : ""}.
              </span>
              <button className="quiet-button" onClick={onClearNasArchitecture} type="button">Return to architecture designer</button>
            </div>
          )}

          <div className="model-architecture-summary">
            <div>
              <span>Model design</span>
              <strong>{architectureName}</strong>
            </div>
            <div>
              <span>Input shape</span>
              <strong>{model?.architecture.input_shape.join(" × ") ?? "28 × 28 × 1"}</strong>
            </div>
            <div>
              <span>Trainable parameters</span>
              <strong>{modelParameters.toLocaleString()}</strong>
            </div>
            <div>
              <span>Output classes</span>
              <strong>10 digits (0–9)</strong>
            </div>
          </div>

          {architectureLayers.length > 0 && (
            <LayeredArchitectureView
              inputShape={model?.architecture.input_shape.join(" × ") ?? "28 × 28 × 1"}
              layerLabels={architectureLayerLabels}
              layers={architectureLayers}
              note={legacyArchitecture
                ? "Legacy layer sizes are illustrative."
                : undefined}
              title={`${architectureName} architecture`}
            />
          )}
          {!canBuildDesign && !model && (
            <div className="alert inline-alert" role="alert">
              The design must finish with a 10-unit linear Dense classifier and stay below 20M parameters.
            </div>
          )}

          {model && (
            <div className="model-integrity">
              <span>Architecture SHA-256 <code>{model.architecture_sha256}</code></span>
              <span>Weights SHA-256 <code>{model.weights_sha256}</code></span>
            </div>
          )}
          {!preprocessing && <p className="model-hint">MNIST preprocessing required.</p>}
          {preprocessing && !workerShardsReady && (
            <p className="model-hint">Worker shard verification incomplete.</p>
          )}
          {model && <p className="model-hint">Installed model · immutable.</p>}
          {model && nasArchitecture && (
            <p className="model-hint">Selected NAS architecture · not installed.</p>
          )}
          {model && !installedArchitecture && (
            <div className="alert inline-alert" role="alert">The installed model does not match a supported MNIST design, so training is disabled.</div>
          )}

          <div className="model-studio-actions">
            {!model && (
              <button className="quiet-button" disabled={!modelCanBeCreated || !canBuildDesign} onClick={() => void createModel()} type="button">
                {busy ? "Building model…" : "Build and install model"}
              </button>
            )}
            {model && (
              <>
                <button
                  className="remove-device-button"
                  disabled={busy || Boolean(activeGroup)}
                  onClick={() => void deleteModel()}
                  type="button"
                >
                  {busy ? "Working…" : "Delete installed model"}
                </button>
                <label className="model-run-setting">
                  Rounds
                  <input max={100} min={1} onChange={(event) => setRounds(Number(event.target.value))} type="number" value={rounds} />
                </label>
                <label className="model-run-setting">
                  Local epochs
                  <input max={50} min={1} onChange={(event) => setEpochs(Number(event.target.value))} type="number" value={epochs} />
                </label>
                <button className="primary-button" disabled={!trainingCanStart} onClick={() => void startTraining()} type="button">
                  {activeGroup ? "Training in progress…" : busy ? "Starting training…" : "Start training"}
                </button>
              </>
            )}
          </div>
        </article>

        <article className="panel-card model-training-card">
          <div className="section-heading">
            <h2>Training status &amp; activity logs</h2>
            {displayedGroup && <span className={`dataset-state ${displayedGroup.status === "running" ? "state-preprocessed" : "state-ready"}`}>{displayedGroup.status}</span>}
          </div>
          {!displayedGroup ? (
            <div className="dataset-empty">
              {results
                ? `Saved training groups: ${savedGroupCount}.`
                : "Waiting for coordinator results."}
            </div>
          ) : (
            <>
              <p>
                Training group <strong>{displayedGroup.group_id}</strong> is {displayedGroup.status}.
                {" "}Dataset: {groupDatasetName(displayedGroup)} · {displayedGroup.topology}
                {" "}· target {displayedGroup.rounds} rounds × {displayedGroup.epochs} epochs
              </p>
              {displayedGroup.error && <div className="alert inline-alert" role="alert">{displayedGroup.error}</div>}
              {lastRound ? (
                <div className="model-worker-metrics">
                  {Object.entries(lastRound.workers).map(([worker, metrics]) => (
                    <div className="model-worker-metric" key={worker}>
                      <strong>{worker}</strong>
                      <span>Round {lastRound.round} · {metrics.status ?? displayedGroup.status}</span>
                      <small>
                        Accuracy {metrics.validation_accuracy === undefined ? "—" : `${(metrics.validation_accuracy * 100).toFixed(2)}%`}
                        {" · "}Macro-F1 {metrics.validation_macro_f1?.toFixed(4) ?? "—"}
                      </small>
                      <small>
                        Runtime {metrics.round_seconds?.toFixed(2) ?? "—"} s
                        {" · "}Sent {metrics.bytes_sent?.toLocaleString() ?? "—"} B
                        {" · "}Received {metrics.bytes_received?.toLocaleString() ?? "—"} B
                      </small>
                    </div>
                  ))}
                </div>
              ) : activeGroup && activeRun?.progress ? null : activeGroup ? (
                <div className="model-round-progress" role="status">
                  <strong>
                    Waiting for round {currentRound} of {activeGroup.rounds} results
                    {elapsedSeconds !== null && ` · ${formatElapsedTime(elapsedSeconds)} elapsed`}
                  </strong>
                  <div className="model-round-progress-track" aria-hidden="true">
                    <span />
                  </div>
                  <small>No round metrics yet.</small>
                </div>
              ) : (
                <p>Run accepted · waiting for round metrics.</p>
              )}
              {activeGroup && activeRun?.progress && (
                <div className="model-round-progress" role="status">
                  <strong>
                    Round {currentRound} of {activeGroup.rounds} · worker progress
                    {elapsedSeconds !== null && ` · ${formatElapsedTime(elapsedSeconds)} elapsed`}
                  </strong>
                  <div className="model-round-progress-track" aria-hidden="true">
                    <span />
                  </div>
                  <ul>
                    {Object.entries(activeRun.progress.workers).map(([worker, progress]) => (
                      <li key={worker}>
                        {worker}: {progress.phase === "peer_exchange"
                          ? "local epochs complete · exchanging weights"
                          : progress.epoch !== null && progress.epochs !== null
                            ? `local epoch ${progress.epoch} of ${progress.epochs}`
                            : "starting local training"}
                      </li>
                    ))}
                  </ul>
                </div>
              )}
              {activeGroup && activeGroup.status === "running" && (
                <button className="danger-button" disabled={busy || stopped} onClick={() => void stopTraining()} type="button">
                  Stop active training
                </button>
              )}
            </>
          )}
          <div className="model-live-log" aria-label="Training activity and round history">
            <div className="model-live-log__heading">
              <div>
                <strong>Round-by-round results</strong>
                <span>{displayedGroup?.runs.length ?? 0} runs · {savedRoundCount} saved rounds · worker metrics</span>
              </div>
              {displayedGroup?.status === "running" && <span className="model-live-log__live">LIVE · refreshes every 3s</span>}
            </div>
            {displayedGroup?.runs.length ? (
              <div className="model-run-history-list">
                {[...displayedGroup.runs].reverse().map((run, index) => (
                  <TrainingRunHistory
                    isActive={run.status === "running" && index === 0}
                    groupError={displayedGroup.error}
                    key={run.run_id}
                    run={run}
                    targetRounds={displayedGroup.rounds ?? 0}
                  />
                ))}
              </div>
            ) : (
              <p>No round results yet.</p>
            )}
            {activityLog.length > 0 && (
              <details className="model-action-history">
                <summary>Recent model actions ({activityLog.length})</summary>
                <ol aria-live="polite" aria-relevant="additions">
                  {[...activityLog].reverse().map((line, index) => (
                    <li key={`${displayedGroup?.group_id ?? "model"}-${index}`}>{line}</li>
                  ))}
                </ol>
              </details>
            )}
          </div>
        </article>
      </div>
    </section>
  );
}
