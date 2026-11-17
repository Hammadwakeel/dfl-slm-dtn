import { useEffect, useRef, useState } from "react";

import {
  dispatchAutonomyAction,
  deleteAgentWorkflows,
  deleteSlmConversation,
  getExperimentResults,
  getAutonomyHistory,
  getAutonomyState,
  getInitialModelReport,
  getLatestPreprocessingReport,
  getSlmConversation,
  getSlmConversations,
  listDatasets,
  recordAutonomyIteration,
  sendSlmMessage,
  startAutonomyLoop,
  stopAutonomyLoop,
  type AutonomyHistoryEntry,
  type AutonomyLoop,
  type ExperimentResults,
  type SlmConversation,
  type SlmConversationSummary,
  type SlmConversationMessage,
  type DatasetMetadata,
  type ExperimentGroup,
  type PreprocessingReport,
} from "../api/client";

interface Message {
  id: number | string;
  role: "user" | "assistant";
  text: string;
  trace?: NonNullable<SlmConversationMessage["response_data"]>["trace"];
}

type WorkflowDataset = "mnist" | "cifar10";
type WorkflowDatasetMetadata = DatasetMetadata & { dataset_name: WorkflowDataset };

const INITIAL_BASELINE_ROUNDS = 5;
const INITIAL_BASELINE_EPOCHS = 10;

function datasetTargetSize(dataset: DatasetMetadata): { height: number; width: number } {
  const counts = new Map<string, number>();
  for (const sizes of Object.values(dataset.image_sizes)) {
    for (const [size, count] of Object.entries(sizes)) {
      counts.set(size, (counts.get(size) ?? 0) + count);
    }
  }
  const mostCommon = [...counts.entries()].sort((left, right) => right[1] - left[1])[0]?.[0];
  if (!mostCommon) throw new Error(`Dataset ${dataset.dataset_name} has no readable image dimensions.`);
  const [width, height] = mostCommon.split("x").map(Number);
  if (!Number.isInteger(width) || !Number.isInteger(height) || width < 1 || height < 1) {
    throw new Error(`Dataset ${dataset.dataset_name} has invalid image dimensions.`);
  }
  return { height, width };
}

function workerShardsReady(report: PreprocessingReport | null): boolean {
  const workers = Object.entries(report?.workers ?? {});
  return workers.length > 0 && workers.every(([workerId, worker]) => {
    const localHash = worker.pc_shard_sha256;
    return Boolean(localHash && report?.uploaded_workers?.[workerId]?.shard_sha256 === localHash);
  });
}

interface AssistantPageProps {
  token: string;
  workflowOnly?: boolean;
}

interface WorkflowDecision {
  id: string;
  conversationId: string;
  workflowId: string | null;
  createdAt: string;
  tool: string;
  decision: string;
  arguments: Record<string, unknown>;
  result: Record<string, unknown>;
  reply: string;
}

export default function AssistantPage({ token, workflowOnly = false }: AssistantPageProps) {
  const [messages, setMessages] = useState<Message[]>([]);
  const [datasets, setDatasets] = useState<WorkflowDatasetMetadata[]>([]);
  const [datasetLoading, setDatasetLoading] = useState(true);
  const [datasetError, setDatasetError] = useState("");
  const [workflowDataset, setWorkflowDataset] = useState("");
  const [stopRule, setStopRule] = useState<"target_metric" | "no_improvement_2" | "max_iterations">("target_metric");
  const [targetAccuracy, setTargetAccuracy] = useState(95);
  const [maxIterations, setMaxIterations] = useState(3);
  const [workflowStatus, setWorkflowStatus] = useState<"idle" | "running" | "completed" | "stopped" | "failed">("idle");
  const [workflowPhase, setWorkflowPhase] = useState("");
  const [workflowProgress, setWorkflowProgress] = useState({ iteration: 0, metric: null as number | null });
  const [workflowError, setWorkflowError] = useState("");
  const [workflowLoopId, setWorkflowLoopId] = useState<string | null>(null);
  const [monitorLoop, setMonitorLoop] = useState<AutonomyLoop | null>(null);
  const [monitorLoops, setMonitorLoops] = useState<AutonomyLoop[]>([]);
  const [monitorResults, setMonitorResults] = useState<ExperimentResults | null>(null);
  const [workflowHistory, setWorkflowHistory] = useState<AutonomyHistoryEntry[]>([]);
  const [workflowDecisions, setWorkflowDecisions] = useState<WorkflowDecision[]>([]);
  const [workflowMonitorError, setWorkflowMonitorError] = useState("");
  const [workflowDeleteError, setWorkflowDeleteError] = useState("");
  const [workflowMonitorRefreshKey, setWorkflowMonitorRefreshKey] = useState(0);
  const [selectedWorkflowIds, setSelectedWorkflowIds] = useState<Set<string>>(new Set());
  const [deletingWorkflows, setDeletingWorkflows] = useState(false);
  const [stoppingObservedLoop, setStoppingObservedLoop] = useState(false);
  const [resumingNasWorkflowId, setResumingNasWorkflowId] = useState<string | null>(null);
  const loadedWorkflowConversations = useRef(new Map<string, {
    summary: SlmConversationSummary;
    conversation: SlmConversation;
  }>());
  const [conversations, setConversations] = useState<SlmConversationSummary[]>([]);
  const [selectedConversationId, setSelectedConversationId] = useState<string | null>(null);
  const [input, setInput] = useState("");
  const [pending, setPending] = useState(false);
  const [requestError, setRequestError] = useState("");
  const [historyError, setHistoryError] = useState("");
  const [historyLoading, setHistoryLoading] = useState(true);
  const [deletingConversationId, setDeletingConversationId] = useState<string | null>(null);
  const [selectedConversationIds, setSelectedConversationIds] = useState<Set<string>>(new Set());
  const sequence = useRef(0);
  const bottom = useRef<HTMLDivElement>(null);
  const stopWorkflowRef = useRef(false);
  const workflowIdRef = useRef<string | null>(null);
  const workflowRunning = workflowStatus === "running";
  const workflowBusy = workflowRunning || monitorLoop?.status === "RUNNING";
  const workflowActivities = collectWorkflowActivities(workflowHistory, workflowDecisions);
  const workflowRuns = groupWorkflowRuns(
    workflowActivities,
    workflowDecisions,
    monitorLoops,
    monitorResults?.groups ?? [],
  );
  const deletableWorkflowRuns = workflowRuns.filter((run) =>
    (run.conversationIds.length > 0 || !run.id.startsWith("conversation:"))
    && run.loop?.status !== "RUNNING"
    && !run.groups.some((group) => group.status === "running"),
  );

  useEffect(() => {
    if (!workflowOnly) return;
    let active = true;
    setDatasetLoading(true);
    listDatasets(token)
      .then((items) => {
        if (!active) return;
        const supported = items
          .filter((item): item is WorkflowDatasetMetadata =>
            item.dataset_name === "mnist" || item.dataset_name === "cifar10")
          .sort((left, right) => ["mnist", "cifar10"].indexOf(left.dataset_name) - ["mnist", "cifar10"].indexOf(right.dataset_name));
        setDatasets(supported);
        setWorkflowDataset((current) =>
          supported.some((item) => item.dataset_name === current)
            ? current
            : supported[0]?.dataset_name ?? "",
        );
        setDatasetError("");
      })
      .catch((cause: unknown) => {
        if (active) setDatasetError(errorText(cause));
      })
      .finally(() => {
        if (active) setDatasetLoading(false);
      });
    return () => {
      active = false;
    };
  }, [token, workflowOnly]);

  useEffect(() => {
    if (!workflowOnly) return;
    let active = true;
    let loading = false;

    async function refreshWorkflowMonitor() {
      if (loading) return;
      loading = true;
      try {
        const [autonomy, results, history, summaries] = await Promise.all([
          getAutonomyState(token),
          getExperimentResults(token),
          getAutonomyHistory(token),
          getSlmConversations(token, "agent"),
        ]);
        const currentLoop = autonomy.loops.find((loop) => loop.status === "RUNNING")
          ?? autonomy.loops[0]
          ?? null;
        const currentGroup = results.groups.find(
          (group) => group.group_id === results.active_group_id,
        ) ?? results.groups.find((group) => group.status === "running") ?? null;
        const workflowSummaries = summaries
          .filter((summary) => summary.workflow_id || isDatasetWorkflowConversation(summary.title))
          .slice(0, 120);
        const changedSummaries = workflowSummaries.filter((summary) => {
          const cached = loadedWorkflowConversations.current.get(summary.id);
          return !cached
            || cached.summary.updated_at !== summary.updated_at
            || cached.summary.message_count !== summary.message_count;
        });
        const updatedConversations = await Promise.all(
          changedSummaries.map(async (summary) => ({
            summary,
            conversation: await getSlmConversation(token, summary.id, "agent"),
          })),
        );
        for (const item of updatedConversations) {
          loadedWorkflowConversations.current.set(item.summary.id, item);
        }
        const workflowConversations = workflowSummaries
          .map((summary) => loadedWorkflowConversations.current.get(summary.id)?.conversation)
          .filter((conversation): conversation is SlmConversation => Boolean(conversation));
        if (!active) return;
        setMonitorLoop(currentLoop);
        setMonitorLoops(autonomy.loops);
        setMonitorResults(results);
        setWorkflowHistory(history);
        setWorkflowDecisions(extractWorkflowDecisions(workflowConversations));
        setWorkflowMonitorError("");
        if (
          currentLoop?.status === "RUNNING"
          && (currentGroup?.dataset_name === "mnist" || currentGroup?.dataset_name === "cifar10")
        ) {
          setWorkflowDataset(currentGroup.dataset_name);
        }
      } catch (cause) {
        if (active) {
          setWorkflowMonitorError(
            cause instanceof Error
              ? cause.message
              : "Could not load persisted workflow activity from the coordinator.",
          );
        }
      } finally {
        loading = false;
      }
    }

    const initialRefresh = window.setTimeout(() => void refreshWorkflowMonitor(), 0);
    const timer = window.setInterval(() => void refreshWorkflowMonitor(), 5000);
    return () => {
      active = false;
      window.clearTimeout(initialRefresh);
      window.clearInterval(timer);
    };
  }, [token, workflowOnly, workflowMonitorRefreshKey]);

  useEffect(() => {
    bottom.current?.scrollIntoView?.({ behavior: "smooth", block: "end" });
  }, [messages, pending]);

  useEffect(() => {
    if (workflowOnly) return;
    let active = true;
    async function loadHistory() {
      setHistoryLoading(true);
      setHistoryError("");
      try {
        const history = await getSlmConversations(token, "human");
        if (!active) return;
        setConversations(history);
        const latest = history[0];
        if (latest) {
          setSelectedConversationId(latest.id);
          const conversation = await getSlmConversation(token, latest.id, "human");
          if (active) setMessages(restoreMessages(conversation.messages));
        } else {
          setSelectedConversationId(null);
          setMessages([]);
        }
      } catch (cause) {
        if (active) {
          setHistoryError(
            cause instanceof Error ? cause.message : "Could not load chat history.",
          );
        }
      } finally {
        if (active) setHistoryLoading(false);
      }
    }
    void loadHistory();
    return () => {
      active = false;
    };
  }, [token, workflowOnly]);

  async function refreshHistory() {
    try {
      setConversations(await getSlmConversations(token, "human"));
      setHistoryError("");
    } catch (cause) {
      setHistoryError(cause instanceof Error ? cause.message : "Could not refresh chat history.");
    }
  }

  async function openConversation(conversationId: string) {
    if (pending || conversationId === selectedConversationId) return;
    setSelectedConversationId(conversationId);
    setMessages([]);
    setHistoryError("");
    setHistoryLoading(true);
    try {
      const conversation = await getSlmConversation(token, conversationId, "human");
      setMessages(restoreMessages(conversation.messages));
    } catch (cause) {
      setHistoryError(cause instanceof Error ? cause.message : "Could not load this conversation.");
    } finally {
      setHistoryLoading(false);
    }
  }

  function startNewConversation() {
    if (pending || deletingConversationId) return;
    setSelectedConversationId(null);
    setMessages([]);
    setInput("");
    setRequestError("");
  }

  async function deleteConversation(conversation: SlmConversationSummary) {
    if (pending || historyLoading || deletingConversationId) return;
    if (!window.confirm(`Permanently delete "${conversation.title}" and its messages?`)) return;
    setDeletingConversationId(conversation.id);
    setHistoryError("");
    try {
      await deleteSlmConversation(token, conversation.id, "human");
      const remaining = conversations.filter((item) => item.id !== conversation.id);
      setConversations(remaining);
      setSelectedConversationIds((current) => {
        const next = new Set(current);
        next.delete(conversation.id);
        return next;
      });
      if (selectedConversationId === conversation.id) {
        setSelectedConversationId(null);
        setMessages([]);
        setInput("");
        const nextConversation = remaining[0];
        if (nextConversation) await openConversation(nextConversation.id);
      }
    } catch (cause) {
      setHistoryError(cause instanceof Error ? cause.message : "Could not delete this conversation.");
    } finally {
      setDeletingConversationId(null);
    }
  }

  async function deleteSelectedConversations() {
    const selected = conversations.filter((conversation) => selectedConversationIds.has(conversation.id));
    if (pending || historyLoading || deletingConversationId || selected.length === 0) return;
    if (!window.confirm(`Permanently delete ${selected.length} selected chat${selected.length === 1 ? "" : "s"} and all their messages?`)) return;
    setDeletingConversationId("bulk");
    setHistoryError("");
    try {
      const outcomes = await Promise.allSettled(
        selected.map((conversation) => deleteSlmConversation(token, conversation.id, "human")),
      );
      const deletedIds = selected
        .filter((_conversation, index) => outcomes[index].status === "fulfilled")
        .map((conversation) => conversation.id);
      const remaining = conversations.filter((conversation) => !deletedIds.includes(conversation.id));
      setConversations(remaining);
      setSelectedConversationIds((current) => new Set(
        [...current].filter((conversationId) => !deletedIds.includes(conversationId)),
      ));
      if (deletedIds.includes(selectedConversationId ?? "")) {
        setSelectedConversationId(null);
        setMessages([]);
        setInput("");
        const nextConversation = remaining[0];
        if (nextConversation) await openConversation(nextConversation.id);
      }
      const failures = outcomes.filter((outcome): outcome is PromiseRejectedResult => (
        outcome.status === "rejected"
      ));
      if (failures.length > 0) {
        const messages = failures
          .map(({ reason }) => reason instanceof Error ? reason.message : String(reason))
          .filter(Boolean);
        setHistoryError(
          `${deletedIds.length} chat${deletedIds.length === 1 ? "" : "s"} deleted; ${failures.length} could not be deleted.${messages.length > 0 ? ` ${messages.join(" ")}` : ""}`,
        );
      }
    } finally {
      setDeletingConversationId(null);
    }
  }

  async function deleteSelectedWorkflows() {
    const selected = workflowRuns.filter((run) => selectedWorkflowIds.has(run.id));
    if (deletingWorkflows || selected.length === 0) return;
    if (!window.confirm(
      `Permanently delete ${selected.length} selected Agent workflow${selected.length === 1 ? "" : "s"}, including their chat, action, and loop history? Experiment results will remain on the Experiments page.`,
    )) return;
    setDeletingWorkflows(true);
    setWorkflowDeleteError("");
    try {
      const workflowIds = selected
        .map((run) => run.id)
        .filter((id) => !id.startsWith("conversation:"));
      const conversationIds = selected.flatMap((run) => [
        ...run.conversationIds,
        ...(run.id.startsWith("conversation:") ? [run.id.slice("conversation:".length)] : []),
      ]);
      await deleteAgentWorkflows(token, [...new Set(workflowIds)], [...new Set(conversationIds)]);
      setSelectedWorkflowIds(new Set());
      for (const conversationId of new Set(conversationIds)) {
        loadedWorkflowConversations.current.delete(conversationId);
      }
      setWorkflowMonitorRefreshKey((current) => current + 1);
    } catch (cause) {
      setWorkflowDeleteError(
        cause instanceof Error ? cause.message : "Could not delete the selected Agent workflows.",
      );
    } finally {
      setDeletingWorkflows(false);
    }
  }

  async function sendMessage(text: string) {
    const content = text.trim();
    if (!content || pending || workflowRunning || historyLoading || deletingConversationId) return;
    sequence.current += 1;
    setMessages((current) => [
      ...current,
      { id: `pending-user-${sequence.current}`, role: "user", text: content },
    ]);
    setInput("");
    setPending(true);
    setRequestError("");
    try {
      const response = await sendSlmMessage(
        token,
        content,
        selectedConversationId ?? undefined,
        "human",
      );
      setSelectedConversationId(response.conversation_id);
      sequence.current += 1;
      setMessages((current) => [
        ...current,
        {
          id: `pending-assistant-${sequence.current}`,
          role: "assistant",
          text: response.reply,
          trace: response.trace,
        },
      ]);
      void refreshHistory();
    } catch (cause) {
      setRequestError(cause instanceof Error ? cause.message : "The SLM request failed.");
      void refreshHistory();
    } finally {
      setPending(false);
    }
  }

  async function resumeSelectedCifarNas(workflowId: string) {
    if (workflowBusy || resumingNasWorkflowId) return;
    const sourceRun = workflowRuns.find((run) => run.id === workflowId);
    const nasGroup = sourceRun ? latestSelectedCifarNasGroup(sourceRun) : null;
    if (!sourceRun || !nasGroup) {
      setWorkflowError("The selected Agent workflow has no completed CIFAR-10 NAS model to resume.");
      return;
    }

    stopWorkflowRef.current = false;
    workflowIdRef.current = workflowId;
    setResumingNasWorkflowId(workflowId);
    setWorkflowDataset("cifar10");
    setWorkflowStatus("running");
    setWorkflowPhase("Checking cluster state before resuming the selected CIFAR-10 NAS model…");
    setWorkflowError("");
    setWorkflowProgress({ iteration: 0, metric: null });
    setWorkflowLoopId(null);
    let loopId: string | null = null;

    try {
      const autonomy = await getAutonomyState(token);
      if (autonomy.stop_requested) {
        throw new Error("Global STOP is active. Reset it before resuming this Agent workflow.");
      }
      if (autonomy.loops.some((loop) => loop.status === "RUNNING")) {
        throw new Error("Another Agent loop is already running. Stop or finish it before resuming this workflow.");
      }
      const before = await getExperimentResults(token);
      if (before.active_group_id || before.groups.some((group) => group.status === "running")) {
        throw new Error("A coordinator experiment is already active. Wait for it to finish before resuming.");
      }

      const loop = await startAutonomyLoop(token, {
        max_iterations: 1,
        time_limit_seconds: 24 * 60 * 60,
        stop_rule: "target_metric",
        target_metric: targetAccuracy / 100,
        workflow_id: workflowId,
      });
      loopId = loop.id;
      setWorkflowLoopId(loop.id);
      setMonitorLoop(loop);
      setMonitorLoops((current) => [loop, ...current.filter((item) => item.id !== loop.id)]);
      const seenGroups = new Set(before.groups.map((group) => group.group_id));
      const prompt = agentWorkflowPrompt(
        "cifar10",
        1,
        "target_metric",
        targetAccuracy,
        maxIterations,
        nasGroup,
      );
      setWorkflowPhase("Agent is selecting the saved NAS architecture for 5-round, 10-epoch retraining…");
      await sendSlmMessage(
        token,
        prompt,
        undefined,
        "agent",
        "cifar10",
        true,
        workflowId,
        nasGroup.group_id,
      );
      ensureWorkflowNotStopped(stopWorkflowRef);
      let results = await getExperimentResults(token);
      let retrainingGroup = [...results.groups].reverse().find(
        (group) => !seenGroups.has(group.group_id),
      );

      if (!retrainingGroup) {
        const activeGroup = results.groups.find(
          (group) => group.group_id === results.active_group_id,
        ) ?? results.groups.find((group) => group.status === "running");
        if (activeGroup) {
          retrainingGroup = activeGroup;
        }
      }

      if (!retrainingGroup) {
        throw new Error("The CIFAR-10 retraining run did not appear in coordinator results.");
      }
      if (retrainingGroup.dataset_name !== "cifar10" || retrainingGroup.kind !== "decentralized") {
        throw new Error(
          `The Agent started ${retrainingGroup.kind} on ${retrainingGroup.dataset_name ?? "an unknown dataset"} instead of CIFAR-10 retraining.`,
        );
      }
      seenGroups.add(retrainingGroup.group_id);
      while (retrainingGroup.status === "running") {
        ensureWorkflowNotStopped(stopWorkflowRef);
        setWorkflowPhase(`Waiting for CIFAR-10 retraining ${retrainingGroup.group_id} to finish…`);
        await waitFor(5000);
        results = await getExperimentResults(token);
        const updated = results.groups.find((group) => group.group_id === retrainingGroup?.group_id);
        if (!updated) {
          throw new Error(`CIFAR-10 retraining ${retrainingGroup.group_id} disappeared from coordinator results.`);
        }
        retrainingGroup = updated;
      }
      if (retrainingGroup.status !== "completed") {
        throw new Error(
          `CIFAR-10 retraining ${retrainingGroup.group_id} ended with status ${retrainingGroup.status}: ${retrainingGroup.error ?? "no error details provided"}`,
        );
      }
      const metric = experimentValidationAccuracy(retrainingGroup);
      if (metric === null) {
        throw new Error(`CIFAR-10 retraining ${retrainingGroup.group_id} completed without validation accuracy.`);
      }
      const updatedLoop = await recordAutonomyIteration(token, loop.id, metric);
      setMonitorLoop(updatedLoop);
      setMonitorLoops((current) => current.map((item) => item.id === updatedLoop.id ? updatedLoop : item));
      setMonitorResults(results);
      setWorkflowProgress({ iteration: 1, metric });
      setWorkflowStatus(metric >= targetAccuracy / 100 ? "completed" : "stopped");
      setWorkflowPhase(
        metric >= targetAccuracy / 100
          ? `CIFAR-10 selected-model retraining reached ${(metric * 100).toFixed(2)}% validation accuracy.`
          : `CIFAR-10 selected-model retraining completed at ${(metric * 100).toFixed(2)}%; the ${targetAccuracy}% target was not reached. This continuation stopped after its single required 5-round, 10-epoch validation run.`,
      );
      setWorkflowMonitorRefreshKey((current) => current + 1);
    } catch (cause) {
      let message = errorText(cause);
      if (loopId) {
        try {
          const state = await getAutonomyState(token);
          const activeLoop = state.loops.find((item) => item.id === loopId && item.status === "RUNNING");
          if (activeLoop) {
            const stopped = await stopAutonomyLoop(
              token,
              loopId,
              persistedLoopFailureReason(message),
            );
            setMonitorLoop(stopped);
            setMonitorLoops((current) => current.map((item) => item.id === stopped.id ? stopped : item));
          }
        } catch (stopCause) {
          message = `${message} The continuation loop could not be stopped cleanly: ${errorText(stopCause)}`;
        }
      }
      setWorkflowError(message);
      setWorkflowPhase("Workflow failed.");
      setWorkflowStatus(stopWorkflowRef.current ? "stopped" : "failed");
    } finally {
      setResumingNasWorkflowId(null);
      setWorkflowMonitorRefreshKey((current) => current + 1);
    }
  }

  async function continueCifarWorkflowFromLatestResult(workflowId: string) {
    if (workflowBusy || resumingNasWorkflowId) return;
    const sourceRun = workflowRuns.find((run) => run.id === workflowId);
    const initialGroup = sourceRun ? latestCompletedCifarGroup(sourceRun) : null;
    if (!sourceRun || !initialGroup || !cifarWorkflowNeedsNas(initialGroup, targetAccuracy)) {
      setWorkflowError("The selected Agent workflow has no completed, below-target CIFAR-10 experiment to continue.");
      return;
    }

    setResumingNasWorkflowId(workflowId);
    setWorkflowDataset("cifar10");
    try {
      await startAgentWorkflow({
        workflowId,
        initialGroup,
      });
    } finally {
      setResumingNasWorkflowId(null);
    }
  }

  async function startAgentWorkflow(resume?: {
    workflowId: string;
    initialGroup: ExperimentGroup;
  }) {
      const dataset = datasets.find(
        (item) => item.dataset_name === (resume?.initialGroup.dataset_name ?? workflowDataset),
      );
      if (!dataset || workflowRunning || datasetLoading) return;
      stopWorkflowRef.current = false;
      setWorkflowStatus("running");
      setWorkflowPhase(resume
        ? `Resuming CIFAR-10 Agent from completed experiment ${resume.initialGroup.group_id}…`
        : "Preparing the selected dataset and running the fixed initial baseline…");
      setWorkflowError("");
      setWorkflowProgress({
        iteration: resume ? 1 : 0,
        metric: resume ? experimentValidationAccuracy(resume.initialGroup) : null,
      });
      setWorkflowLoopId(null);
      const workflowId = resume?.workflowId ?? crypto.randomUUID();
      workflowIdRef.current = workflowId;
      let loopId: string | null = null;
      try {
        const autonomy = await getAutonomyState(token);
        if (autonomy.stop_requested) {
          throw new Error("Global STOP is active. Reset it from the autonomy controls before starting this workflow.");
        }
        const existingLoop = autonomy.loops.find((loop) => loop.status === "RUNNING");
        if (existingLoop) {
          setMonitorLoops(autonomy.loops);
          setMonitorLoop(existingLoop);
          setWorkflowStatus("idle");
          setWorkflowPhase(`Loop ${existingLoop.id} is already running. Review its details below or stop it before starting another workflow.`);
          return;
        }
        const before = await getExperimentResults(token);
        if (before.active_group_id || before.groups.some((group) => group.status === "running")) {
          throw new Error(`Training group ${before.active_group_id ?? "unknown"} is already active. Wait for it to finish before starting.`);
        }
        const resumedGroup = resume
          ? before.groups.find((group) => group.group_id === resume.initialGroup.group_id) ?? null
          : null;
        if (resume && (
          !resumedGroup
          || resumedGroup.status !== "completed"
          || resumedGroup.dataset_name !== dataset.dataset_name
        )) {
          throw new Error(`Completed resume experiment ${resume.initialGroup.group_id} is no longer available for ${dataset.dataset_name}.`);
        }
        ensureWorkflowNotStopped(stopWorkflowRef);

        const loop = await startAutonomyLoop(token, {
          max_iterations: stopRule === "max_iterations" ? maxIterations : 100,
          time_limit_seconds: 24 * 60 * 60,
          stop_rule: "max_iterations",
          workflow_id: workflowId,
        });
        loopId = loop.id;
        setWorkflowLoopId(loop.id);
        setMonitorLoop(loop);
        setMonitorLoops((current) => [
          loop,
          ...current.filter((item) => item.id !== loop.id),
        ]);

        const seenGroups = new Set(before.groups.map((group) => group.group_id));
        let conversationId: string | undefined;
        let completedIterations = resumedGroup ? 1 : 0;
        let previousGroup: ExperimentGroup | null = resumedGroup;
        let decisionTurns = 0;

        if (!resume) {
          ensureWorkflowNotStopped(stopWorkflowRef);
          const preprocessing = await getLatestPreprocessingReport(token, dataset.dataset_name);
          if (!workerShardsReady(preprocessing)) {
            setWorkflowPhase(`Preparing and verifying ${dataset.dataset_name} shards on the workers…`);
            const prepared = await dispatchAutonomyAction(
              token,
              {
                action: "preprocess_dataset",
                parameters: {
                  dataset_name: dataset.dataset_name,
                  normalization: "standard",
                  dataset_split: "stratified",
                  target_size: datasetTargetSize(dataset),
                  seed: 42,
                  replace_existing: true,
                },
              },
              `Prepare and verify the fixed ${dataset.dataset_name} dataset shards before the Agent workflow baseline.`,
              {
                source: "agent_workflow",
                workflow_id: workflowId,
                dataset_name: dataset.dataset_name,
              },
            );
            if (!prepared.result.available) {
              throw new Error(prepared.result.message ?? `Could not prepare ${dataset.dataset_name} shards.`);
            }
          }

          ensureWorkflowNotStopped(stopWorkflowRef);
          const initialModel = await getInitialModelReport(token, dataset.dataset_name);
          if (!initialModel) {
            setWorkflowPhase(`Creating and distributing the fixed initial ${dataset.dataset_name} model…`);
            const created = await dispatchAutonomyAction(
              token,
              {
                action: "create_initial_model",
                parameters: { model_name: dataset.dataset_name, seed: 42 },
              },
              `Create and distribute the fixed initial ${dataset.dataset_name} model before the Agent workflow baseline.`,
              {
                source: "agent_workflow",
                workflow_id: workflowId,
                dataset_name: dataset.dataset_name,
              },
            );
            if (!created.result.available) {
              throw new Error(created.result.message ?? `Could not create the ${dataset.dataset_name} initial model.`);
            }
          }

          ensureWorkflowNotStopped(stopWorkflowRef);
          setWorkflowPhase(
            `Running the fixed ${INITIAL_BASELINE_ROUNDS}-round, ${INITIAL_BASELINE_EPOCHS}-epoch ${dataset.dataset_name} baseline…`,
          );
          const training = await dispatchAutonomyAction(
            token,
            {
              action: "start_training",
              parameters: {
                dataset_name: dataset.dataset_name,
                rounds: INITIAL_BASELINE_ROUNDS,
                epochs: INITIAL_BASELINE_EPOCHS,
              },
            },
            `Run the fixed initial ${INITIAL_BASELINE_ROUNDS}-round, ${INITIAL_BASELINE_EPOCHS}-epoch ${dataset.dataset_name} baseline before Agent-selected follow-up actions.`,
            {
              source: "agent_workflow",
              workflow_id: workflowId,
              dataset_name: dataset.dataset_name,
              rounds: INITIAL_BASELINE_ROUNDS,
              epochs: INITIAL_BASELINE_EPOCHS,
            },
          );
          if (!training.result.available) {
            throw new Error(training.result.message ?? `Could not start the ${dataset.dataset_name} baseline.`);
          }
          const baselineGroupId = training.result.data?.group_id;
          if (typeof baselineGroupId !== "string" || !baselineGroupId) {
            throw new Error("The initial baseline was accepted without a verifiable experiment group ID.");
          }

          const baselineStartedAt = Date.now();
          let baselineGroup: ExperimentGroup | undefined;
          while (!baselineGroup) {
            ensureWorkflowNotStopped(stopWorkflowRef);
            const results = await getExperimentResults(token);
            baselineGroup = results.groups.find((group) => group.group_id === baselineGroupId);
            if (!baselineGroup && Date.now() - baselineStartedAt >= 30_000) {
              throw new Error(`Baseline experiment ${baselineGroupId} did not appear in coordinator results.`);
            }
            if (!baselineGroup) await waitFor(1000);
          }
          if (baselineGroup.dataset_name !== dataset.dataset_name) {
            throw new Error(`Baseline experiment ${baselineGroupId} used ${baselineGroup.dataset_name ?? "an unknown dataset"} instead of ${dataset.dataset_name}.`);
          }
          while (baselineGroup.status === "running") {
            ensureWorkflowNotStopped(stopWorkflowRef);
            setWorkflowPhase(`Waiting for initial baseline ${baselineGroup.group_id} to finish…`);
            await waitFor(5000);
            const results = await getExperimentResults(token);
            const updated = results.groups.find((group) => group.group_id === baselineGroupId);
            if (!updated) throw new Error(`Baseline experiment ${baselineGroupId} disappeared from coordinator results.`);
            baselineGroup = updated;
          }

          seenGroups.add(baselineGroup.group_id);
          previousGroup = baselineGroup;
          const metric = experimentValidationAccuracy(baselineGroup);
          const updatedLoop = await recordAutonomyIteration(
            token,
            loop.id,
            metric ?? undefined,
          );
          setMonitorLoop(updatedLoop);
          setMonitorLoops((current) => current.map((item) => item.id === updatedLoop.id ? updatedLoop : item));
          completedIterations += 1;
          setWorkflowProgress({ iteration: completedIterations, metric });
          setWorkflowPhase(
            `Initial baseline ${baselineGroup.group_id} ended ${baselineGroup.status}; the SLM will decide the next action from its results.`,
          );
          if (updatedLoop.status !== "RUNNING") {
            setWorkflowStatus("stopped");
            setWorkflowPhase(`Workflow safety limit reached after the initial baseline: ${updatedLoop.failure_reason ?? updatedLoop.status}.`);
            return;
          }
        }

        while (true) {
          ensureWorkflowNotStopped(stopWorkflowRef);
          if (decisionTurns >= 100) {
            throw new Error("The Agent reached the 100-decision workflow safety limit without selecting termination.");
          }
          decisionTurns += 1;
          const actionPrompt = agentWorkflowPrompt(
            dataset.dataset_name,
            completedIterations,
            stopRule,
            targetAccuracy,
            maxIterations,
            previousGroup,
          );
          setWorkflowPhase(
            `Waiting for the SLM to choose the next action (decision ${decisionTurns})…`,
          );

          const response = await sendSlmMessage(
            token,
            actionPrompt,
            conversationId,
            "agent",
            dataset.dataset_name,
            false,
            workflowId,
            previousGroup?.group_id,
          );
          conversationId = response.conversation_id;
          ensureWorkflowNotStopped(stopWorkflowRef);
          const decision = [...(response.trace ?? [])].reverse().find((item) => item.decision)?.decision;
          if (decision) setWorkflowPhase(`SLM decision: ${decision}`);
          if (response.terminated) {
            const termination = [...(response.trace ?? [])].reverse().find(
              (item) => item.tool === "finish_agent_workflow",
            );
            const outcome = termination?.result.outcome;
            const summary = termination?.result.summary;
            const reason = `SLM selected termination (${String(outcome ?? "unknown")}): ${String(summary ?? response.reply)}`;
            const stopped = await stopAutonomyLoop(token, loop.id, reason);
            setMonitorLoop(stopped);
            setMonitorLoops((current) => current.map((item) => item.id === stopped.id ? stopped : item));
            setWorkflowStatus("completed");
            setWorkflowPhase(reason);
            setWorkflowMonitorRefreshKey((current) => current + 1);
            return;
          }
          const results = await getExperimentResults(token);
          const newGroup = [...results.groups].reverse().find(
            (group) => !seenGroups.has(group.group_id),
          );
          if (newGroup) seenGroups.add(newGroup.group_id);
          if (!newGroup) {
            setWorkflowPhase(
              decision
                ? `SLM selected and completed a non-experiment action: ${decision}`
                : "No new experiment was started. Observing the latest state before the next SLM decision.",
            );
            continue;
          }
          let observedGroup = newGroup;
          if (observedGroup.dataset_name !== dataset.dataset_name) {
            if (observedGroup.status === "running") {
              await dispatchAutonomyAction(
                token,
                { action: "stop_training", parameters: {} },
                "Stop an Agent workflow experiment that used the wrong dataset.",
                {
                  group_id: observedGroup.group_id,
                  source: "agent_workflow",
                  workflow_id: workflowId,
                },
              );
            }
            throw new Error(`The Agent started an experiment for ${observedGroup.dataset_name ?? "an unknown dataset"} instead of ${dataset.dataset_name}.`);
          }
          while (observedGroup.status === "running") {
            ensureWorkflowNotStopped(stopWorkflowRef);
            setWorkflowPhase(
              `Waiting for Agent experiment ${observedGroup.group_id} to finish…`,
            );
            await waitFor(5000);
            const results = await getExperimentResults(token);
            const updated = results.groups.find((group) => group.group_id === observedGroup.group_id);
            if (!updated) throw new Error(`Experiment ${observedGroup.group_id} disappeared from coordinator results.`);
            observedGroup = updated;
          }
          if (observedGroup.status !== "completed") {
            const metric = experimentValidationAccuracy(observedGroup);
            previousGroup = observedGroup;
            const updatedLoop = await recordAutonomyIteration(
              token,
              loop.id,
              metric ?? undefined,
            );
            setMonitorLoop(updatedLoop);
            setMonitorLoops((current) => current.map((item) => item.id === updatedLoop.id ? updatedLoop : item));
            completedIterations += 1;
            setWorkflowProgress({ iteration: completedIterations, metric });
            setWorkflowPhase(
              `SLM action result: experiment ${observedGroup.group_id} ended ${observedGroup.status}. The failure is recorded for the next SLM decision.`,
            );
            if (updatedLoop.status !== "RUNNING") {
              setWorkflowStatus("stopped");
              setWorkflowPhase(
                `Workflow safety limit reached after experiment ${completedIterations}: ${updatedLoop.failure_reason ?? updatedLoop.status}. The SLM did not select termination.`,
              );
              return;
            }
            continue;
          }
          const metric = experimentValidationAccuracy(observedGroup);
          previousGroup = observedGroup;
          const updatedLoop = await recordAutonomyIteration(
            token,
            loop.id,
            metric ?? undefined,
          );
          setMonitorLoop(updatedLoop);
          setMonitorLoops((current) => current.map((item) => item.id === updatedLoop.id ? updatedLoop : item));
          completedIterations += 1;
          setWorkflowProgress({ iteration: completedIterations, metric });
          setWorkflowPhase(
            `Observed ${observedGroup.kind} ${observedGroup.group_id} as completed; the SLM will decide what to do next.`,
          );
          if (updatedLoop.status !== "RUNNING") {
            setWorkflowStatus("stopped");
            setWorkflowPhase(
              `Workflow safety limit reached after experiment ${completedIterations}: ${updatedLoop.failure_reason ?? updatedLoop.status}. The SLM did not select termination.`,
            );
            return;
          }
        }
      } catch (cause) {
        const message = errorText(cause);
        if (stopWorkflowRef.current) {
          setWorkflowStatus("stopped");
          setWorkflowPhase("Workflow stopped by user.");
          try {
            const activeResults = await getExperimentResults(token);
            if (activeResults.active_group_id) {
              await dispatchAutonomyAction(
                token,
                { action: "stop_training", parameters: {} },
                "Stop any experiment started while the Agent stop request was in flight.",
                {
                  group_id: activeResults.active_group_id,
                  source: "agent_workflow",
                  workflow_id: workflowIdRef.current,
                },
              );
            }
          } catch (stopError) {
            setWorkflowError(`The workflow stopped, but the active experiment could not be verified or stopped: ${errorText(stopError)}`);
          }
        } else {
          if (/another agent loop is already running|a loop is already running/i.test(message)) {
            try {
              const autonomy = await getAutonomyState(token);
              const runningLoop = autonomy.loops.find((loop) => loop.status === "RUNNING");
              if (runningLoop) {
                setMonitorLoops(autonomy.loops);
                setMonitorLoop(runningLoop);
                setWorkflowStatus("idle");
                setWorkflowPhase(`Loop ${runningLoop.id} is already running. Review its details below or stop it before starting another workflow.`);
                setWorkflowError("");
              } else {
                setWorkflowStatus("failed");
                setWorkflowPhase("Workflow did not reach its termination criterion.");
                setWorkflowError(message);
              }
            } catch (refreshError) {
              setWorkflowStatus("failed");
              setWorkflowPhase("Could not verify the already-running Agent loop.");
              setWorkflowError(`${message} ${errorText(refreshError)}`);
            }
          } else {
            setWorkflowStatus("failed");
            setWorkflowPhase("Workflow did not reach its termination criterion.");
            setWorkflowError(message);
          }
        }
        if (loopId && !stopWorkflowRef.current) {
          try {
            await stopAutonomyLoop(
              token,
              loopId,
              stopWorkflowRef.current ? "Stopped by user" : persistedLoopFailureReason(message),
            );
          } catch (stopError) {
            setWorkflowError((current) =>
              [current, `Could not update the persisted loop state: ${errorText(stopError)}`]
                .filter(Boolean)
                .join(" "),
            );
          }
        }
      }
  }

  async function stopObservedWorkflow() {
    const activeLoop = monitorLoops.find((loop) => loop.status === "RUNNING")
      ?? (monitorLoop?.status === "RUNNING" ? monitorLoop : null)
      ?? (workflowLoopId ? monitorLoops.find((loop) => loop.id === workflowLoopId) ?? null : null);
    if (!activeLoop || activeLoop.status !== "RUNNING" || stoppingObservedLoop) return;
    const isCurrentTabWorkflow = Boolean(
      activeLoop.workflow_id && activeLoop.workflow_id === workflowIdRef.current,
    );
    if (isCurrentTabWorkflow) {
      stopWorkflowRef.current = true;
      setWorkflowPhase(`Stopping loop ${activeLoop.id} and its experiment…`);
    }
    setStoppingObservedLoop(true);
    setWorkflowMonitorError("");
    try {
      const stoppedLoop = await stopAutonomyLoop(
        token,
        activeLoop.id,
        "Stopped by user from the Agent Runs monitor.",
      );
      setMonitorLoop(stoppedLoop);
      setMonitorLoops((loops) => loops.map((loop) => (
        loop.id === stoppedLoop.id ? stoppedLoop : loop
      )));
      const results = await getExperimentResults(token);
      const history = await getAutonomyHistory(token);
      const activeGroupBelongsToLoop = Boolean(
        results.active_group_id
        && activeLoop.workflow_id
        && (
          isCurrentTabWorkflow
          || history.some((entry) => {
            const details = recordValue(entry.details);
            const evidence = recordValue(details.evidence);
            const recordedResult = recordValue(details.result);
            const resultData = recordValue(recordedResult.data);
            return evidence.workflow_id === activeLoop.workflow_id
              && (
                details.run_id === results.active_group_id
                || recordedResult.group_id === results.active_group_id
                || resultData.group_id === results.active_group_id
              );
          })
        ),
      );
      if (activeGroupBelongsToLoop && results.active_group_id) {
        await dispatchAutonomyAction(
          token,
          { action: "stop_training", parameters: {} },
          "User stopped the experiment belonging to this Agent loop.",
          {
            group_id: results.active_group_id,
            source: "agent_workflow",
            workflow_id: activeLoop.workflow_id,
          },
        );
      }
      setWorkflowStatus("stopped");
      setWorkflowPhase(
        activeGroupBelongsToLoop
          ? `Loop ${activeLoop.id} and its experiment were stopped.`
          : `Loop ${activeLoop.id} was stopped. No active experiment could be verified as belonging to this loop, so other training was left untouched.`,
      );
    } catch (cause) {
      setWorkflowMonitorError(
        `Could not complete the stop request for loop ${activeLoop.id}: ${errorText(cause)}`,
      );
    } finally {
      setStoppingObservedLoop(false);
    }
  }

  const workflowControls = (
    <section className="agent-workflow" aria-label="Agent workflow controls">
      <div className="agent-workflow-heading">
        <div>
          <span className="eyebrow">AUTONOMOUS RUN</span>
          <h2>Dataset-to-learning workflow</h2>
        </div>
      </div>
      <div className="agent-workflow-controls">
        <label>
          Dataset
          <select
            aria-label="Workflow dataset"
            disabled={workflowBusy || datasetLoading || datasets.length === 0}
            onChange={(event) => setWorkflowDataset(event.target.value)}
            value={workflowDataset}
          >
            {datasetLoading && <option value="">Loading datasets…</option>}
            {!datasetLoading && datasets.length === 0 && <option value="">No supported datasets</option>}
            {datasets.map((dataset) => (
              <option key={dataset.dataset_name} value={dataset.dataset_name}>
                {datasetLabel(dataset.dataset_name)}
              </option>
            ))}
          </select>
        </label>
        <label>
          Termination criterion
          <select
            aria-label="Termination criterion"
            disabled={workflowBusy}
            onChange={(event) => setStopRule(event.target.value as typeof stopRule)}
            value={stopRule}
          >
            <option value="target_metric">Validation accuracy target</option>
            <option value="no_improvement_2">No improvement for 2 experiments</option>
            <option value="max_iterations">Maximum completed experiments</option>
          </select>
        </label>
        {stopRule === "target_metric" && (
          <label className="agent-workflow-number">
            Target accuracy
            <span>
              <input
                aria-label="Target validation accuracy percent"
                disabled={workflowBusy}
                max={100}
                min={1}
                onChange={(event) => setTargetAccuracy(Number(event.target.value))}
                type="number"
                value={targetAccuracy}
              />%
            </span>
          </label>
        )}
        {stopRule === "max_iterations" && (
          <label className="agent-workflow-number">
            Completed experiments
            <input
              aria-label="Maximum completed experiments"
              disabled={workflowBusy}
              max={100}
              min={1}
              onChange={(event) => setMaxIterations(Number(event.target.value))}
              type="number"
              value={maxIterations}
            />
          </label>
        )}
        <div className="agent-workflow-actions">
          <button
            className="send-button"
            disabled={workflowBusy || datasetLoading || !workflowDataset || Boolean(datasetError || workflowMonitorError)}
            onClick={() => void startAgentWorkflow()}
            type="button"
          >
            {workflowRunning ? "Agent running…" : "Start Agent"}
          </button>
          {workflowRunning ? (
            <button
              className="agent-stop-button"
              disabled={stoppingObservedLoop}
              onClick={() => void stopObservedWorkflow()}
              aria-label={`Kill loop ${workflowLoopId ?? monitorLoop?.id ?? ""}`}
              type="button"
            >
              {stoppingObservedLoop ? "Stopping…" : "Stop loop"}
            </button>
          ) : monitorLoop?.status === "RUNNING" && (
            <button
              className="agent-stop-button"
              disabled={stoppingObservedLoop}
              onClick={() => void stopObservedWorkflow()}
              aria-label={`Kill loop ${monitorLoop.id}`}
              type="button"
            >
              {stoppingObservedLoop ? "Stopping…" : "Stop loop"}
            </button>
          )}
        </div>
      </div>
      <p className="agent-workflow-hint">
        Safety limits: 100 experiments · 100 decision turns · 24 hours.
      </p>
      {datasetError && datasetError !== workflowMonitorError && (
        <p className="assistant-error" role="alert">{datasetError}</p>
      )}
      {(workflowStatus !== "idle" || workflowError || monitorLoop?.status === "RUNNING") && (
        <div className={`agent-workflow-status ${workflowStatus}`} aria-live="polite">
          <strong>
            {workflowStatus === "running" ? "Workflow running" :
              workflowStatus === "completed" ? "Workflow complete" :
                workflowStatus === "stopped" ? "Workflow stopped" :
                  workflowStatus === "failed" ? "Workflow failed" :
                    monitorLoop?.status === "RUNNING" ? `Loop ${monitorLoop.id} is running` : ""}
          </strong>
          {monitorLoop?.status === "RUNNING" && (
            <span>
              Active loop: {monitorLoop.id} · {monitorLoop.iteration}/{monitorLoop.max_iterations} experiments
              {monitorLoop.workflow_id ? ` · workflow ${monitorLoop.workflow_id}` : ""}
            </span>
          )}
          <span>{workflowPhase}</span>
          {workflowProgress.iteration > 0 && (
            <span>
              Completed experiments: {workflowProgress.iteration}
              {workflowProgress.metric !== null
                ? ` · latest validation accuracy ${(workflowProgress.metric * 100).toFixed(2)}%`
                : ""}
            </span>
          )}
          {workflowError && <span role="alert">{workflowError}</span>}
        </div>
      )}
    </section>
  );

  if (workflowOnly) {
    return (
      <section className="content agent-workflow-page">
        {workflowControls}
        <div className="workflow-bulk-actions">
          <label>
            <input
              aria-label="Select all Agent workflows"
              checked={deletableWorkflowRuns.length > 0 && deletableWorkflowRuns.every((run) => selectedWorkflowIds.has(run.id))}
              disabled={deletableWorkflowRuns.length === 0 || deletingWorkflows}
              onChange={(event) => setSelectedWorkflowIds(
                event.target.checked
                  ? new Set(deletableWorkflowRuns.map((run) => run.id))
                  : new Set(),
              )}
              type="checkbox"
            />
            Select all
          </label>
          <span>{selectedWorkflowIds.size} selected</span>
          <button
            className="remove-device-button"
            disabled={selectedWorkflowIds.size === 0 || deletingWorkflows}
            onClick={() => void deleteSelectedWorkflows()}
            type="button"
          >
            {deletingWorkflows ? "Deleting…" : "Delete selected"}
          </button>
        </div>
        {workflowDeleteError && <p className="assistant-error" role="alert">{workflowDeleteError}</p>}
        <WorkflowMonitor
          error={workflowMonitorError}
          runs={workflowRuns}
          selectedIds={selectedWorkflowIds}
          resumingWorkflowId={resumingNasWorkflowId}
          workflowBusy={workflowBusy}
          onResumeCifarNas={(id) => void resumeSelectedCifarNas(id)}
          onContinueCifar={(id) => void continueCifarWorkflowFromLatestResult(id)}
          onSelectionChange={(id, selected) => setSelectedWorkflowIds((current) => {
            const next = new Set(current);
            if (selected) next.add(id);
            else next.delete(id);
            return next;
          })}
          deleting={deletingWorkflows}
        />
      </section>
    );
  }

  return (
    <section className="content assistant-content">
      <div className="page-heading">
        <h1>SLM Assistant</h1>
        <span className="assistant-model"><span className="status-dot" /> Master connected</span>
      </div>

      <div className="assistant-panel">
        <div className="chat-workspace">
          <aside className="chat-history" aria-label="Chat history">
            <div className="chat-history-heading">
              <div>
                <span className="eyebrow">YOUR WORKSPACE</span>
                <h2>Chat history</h2>
              </div>
              <button
                type="button"
                className="new-chat-button"
                disabled={pending || workflowRunning || historyLoading || deletingConversationId !== null}
                onClick={startNewConversation}
              >
                <span>＋</span> New chat
              </button>
            </div>
            {historyError && <div className="assistant-error" role="alert">{historyError}</div>}
            <div className="chat-history-list">
              {conversations.length > 0 && (
                <div className="chat-history-bulk-actions">
                  <label>
                    <input
                      aria-label="Select all chats"
                      checked={conversations.length > 0 && conversations.every((conversation) => selectedConversationIds.has(conversation.id))}
                      disabled={pending || workflowRunning || historyLoading || deletingConversationId !== null}
                      onChange={(event) => setSelectedConversationIds(
                        event.target.checked
                          ? new Set(conversations.map((conversation) => conversation.id))
                          : new Set(),
                      )}
                      type="checkbox"
                    />
                    Select all
                  </label>
                  <span>{selectedConversationIds.size} selected</span>
                  <button
                    className="chat-history-bulk-delete"
                    disabled={pending || workflowRunning || historyLoading || deletingConversationId !== null || selectedConversationIds.size === 0}
                    onClick={() => void deleteSelectedConversations()}
                    type="button"
                  >
                    {deletingConversationId === "bulk" ? "Deleting…" : "Delete selected"}
                  </button>
                </div>
              )}
              {historyLoading && conversations.length === 0 && (
                <p className="chat-history-note" role="status">Loading saved chats…</p>
              )}
              {!historyLoading && conversations.length === 0 && !historyError && (
                <p className="chat-history-note">No saved chats.</p>
              )}
              {conversations.map((conversation) => (
                <div className="chat-history-entry" key={conversation.id}>
                  <input
                    aria-label={`Select chat: ${conversation.title}`}
                    checked={selectedConversationIds.has(conversation.id)}
                    disabled={pending || workflowRunning || historyLoading || deletingConversationId !== null}
                    onChange={(event) => setSelectedConversationIds((current) => {
                      const next = new Set(current);
                      if (event.target.checked) next.add(conversation.id);
                      else next.delete(conversation.id);
                      return next;
                    })}
                    type="checkbox"
                  />
                  <button
                    type="button"
                    className={`chat-history-item${conversation.id === selectedConversationId ? " active" : ""}`}
                    disabled={pending || workflowRunning || historyLoading || deletingConversationId !== null}
                    onClick={() => void openConversation(conversation.id)}
                    aria-current={conversation.id === selectedConversationId ? "page" : undefined}
                  >
                    <strong>{conversation.title}</strong>
                    <span>{formatConversationDate(conversation.updated_at)}</span>
                  </button>
                  <button
                    type="button"
                    className="chat-history-delete"
                    aria-label={`Delete chat: ${conversation.title}`}
                    title="Delete conversation"
                    disabled={pending || workflowRunning || historyLoading || deletingConversationId !== null}
                    onClick={() => void deleteConversation(conversation)}
                  >
                    {deletingConversationId === conversation.id ? "…" : "×"}
                  </button>
                </div>
              ))}
            </div>
          </aside>
          <div className="chat-main">
            <div className="conversation" aria-live="polite">
              {historyLoading && messages.length === 0 && (
                <div className="conversation-empty" role="status">Loading conversation…</div>
              )}
              {messages.length === 0 && !historyLoading && (
                <div className="conversation-empty">No messages yet.</div>
              )}

              {messages.map((message) => (
                <article className={`chat-message ${message.role}`} key={message.id}>
                  <div className="chat-avatar">{message.role === "assistant" ? "✦" : "H"}</div>
                  <div className="chat-message-content">
                    <span className="chat-speaker">{message.role === "assistant" ? "SLM Assistant" : "You"}</span>
                    <p>{message.text}</p>
                    {message.trace && message.trace.length > 0 && (
                      <AgentTrace trace={message.trace} />
                    )}
                  </div>
                </article>
              ))}

              {pending && (
                <div className="chat-message assistant" role="status">
                  <div className="chat-avatar">✦</div>
                  <div className="chat-message-content">
                    <span className="chat-speaker">SLM Assistant</span>
                    <p className="thinking"><span /><span /><span />
                      Thinking on dfl-master…
                    </p>
                  </div>
                </div>
              )}
              {requestError && <div className="assistant-error" role="alert">{requestError}</div>}
              <div ref={bottom} />
            </div>

            <form
              className="chat-composer"
              onSubmit={(event) => {
                event.preventDefault();
                void sendMessage(input);
              }}
            >
              <textarea
                aria-label="Message the SLM assistant"
                disabled={historyLoading || workflowRunning || pending}
                maxLength={4000}
                onChange={(event) => setInput(event.target.value)}
                onKeyDown={(event) => {
                  if (event.key === "Enter" && !event.shiftKey) {
                    event.preventDefault();
                    void sendMessage(input);
                  }
                }}
                placeholder="Message…"
                rows={2}
                value={input}
              />
              <div className="composer-footer">
                <span>Enter to send · Shift+Enter for a new line · max 4,000 characters</span>
                <button
                  className="send-button"
                  disabled={!input.trim() || pending || workflowRunning || historyLoading}
                  type="submit"
                >
                  {pending ? "Working…" : "Send"} <span>↑</span>
                </button>
              </div>
            </form>
          </div>
        </div>
      </div>
    </section>
  );
}

export function AgentWorkflowPanel({ token }: { token: string }) {
  return <AssistantPage token={token} workflowOnly />;
}

interface WorkflowActivity {
  id: string;
  workflowId: string | null;
  conversationId: string | null;
  createdAt: string;
  action: string;
  status: string;
  parameters: Record<string, unknown>;
  result: Record<string, unknown>;
  reason: string;
  decision: string;
  reply: string;
}

interface WorkflowRunView {
  id: string;
  startedAt: string;
  conversationIds: string[];
  activities: WorkflowActivity[];
  loop: AutonomyLoop | null;
  groups: ExperimentGroup[];
}

function latestSelectedCifarNasGroup(run: WorkflowRunView): ExperimentGroup | null {
  return [...run.groups].reverse().find(
    (group) => group.dataset_name === "cifar10"
      && (group.kind === "dataset_architecture_search" || group.kind === "neural_architecture_search")
      && group.status === "completed"
      && Boolean(group.selected_architecture),
  ) ?? null;
}

function latestCompletedCifarGroup(run: WorkflowRunView): ExperimentGroup | null {
  return [...run.groups].reverse().find(
    (group) => group.dataset_name === "cifar10" && group.status === "completed",
  ) ?? null;
}

function hasCifarNasRetrainingAfterLatestSearch(run: WorkflowRunView): boolean {
  const nasGroup = latestSelectedCifarNasGroup(run);
  const nasIndex = nasGroup ? run.groups.findIndex((group) => group.group_id === nasGroup.group_id) : -1;
  return nasIndex >= 0 && run.groups.slice(nasIndex + 1).some(
    (group) => group.kind === "decentralized" && group.dataset_name === "cifar10",
  );
}

function isDatasetWorkflowConversation(title: string): boolean {
  return /^(Start exactly one (?:baseline decentralized training run|new experiment)|The initial baseline has not started yet|No experiment started yet)/i.test(title);
}

function extractWorkflowDecisions(conversations: SlmConversation[]): WorkflowDecision[] {
  return conversations.flatMap((conversation) =>
    conversation.messages.flatMap((message) => {
      if (message.role !== "assistant") return [];
      const trace = message.response_data?.trace ?? [];
      return trace.map((item, index) => ({
        id: `${conversation.id}-${message.id}-${index}`,
        conversationId: conversation.id,
        workflowId: conversation.workflow_id ?? message.response_data?.workflow_id ?? null,
        createdAt: message.created_at,
        tool: item.tool,
        decision: item.decision ?? "",
        arguments: item.arguments,
        result: item.result,
        reply: message.content,
      }));
    }),
  );
}

function collectWorkflowActivities(
  history: AutonomyHistoryEntry[],
  decisions: WorkflowDecision[],
): WorkflowActivity[] {
  const workflowConversationIds = new Set(decisions.map((decision) => decision.conversationId));
  const decisionsByAction = new Map<string, WorkflowDecision[]>();
  for (const decision of decisions) {
    const key = `${decision.conversationId}:${decision.tool}`;
    decisionsByAction.set(key, [...(decisionsByAction.get(key) ?? []), decision]);
  }
  const matchedDecisions = new Set<string>();
  const activities: WorkflowActivity[] = [];

  for (const entry of [...history].sort(
    (left, right) => new Date(left.created_at).getTime() - new Date(right.created_at).getTime(),
  )) {
    if (!entry.action) continue;
    const details = entry.details ?? {};
    const evidence = recordValue(details.evidence);
    const conversationId = typeof evidence.conversation_id === "string"
      ? evidence.conversation_id
      : "";
    const workflowId = typeof evidence.workflow_id === "string" ? evidence.workflow_id : null;
    const isWorkflowSetup = evidence.source === "agent_workflow";
    if (
      !isWorkflowSetup
      && !workflowId
      && !workflowConversationIds.has(conversationId)
    ) continue;
    const decision = conversationId
      ? decisionsByAction.get(`${conversationId}:${entry.action}`)?.shift()
      : undefined;
    if (decision) matchedDecisions.add(decision.id);
    const id = `${conversationId || "workflow"}:${entry.action}:${entry.id}`;
    activities.push({
      id,
      workflowId: workflowId ?? decision?.workflowId ?? null,
      conversationId: conversationId || null,
      createdAt: entry.created_at,
      action: entry.action,
      status: entry.result_status ?? "recorded",
      parameters: recordValue(details.parameters),
      result: recordValue(details.result),
      reason: typeof details.reason === "string" ? details.reason : "",
      decision: decision?.decision ?? "",
      reply: decision?.reply ?? "",
    });
  }

  for (const decision of decisions) {
    if (matchedDecisions.has(decision.id)) continue;
    const id = `${decision.conversationId}:${decision.tool}:${decision.id}`;
    const nestedResult = recordValue(decision.result.result);
    activities.push({
      id,
      workflowId: decision.workflowId,
      conversationId: decision.conversationId,
      createdAt: decision.createdAt,
      action: decision.tool,
      status: typeof decision.result.status === "string" ? decision.result.status : "recorded",
      parameters: decision.arguments,
      result: nestedResult,
      reason: "",
      decision: decision.decision,
      reply: decision.reply,
    });
  }

  return activities.sort(
    (left, right) => new Date(left.createdAt).getTime() - new Date(right.createdAt).getTime(),
  );
}

function groupWorkflowRuns(
  activities: WorkflowActivity[],
  decisions: WorkflowDecision[],
  loops: AutonomyLoop[],
  experimentGroups: ExperimentGroup[],
): WorkflowRunView[] {
  const grouped = new Map<string, WorkflowRunView>();
  const ensureRun = (id: string, createdAt: string) => {
    let run = grouped.get(id);
    if (!run) {
      run = { id, startedAt: createdAt, conversationIds: [], activities: [], loop: null, groups: [] };
      grouped.set(id, run);
    } else if (new Date(createdAt).getTime() < new Date(run.startedAt).getTime()) {
      run.startedAt = createdAt;
    }
    return run;
  };

  for (const activity of activities) {
    const id = activity.workflowId
      ?? (activity.conversationId ? `conversation:${activity.conversationId}` : "legacy-unlinked");
    const run = ensureRun(id, activity.createdAt);
    run.activities.push(activity);
    if (activity.conversationId && !run.conversationIds.includes(activity.conversationId)) {
      run.conversationIds.push(activity.conversationId);
    }
  }
  for (const decision of decisions) {
    const id = decision.workflowId ?? `conversation:${decision.conversationId}`;
    const run = ensureRun(id, decision.createdAt);
    if (!run.conversationIds.includes(decision.conversationId)) {
      run.conversationIds.push(decision.conversationId);
    }
  }
  for (const loop of loops) {
    if (!loop.workflow_id) continue;
    const run = ensureRun(loop.workflow_id, loop.created_at);
    if (!run.loop || new Date(loop.created_at).getTime() > new Date(run.loop.created_at).getTime()) {
      run.loop = loop;
    }
  }
  for (const run of grouped.values()) {
    const groupIds = new Set(
      run.activities.flatMap((activity) => [
        ...collectExperimentIds(activity.parameters),
        ...collectExperimentIds(activity.result),
      ]),
    );
    run.groups = experimentGroups.filter((group) => groupIds.has(group.group_id));
  }

  return [...grouped.values()].sort(
    (left, right) => new Date(left.startedAt).getTime() - new Date(right.startedAt).getTime(),
  );
}

function collectExperimentIds(value: unknown): string[] {
  if (Array.isArray(value)) return value.flatMap(collectExperimentIds);
  if (typeof value !== "object" || value === null) return [];
  return Object.entries(value).flatMap(([key, item]) => {
    if ((key === "group_id" || key === "run_id") && typeof item === "string") {
      return [item];
    }
    return collectExperimentIds(item);
  });
}

function workflowRunHealth(run: WorkflowRunView): { label: string; detail: string; state: string } {
  const failures = run.activities.filter((activity) => {
    const status = activity.status.toLowerCase();
    return /failed|invalid|error/.test(status)
      || activity.result.available === false
      || typeof activity.result.error === "string";
  });
  if (failures.length > 0) {
    return {
      label: "Needs attention",
      detail: `${failures.length} recorded action${failures.length === 1 ? "" : "s"} failed or were rejected. Review the error details before continuing.`,
      state: "failed",
    };
  }
  if (run.loop?.status === "PAUSED") {
    return {
      label: "Needs attention",
      detail: run.loop.failure_reason ?? "The workflow paused before its stop criterion was reached.",
      state: "failed",
    };
  }
  const normalStopReasons = new Set([
    "target metric reached",
    "no improvement for two iterations",
    "maximum iterations reached",
  ]);
  if (
    run.loop?.status === "STOPPED"
    && run.loop.failure_reason
    && !normalStopReasons.has(run.loop.failure_reason)
    && !/^stopped by user/i.test(run.loop.failure_reason)
  ) {
    return {
      label: "Needs attention",
      detail: run.loop.failure_reason,
      state: "failed",
    };
  }
  if (run.loop?.status === "RUNNING") {
    if (run.activities.length === 0) {
      return {
        label: "Insufficient evidence",
        detail: "The loop is running, but no action results are linked to this workflow yet.",
        state: "stopped",
      };
    }
    return {
      label: "No tool errors recorded",
      detail: `${run.activities.length} action record${run.activities.length === 1 ? "" : "s"} checked; there are no recorded failed or rejected actions so far.`,
      state: "completed",
    };
  }
  if (run.loop?.failure_reason === "target metric reached") {
    return {
      label: "Target reached",
      detail: `Best recorded validation accuracy: ${run.loop.best_metric === null ? "not recorded" : `${(run.loop.best_metric * 100).toFixed(2)}%`}.`,
      state: "completed",
    };
  }
  if (run.loop?.failure_reason === "no improvement for two iterations") {
    return {
      label: "Stopped at plateau",
      detail: `The stop rule triggered after two experiments without improvement. Best recorded validation accuracy: ${run.loop.best_metric === null ? "not recorded" : `${(run.loop.best_metric * 100).toFixed(2)}%`}.`,
      state: "stopped",
    };
  }
  if (run.loop?.failure_reason === "maximum iterations reached") {
    return {
      label: "Experiment limit reached",
      detail: `${run.loop.iteration} experiments completed; best recorded validation accuracy: ${run.loop.best_metric === null ? "not recorded" : `${(run.loop.best_metric * 100).toFixed(2)}%`}.`,
      state: "stopped",
    };
  }
  if (run.loop?.status === "STOPPED") {
    return {
      label: "Stopped",
      detail: run.loop.failure_reason ?? "The workflow was stopped before a recorded success criterion.",
      state: "stopped",
    };
  }
  return {
    label: run.activities.length > 0 ? "No recorded failures" : "Insufficient evidence",
    detail: run.activities.length > 0
      ? `${run.activities.length} action record${run.activities.length === 1 ? "" : "s"} checked; no saved loop outcome is linked to this run.`
      : "This workflow has no saved action results to assess yet.",
    state: "stopped",
  };
}

function recordValue(value: unknown): Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value)
    ? value as Record<string, unknown>
    : {};
}

type WorkflowRoundMetric = {
  worker: string;
  round: number;
  accuracy?: number;
  precision?: number;
  recall?: number;
  f1?: number;
  loss?: number;
};

export function workflowLatestRoundMetrics(group: ExperimentGroup): WorkflowRoundMetric[] {
  return group.runs.flatMap((run) => {
    const latestRound = run.rounds.reduce<ExperimentGroup["runs"][number]["rounds"][number] | undefined>(
      (latest, round) => !latest || round.round > latest.round ? round : latest,
      undefined,
    );
    if (!latestRound) return [];
    return Object.entries(latestRound.workers).map(([worker, metrics]) => ({
      worker,
      round: latestRound.round,
      ...(typeof metrics.validation_accuracy === "number"
        ? { accuracy: metrics.validation_accuracy }
        : typeof metrics.accuracy === "number"
          ? { accuracy: metrics.accuracy }
          : {}),
      ...(typeof metrics.validation_macro_precision === "number"
        ? { precision: metrics.validation_macro_precision }
        : typeof metrics.macro_precision === "number"
          ? { precision: metrics.macro_precision }
          : {}),
      ...(typeof metrics.validation_macro_recall === "number"
        ? { recall: metrics.validation_macro_recall }
        : typeof metrics.macro_recall === "number"
          ? { recall: metrics.macro_recall }
          : {}),
      ...(typeof metrics.validation_macro_f1 === "number"
        ? { f1: metrics.validation_macro_f1 }
        : typeof metrics.macro_f1 === "number"
          ? { f1: metrics.macro_f1 }
          : {}),
      ...(typeof metrics.validation_loss === "number"
        ? { loss: metrics.validation_loss }
        : typeof metrics.loss === "number"
          ? { loss: metrics.loss }
          : {}),
    }));
  });
}

function formatWorkflowMetric(value: number | undefined, percent = false): string {
  if (value === undefined || !Number.isFinite(value)) return "—";
  return percent ? `${(value * 100).toFixed(2)}%` : value.toFixed(4);
}

function WorkflowExperiment({ group }: { group: ExperimentGroup }) {
  const activeRun = group.runs.find((run) => run.status === "running");
  const workerProgress = activeRun?.progress?.workers ?? {};
  const latestRoundMetrics = workflowLatestRoundMetrics(group);
  const meanMetric = (select: (metric: WorkflowRoundMetric) => number | undefined) => {
    const values = latestRoundMetrics
      .map(select)
      .filter((value): value is number => value !== undefined && Number.isFinite(value));
    return values.length > 0
      ? values.reduce((total, value) => total + value, 0) / values.length
      : undefined;
  };
  const latestRound = latestRoundMetrics.reduce(
    (latest, metric) => Math.max(latest, metric.round),
    0,
  );
  const completedRounds = group.runs.reduce((total, run) => total + run.rounds.length, 0);
  const expectedRounds = group.rounds === undefined
    ? null
    : group.rounds * Math.max(group.seeds?.length ?? group.runs.length, 1);

  return (
    <article className="workflow-monitor__experiment">
      <header>
        <strong>{group.kind.replace(/_/g, " ")}</strong>
        <span className={`agent-run-status ${group.status}`}>{group.status}</span>
      </header>
      <div className="workflow-monitor__metrics">
        <div><span>Experiment ID</span><strong>{group.group_id}</strong></div>
        <div><span>Dataset / split</span><strong>{group.dataset_name ?? "Not recorded"} / {group.dataset_split ?? "Not recorded"}</strong></div>
        <div><span>Topology</span><strong>{group.topology.replace(/_/g, " ")}</strong></div>
        <div><span>Configured rounds / local epochs</span><strong>{group.rounds ?? "Search"} rounds / {group.epochs ?? "Search"} epochs</strong></div>
        <div><span>Seed runs</span><strong>{group.seeds?.join(", ") ?? (group.runs.map((run) => run.seed).join(", ") || "Not reported")}</strong></div>
        {group.rounds !== undefined && (
          <div><span>Rounds completed</span><strong>{completedRounds}{expectedRounds === null ? "" : ` / ${expectedRounds}`}</strong></div>
        )}
        {latestRoundMetrics.length > 0 && (
          <>
            <div><span>Latest metrics · round {latestRound}</span><strong>{latestRoundMetrics.length} worker results</strong></div>
            <div><span>Mean validation accuracy</span><strong>{formatWorkflowMetric(meanMetric((metric) => metric.accuracy), true)}</strong></div>
            <div><span>Mean validation precision</span><strong>{formatWorkflowMetric(meanMetric((metric) => metric.precision), true)}</strong></div>
            <div><span>Mean validation recall</span><strong>{formatWorkflowMetric(meanMetric((metric) => metric.recall), true)}</strong></div>
            <div><span>Mean validation F1</span><strong>{formatWorkflowMetric(meanMetric((metric) => metric.f1), true)}</strong></div>
            <div><span>Mean validation loss</span><strong>{formatWorkflowMetric(meanMetric((metric) => metric.loss))}</strong></div>
          </>
        )}
      </div>
      {group.selected_configuration && (
        <p className="workflow-monitor__next">
          <strong>Selected topology:</strong> {group.selected_configuration.topology},
          alpha {group.selected_configuration.alpha},
          split {group.selected_configuration.dataset_split}.
        </p>
      )}
      {group.selected_architecture && (
        <details className="workflow-monitor__configuration">
          <summary>Selected NAS model for retraining</summary>
          <pre>{JSON.stringify(group.selected_architecture, null, 2)}</pre>
        </details>
      )}
      {group.worker_search_progress && Object.keys(group.worker_search_progress).length > 0 && (
        <div className="workflow-monitor__workers">
          <h4>NAS worker progress</h4>
          <div className="table-scroll">
            <table>
              <thead><tr><th>Worker</th><th>Status</th><th>Generation</th><th>Epoch fidelity</th><th>Candidates</th></tr></thead>
              <tbody>
                {Object.entries(group.worker_search_progress).map(([worker, progress]) => (
                  <tr key={worker}>
                    <td>{worker}</td>
                    <td>{progress.status}</td>
                    <td>{progress.generation ?? "—"}</td>
                    <td>{progress.fidelity_epochs ?? "—"}</td>
                    <td>{progress.candidate_count ?? "—"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
      {group.nas_candidates && group.nas_candidates.length > 0 && (
        <details className="workflow-monitor__configuration">
          <summary>NAS candidate results · {group.nas_candidates.length}</summary>
          <ol>
            {[...group.nas_candidates]
              .sort((left, right) => right.mean_validation_accuracy - left.mean_validation_accuracy)
              .slice(0, 5)
              .map((candidate) => (
                <li key={candidate.architecture.id}>
                  {candidate.architecture.id}: {(candidate.mean_validation_accuracy * 100).toFixed(2)}% validation accuracy
                  {candidate.mean_validation_macro_f1 === undefined
                    ? ""
                    : ` · ${(candidate.mean_validation_macro_f1 * 100).toFixed(2)}% macro-F1`}
                  {candidate.fidelity_epochs === undefined ? "" : ` · ${candidate.fidelity_epochs} epochs`}
                </li>
              ))}
          </ol>
        </details>
      )}
      {group.candidates && group.candidates.length > 0 && (
        <details className="workflow-monitor__configuration">
          <summary>Topology candidates · {group.candidates.length}</summary>
          <ol>
            {group.candidates.map((candidate) => (
              <li key={`${candidate.topology}-${candidate.alpha}`}>
                {candidate.topology} · alpha {candidate.alpha}
                {candidate.mean_final_accuracy === undefined
                  ? ""
                  : ` · ${(candidate.mean_final_accuracy * 100).toFixed(2)}% final accuracy`}
                {candidate.mean_reward === undefined
                  ? ""
                  : ` · reward ${candidate.mean_reward.toFixed(4)}`}
              </li>
            ))}
          </ol>
        </details>
      )}
      {group.architecture && (
        <details className="workflow-monitor__configuration">
          <summary>Model configuration</summary>
          <pre>{JSON.stringify(group.architecture, null, 2)}</pre>
        </details>
      )}
      {Object.entries(workerProgress).length > 0 && (
        <div className="workflow-monitor__workers">
          <h4>Worker round / epoch progress</h4>
          <div className="table-scroll">
            <table>
              <thead><tr><th>Worker</th><th>Round</th><th>Local epoch</th><th>Phase</th></tr></thead>
              <tbody>
                {Object.entries(workerProgress).map(([worker, progress]) => (
                  <tr key={worker}>
                    <td>{worker}</td>
                    <td>{activeRun?.progress?.round ?? "—"} / {group.rounds ?? "—"}</td>
                    <td>{progress.epoch ?? "—"} / {progress.epochs ?? group.epochs ?? "—"}</td>
                    <td>{progress.phase?.replace(/_/g, " ") ?? "waiting"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
      {latestRoundMetrics.length > 0 && (
        <div className="workflow-monitor__workers">
          <h4>Latest per-worker validation metrics · round {latestRound}</h4>
          <div className="table-scroll">
            <table>
              <thead>
                <tr>
                  <th>Worker</th>
                  <th>Round</th>
                  <th>Accuracy</th>
                  <th>Precision</th>
                  <th>Recall</th>
                  <th>F1</th>
                  <th>Loss</th>
                </tr>
              </thead>
              <tbody>
                {latestRoundMetrics.map((metric, index) => (
                  <tr key={`${metric.worker}-${metric.round}-${index}`}>
                    <td>{metric.worker}</td>
                    <td>{metric.round} / {group.rounds ?? "—"}</td>
                    <td>{formatWorkflowMetric(metric.accuracy, true)}</td>
                    <td>{formatWorkflowMetric(metric.precision, true)}</td>
                    <td>{formatWorkflowMetric(metric.recall, true)}</td>
                    <td>{formatWorkflowMetric(metric.f1, true)}</td>
                    <td>{formatWorkflowMetric(metric.loss)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
      {group.error && <p className="assistant-error" role="alert">{group.error}</p>}
    </article>
  );
}

function WorkflowMonitor({
  deleting,
  error,
  onContinueCifar,
  onResumeCifarNas,
  onSelectionChange,
  resumingWorkflowId,
  runs: workflowRuns,
  selectedIds,
  workflowBusy,
}: {
  error: string;
  runs: WorkflowRunView[];
  selectedIds: Set<string>;
  onSelectionChange: (id: string, selected: boolean) => void;
  deleting: boolean;
  onContinueCifar: (workflowId: string) => void;
  onResumeCifarNas: (workflowId: string) => void;
  resumingWorkflowId: string | null;
  workflowBusy: boolean;
}) {
  return (
    <section className="workflow-monitor" aria-label="Workflow details and activity">
      <header className="workflow-monitor__header">
        <div>
          <span className="eyebrow">DATABASE-BACKED MONITOR</span>
          <h2>Agent workflows · {workflowRuns.length}</h2>
        </div>
      </header>
      {error && <p className="assistant-error" role="alert">Could not refresh saved workflow details: {error}</p>}
      {workflowRuns.length === 0 ? (
        <p className="workflow-monitor__empty">
          {error
            ? "Coordinator unavailable."
            : "No saved workflows."}
        </p>
      ) : (
        <div className="workflow-monitor__sessions">
          {[...workflowRuns].reverse().map((run) => {
            const health = workflowRunHealth(run);
            const resumableNasGroup = latestSelectedCifarNasGroup(run);
            const latestCifarGroup = latestCompletedCifarGroup(run);
            const shortContinuationLoop = run.loop?.failure_reason === "maximum iterations reached"
              && run.loop.max_iterations === 1;
            const runTargetAccuracy = run.loop?.target_metric !== null && run.loop?.target_metric !== undefined
              ? run.loop.target_metric * 100
              : 80;
            const canResumeNas = Boolean(
              resumableNasGroup
              && run.loop?.status !== "RUNNING"
              && !run.groups.some((group) => group.status === "running")
              && !hasCifarNasRetrainingAfterLatestSearch(run),
            );
            const canContinueCifar = Boolean(
              latestCifarGroup
              && latestCifarGroup.kind === "decentralized"
              && cifarWorkflowNeedsNas(latestCifarGroup, runTargetAccuracy)
              && run.loop?.status !== "RUNNING"
              && !run.groups.some((group) => group.status === "running")
              && (run.loop?.failure_reason !== "maximum iterations reached" || shortContinuationLoop),
            );
            const canDelete = (run.conversationIds.length > 0 || !run.id.startsWith("conversation:"))
              && run.loop?.status !== "RUNNING"
              && !run.groups.some((group) => group.status === "running");
            const rule = run.loop?.stop_rule === "target_metric" && run.loop.target_metric !== null
              ? `Validation accuracy ≥ ${(run.loop.target_metric * 100).toFixed(1)}%`
              : run.loop?.stop_rule === "no_improvement_2"
                ? "Stop after 2 experiments without improvement"
                : run.loop
                  ? `${run.loop.max_iterations} experiment limit`
                  : "No saved stop criterion";
            return (
              <div className="workflow-monitor__run-row" key={run.id}>
                <input
                  aria-label={`Select Agent workflow ${run.id}`}
                  checked={selectedIds.has(run.id)}
                  disabled={!canDelete || deleting}
                  onChange={(event) => onSelectionChange(run.id, event.target.checked)}
                  type="checkbox"
                />
                <details className="workflow-monitor__run">
                <summary>
                  <strong>Agent ID: {run.id}</strong>
                  <span className={`agent-run-status ${run.loop?.status.toLowerCase() ?? health.state}`}>
                    {run.loop?.status.toLowerCase() ?? health.label}
                  </span>
                  <span>{run.groups.length} experiments · {run.activities.length} actions</span>
                  <time>{formatWorkflowDate(run.startedAt)}</time>
                </summary>
                <div className="workflow-monitor__run-content">
                  <p className="workflow-monitor__run-health">
                    <strong>Evidence-based check:</strong>{" "}
                    <span className={`agent-run-status ${health.state}`}>{health.label}</span> {health.detail}
                  </p>
                  {canResumeNas && (
                    <button
                      className="send-button"
                      disabled={workflowBusy || resumingWorkflowId !== null || deleting}
                      onClick={() => onResumeCifarNas(run.id)}
                      type="button"
                    >
                      {resumingWorkflowId === run.id
                        ? "Resuming selected model…"
                        : "Resume with selected CIFAR-10 NAS model"}
                    </button>
                  )}
                  {canContinueCifar && (
                    <button
                      className="send-button"
                      disabled={workflowBusy || resumingWorkflowId !== null || deleting}
                      onClick={() => onContinueCifar(run.id)}
                      type="button"
                    >
                      {resumingWorkflowId === run.id
                        ? "Continuing CIFAR-10 Agent…"
                        : "Continue CIFAR-10 Agent from latest result"}
                    </button>
                  )}
                  <section className="workflow-monitor__section" aria-label={`Agent ${run.id} loop progress`}>
                    <h3>Loop plan & progress</h3>
                    {run.loop ? (
                      <>
                        <div className="workflow-monitor__metrics">
                          <div><span>Workflow ID</span><strong>{run.id}</strong></div>
                          <div><span>Loop ID</span><strong>{run.loop.id}</strong></div>
                          <div><span>Experiments completed</span><strong>{run.loop.iteration} / {run.loop.max_iterations}</strong></div>
                          <div><span>Termination criterion</span><strong>{rule}</strong></div>
                          <div><span>Best validation accuracy</span><strong>{run.loop.best_metric === null ? "Not recorded" : `${(run.loop.best_metric * 100).toFixed(2)}%`}</strong></div>
                          <div><span>Workflow deadline</span><strong>{formatWorkflowDate(run.loop.deadline_at)}</strong></div>
                        </div>
                        {run.loop.failure_reason && (
                          <p className="workflow-monitor__next"><strong>Loop outcome:</strong> {run.loop.failure_reason}</p>
                        )}
                        {run.loop.status === "RUNNING" && (
                          <p className="workflow-monitor__next">
                            <strong>Currently running:</strong> loop {run.loop.id}.
                            {run.groups.find((group) => group.status === "running")
                              ? ` Experiment ${run.groups.find((group) => group.status === "running")?.group_id} is active.`
                              : " Waiting for this Agent to submit its next experiment."}
                          </p>
                        )}
                      </>
                    ) : (
                      <p className="workflow-monitor__empty">No linked loop.</p>
                    )}
                  </section>
                  <section className="workflow-monitor__section" aria-label={`Agent ${run.id} experiments`}>
                    <h3>Experiments · {run.groups.length}</h3>
                    {run.groups.length === 0 ? (
                      <p className="workflow-monitor__empty">No linked experiment results.</p>
                    ) : (
                      <div className="workflow-monitor__experiments">
                        {run.groups.map((group) => <WorkflowExperiment group={group} key={group.group_id} />)}
                      </div>
                    )}
                  </section>
                  <section className="workflow-monitor__section workflow-monitor__activity" aria-label={`Agent ${run.id} actions`}>
                    <h3>Actions · {run.activities.length}</h3>
                    {run.activities.length === 0 ? (
                      <p className="workflow-monitor__empty">No linked action records.</p>
                    ) : (
                      <ol>
                        {[...run.activities].reverse().map((activity, index) => (
                          <li key={activity.id}>
                            <span className="workflow-monitor__index">{run.activities.length - index}</span>
                            <div>
                              <header>
                                <strong>{activity.action.replace(/_/g, " ")}</strong>
                                <span className={`agent-run-status ${activity.status === "executed" ? "completed" : activity.status}`}>
                                  {activity.status.replace(/_/g, " ")}
                                </span>
                                <time>{formatWorkflowDate(activity.createdAt)}</time>
                              </header>
                              {activity.decision ? (
                                <p><strong>Why the Agent selected it:</strong> {activity.decision}</p>
                              ) : activity.reason ? (
                                <p><strong>Recorded workflow instruction:</strong> {activity.reason}</p>
                              ) : (
                                <p className="workflow-monitor__missing-reason">
                                  No decision summary recorded.
                                </p>
                              )}
                              <details>
                                <summary>Action parameters and result</summary>
                                <pre>{JSON.stringify({ parameters: activity.parameters, result: activity.result }, null, 2)}</pre>
                              </details>
                              {activity.reply && activity.reply !== activity.decision && (
                                <details>
                                  <summary>Agent response</summary>
                                  <p>{activity.reply}</p>
                                </details>
                              )}
                            </div>
                          </li>
                        ))}
                      </ol>
                    )}
                  </section>
                </div>
                </details>
              </div>
            );
          })}
        </div>
      )}
    </section>
  );
}

function formatWorkflowDate(value: string): string {
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? "time not recorded" : date.toLocaleString();
}

function restoreMessages(storedMessages: SlmConversationMessage[]): Message[] {
  return storedMessages.map((message) => ({
    id: message.id,
    role: message.role,
    text: message.content,
    trace: message.response_data?.trace,
  }));
}

function AgentTrace({
  trace,
}: {
  trace: NonNullable<SlmConversationMessage["response_data"]>["trace"];
}) {
  if (!trace?.length) return null;
  return (
    <details className="agent-trace">
      <summary>Agent tools · {trace.length}</summary>
      <ul>
        {trace.map((item, index) => {
          const error = item.result.error;
          const status = item.result.status;
          return (
            <li key={`${item.tool}-${index}`}>
              <strong>{item.tool}</strong>
              <span>{typeof error === "string" ? error : typeof status === "string" ? status : "completed"}</span>
            </li>
          );
        })}
      </ul>
    </details>
  );
}

function formatConversationDate(value: string): string {
  const date = new Date(value);
  return Number.isNaN(date.getTime())
    ? "Saved conversation"
    : date.toLocaleDateString(undefined, { month: "short", day: "numeric" });
}

function errorText(cause: unknown): string {
  return cause instanceof Error ? cause.message : "The coordinator request failed.";
}

function datasetLabel(name: string): string {
  return name === "cifar10" ? "CIFAR-10" : "MNIST";
}

export function agentWorkflowPrompt(
  datasetName: string,
  completedIterations: number,
  stopRule: "target_metric" | "no_improvement_2" | "max_iterations",
  targetAccuracy: number,
  maxIterations: number,
  previousGroup: ExperimentGroup | null,
): string {
  return [
    `Operate an Agent workflow for dataset ${datasetName}.`,
    `Objective: ${stopRule === "target_metric"
      ? "reach validation accuracy of at least " + targetAccuracy + "%."
      : stopRule === "no_improvement_2"
        ? "stop when two completed experiments in succession fail to improve the best validation accuracy."
        : "complete no more than " + maxIterations + " experiments."}`,
    `Completed experiments so far: ${completedIterations}.`,
    previousGroup
      ? `Most recently observed experiment: ${JSON.stringify({
        group_id: previousGroup.group_id,
        kind: previousGroup.kind,
        status: previousGroup.status,
        accuracy: experimentValidationAccuracy(previousGroup),
        error: previousGroup.error,
        selected_configuration: previousGroup.selected_configuration,
        selected_architecture: previousGroup.selected_architecture,
      })}.`
      : "No new workflow experiment has completed yet.",
    "The selected dataset and initial baseline are fixed workflow setup and have already been handled. Treat that baseline as evidence, not as the next action to repeat. Review the current system snapshot and conversation tool results, then decide the next useful action yourself from the available tools; follow-up training, topology optimization, architecture search, monitoring, and stopping are not prescribed in advance.",
    "Call one tool at a time, explain your choice in a concise evidence-based rationale, and use only values permitted by its schema. If a tool fails or a safety rule rejects it, use the returned error to choose an alternative; do not claim a rejected action succeeded.",
    "After observing results, decide whether to take another action or call finish_agent_workflow with an evidence-backed outcome. Use the termination tool when the objective is met, no worthwhile action remains, or a blocker prevents progress. Safety limits are enforced separately.",
  ].join(" ");
}

export function persistedLoopFailureReason(message: string): string {
  return `Workflow failed: ${message}`.slice(0, 1000);
}

export function cifarWorkflowNeedsNas(
  previousGroup: ExperimentGroup,
  targetAccuracy: number,
): boolean {
  const validationAccuracy = experimentValidationAccuracy(previousGroup);
  return previousGroup.kind === "decentralized"
    && (validationAccuracy === null || validationAccuracy < targetAccuracy / 100);
}

function experimentValidationAccuracy(group: ExperimentGroup): number | null {
  const candidateAccuracies = [
    ...(group.nas_candidates ?? []).map((candidate) => candidate.mean_validation_accuracy),
    ...(group.candidates ?? [])
      .map((candidate) => candidate.mean_final_accuracy)
      .filter((value): value is number => typeof value === "number"),
  ].filter((value) => Number.isFinite(value) && value >= 0 && value <= 1);
  if (candidateAccuracies.length > 0) return Math.max(...candidateAccuracies);

  const accuracies: number[] = [];
  for (const run of group.runs) {
    const finalRound = run.rounds[run.rounds.length - 1];
    if (!finalRound) continue;
    for (const worker of Object.values(finalRound.workers)) {
      const accuracy = worker.validation_accuracy;
      if (typeof accuracy === "number" && Number.isFinite(accuracy) && accuracy >= 0 && accuracy <= 1) {
        accuracies.push(accuracy);
      }
    }
  }
  return accuracies.length > 0
    ? accuracies.reduce((sum, accuracy) => sum + accuracy, 0) / accuracies.length
    : null;
}

function ensureWorkflowNotStopped(stopRef: { current: boolean }): void {
  if (stopRef.current) throw new Error("Agent workflow stopped by user.");
}

function waitFor(milliseconds: number): Promise<void> {
  return new Promise((resolve) => window.setTimeout(resolve, milliseconds));
}
