import { cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import AssistantPage, {
  AgentWorkflowPanel,
  agentWorkflowPrompt,
  persistedLoopFailureReason,
  workflowLatestRoundMetrics,
} from "./AssistantPage";
import type { ExperimentGroup } from "../api/client";

const masterUrl = "http://192.168.11.15:5003";

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

describe("SLM Assistant", () => {
  it("bounds persisted loop failure reasons to the coordinator limit", () => {
    const reason = persistedLoopFailureReason("x".repeat(1200));

    expect(reason).toHaveLength(1000);
    expect(reason).toMatch(/^Workflow failed: /);
  });

  it("leaves below-target CIFAR-10 tool selection to the SLM", () => {
    const prompt = agentWorkflowPrompt("cifar10", 1, "target_metric", 80, 100, null);

    expect(prompt).toContain("decide the next useful action yourself from the available tools");
    expect(prompt).toContain("not prescribed in advance");
    expect(prompt).toContain("call finish_agent_workflow");
    expect(prompt).not.toContain("enable_nas");
  });

  it("shows a completed NAS result without prescribing retraining", () => {
    const selectedArchitecture = {
      id: "cifar-resume",
      blocks: 2,
      channels: 32,
      kernel_size: 3,
    };
    const nasGroup: ExperimentGroup = {
      group_id: "cifar-nas-resume",
      kind: "dataset_architecture_search",
      dataset_name: "cifar10",
      topology: "directed_ring",
      status: "completed",
      error: null,
      runs: [],
      selected_architecture: selectedArchitecture,
    };
    const prompt = agentWorkflowPrompt("cifar10", 1, "target_metric", 80, 100, nasGroup);

    expect(prompt).toContain(JSON.stringify(selectedArchitecture));
    expect(prompt).toContain("decide the next useful action yourself");
    expect(prompt).not.toContain("start_training");
  });

  it("summarizes each worker's metrics from the latest available round", () => {
    const group: ExperimentGroup = {
      group_id: "mnist-metrics",
      kind: "decentralized",
      topology: "directed_ring",
      rounds: 5,
      status: "running",
      error: null,
      runs: [{
        run_id: "mnist-metrics-run",
        seed: 2026,
        topology: "directed_ring",
        status: "running",
        rounds: [{
          round: 1,
          topology: "directed_ring",
          workers: {
            worker1: {
              node_id: "worker1",
              run_id: "mnist-metrics-run",
              round: 1,
              topology: "directed_ring",
              validation_accuracy: 0.8,
              peak_process_rss_mb: 0,
            },
          },
        }, {
          round: 2,
          topology: "directed_ring",
          workers: {
            worker1: {
              node_id: "worker1",
              run_id: "mnist-metrics-run",
              round: 2,
              topology: "directed_ring",
              validation_accuracy: 0.92,
              validation_macro_precision: 0.93,
              validation_macro_recall: 0.91,
              validation_macro_f1: 0.92,
              validation_loss: 0.2,
              peak_process_rss_mb: 0,
            },
          },
        }],
      }],
    };

    expect(workflowLatestRoundMetrics(group)).toEqual([{
      worker: "worker1",
      round: 2,
      accuracy: 0.92,
      precision: 0.93,
      recall: 0.91,
      f1: 0.92,
      loss: 0.2,
    }]);
  });

  it("loads saved master conversations as plain chat without action cards", async () => {
    vi.stubGlobal("fetch", vi.fn().mockImplementation((input: RequestInfo | URL) => {
      const url = String(input);
      if (url === `${masterUrl}/rag/conversations`) {
        return Promise.resolve({
          ok: true,
          json: async () => [{
            id: "conversation-saved",
            title: "What should I do next?",
            updated_at: "2026-11-03T04:00:00Z",
            message_count: 2,
          }],
        });
      }
      return Promise.resolve({
        ok: true,
        json: async () => ({
          id: "conversation-saved",
          title: "What should I do next?",
          created_at: "2026-11-03T04:00:00Z",
          updated_at: "2026-11-03T04:00:00Z",
          messages: [
            {
              id: 1,
              role: "user",
              content: "What should I do next?",
              created_at: "2026-11-03T04:00:00Z",
            },
            {
              id: 2,
              role: "assistant",
              content: "Create an initial model, then verify its distribution to all five workers.",
              created_at: "2026-11-03T04:00:02Z",
            },
          ],
        }),
      });
    }));
    render(<AssistantPage token="dtn-token" />);

    expect(await screen.findByRole("button", {
      name: "Delete chat: What should I do next?",
    })).toBeInTheDocument();
    expect(await screen.findByText(
      "Create an initial model, then verify its distribution to all five workers.",
    )).toBeInTheDocument();
    expect(screen.queryByText("Validated action")).not.toBeInTheDocument();
    expect(screen.queryByText("Action JSON")).not.toBeInTheDocument();
    expect(screen.queryByText("Technical details")).not.toBeInTheDocument();
  });

  it("sends the original query directly to the master RAG endpoint", async () => {
    const fetchMock = vi.fn().mockImplementation((input: RequestInfo | URL) => {
      const url = String(input);
      if (url === `${masterUrl}/rag/conversations`) {
        return Promise.resolve({ ok: true, json: async () => [] });
      }
      return Promise.resolve({
        ok: true,
        json: async () => ({
          conversation_id: "conversation-1",
          reply: "The preprocessing result shows five workers received the dataset. Create the initial model next.",
        }),
      });
    });
    vi.stubGlobal("fetch", fetchMock);
    render(<AssistantPage token="dtn-token" />);
    expect(await screen.findByText("No saved chats.")).toBeInTheDocument();

    fireEvent.change(screen.getByLabelText("Message the SLM assistant"), {
      target: { value: "The dataset was shipped to all 5 workers. What should I do next?" },
    });
    fireEvent.click(screen.getByRole("button", { name: /send/i }));

    expect(await screen.findByText(/Create the initial model next/)).toBeInTheDocument();
    const chatCall = fetchMock.mock.calls.find(([url]) => String(url) === `${masterUrl}/rag/chat`);
    expect(chatCall).toBeDefined();
    const request = chatCall?.[1] as RequestInit;
    expect(request.method).toBe("POST");
    expect(JSON.parse(String(request.body))).toEqual({
      message: "The dataset was shipped to all 5 workers. What should I do next?",
    });
    expect(fetchMock.mock.calls.some(([url]) => String(url).includes(":8000/api/slm/"))).toBe(false);
    expect(screen.queryByText("Validated action")).not.toBeInTheDocument();
    expect(screen.queryByText("Action JSON")).not.toBeInTheDocument();
  });

  it("keeps agent controls out of the SLM Assistant", async () => {
    const fetchMock = vi.fn().mockImplementation((input: RequestInfo | URL) => {
      const url = String(input);
      if (url === `${masterUrl}/rag/conversations`) {
        return Promise.resolve({ ok: true, json: async () => [] });
      }
      return Promise.resolve({ ok: true, json: async () => ({}) });
    });
    vi.stubGlobal("fetch", fetchMock);
    render(<AssistantPage token="dtn-token" />);
    expect(await screen.findByText("No saved chats.")).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Agent mode" })).not.toBeInTheDocument();
    expect(screen.queryByText("Ask questions about your cluster history. Choose and start workflows from Agent Runs.")).not.toBeInTheDocument();
    expect(screen.getByText("No messages yet.")).toBeInTheDocument();
  });

  it("runs a fixed dataset/model/baseline bootstrap before the SLM chooses follow-up actions", async () => {
    const calls: Array<{ url: string; init?: RequestInit }> = [];
    let trainingDispatched = false;
    let agentCalls = 0;
    const group = {
      group_id: "agent-run-1",
      dataset_name: "mnist",
      kind: "decentralized",
      topology: "ring",
      status: "completed",
      error: null,
      runs: [{
        run_id: "mnist-run-1",
        seed: 42,
        topology: "ring",
        status: "completed",
        rounds: [{
          round: 1,
          topology: "ring",
          workers: {
            worker1: { validation_accuracy: 0.96 },
            worker2: { validation_accuracy: 0.95 },
          },
        }],
      }],
    };
    vi.stubGlobal("fetch", vi.fn().mockImplementation((input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      calls.push({ url, init });
      if (url === `${masterUrl}/rag/conversations`) {
        return Promise.resolve({ ok: true, json: async () => [] });
      }
      if (url.endsWith("/api/datasets/mnist/initial-model")) {
        return Promise.resolve({
          ok: false,
          status: 404,
          json: async () => ({ detail: "Initial model not found" }),
        });
      }
      if (url.endsWith("/api/datasets/mnist/preprocessing-report")) {
        return Promise.resolve({
          ok: false,
          status: 404,
          json: async () => ({ detail: "Preprocessing report not found" }),
        });
      }
      if (url.endsWith("/api/datasets")) {
        return Promise.resolve({
          ok: true,
          json: async () => [{
            dataset_name: "mnist",
            classes: ["0", "1"],
            class_counts: { "0": 2, "1": 2 },
            image_sizes: { "0": { "28x28": 2 }, "1": { "28x28": 2 } },
            image_count: 4,
            expanded_image_bytes: 0,
          }],
        });
      }
      if (url.endsWith("/api/autonomy/state")) {
        return Promise.resolve({ ok: true, json: async () => ({ stop_requested: false, loops: [] }) });
      }
      if (url.includes("/api/autonomy/history")) {
        return Promise.resolve({ ok: true, json: async () => [] });
      }
      if (url.endsWith("/api/slm/conversations")) {
        return Promise.resolve({ ok: true, json: async () => [] });
      }
      if (url.endsWith("/api/slm/results")) {
        return Promise.resolve({
          ok: true,
          json: async () => ({
            configuration: { rounds: 1, epochs: 1, topology: "ring", dataset_split: "stratified" },
            active_group_id: null,
            topologies: {},
            groups: trainingDispatched ? [group] : [],
          }),
        });
      }
      if (url.endsWith("/api/autonomy/dispatch")) {
        const body = JSON.parse(String(init?.body)) as {
          action: { action: string; parameters?: Record<string, unknown> };
        };
        if (body.action.action === "start_training") trainingDispatched = true;
        return Promise.resolve({
          ok: true,
          json: async () => ({
            status: "executed",
            result: {
              available: true,
              data: body.action.action === "preprocess_dataset"
                ? { uploaded_workers: {} }
                : body.action.action === "start_training"
                  ? { group_id: group.group_id }
                  : {},
            },
          }),
        });
      }
      if (url.endsWith("/api/autonomy/loops") && init?.method === "POST") {
        return Promise.resolve({
          ok: true,
          json: async () => ({
            id: "loop-1",
            status: "RUNNING",
            iteration: 0,
            max_iterations: 100,
            stop_rule: "target_metric",
            target_metric: null,
            best_metric: null,
            failure_reason: null,
          }),
        });
      }
      if (url.endsWith("/api/slm/agent/chat")) {
        agentCalls += 1;
        return Promise.resolve({
          ok: true,
          json: async () => ({
            conversation_id: "workflow-conversation",
            reply: "The result met the objective; I am ending the workflow.",
            tool_calls: 1,
            terminated: true,
            trace: [{
              tool: "finish_agent_workflow",
              decision: "The observed validation accuracy meets the selected target.",
              arguments: {
                outcome: "objective_achieved",
                summary: "The objective was achieved.",
                evidence: "The completed experiment has 95.5% validation accuracy.",
              },
              result: {
                status: "workflow_complete",
                outcome: "objective_achieved",
                summary: "The objective was achieved.",
                evidence: "The completed experiment has 95.5% validation accuracy.",
              },
            }],
          }),
        });
      }
      if (url.endsWith("/api/autonomy/loops/loop-1/iterations")) {
        return Promise.resolve({
          ok: true,
          json: async () => ({
            id: "loop-1",
            status: "RUNNING",
            iteration: 1,
            max_iterations: 100,
            stop_rule: "target_metric",
            target_metric: null,
            best_metric: 0.955,
            failure_reason: null,
          }),
        });
      }
      return Promise.resolve({ ok: true, json: async () => ({}) });
    }));

    render(<AgentWorkflowPanel token="dtn-token" />);
    const startButton = await screen.findByRole("button", { name: "Start Agent" });
    await waitFor(() => expect(startButton).toBeEnabled());
    fireEvent.click(startButton);

    expect(await screen.findByText(/SLM selected termination \(objective_achieved\)/)).toBeInTheDocument();
    const requests = calls.map(({ url, init }) => ({
      path: new URL(url).pathname,
      body: init?.body ? JSON.parse(String(init.body)) : undefined,
    }));
    const loopStart = requests.find((request) => request.path === "/api/autonomy/loops");
    const iteration = requests.find((request) => request.path === "/api/autonomy/loops/loop-1/iterations");
    const agentRequests = requests.filter((request) => request.path === "/api/slm/agent/chat");
    const loopStartIndex = requests.findIndex((request) => request.path === "/api/autonomy/loops");
    const agentIndex = requests.findIndex((request) => request.path === "/api/slm/agent/chat");
    const bootstrapActions = requests
      .filter((request) => request.path === "/api/autonomy/dispatch")
      .map((request) => request.body?.action?.action);
    expect(loopStartIndex).toBeLessThan(agentIndex);
    expect(bootstrapActions).toEqual(["preprocess_dataset", "create_initial_model", "start_training"]);
    expect(agentRequests).toHaveLength(1);
    expect(agentCalls).toBe(1);
    expect(agentRequests[0].body).toMatchObject({ workflow_dataset: "mnist" });
    expect(agentRequests[0].body.message).toContain("Completed experiments so far: 1.");
    expect(agentRequests[0].body.message).toContain("initial baseline are fixed workflow setup");
    expect(loopStart?.body).toMatchObject({ stop_rule: "max_iterations", max_iterations: 100 });
    expect(loopStart?.body.workflow_id).toMatch(
      /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i,
    );
    expect(iteration?.body).toEqual({ metric: 0.955 });
  });

  it("does not prescribe automatic CIFAR recovery after a failed experiment", () => {
    const failedNas: ExperimentGroup = {
      group_id: "cifar-nas-failed",
      kind: "dataset_architecture_search",
      dataset_name: "cifar10",
      topology: "candidate_sweep",
      status: "failed",
      error: "The worker does not have enough available memory.",
      runs: [],
    };
    const prompt = agentWorkflowPrompt("cifar10", 1, "target_metric", 80, 100, failedNas);

    expect(prompt).toContain("decide the next useful action yourself");
    expect(prompt).not.toContain("reboot");
    expect(prompt).not.toContain("retry");
    expect(prompt).not.toContain("start_training");
  });

  it("shows a loop returned by a start conflict instead of reporting the workflow as failed", async () => {
    let anotherLoopIsRunning = false;
    const activeLoop = {
      id: "already-running-loop",
      workflow_id: "55555555-5555-4555-8555-555555555555",
      status: "RUNNING",
      iteration: 1,
      max_iterations: 10,
      time_limit_seconds: 3600,
      stop_rule: "target_metric",
      target_metric: 0.8,
      best_metric: 0.61,
      no_improvement_count: 0,
      failure_reason: null,
      created_at: "2026-11-09T04:00:00Z",
      deadline_at: "2026-11-09T05:00:00Z",
      updated_at: "2026-11-09T04:10:00Z",
    };
    const fetchMock = vi.fn().mockImplementation((input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      if (url.endsWith("/api/datasets")) {
        return Promise.resolve({
          ok: true,
          json: async () => [{
            dataset_name: "mnist",
            classes: ["0", "1"],
            class_counts: { "0": 1, "1": 1 },
            image_sizes: { "0": { "28x28": 1 }, "1": { "28x28": 1 } },
            image_count: 2,
            expanded_image_bytes: 0,
          }],
        });
      }
      if (url.endsWith("/api/autonomy/state")) {
        return Promise.resolve({
          ok: true,
          json: async () => ({
            stop_requested: false,
            loops: anotherLoopIsRunning ? [activeLoop] : [],
          }),
        });
      }
      if (url.endsWith("/api/slm/results")) {
        return Promise.resolve({
          ok: true,
          json: async () => ({
            configuration: { rounds: 1, epochs: 1, topology: "ring", dataset_split: "stratified" },
            active_group_id: null,
            topologies: {},
            groups: [],
          }),
        });
      }
      if (url.endsWith("/api/datasets/mnist/initial-model")) {
        return Promise.resolve({ ok: true, json: async () => ({ model_name: "mnist" }) });
      }
      if (url.endsWith("/api/autonomy/dispatch")) {
        return Promise.resolve({
          ok: true,
          json: async () => ({
            status: "executed",
            result: { available: true, data: { uploaded_workers: {} } },
          }),
        });
      }
      if (url.endsWith("/api/autonomy/loops") && init?.method === "POST") {
        anotherLoopIsRunning = true;
        return Promise.resolve({
          ok: false,
          status: 409,
          json: async () => ({ detail: "A loop is already running" }),
        });
      }
      if (url.includes("/api/autonomy/history") || url.endsWith("/api/slm/conversations")) {
        return Promise.resolve({ ok: true, json: async () => [] });
      }
      return Promise.resolve({ ok: true, json: async () => ({}) });
    });
    vi.stubGlobal("fetch", fetchMock);
    render(<AgentWorkflowPanel token="dtn-token" />);

    const startButton = await screen.findByRole("button", { name: "Start Agent" });
    await waitFor(() => expect(startButton).toBeEnabled());
    fireEvent.click(startButton);

    expect(await screen.findByText("Loop already-running-loop is running")).toBeInTheDocument();
    const stopLoopButton = screen.getByRole("button", { name: "Kill loop already-running-loop" });
    expect(stopLoopButton).toBeEnabled();
    expect(stopLoopButton).toHaveTextContent("Stop loop");
    expect(screen.queryByText("Workflow failed")).not.toBeInTheDocument();
    expect(screen.getByText(/Loop already-running-loop is already running/)).toBeInTheDocument();
    expect(fetchMock.mock.calls.some(([url]) => String(url).includes("/api/autonomy/dispatch"))).toBe(false);
  });

  it("restores persisted workflow settings, progress, action parameters, and Agent rationale", async () => {
    vi.stubGlobal("confirm", vi.fn(() => true));
    const firstWorkflowId = "11111111-1111-4111-8111-111111111111";
    const secondWorkflowId = "22222222-2222-4222-8222-222222222222";
    const thirdWorkflowId = "33333333-3333-4333-8333-333333333333";
    const fetchMock = vi.fn().mockImplementation((input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      if (url.endsWith("/api/datasets")) {
        return Promise.resolve({ ok: true, json: async () => [] });
      }
      if (url.endsWith("/api/autonomy/state")) {
        return Promise.resolve({
          ok: true,
          json: async () => ({
            stop_requested: false,
            loops: [{
              id: "loop-persisted",
              workflow_id: secondWorkflowId,
              status: "RUNNING",
              iteration: 3,
              max_iterations: 20,
              time_limit_seconds: 3600,
              stop_rule: "target_metric",
              target_metric: 0.9,
              best_metric: 0.72,
              no_improvement_count: 0,
              failure_reason: null,
              created_at: "2026-11-03T04:00:00Z",
              deadline_at: "2026-11-03T05:00:00Z",
              updated_at: "2026-11-03T04:20:00Z",
            }, {
              id: "loop-previous",
              workflow_id: firstWorkflowId,
              status: "STOPPED",
              iteration: 2,
              max_iterations: 2,
              time_limit_seconds: 3600,
              stop_rule: "max_iterations",
              target_metric: null,
              best_metric: 0.81,
              no_improvement_count: 0,
              failure_reason: "maximum iterations reached",
              created_at: "2026-11-02T03:00:00Z",
              deadline_at: "2026-11-02T04:00:00Z",
              updated_at: "2026-11-02T03:20:00Z",
            }],
          }),
        });
      }
      if (url.endsWith("/api/slm/results")) {
        return Promise.resolve({
          ok: true,
          json: async () => ({
            configuration: { rounds: 3, epochs: 2, topology: "ring", dataset_split: "stratified" },
            active_group_id: "group-persisted",
            topologies: {},
            groups: [{
              group_id: "group-persisted",
              dataset_name: "cifar10",
              dataset_split: "stratified",
              kind: "decentralized",
              topology: "ring",
              rounds: 5,
              epochs: 10,
              seeds: [2026, 2027],
              status: "running",
              error: null,
              runs: [{
                run_id: "run-persisted",
                seed: 2026,
                topology: "ring",
                status: "running",
                progress: {
                  round: 4,
                  workers: {
                    worker1: { epoch: 2, epochs: 10, phase: "local_training" },
                  },
                },
                rounds: [
                  { round: 1, topology: "ring", workers: {} },
                  { round: 2, topology: "ring", workers: {} },
                  { round: 3, topology: "ring", workers: {} },
                  { round: 4, topology: "ring", workers: {} },
                ],
              }],
            }, {
              group_id: "group-previous",
              dataset_name: "mnist",
              dataset_split: "stratified",
              kind: "decentralized",
              topology: "ring",
              rounds: 5,
              epochs: 10,
              seeds: [42],
              status: "completed",
              error: null,
              runs: [],
            }],
          }),
        });
      }
      if (url.endsWith("/api/autonomy/loops/loop-persisted/stop") && init?.method === "POST") {
        return Promise.resolve({
          ok: true,
          json: async () => ({
            id: "loop-persisted",
            workflow_id: secondWorkflowId,
            status: "STOPPED",
            iteration: 3,
            max_iterations: 20,
            time_limit_seconds: 3600,
            stop_rule: "target_metric",
            target_metric: 0.9,
            best_metric: 0.72,
            no_improvement_count: 0,
            failure_reason: "Stopped by user from the Agent Runs monitor.",
            created_at: "2026-11-03T04:00:00Z",
            deadline_at: "2026-11-03T05:00:00Z",
            updated_at: "2026-11-03T04:20:00Z",
          }),
        });
      }
      if (url.includes("/api/autonomy/history")) {
        return Promise.resolve({
          ok: true,
          json: async () => [{
            id: "execution-1",
            decision_id: null,
            event: "executed automatically",
            action: "start_training",
            result_status: "executed",
            created_at: "2026-11-03T04:10:00Z",
            details: {
              parameters: { rounds: 5, epochs: 10 },
              result: { group_id: "group-persisted" },
              reason: "Evaluate whether the selected ring experiment improves validation accuracy.",
              evidence: {
                conversation_id: "workflow-conversation",
                workflow_id: secondWorkflowId,
              },
            },
          }, {
            id: "execution-previous-training",
            decision_id: null,
            event: "executed automatically",
            action: "start_training",
            result_status: "executed",
            created_at: "2026-11-02T03:12:00Z",
            details: {
              parameters: { rounds: 5, epochs: 10 },
              result: { group_id: "group-previous" },
              reason: "Validate the previous topology selection.",
              evidence: {
                conversation_id: "workflow-conversation-previous",
                workflow_id: firstWorkflowId,
              },
            },
          }, {
            id: "execution-previous",
            decision_id: null,
            event: "executed automatically",
            action: "set_topology",
            result_status: "executed",
            created_at: "2026-11-02T03:10:00Z",
            details: {
              parameters: { topology: "ring" },
              result: { available: true },
              reason: "Compare with the previous completed run.",
              evidence: {
                conversation_id: "workflow-conversation-previous",
                workflow_id: firstWorkflowId,
              },
            },
          }, {
            id: "execution-failed",
            decision_id: null,
            event: "automatic execution failed",
            action: "start_training",
            result_status: "failed",
            created_at: "2026-11-01T02:10:00Z",
            details: {
              parameters: { dataset_name: "mnist" },
              result: { error: "No dataset shard was available." },
              reason: "Start the requested run.",
              evidence: {
                conversation_id: "workflow-conversation-failed",
                workflow_id: thirdWorkflowId,
              },
            },
          }],
        });
      }
      if (url.endsWith("/api/slm/conversations")) {
        return Promise.resolve({
          ok: true,
          json: async () => [{
            id: "workflow-conversation",
            title: "Start exactly one new experiment on dataset cifar10.",
            mode: "agent",
            workflow_id: secondWorkflowId,
            updated_at: "2026-11-03T04:10:00Z",
            message_count: 2,
          }, {
            id: "workflow-conversation-previous",
            title: "Start exactly one new experiment on dataset mnist.",
            mode: "agent",
            workflow_id: firstWorkflowId,
            updated_at: "2026-11-02T03:10:00Z",
            message_count: 2,
          }, {
            id: "workflow-conversation-failed",
            title: "Start exactly one baseline decentralized training run on dataset mnist.",
            mode: "agent",
            workflow_id: thirdWorkflowId,
            updated_at: "2026-11-01T02:10:00Z",
            message_count: 2,
          }],
        });
      }
      if (url.endsWith("/api/slm/conversations/workflow-conversation-previous")) {
        return Promise.resolve({
          ok: true,
          json: async () => ({
            id: "workflow-conversation-previous",
            title: "Start exactly one new experiment on dataset mnist.",
            mode: "agent",
            workflow_id: firstWorkflowId,
            created_at: "2026-11-02T03:00:00Z",
            updated_at: "2026-11-02T03:10:00Z",
            message_count: 2,
            messages: [{
              id: 2,
              role: "assistant",
              content: "The prior run selected ring to compare topology results.",
              created_at: "2026-11-02T03:10:00Z",
              response_data: {
                workflow_id: firstWorkflowId,
                trace: [{
                  tool: "set_topology",
                  decision: "The previous completed run supports a ring comparison.",
                  arguments: { topology: "ring" },
                  result: { status: "executed", result: { available: true } },
                }],
              },
            }],
          }),
        });
      }
      if (url.endsWith("/api/slm/conversations/workflow-conversation-failed")) {
        return Promise.resolve({
          ok: true,
          json: async () => ({
            id: "workflow-conversation-failed",
            title: "Start exactly one baseline decentralized training run on dataset mnist.",
            mode: "agent",
            workflow_id: thirdWorkflowId,
            created_at: "2026-11-01T02:00:00Z",
            updated_at: "2026-11-01T02:10:00Z",
            message_count: 2,
            messages: [{
              id: 2,
              role: "assistant",
              content: "The training request was rejected because no shard was available.",
              created_at: "2026-11-01T02:10:00Z",
              response_data: {
                workflow_id: thirdWorkflowId,
                trace: [{
                  tool: "start_training",
                  decision: "The dataset was prepared, so start its baseline run.",
                  arguments: { dataset_name: "mnist" },
                  result: {
                    status: "failed",
                    result: { error: "No dataset shard was available." },
                  },
                }],
              },
            }],
          }),
        });
      }
      if (url.endsWith("/api/slm/conversations/workflow-conversation")) {
        return Promise.resolve({
          ok: true,
          json: async () => ({
            id: "workflow-conversation",
            title: "Start exactly one new experiment on dataset cifar10.",
            mode: "agent",
            workflow_id: secondWorkflowId,
            created_at: "2026-11-03T04:00:00Z",
            updated_at: "2026-11-03T04:10:00Z",
            message_count: 2,
            messages: [{
              id: 2,
              role: "assistant",
              content: "The CIFAR-10 ring experiment targets the current validation gap.",
              created_at: "2026-11-03T04:10:00Z",
              response_data: {
                workflow_id: secondWorkflowId,
                trace: [{
                  tool: "start_training",
                  decision: "The current validation accuracy is below target, so test the ring configuration next.",
                  arguments: { rounds: 5, epochs: 10 },
                  result: { status: "executed", result: { group_id: "group-persisted" } },
                }],
              },
            }],
          }),
        });
      }
      return Promise.resolve({ ok: true, json: async () => ({}) });
    });
    vi.stubGlobal("fetch", fetchMock);
    render(<AgentWorkflowPanel token="dtn-token" />);

    expect(await screen.findByText("The current validation accuracy is below target, so test the ring configuration next.")).toBeInTheDocument();
    expect(screen.getByText(`Agent ID: ${firstWorkflowId}`)).toBeInTheDocument();
    expect(screen.getByText(`Agent ID: ${secondWorkflowId}`)).toBeInTheDocument();
    expect(screen.getByText(`Agent ID: ${thirdWorkflowId}`)).toBeInTheDocument();
    expect(screen.getByText("Loop loop-persisted is running")).toBeInTheDocument();
    expect(screen.getByText(/Active loop: loop-persisted/)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Kill loop loop-persisted" })).toBeEnabled();
    expect(screen.getByText("The previous completed run supports a ring comparison.")
      .closest("details")).not.toHaveAttribute("open");

    fireEvent.click(screen.getByText(`Agent ID: ${secondWorkflowId}`));
    expect(screen.getByText("The current validation accuracy is below target, so test the ring configuration next.")).toBeInTheDocument();
    expect(screen.getByText("No tool errors recorded")).toBeInTheDocument();
    expect(screen.getByText("3 / 20")).toBeInTheDocument();
    expect(screen.getAllByText("Validation accuracy ≥ 90.0%")).toHaveLength(1);
    const secondWorkflow = screen.getByText(`Agent ID: ${secondWorkflowId}`).closest("details");
    expect(secondWorkflow).not.toBeNull();
    expect(within(secondWorkflow as HTMLElement).getByText("cifar10 / stratified")).toBeInTheDocument();
    expect(within(secondWorkflow as HTMLElement).getByText("5 rounds / 10 epochs")).toBeInTheDocument();
    expect(within(secondWorkflow as HTMLElement).getByText("4 / 10")).toBeInTheDocument();
    expect(within(secondWorkflow as HTMLElement).getByText("worker1")).toBeInTheDocument();
    expect(within(secondWorkflow as HTMLElement).getAllByText("Action parameters and result")).toHaveLength(1);
    expect(within(secondWorkflow as HTMLElement).getByText("group-persisted")).toBeInTheDocument();
    expect(within(secondWorkflow as HTMLElement).queryByText("group-previous")).not.toBeInTheDocument();

    fireEvent.click(screen.getByText(`Agent ID: ${firstWorkflowId}`));
    expect(screen.getByText("The previous completed run supports a ring comparison.")).toBeInTheDocument();
    expect(screen.getByText("Experiment limit reached")).toBeInTheDocument();
    const firstWorkflow = screen.getByText(`Agent ID: ${firstWorkflowId}`).closest("details");
    expect(firstWorkflow).not.toBeNull();
    expect(within(firstWorkflow as HTMLElement).getAllByText("Action parameters and result")).toHaveLength(2);
    expect(within(firstWorkflow as HTMLElement).getByText("group-previous")).toBeInTheDocument();

    fireEvent.click(screen.getByText(`Agent ID: ${thirdWorkflowId}`));
    expect(screen.getByText(/1 recorded action failed or were rejected/)).toBeInTheDocument();

    expect(screen.getByRole("checkbox", {
      name: `Select Agent workflow ${secondWorkflowId}`,
    })).toBeDisabled();
    fireEvent.click(screen.getByRole("checkbox", {
      name: `Select Agent workflow ${firstWorkflowId}`,
    }));
    fireEvent.click(screen.getByRole("button", { name: "Delete selected" }));
    await waitFor(() => {
      const deleteRequest = fetchMock.mock.calls.find(([url, init]) => (
        String(url).endsWith("/api/autonomy/workflows") && init?.method === "DELETE"
      ));
      expect(deleteRequest).toBeDefined();
      expect(JSON.parse(String(deleteRequest?.[1]?.body)).workflow_ids).toEqual([firstWorkflowId]);
    });

    fireEvent.click(screen.getByRole("button", { name: "Kill loop loop-persisted" }));

    await waitFor(() => {
      expect(fetchMock).toHaveBeenCalledWith(
        expect.stringContaining("/api/autonomy/loops/loop-persisted/stop"),
        expect.objectContaining({ method: "POST" }),
      );
    });
    const stopRequest = fetchMock.mock.calls.find(([url]) => (
      String(url).endsWith("/api/autonomy/loops/loop-persisted/stop")
    ));
    expect(JSON.parse(String(stopRequest?.[1]?.body))).toEqual({
      reason: "Stopped by user from the Agent Runs monitor.",
    });
    const stopTrainingRequests = fetchMock.mock.calls.filter(([url, init]) => (
      String(url).endsWith("/api/autonomy/dispatch")
      && JSON.parse(String(init?.body)).action.action === "stop_training"
    ));
    expect(stopTrainingRequests).toHaveLength(1);
    expect(JSON.parse(String(stopTrainingRequests[0][1]?.body)).evidence.workflow_id).toBe(secondWorkflowId);
  });

  it("deletes a master conversation and clears the active chat after confirmation", async () => {
    vi.stubGlobal("confirm", vi.fn(() => true));
    const fetchMock = vi.fn().mockImplementation((input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      if (url === `${masterUrl}/rag/conversations`) {
        return Promise.resolve({
          ok: true,
          json: async () => [{
            id: "conversation-saved",
            title: "Saved cluster question",
            updated_at: "2026-11-03T04:00:00Z",
            message_count: 2,
          }],
        });
      }
      if (url.endsWith("/conversation-saved") && init?.method === "DELETE") {
        return Promise.resolve({ ok: true, status: 204 });
      }
      return Promise.resolve({
        ok: true,
        json: async () => ({
          id: "conversation-saved",
          title: "Saved cluster question",
          created_at: "2026-11-03T04:00:00Z",
          updated_at: "2026-11-03T04:00:00Z",
          messages: [{
            id: 1,
            role: "user",
            content: "Saved cluster question",
            created_at: "2026-11-03T04:00:00Z",
          }],
        }),
      });
    });
    vi.stubGlobal("fetch", fetchMock);
    render(<AssistantPage token="dtn-token" />);

    fireEvent.click(await screen.findByRole("button", {
      name: "Delete chat: Saved cluster question",
    }));

    expect(await screen.findByText("No saved chats.")).toBeInTheDocument();
    expect(screen.getByText("No messages yet.")).toBeInTheDocument();
    const deleteCall = fetchMock.mock.calls.find(
      ([url, init]) => String(url).endsWith("/conversation-saved") && init?.method === "DELETE",
    );
    expect(deleteCall).toBeDefined();
    expect(vi.mocked(window.confirm)).toHaveBeenCalled();
  });

  it("deletes multiple selected master conversations", async () => {
    vi.stubGlobal("confirm", vi.fn(() => true));
    const conversations = [
      {
        id: "conversation-first",
        title: "First saved chat",
        updated_at: "2026-11-03T04:00:00Z",
        message_count: 2,
      },
      {
        id: "conversation-second",
        title: "Second saved chat",
        updated_at: "2026-11-02T04:00:00Z",
        message_count: 2,
      },
    ];
    const deleteCalls: string[] = [];
    const fetchMock = vi.fn().mockImplementation((input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      if (url === `${masterUrl}/rag/conversations`) {
        return Promise.resolve({ ok: true, json: async () => conversations });
      }
      if (init?.method === "DELETE") {
        deleteCalls.push(url);
        return Promise.resolve({ ok: true, status: 204 });
      }
      const conversation = conversations.find((item) => url.endsWith(`/${item.id}`));
      return Promise.resolve({
        ok: true,
        json: async () => ({
          ...conversation,
          created_at: conversation?.updated_at,
          messages: [{
            id: 1,
            role: "user",
            content: conversation?.title,
            created_at: conversation?.updated_at,
          }],
        }),
      });
    });
    vi.stubGlobal("fetch", fetchMock);
    render(<AssistantPage token="dtn-token" />);

    fireEvent.click(await screen.findByRole("checkbox", { name: "Select all chats" }));
    fireEvent.click(screen.getByRole("button", { name: "Delete selected" }));

    await waitFor(() => expect(deleteCalls).toHaveLength(2));
    expect(screen.getByText("No saved chats.")).toBeInTheDocument();
    expect(deleteCalls.sort()).toEqual([
      `${masterUrl}/rag/conversations/conversation-first`,
      `${masterUrl}/rag/conversations/conversation-second`,
    ]);
  });

  it("deletes the legacy unlinked Agent workflow", async () => {
    vi.stubGlobal("confirm", vi.fn(() => true));
    let legacyActionsAvailable = true;
    const fetchMock = vi.fn().mockImplementation((input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      if (url.includes("/api/autonomy/history")) {
        return Promise.resolve({
          ok: true,
          json: async () => legacyActionsAvailable ? [{
            id: 1,
            action: "preprocess_dataset",
            created_at: "2026-11-07T16:52:17Z",
            result_status: "executed",
            details: {
              parameters: { dataset_name: "mnist" },
              evidence: { source: "agent_workflow" },
              reason: "Agent workflow preprocessing",
            },
          }] : [],
        });
      }
      if (url.endsWith("/api/autonomy/state")) {
        return Promise.resolve({ ok: true, json: async () => ({ stop_requested: false, loops: [] }) });
      }
      if (url.endsWith("/api/slm/conversations")) {
        return Promise.resolve({ ok: true, json: async () => [] });
      }
      if (url.endsWith("/api/slm/results")) {
        return Promise.resolve({
          ok: true,
          json: async () => ({ active_group_id: null, groups: [], topologies: {} }),
        });
      }
      if (url.endsWith("/api/datasets")) {
        return Promise.resolve({ ok: true, json: async () => [] });
      }
      if (url.endsWith("/api/autonomy/workflows") && init?.method === "DELETE") {
        legacyActionsAvailable = false;
        return Promise.resolve({ ok: true, json: async () => ({ deleted_actions: 1 }) });
      }
      return Promise.resolve({ ok: true, json: async () => ({}) });
    });
    vi.stubGlobal("fetch", fetchMock);
    render(<AgentWorkflowPanel token="dtn-token" />);

    const legacyCheckbox = await screen.findByRole("checkbox", {
      name: "Select Agent workflow legacy-unlinked",
    });
    expect(legacyCheckbox).toBeEnabled();
    fireEvent.click(legacyCheckbox);
    fireEvent.click(screen.getByRole("button", { name: "Delete selected" }));

    await waitFor(() => {
      const deleteRequest = fetchMock.mock.calls.find(([url, init]) => (
        String(url).endsWith("/api/autonomy/workflows") && init?.method === "DELETE"
      ));
      expect(deleteRequest).toBeDefined();
      expect(JSON.parse(String(deleteRequest?.[1]?.body)).workflow_ids).toEqual(["legacy-unlinked"]);
    });
    await waitFor(() => expect(screen.queryByText("Agent ID: legacy-unlinked")).not.toBeInTheDocument());
  });

  it("stops the persisted loop when workflow setup fails", async () => {
    let loopStarted = false;
    const fetchMock = vi.fn().mockImplementation((input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      const path = new URL(url).pathname;
      if (path === "/api/datasets") {
        return Promise.resolve({
          ok: true,
          json: async () => [{
            dataset_name: "mnist",
            classes: ["0", "1"],
            class_counts: { "0": 1, "1": 1 },
            image_sizes: { "0": { "28x28": 1 }, "1": { "28x28": 1 } },
            image_count: 2,
            expanded_image_bytes: 0,
          }],
        });
      }
      if (path === "/api/autonomy/state") {
        return Promise.resolve({ ok: true, json: async () => ({ stop_requested: false, loops: [] }) });
      }
      if (path === "/api/slm/results") {
        return Promise.resolve({
          ok: true,
          json: async () => ({ active_group_id: null, groups: [], topologies: {} }),
        });
      }
      if (path === "/api/autonomy/loops" && init?.method === "POST") {
        loopStarted = true;
        return Promise.resolve({
          ok: true,
          json: async () => ({
            id: "loop-setup-failure",
            status: "RUNNING",
            iteration: 0,
            max_iterations: 100,
            stop_rule: "target_metric",
            target_metric: 0.8,
            best_metric: null,
            failure_reason: null,
          }),
        });
      }
      if (path === "/api/autonomy/dispatch") {
        expect(loopStarted).toBe(true);
        return Promise.resolve({
          ok: false,
          status: 500,
          json: async () => ({ detail: "Dataset setup failed" }),
        });
      }
      if (path === "/api/autonomy/loops/loop-setup-failure/stop") {
        return Promise.resolve({
          ok: true,
          json: async () => ({ id: "loop-setup-failure", status: "STOPPED" }),
        });
      }
      if (path === "/api/autonomy/history" || path === "/api/slm/conversations") {
        return Promise.resolve({ ok: true, json: async () => [] });
      }
      return Promise.resolve({ ok: true, json: async () => ({}) });
    });
    vi.stubGlobal("fetch", fetchMock);
    render(<AgentWorkflowPanel token="dtn-token" />);

    const startButton = await screen.findByRole("button", { name: "Start Agent" });
    await waitFor(() => expect(startButton).toBeEnabled());
    fireEvent.click(startButton);

    await waitFor(() => {
      expect(fetchMock).toHaveBeenCalledWith(
        expect.stringContaining("/api/autonomy/loops/loop-setup-failure/stop"),
        expect.objectContaining({ method: "POST" }),
      );
    });
    expect(screen.getByText("Workflow did not reach its termination criterion.")).toBeInTheDocument();
  });
});
