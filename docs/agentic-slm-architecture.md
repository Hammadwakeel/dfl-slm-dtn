# LangGraph SLM agent

The Assistant supports two independent conversation modes. **Human mode** keeps the
existing read-only master RAG chat. **Agent mode** sends turns to the authenticated
DTN backend, where LangGraph can select and execute coordinator tools without
asking for confirmation. Each conversation stores its mode and history.

## Request and control flow

1. The frontend selects the mode for the conversation. Human turns continue to use
   the master's existing RAG endpoints; agent turns use `POST /api/slm/agent/chat`.
2. The backend builds one bounded snapshot from every registered node's latest
   polled status, available dataset metadata and preprocessing reports, recent
   experiment state, global STOP state, and recent audited actions. It adds the
   conversation history and current user request as separate messages.
3. The backend calls the authenticated master's `/agent/model` endpoint. The master
   uses Ollama's native function-call interface; LangGraph owns the model/tool loop.
4. Tool definitions are generated from `common/slm_action_schema.json`. Actions
   marked `x-agent-tool` are offered as functions; their parameter schemas and
   descriptions are the source of truth. No keyword/intent router selects actions.
5. LangGraph validates every proposed action against the shared action schema and
   dispatches it through the existing coordinator controller. Writes obey global
   STOP, action bounds, dataset preflight, and worker-side execution checks; agent
   executions are recorded with mode `AGENT`. Tool results are returned to the
   model for analysis before its user-facing response is saved.

The tool contract includes model creation/distribution, dataset preprocessing,
distributed training, NAS, topology optimization, current status, experiment
results, logs, and stopping an active training workflow. The agent can inspect a
running experiment with status/results tools and can stop it with the stop tool.
Model creation accepts an explicit architecture or uses the existing standard
small-CNN default. Training work itself remains managed by the existing
master/worker services.
Both MNIST and CIFAR-10 are valid dataset targets for model creation and
distributed training after their shards have been preprocessed and shipped.
Agent mode also provides a dataset-to-learning workflow. The selected dataset and
termination criterion are supplied as goals, while the SLM chooses each next tool
and its parameters from the current snapshot and previous tool results. The
frontend refreshes experiment state between bounded SLM turns and reports failed
actions as well as successful results back to the model. The SLM explicitly ends
the workflow with `finish_agent_workflow`; the frontend enforces a separate
experiment-count and elapsed-time safety limit. Workflow tool calls are bound to
the selected dataset; topology optimization is rejected for CIFAR-10 because that
existing optimization action currently uses MNIST. The workflow does not
automatically prescribe a baseline, NAS retry, worker reboot, or selected-model
retraining after an experiment result.

Within a turn, schema-invalid calls and explicitly retryable no-op rejections may
be corrected by the model up to a small fixed bound. A successful action still
ends the turn, and workflow-policy rejections are not retried. Dataset-bound
turns use the fresh snapshot instead of spending experiment turns on status,
logs, or results lookups; rejected stop requests and reused initial models are
returned as errors rather than success-shaped results.

## Operational limits and extension points

`DTN_AGENT_MAX_TOOL_CALLS`, `DTN_AGENT_HISTORY_MESSAGES`, and
`DTN_AGENT_ACTION_HISTORY_LIMIT` bound one turn's tool loop and prompt history.
Dataset reports, action history, and tool results are summarized before they enter
the model context, and chat history also has a character limit to keep multi-step
turns within the local model's context window.
The LangGraph runtime is stateless between turns; durable conversation history
and action audit records are stored in the DTN database. The current agent can
inspect a running workflow during its turn and on later Agent-mode turns; it does
not run an always-on background supervisor after a chat response completes. The
dataset-to-learning workflow is supervised by the Assistant page and must remain
open while it runs; its loop and termination state are recorded by the coordinator.
Manual stop, failed actions, 100 completed experiments, and a 24-hour safety
limit are surfaced separately from successful completion of the selected
criterion.

For deployment at larger scale, the next extension is a durable LangGraph
checkpointer and a separate worker-backed monitor service that subscribes to
active experiment IDs, publishes progress events to the UI, and resumes bounded
monitor graphs after backend restarts. That extension should reuse the same
schema-derived tools and controller rather than adding a second execution path.
