# Agentic SLM architecture (v2 supervisor)

The DTN has two operating modes:

- **Human mode** (SLM Assistant page). The operator performs actions in the UI and asks the
  on-device SLM for advice. The browser calls the master's read-only `/rag/chat`; the master
  retrieves its action history and live training results and answers with Qwen3.5-4B through
  Ollama. If Ollama is unavailable, it returns a rule-based next-step suggestion instead of an
  error. Human mode never executes actions.
- **Agentic mode** (Agent Supervisor page). The operator gives a dataset and termination
  criteria; the supervisor agent runs the whole workflow by itself and can be asked questions
  at any time.

## Why v2

The v1 agent offered the SLM all 14 coordinator tools and was driven by a browser page. Two
problems followed:

1. Tool calls did not make progress. On the master Jetson, Qwen3.5-4B chose the read-only
   `get_results` tool in 10 of 10 trials when offered the full toolset (at 4,096 or 8,192
   context), but chose a useful step (`run_nas`) in 10 of 10 trials when offered five compact
   steps. The prompt was not truncated (2,822 tokens), so the cause was the size of the tool
   set, not the context window. Data: `docs/evidence/agent-toolcall-benchmark-2026-11-17.json`.
2. Closing or refreshing the browser ended the workflow ("frontend workflow session lost").

## Supervisor design

`dtn/backend/app/services/supervisor.py` runs in the coordinator process as one background
thread. Every run is stored in `agent_runs` (goal and resumable state as JSON) with an event
timeline in `agent_run_events`, so a backend restart resumes the run from its saved stage.

Each tick (default 10 s) the supervisor:

1. **Monitors** worker readiness and free memory from the device poller, and the active
   experiment from the master's `/results`.
2. **Admits** a step only when every training worker is online and has at least
   `DTN_SUPERVISOR_MIN_FREE_MEMORY_MB` (default 1,200 MB) free. On low memory it runs the
   worker memory cleanup once, then waits; after 10 minutes it ends the run as `blocked`.
3. **Remediates** a transient failure (timeout, memory, connection, peer errors) by cleaning
   worker memory and retrying the step once. Three consecutive failures end the run.
4. **Decides** the next step at decision points:
   - deterministic rules first: finish when the measured mean worker validation accuracy meets
     the target, when the time budget is gone, or when no step is left;
   - otherwise the SLM picks one step from a compact, state-filtered tool set
     (`run_nas`, `optimize_topology` (MNIST), `train_final`, `finish_workflow`). The reply is
     validated against the tool's JSON schema; a tool call written as text is recovered; one
     invalid reply gets a repair prompt; if the SLM is still invalid or unreachable, a
     deterministic fallback policy (NAS → topology → retrain) is used.
5. **Executes** the step through the audited `AutonomyController` (global STOP is honoured,
   every action is recorded with `evidence.source = "supervisor"`).

The fixed workflow is: preflight → preprocess and ship shards → distribute one initial model →
baseline DFL → SLM-chosen improvement steps → finish. `train_final` redistributes the searched
architecture as the initial model before training, because workers train their installed
initial model and would otherwise silently ignore the requested architecture.

Decision provenance is recorded per run (`counters`: SLM, SLM after repair, fallback, rule),
which is the measure of how much the on-device SLM actually decided.

## API

All routes need the DTN bearer token.

| Route | Purpose |
|---|---|
| `GET /api/supervisor/runs` | Recent runs |
| `POST /api/supervisor/runs` | Start a run (only one active run at a time) |
| `GET /api/supervisor/runs/{id}` | Run detail with the event timeline |
| `POST /api/supervisor/runs/{id}/stop` | Stop the run and its active experiment |
| `POST /api/supervisor/runs/{id}/ask` | Ask the SLM about the run (read-only, no tools) |
| `GET /api/supervisor/cluster` | Worker readiness, free memory, global STOP |

The master's `/agent/model` accepts the caller's `tools` list (1-12 definitions, under 16 KB)
or `allow_tools: false`; it never offers the full action schema to the model.

## Model runtime on the master

One Ollama model serves both modes. The former Llama-3.2-1B reply-style LoRA service is
retired, so the master holds one language model. `DFL_OLLAMA_NUM_CTX` (default 4,096) must stay
the same for every call: changing it forces a ~61 s model reload. 8,192 would cost only about
132 MB more, because Qwen3.5 keeps a KV cache for 8 of its 32 layers.
