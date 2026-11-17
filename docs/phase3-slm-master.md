# Phase 3: SLM Master

## Topology and deployment

The Waitress-served SLM API runs on `dfl-master` (`192.168.11.15`) at TCP `5003`. The Phase 1 authenticated device-status service continues on TCP `5002`; the services use separate systemd units, environment files, ports, and API tokens:

- `dfl-slm-status.service`: TCP `5002`, `/etc/dfl-slm-node.env`, `MASTER_API_TOKEN`.
- `dfl-slm-master.service`: TCP `5003`, `/etc/dfl-slm-master.env`, `MASTER_SLM_API_TOKEN`.
- `dfl-slm-reply.service`: loopback TCP `5004`, `/etc/dfl-slm-reply.env`, the same server-side SLM token; this hosts the reply-style LoRA without replacing the action model.

The coordinator is configured for Ollama `qwen3.5:4b`. The legacy action endpoint calls `/api/chat` with JSON-schema constrained output; direct RAG chat uses a natural-language prompt containing the retrieved context and original question. Both use `num_ctx=4096`, `keep_alive=5m`, and `think=false`; routing temperatures are 0.0 for schema actions and 0.1 for Agent tool calls. The Agent endpoint passes the existing JSON tool schemas through Ollama's standard chat API. The migration installer pulls and inspects the configured model before restarting the SLM service; if the installed Ollama version cannot load it, installation stops before replacing the running service. The separately hosted reply-style LoRA remains based on Llama 3.2 1B and is not the coordinator model.

The coordinator has been migrated to Ollama `qwen3.5:4b` (Q4_K_M); schema-constrained JSON and tool-call smoke tests passed on Ollama 0.34.0, and the old `llama3.2:1b` Ollama tag was removed. With Qwen loaded, the 8 GB master had about 1.2 GiB available in the latest check, so monitor memory during experiments. The separately hosted reply-style LoRA still uses its Hugging Face Llama 3.2 1B base; keep that cache while the reply service is enabled.

From Windows PowerShell at the repository root, deploy the SLM service with the configured SSH key and passwordless sudo:

```powershell
.\deploy\deploy_master_slm.ps1
```

The script creates a cryptographically random `MASTER_SLM_API_TOKEN` in the ignored root `.env` if one is not configured, then stages the SLM application, shared action schema, installer, service units, and protected environment files to the master. It installs the runtime dependencies into `/opt/dfl-slm-dtn/master/venv`, writes the service environments with mode `0600`, enables the action and LoRA services, and verifies both the existing status API on port `5002` and the reply model on loopback port `5004`. The installer does not restart, disable, or replace the Phase 1 unit.

The reply service uses the cached Llama 3.2 1B base model and the LoRA adapter at `/home/hammad/slm-ft-outputs/reply-style-20261102T2158JST/adapter`. Each request runs in a bounded one-shot inference process; this releases model allocations when the process exits, at the cost of loading the model for each reply. The dashboard coordinator remains responsible for dashboard and worker workflows. The Assistant chat itself sends the original user question directly to the master SLM; it does not call the coordinator's intent matcher or action dispatcher.

`deploy_master_slm.ps1` configures a dedicated chat-only token from `DTN_API_TOKEN` and the allowed UI origins. The browser already holds that token after login. The master accepts it only on `/rag/chat` and `/rag/conversations`; it cannot access `/command`, status, logs, or other privileged master routes. The privileged `MASTER_SLM_API_TOKEN` remains server-side. The master permits browser requests only from configured Vite origins; add any other exact origin to `DFL_CHAT_ALLOWED_ORIGINS` before deployment. `VITE_MASTER_SLM_BASE_URL` can override the default `http://192.168.11.15:5003` frontend address.

## Authenticated API and action safety

Privileged master endpoints require the master SLM bearer token. Read-only RAG endpoints use the scoped browser chat token:

- Assistant chat: `POST /rag/chat`, `GET /rag/conversations`, `GET /rag/conversations/{id}`, `DELETE /rag/conversations/{id}`.
- Privileged action API: `POST /chat`, `POST /command`, `GET /status`, `GET /logs`, and `POST /action-history`.

`POST /rag/chat` takes `{"message":"...","conversation_id":"optional-id"}` and returns a natural-language reply plus the master-owned conversation ID. The master retrieves recent turns in that conversation and relevant successful action records—including their saved results—from SQLite, then includes that bounded context and the exact current question in the SLM prompt. The model may recommend a next step based on evidence, but this endpoint never executes actions or returns action JSON. Conversations and messages are stored alongside action history in `/var/lib/dfl-slm-master/action_history.sqlite3`. Existing coordinator-side conversations are not migrated; new direct-master conversations are persisted and restored here.

Successful master actions, including read-only queries, are persisted with their results. The coordinator forwards full results for successful preprocessing and initial-model actions so the RAG chat can refer to those too. Older action rows created before result persistence have no recoverable result payload; new records retain it. `POST /chat` remains a separate legacy action-schema API, while the Assistant UI uses only the RAG endpoints.

`common/slm_action_schema.json` allows only:

| Action | Parameters / validation | Phase 3 behavior |
|---|---|---|
| `clarify` | No parameters; asks for missing/ambiguous details or which safe step to perform first | Safe conversational outcome; no action dispatched |
| `refuse` | No parameters; declines unsafe, out-of-scope, or out-of-range requests | Safe conversational outcome; no action dispatched |
| `redirect` | No parameters; redirects unrelated chat to DFL operations | Safe conversational outcome; no action dispatched |
| `upload_dataset` | Dataset name matching the safe identifier pattern | Not available yet |
| `preprocess_dataset` | Dataset name; optional `standard`, `minmax`, or `none` normalization | Not available yet |
| `create_initial_model` | Safe model name | Not available yet |
| `start_training` | 1–100 rounds and 1–50 epochs | Not available yet |
| `stop_training` | No parameters | Not available yet |
| `set_topology` | `ring`, `directed_ring`, `bidirectional_ring`, `mesh`, `chain`, or `star` | Not available yet |
| `set_rounds` | 1–100 | Not available yet |
| `set_epochs` | 1–50 | Not available yet |
| `enable_nas` | Boolean `enabled`; optional rounds 1–100, epochs 1–50, 1–3 unique uint32 seeds, and search time 30–300 seconds | Not available yet |
| `optimize_topology` | Optional rounds 1–100, epochs 1–50, 1–3 unique uint32 seeds; currently uses Dirichlet alpha 0.3 only | Not available yet |
| `get_status` | No parameters | Reads the real local authenticated status agent on port 5002 |
| `get_logs` | Optional 1–100 log lines | Reads the SLM service's real rotated JSONL event log |
| `get_results` | No parameters | Not available yet |
| `compare_runs` | Exactly two unique safe run IDs (1–64 characters each) | Not available yet |
| `suggest_next_action` | No parameters | Not available yet |

Unavailable actions return `available: false` and an explicit explanation. There is no worker training API in Phase 3, and configuration-shaped actions do not mutate a training system. User text, model output, and action parameters are never passed to a shell or evaluated as code. The service logs action names, outcomes, timing, and RAM metrics, not full user prompts or API tokens.

This table records the Phase 3 acceptance state. Phase 4 subsequently implemented worker-backed `start_training`, `stop_training`, `set_topology` (Ring/Mesh), `set_rounds`, `set_epochs`, and `get_results`; dataset upload/preprocessing, model creation, and NAS remain unavailable. See [phase4-worker-training.md](./phase4-worker-training.md) for the deployed services and real experiment output.

## Tests

Run the backend suite (includes authenticated SLM proxy coverage):

```powershell
Set-Location 'D:\Lab\MS Research Project\dfl-slm-dtn\dtn\backend'
python -m pytest -q
```

Run the local SLM action schema and Flask API tests:

```powershell
Set-Location 'D:\Lab\MS Research Project\dfl-slm-dtn'
python -m pip install -r master\requirements.txt
python -m pytest -q master\tests
```

Run the dashboard tests and production build:

```powershell
Set-Location 'D:\Lab\MS Research Project\dfl-slm-dtn\dtn\frontend'
npm test
npm run build
```

## Live verification (2026-10-24)

The deployed SLM was exercised through the coordinator at `192.168.11.12:8000`, not by calling Ollama directly. All 20 natural-language prompts returned one action that passed independent validation against the shared JSON Schema: **20/20 valid (100%)**, with no invalid or failed responses. Eight actions were available and executed against real status/log sources; the other twelve returned `available: false` because their worker/training functionality is not implemented.

| Returned action | Prompt count | Available |
|---|---:|---:|
| `get_status` | 6 | 6 |
| `get_logs` | 2 | 2 |
| `get_results` | 3 | 0 |
| `upload_dataset` | 1 | 0 |
| `preprocess_dataset` | 1 | 0 |
| `start_training` | 1 | 0 |
| `stop_training` | 1 | 0 |
| `set_topology` | 1 | 0 |
| `set_rounds` | 2 | 0 |
| `enable_nas` | 1 | 0 |
| `compare_runs` | 1 | 0 |

Warm inference latency averaged **1,696 ms** (range **1,336–2,081 ms**) across the 20 responses. The API's system-wide used-RAM delta averaged **+2.6 MB** (largest observed increase **+35.9 MB**) during those warm generations. A separate cold-start response took **54,020 ms** while Ollama loaded the model; system RAM use rose from about **1,485 MB to 3,528 MB** (about **+2,043 MB**). The model file is approximately **1.321 GB**; the master reported about **6,433 MB available** before that cold inference. RAM deltas are host-wide measurements around each request, not process-isolated SLM RSS.

Verification completed: master API tests **11 passed**; coordinator backend tests **16 passed**; frontend tests **5 passed**; TypeScript check and Vite production build succeeded. Authenticated coordinator `/api/slm/status` and `/api/slm/logs` returned HTTP 200; unauthenticated requests to all four master endpoints returned HTTP 401. The live status route confirmed `dfl-master` and Ollama model availability. The status agent and SLM services remain separate on ports **5002** and **5003**, respectively.

## Accuracy check rerun (2026-10-24)

The same 20 prompts were sent again through the live coordinator. Intent accuracy means the returned action matches the action appropriate to the prompt. Parameter accuracy requires the prompt's specified values to match; an omitted optional value in prompt 5 is unconstrained (the returned limit must only be schema-valid). This distinguishes semantic correctness from JSON Schema validity.

| # | Prompt | Expected action and parameters | SLM returned action and parameters | Intent | Parameters |
|---:|---|---|---|:---:|:---:|
| 1 | Report live status of the master Jetson. | `get_status {}` | `get_status {}` | Pass | Pass |
| 2 | Check whether dfl-master is online and report its CPU temperature. | `get_status {}` | `get_status {}` | Pass | Pass |
| 3 | Show the current health readings from the master device. | `get_status {}` | `get_status {}` | Pass | Pass |
| 4 | Fetch the latest status metrics for dfl-master. | `get_status {}` | `get_status {}` | Pass | Pass |
| 5 | List the most recent assistant service logs. | `get_logs {limit: unspecified}` | `get_logs {limit: 100}` | Pass | Pass |
| 6 | Show me the last five SLM API log entries. | `get_logs {limit: 5}` | `get_results {}` | Fail | Fail |
| 7 | Read 3 recent events from your logs. | `get_logs {limit: 3}` | `get_logs {limit: 10}` | Pass | Fail |
| 8 | Upload dataset named indoor_co2. | `upload_dataset {dataset_name: indoor_co2}` | `upload_dataset {dataset_name: indoor_co2}` | Pass | Pass |
| 9 | Preprocess indoor_co2 with standard normalization. | `preprocess_dataset {dataset_name: indoor_co2, normalization: standard}` | `preprocess_dataset {dataset_name: indoor_co2, normalization: standard}` | Pass | Pass |
| 10 | Create an initial model called co2_baseline. | `create_initial_model {model_name: co2_baseline}` | `get_results {}` | Fail | Fail |
| 11 | Start training using 4 rounds and 2 epochs. | `start_training {rounds: 4, epochs: 2}` | `start_training {rounds: 4, epochs: 2}` | Pass | Pass |
| 12 | Stop the currently running distributed training job. | `stop_training {}` | `stop_training {}` | Pass | Pass |
| 13 | Set cluster topology to ring. | `set_topology {topology: ring}` | `set_topology {topology: ring}` | Pass | Pass |
| 14 | Change the training plan to 8 rounds. | `set_rounds {rounds: 8}` | `set_rounds {rounds: 8}` | Pass | Pass |
| 15 | Use 3 local epochs per round. | `set_epochs {epochs: 3}` | `set_rounds {rounds: 3}` | Fail | Fail |
| 16 | Enable neural architecture search. | `enable_nas {enabled: true}` | `enable_nas {enabled: true}` | Pass | Pass |
| 17 | Check the latest training results. | `get_results {}` | `get_results {}` | Pass | Pass |
| 18 | Compare runs run_001 and run_002. | `compare_runs {run_ids: [run_001, run_002]}` | `compare_runs {run_ids: [run_001, run_002]}` | Pass | Pass |
| 19 | Based on the current cluster setup, suggest the next action. | `suggest_next_action {}` | `get_status {}` | Fail | Fail |
| 20 | Review dfl-master status once more and tell me if the service is up. | `get_status {}` | `get_status {}` | Pass | Pass |

**Intent accuracy: 16/20 (80%). Parameter accuracy: 15/20 (75%)** over all prompts; among the 16 intent-correct prompts, parameter accuracy was 15/16 (93.8%). Schema validity remained 20/20 (100%), which does not imply semantic accuracy. In particular, prompt 15 confused epochs with rounds; prompts 6, 10, and 19 selected the wrong action. Prompt 7 selected the right action but returned the wrong requested limit. The replies also occasionally contained misleading status or success claims even when the action indicated unavailability; callers must treat the structured action and result as authoritative, not the free-text reply.

## Next-action recommendations

The coordinator recognizes common next-step phrasings before forwarding a request to the action model, so questions such as “what is the next action?”, “what should I do now?”, and “what next?” are not misclassified as master-status queries. When there are no saved MNIST runs, it advises verifying the worker shards, creating and distributing an initial model in Model Studio, and then starting distributed training. When runs exist, the recommendation follows the latest run state and saved metrics.

The master SLM stores successfully executed state-changing actions in `/var/lib/dfl-slm-master/action_history.sqlite3` (override with `DFL_ACTION_HISTORY_PATH`). It retrieves up to ten recent entries itself and supplies their action names, parameters, timestamps, and run/dataset identifiers to both SLM models as factual context. Failed actions, raw prompts, and execution evidence are excluded. Coordinator-executed preprocessing/model-creation actions are forwarded once to the master's authenticated `/action-history` endpoint; the coordinator does not retrieve or construct SLM history context.

## Security and operations

Both the SLM API and coordinator proxy use bearer credentials on the trusted lab LAN over HTTP, not TLS. Keep these ports on the protected network and do not expose them to the public Internet. The Ollama API is configured at loopback on the master and is not exposed to the PC or browser. Use `sudo systemctl status dfl-slm-master.service`, `sudo journalctl -u dfl-slm-master.service`, and `sudo systemctl status dfl-slm-status.service` for operational checks.
