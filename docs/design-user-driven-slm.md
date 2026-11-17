# User-driven SLM product design plan

> Superseded by the current [LangGraph SLM agent architecture](./agentic-slm-architecture.md).
> This earlier proposal remains as historical context; the current product supports
> a read-only Human mode and a tool-enabled Agent mode.

**Status:** historical design proposal.

## Product decision

The DTN UI is the only product entry point for starting work. Remove autonomous execution and `FULL_ACCESS`; the SLM is read-only and advisory. It answers questions from retrieved DTN records and recommends a module, but cannot dispatch, approve, schedule, or execute work. A user must navigate to a module and explicitly start its run.

Keep a global STOP, an all-nodes-idle preflight before every hardware run, an explicit confirmation for long runs (including the roughly 59-minute sweep), and a durable audit record of each user-initiated run. The confirmation is a UI confirmation dialog, not an SLM “pending decision.”

## A. Inventory and disposition

This inventory is based on the current repository source. The root and backend code have in-progress local changes; this plan does not overwrite or interpret them as an accepted baseline.

| Area / current surface | Current dependency | Proposed disposition | Test / documentation impact |
|---|---|---|---|
| `dtn/backend/app/services/autonomy.py` | Persists `ASK_HUMAN` / `FULL_ACCESS`; dispatch can execute state changes directly in `FULL_ACCESS`; has pending decisions, loops, STOP, and a `FULL_ACCESS`-specific execution audit. | **Disable first:** reject `FULL_ACCESS` at the service boundary and remove all direct-autonomy execution paths. Retain global STOP. Replace approval/loop execution with user-run lifecycle and audit. Remove dead decision/loop code after callers migrate. | Rewrite `dtn/backend/tests/test_autonomy.py`: preserve auth, STOP, persistence, and validation; replace FULL_ACCESS, auto-loop, and SLM-approval expectations with refusal and user-run lifecycle assertions. |
| `dtn/backend/app/routers/autonomy.py` | Exposes mode mutation, dispatch, pending-decision approval/edit/reject, loop controls, and STOP. | **Disable first:** keep STOP and read-only state/history; reject/remove mode, generic dispatch, decision approval/edit/reject, and loop-start/iteration routes from the product API. Add authenticated run-start/confirm/status routes only for UI-initiated module work. | Update `dtn/backend/tests/test_api.py` (8 current tests) and `dtn/backend/tests/test_autonomy.py` (18) to assert removed endpoints fail closed (404/405 or explicit 409) and STOP stays global, durable, and effective. |
| `dtn/backend/app/routers/slm.py` and `dtn/backend/app/services/feedback.py` | `/chat`, `/command`, and `/suggest-next-action` can produce schema actions, dispatch them, or turn feedback into an executable suggestion. | **Disable first:** make chat an RAG question-answer route and recommendation route only. Remove command/feedback dispatch and action-schema validation from the SLM path. Keep deterministic retrieval and response verification in backend code. | Rewrite `dtn/backend/tests/test_slm.py` and feedback tests to assert no dispatch is possible, answers cite retrieved evidence, recommendations obey the fixed schema, and unsupported claims are rejected. |
| `common/slm_action_schema.json` | Allow-listed action contract includes state-changing actions and suggestion/clarification actions. | **Remove from SLM product flow:** replace with strict answer and recommendation response schemas. A user-run API gets a separate, narrow request schema; it is not exposed to model output. | Schema contract tests for exact fields, allowed modules, no extra module, and no execution-shaped `next_action`. |
| `dtn/backend/app/db/models.py`, migrations `0002_autonomy_control.py` and `0003_full_access_execution_audit.py` | `autonomy_state`, pending decisions/history, loops, and an audit table whose `mode` only records `FULL_ACCESS`; current bearer token does not identify a human actor. | **Keep and extend audit history; phase out autonomy tables later.** Add actor identity, module, parameters/config hash, request/confirmation/start/end timestamps, status, result/run ID, and STOP/preflight/confirmation facts. Do not erase historical audit rows. Add a schema migration; retain STOP persistence. | Migration and API cases in `dtn/backend/tests/test_autonomy.py` and `test_api.py` cover old records and new required audit fields. Auth tests must prove a real identity is recorded instead of a shared token or invented name. |
| `dtn/backend/app/core/security.py` | Authenticates a shared API bearer token, not an individual user. | **Extend before user-attributed audit is claimed:** establish authenticated user identity (or a trusted identity-provider subject) and propagate it through UI/API. Until then, “who” is unavailable and hardware run start must be blocked or explicitly attributed through an approved identity mechanism. | Auth integration tests for identity propagation, missing identity, and authorization to confirm/start. |
| `dtn/frontend/src/api/client.ts`, `dtn/frontend/src/pages/DatasetsPage.tsx` | Frontend models autonomy mode, sends dispatch requests, and renders pending-decision approval/edit/reject flows. | **Disable first:** remove the mode switch and hide all auto-mode controls. Replace action dispatch/approval UX with module-owned start buttons, preflight status, long-run confirmation, live status, and audit result. Keep STOP visible and global. | Update `DatasetsPage.test.tsx` (7 current tests) for no mode switch/approval, explicit user start, idle preflight, confirmation, STOP, and audit result; update API-client contract tests. |
| `dtn/frontend/src/pages/AssistantPage.tsx` | SLM page promises bounded validated actions and currently displays action results. | **Keep the chat page, change its role:** RAG answers and recommendation card only. Remove action-result rendering and action-oriented suggestions. Add evidence links, provenance/split labels, and “open module” navigation. | Update `AssistantPage.test.tsx`; assert chat cannot start work and unsupported facts become “insufficient evidence.” |
| Topology/APG module UI | APG specification and a topology-panel test exist; no standalone `*Topology*.tsx` or `*APG*.tsx` page was found in the frontend source inventory. | **Keep/complete the user-driven APG module UI:** retain the simulation button, node map, and verdict box if present in the in-progress UI; replace “pending decision” language with a confirmation dialog. Add a distinct hardware-start path with preflight and long-run confirmation. The UI implementation location is not yet established. | Inspect the actual in-progress topology component before implementation; add UI tests for simulation vs hardware, confirmation, node map, verdict, STOP, and no SLM-triggered start. |
| `scripts/run_mnist_full_access.py`, `scripts/m5_feedback_loop.py` | Bounded scripts switch to `FULL_ACCESS`, start loops, or run feedback dispatch. | **Disable from the product flow immediately.** Preserve only as archived/manual historical acceptance evidence if required; remove or rewrite later so they cannot enable FULL_ACCESS or start work without UI-originated identity and confirmation. | Remove hardware execution from automated test paths. `tests/test_mnist_execution_history.py` has 1 current test for the legacy driver/controller history shape; retain only if legacy history must remain readable, otherwise replace it with user-run audit coverage. |
| Other autonomy-related scripts | The source search also finds `scripts/check_backend_listener.py`, `demo_e2e.py`, `deploy_master_nas.py`, `diagnose_dataset_nas.py`, `inspect_task_k_failure.py`, `m4_hardware_check.py`, `m5_preflight.py`, `m7_backend_restart_state.py`, `m7_cleanup_failed_run.py`, `m7_recovery_status.py`, `m7_worker_failure.py`, `m7_worker_reboot.py`, `mnist_dataset_acceptance.py`, `mnist_demo2_telemetry.py`, `mnist_task_k_acceptance.py`, `prepare_backend_restart.py`, `recover_task_k_nas.py`, `report_m5_feedback_evidence.py`, `restore_task_k_idle.py`, `retry_mnist_nas_telemetry.py`, `slm_dataset.py`, `summarize_task_k_results.py`, `upload_preprocess_cifar10k.py`, `validate_chatgpt_batch.py`, `verify_backend_restart.py`, and `verify_backend_stability.py`. These cover state inspection, preflight, recovery, deployment, data preparation, acceptance, and results reporting. | **Keep read-only diagnostics/reports; disable hardware/action-running and mode-switching tools from the product path.** Operator-only recovery/deployment tools require explicit classification and the preflight/identity/confirmation/audit contract; do not make them callable by the SLM or UI. APG research drivers remain outside the product flow. | Audit script entry points individually; tests cover any retained backend calls and assert product/API/model responses cannot invoke these scripts. No script is to be executed as part of this design step. |
| `README.md`, `docs/phase3-slm-master.md`, `docs/phase8-implementation.md`, `docs/results-report.md`, `docs/e1-422-diagnosis.md`, `docs/slm-dataset-card.md`, autonomy and acceptance docs | Describe ASK_HUMAN/FULL_ACCESS, legacy command/suggestion actions, bounded acceptance, or automatic execution events. | **Update after implementation:** mark measured historical results and the old SLM dataset as historical; remove operational advice to enter FULL_ACCESS; document UI-only starts, STOP, preflight, confirmation, actor audit, RAG boundaries, and unsupported modules. Keep old evidence but clearly label obsolete behavior. | Documentation searches/review ensure no active instruction suggests enabling auto mode or treats historical automatic actions as the new workflow. |

### Test-count baseline and expected scope

The last recorded suites (not rerun for this plan) were **173 root Python tests, 79 backend tests, and 23 frontend tests**, all passing at that time. A source-level count of directly affected test declarations is **71 across 7 files**: `dtn/backend/tests/test_autonomy.py` (18), `test_slm.py` (18), `test_api.py` (8), `test_datasets.py` (17), `tests/test_mnist_execution_history.py` (1), `dtn/frontend/src/pages/DatasetsPage.test.tsx` (7), and `AssistantPage.test.tsx` (2). The whole frontend suite may need adjustments if navigation/client contracts change. These counts come from source declarations, not test execution; they are tests to update/replace, not a promise that the same total will remain after retiring loop/decision behavior. This plan did not run tests or check whether any matrix process is active.

## B. SLM RAG design

### Retrieval map

Use the authenticated backend as the only retriever. The model receives a compact, typed evidence packet with source identifiers and timestamps, not raw database access.

| Question | Authoritative retrieval |
|---|---|
| “Are nodes online / how many active nodes?” | `devices` and recent `device_status_history` rows from the DTN database. Define “active” in code as a registered node with a fresh successful heartbeat inside a documented freshness window; report the as-of time, the window, and stale/offline counts. Do not count configured nodes as active. |
| “Was training done / what is running?” | Current master run/group state and persisted run history returned by the master results/status API; join to DTN run/audit records by stable run ID. Retrieve status timestamps and terminal result status. If the master does not persist a run state or a matching ID, say “insufficient evidence.” |
| “How were results?” | Persisted per-round run metrics and saved predictions, keyed by run ID and split. Only calculate accuracy, loss, macro metrics, confusion matrix, and round deltas in deterministic backend code. Include data setting, topology, seed, model/config identifier, completion status, provenance, and explicit `VALIDATION` or `TEST`. |
| “How did APG/topology learning perform?” | APG per-run inputs, selected edges, per-round metrics, and the matching fixed-topology comparisons from saved run outputs. `results_apg` is research/simulation evidence and must be labeled `SIMULATED`; hybrid-shaped emulation is `HYBRID` only when that is genuinely the artifact provenance. It is not hardware evidence. |
| “What did the user start and what happened?” | User-run audit records, module run records, current global STOP state, preflight record, and the module result keyed by run ID. Existing autonomy decision/execution history is legacy evidence only and does not establish a human identity. |
| “What should happen next?” | The typed, code-computed state vector described in section C plus source records above. Missing/expired features make the relevant recommendation unavailable. |

### Numeric and claim integrity

Backend code computes every number before prompting: active-node count; accuracy/loss and their deltas; round counts; node variance; APG reward; latency; training-stability signals; and any aggregate metrics. Every metric carries its value, unit, source/run ID, timestamp, provenance (`SIMULATED`, `HYBRID`, or `HARDWARE`), and split label where applicable. The SLM may explain those values but may not calculate new values or substitute one split for another.

Use a structured answer contract with claims linked to evidence IDs. A verifier compares every numeric claim and metric label in the proposed answer to retrieved values (including unit, rounding tolerance, provenance, and split). Reject an answer if any number is absent, mismatched, stale, unlabeled, or refers to TEST where only VALIDATION evidence was requested. Do not silently “repair” it by guessing; return a deterministic “insufficient evidence” response with the missing field/source. Answers must label `VALIDATION` and `TEST` separately, never say only “accuracy” when the split is known.

## C. Recommendation contract

Only these module values are allowed:

1. `Topology Learning (APG)`
2. `NAS (current dataset NAS)`
3. `Continue Training`
4. `Insufficient Evidence`

`Node Selection` and `Adaptive Aggregation` are not implemented and must never be suggested.

The backend builds the input state, with source and freshness metadata, and the SLM returns JSON of this shape:

```json
{
  "recommended_modules": [
    {
      "module": "Topology Learning (APG)",
      "priority": 1,
      "confidence": 0.0,
      "evidence": [
        {
          "evidence_id": "run-or-metric-id",
          "claim": "short, source-grounded explanation"
        }
      ]
    }
  ],
  "reason": "concise rationale grounded in the evidence list",
  "next_action": {
    "type": "open_module",
    "module": "Topology Learning (APG)"
  }
}
```

`next_action` is navigation only. It contains no run parameters, command, approval, or dispatch request. The UI opens the named module page; the user starts work there.

Input features are `active_nodes`, `accuracy`, `accuracy_delta_last_5_rounds`, `loss`, `plateau_flag`, `node_variance`, `topology_reward`, `communication_latency`, `training_stability`, and an explicit split label (`VALIDATION` or a separately requested operational status). Every feature includes provenance, timestamp, and source ID. TEST values are prohibited as recommendation/tuning inputs. If a necessary feature is absent or not fresh, the recommender must return `Insufficient Evidence` or omit a lower-priority option with an explicit reason; it must not infer the missing value.

Define numerical thresholds and missing-data behavior in deterministic code and version that policy. The model ranks only the eligible module options and explains the code-computed evidence; it does not invent thresholds or execute anything.

## D. Fine-tuning data and evaluation plan

### Honest labeled examples

- **APG vs Ring/Mesh:** The saved Phase B matrix has 2 data settings × 3 seeds. That gives **6 independent setting/seed blocks** and up to **12 paired comparisons** (APG-vs-Ring and APG-vs-Mesh per block). Two simulation-mode records have identical outcomes and are not extra independent examples. These examples are simulation evidence, not hardware outcomes. Round-level points within a run are correlated and must not be counted as independent labeled runs.
- **NAS vs fixed architecture:** Existing NAS/baseline history may be usable only after joining completed runs to the same dataset, split, seed, and evaluation protocol. The honest usable count is **not established by the current inventory**; count after artifact/provenance audit. Missing matched baselines are not examples.
- **Continue-training curves:** Use complete persisted round curves with their run IDs, dataset/split labels, and stopping outcomes. The honest usable count is **not established until run-history completeness and provenance are audited**. Individual rounds from one run do not become independent examples.
- **Hardware examples:** No hardware APG-vs-baseline outcome is established by the saved APG matrix. Do not label simulated or hybrid-shaped runs as measured hardware. Collect user-confirmed, audited hardware outcomes prospectively.
- **Synthetic examples:** Any generated prompts, scenarios, labels, or simulated outcomes must carry an unmistakable `SYNTHETIC` label and be reported separately. Never present them as measured results or combine them silently with real data.

Partition by whole run (and by dataset/session where relevant), not by example or round, into train/validation/test. Keep all seeds/rounds/prompt variants derived from a run in one partition. Deduplicate identical simulation-mode artifacts before counting examples. Prevent TEST metric leakage into training, threshold tuning, or recommendation policy selection.

### Evaluation and deployment gate

Train a LoRA adapter on the PC. On a held-out test set of runs, report recommendation accuracy, per-module precision/recall/F1, confusion matrix, and sample counts/support per module. Separately test evidence faithfulness: feed known retrieved values, perturb/remove values, and verify answers neither alter numbers nor omit the `VALIDATION`/`TEST` and provenance labels; unsupported claims must be rejected into “insufficient evidence.” Include abstention/insufficient-evidence quality and calibration, not just classification accuracy.

Compare the candidate to the **base Llama 3.2 1B** using the same held-out runs, exact prompts, retrieved evidence, decoding settings, schema validator, and scoring procedure. Report paired per-run deltas and failures; do not tune on the test set. Preserve a base-model fallback.

Package the evaluated adapter and prompt/policy version on the PC, calculate SHA-256, transfer to the master Jetson through the approved deployment path, and verify the hash before loading. The master continues serving the base model until the candidate passes the held-out evaluation and an explicit deployment gate. Rollback must restore the base model without deleting audit/results records.

## E. UI changes

- **SLM chat panel:** retain conversational RAG; show answer evidence, as-of time, and `SIMULATED` / `HYBRID` / `HARDWARE` provenance. Render split labels for every metric. Add an `Insufficient Evidence` state rather than a confident unsupported answer.
- **Active nodes card:** show active count, node names/status, freshness window, and last update. Compute count in backend code; stale status is visible, not silently counted active.
- **Recommendation card:** show allowed module, priority/confidence, evidence links, missing/stale inputs, and a button that opens that module. The card does not start the module. Avoid presenting a recommendation as approval.
- **Module start UX:** user presses the relevant module button. Before any hardware run, show per-node idle/preflight result and block if any node is not idle or status is stale. Ask for explicit confirmation for long runs (including the ~59-minute sweep), including configuration and estimated duration. Simulation and hardware runs must be clearly distinct.
- **Global STOP:** persistent and available from all relevant screens; one global stop request immediately blocks new starts and signals active work to stop. Reset is a separate authenticated user action and audited.
- **Audit trail:** list actor, module, config, confirmation, preflight, timestamps, run ID, status/result, and provenance. Show failed/cancelled/stopped runs as outcomes, not as success.
- **APG UI:** retain simulation button, node map, and verdict box from the earlier design. Replace “pending decision” wording with the explicit user confirmation dialog. Clearly label simulation verdicts; they cannot claim measured hardware benefit. The current component location and completeness need confirmation before implementation.

## F. Order of work, risks, unavailable items, and questions

### Proposed implementation order

1. **Freeze the boundary:** enumerate API routes, schema actions, scripts, frontend controls, and deployment entry points. Add a backend test proving no SLM/model response can cause a state change. Disable `FULL_ACCESS` and generic SLM dispatch server-side before hiding the UI switch.
2. **Preserve safety:** keep global STOP; define node-idle/freshness contract, preflight response, explicit confirmation token/expiry, and STOP behavior during a run. Require authenticated actor identity.
3. **Run lifecycle and audit:** introduce UI-originated module run records, confirmation/preflight checks, per-run status/result persistence, and migration/backfill behavior. Never infer an actor for legacy events.
4. **RAG and verifier:** implement typed retrieval, code-computed features, evidence IDs, provenance/split labels, numeric verification, and deterministic abstention. Add adversarial/freshness tests.
5. **Recommendation and UI:** add strict JSON contract and module allow-list; deliver chat, active-nodes, recommendation, module navigation, and user-driven APG/simulation/hardware confirmations.
6. **Legacy removal:** after product callers and tests are migrated, remove dead autonomy modes, generic dispatch, auto loops, old decision approval UI, and unused schema/migration code as a deliberate compatibility change. Preserve old audit history and any required read-only diagnostics.
7. **Fine-tune readiness:** audit artifacts and count independent run examples; establish run-level splits and base-model baseline; train/evaluate LoRA on PC; deploy only after hash and evaluation gates pass.
8. **Release verification:** backend/frontend tests, API contract tests, migration tests on a copy, UI flows with fake nodes, STOP/preflight/confirmation failure tests, and security review. Hardware release is a separate explicitly authorized step; no live hardware validation is implied by this plan.

### Risks and mitigations

- **Shared-token identity gap:** current bearer auth does not identify the human. Do not claim actor-level audit or permit hardware run start until identity is trustworthy.
- **Status freshness and idle semantics:** stale heartbeats or incomplete worker state can create false “idle” preflight. Define freshness, require all expected nodes, and fail closed.
- **Evidence leakage/hallucinated metrics:** bind each claim to evidence and verify numbers, units, split, timestamp, and provenance in code; abstain on mismatch.
- **Metric leakage:** keep TEST out of recommendations and training; partition fine-tuning/evaluation by run.
- **Small and correlated data:** the APG set has only six independent setting/seed blocks and simulated provenance. Treat fine-tuned results as exploratory; prefer base-model comparison and measured prospective data.
- **Legacy scripts and routes:** hiding a UI control is insufficient. Server-side refusal and script inventory are required before the UI change is considered safe.
- **STOP semantics:** stopping new work and stopping an already active node operation are distinct; define both and audit the result.
- **Long-run confirmation:** a confirmation that can be replayed or refers to stale config is unsafe; bind it to actor, run configuration, expiry, preflight snapshot, and one-time use.
- **Model deployment:** verify artifact hash, preserve base model, require held-out pass and rollback path. A passing schema check alone is not model acceptance.

### Not available / not established

- Human identity is not available from the current shared bearer-token authentication.
- A canonical DB relation joining all master runs, datasets, validation/test predictions, topology, provenance, and audit events has not been established.
- A reliable, persisted active-training status and all-nodes-idle/preflight contract has not been established for every node and run type.
- Fresh per-node validation metrics, node variance, communication latency, training stability, and topology reward are not guaranteed to exist as current code-computed features.
- The exact number of matched NAS-vs-fixed-architecture and continue-training labeled runs has not been audited.
- Hardware APG performance evidence is not present in the saved APG matrix; that matrix is simulated/hybrid-shaped.
- The exact APG UI component/page to retain was not located in the current frontend source inventory.
- A labeled set large enough to train and validate a reliable recommender is not yet established. Synthetic examples cannot close this gap while being described as real.
- Per-user authorization roles, long-run time estimates, confirmation expiry, and STOP/reset ownership need product/security decisions.

### Questions to resolve before implementation

1. Which identity provider or login will replace the shared-token-only identity for per-user audit?
2. What heartbeat age and complete-node set define “active” and “all nodes idle”? What is the fail-closed behavior when a node cannot be reached?
3. Which module operations are permitted to start hardware runs, and which runs require a confirmation threshold beyond the ~59-minute sweep?
4. Should legacy pending decisions/loops be retained read-only for historical audit, and for how long?
5. Which authoritative master data store/API will become the canonical run, round-metric, prediction, split, and provenance source?
6. What deterministic plateau, node-variance, topology-reward, latency, and stability definitions/thresholds should the recommender use?
7. What held-out evidence/faithfulness pass criteria are required before LoRA can replace the base model, and who approves deployment?
8. Where is the in-progress APG UI located, and is the earlier UI mockup the source of truth for the node map and verdict behavior?

**Stop point:** this is a plan only. No code, tests, process state, matrix run, database/data directory, or hardware was accessed or changed by this step.
