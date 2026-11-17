# Phase 8 and Phase 9 implementation log

This report is updated as each implementation milestone passes. Test fixtures are identified as fixtures; no fixture measurements are presented as hardware or user-data results. No credentials or token values are recorded.

> **Current dataset policy:** The upload workflow documented below is historical. Dataset uploads remain disabled; the coordinator catalog exposes only prepared official MNIST and CIFAR-10. Run `scripts/prepare_mnist_dataset.py` and `scripts/prepare_cifar10_dataset.py` to prepare them. The Datasets page can preprocess either dataset on the PC and ship verified shards to workers while preserving each official test split.

> **Current training results:** New experiment groups are persisted in SQLite by the master training coordinator; an existing JSON results file is imported once and retained. The MNIST experiment page displays worker-local validation metrics (including precision, recall, F1, and confusion matrices), resource/peer-wait/transfer measurements, and a downloadable accuracy chart. These are validation results, not held-out test-set measurements.

## M1 - Autonomy switch - PASS

Implemented authenticated, backend-enforced `ASK_HUMAN` (default) and `FULL_ACCESS` modes with persistent SQLite state. State-changing actions are schema-validated against `common/slm_action_schema.json`; in ASK_HUMAN they create expiring decisions with reasons and evidence, and in FULL_ACCESS they execute through the master command API. Read-only actions continue immediately in either mode.

Added approve, reject, schema-revalidated edit, decision history, explicit mode change, global STOP/reset, and persistent loop controls. Loops persist their iteration, deadline, max-iteration count, stop rule, and failure state. Maximum iterations default to 10; supported stop rules are maximum iterations, target metric, and two consecutive non-improvements. Worker failure pauses a loop. A backend restart reloads decisions, mode, and loop records from SQLite.

The PC chat route now calls an authenticated, side-effect-free master `/interpret` endpoint and only then submits its validated action to backend policy. This prevents master chat from executing a state-changing SLM action before ASK_HUMAN policy runs. The master model selection is configurable with `SLM_MODEL_NAME`, defaulting to `llama3.2:1b`.

Validation results: **121 passed** in `python -m pytest --import-mode=importlib -q tests master\tests`; **32 passed** in the M1 backend suite; the clean two-revision Alembic chain completed and created all four autonomy tables. Python compile validation passed. The focused M1 tests initially found test-request and endpoint expectation issues; those were corrected before acceptance. No M1 hardware test was required or run.

## M2 - Dataset flow - PASS

Added authenticated PC-backend image dataset endpoints:

- `POST /api/datasets/upload` accepts a ZIP archive with class subfolders.
- `POST /api/datasets/upload-folder` accepts individual files with relative class-folder paths.
- `GET /api/datasets` and `GET /api/datasets/{dataset_name}` expose stored metadata.

Names are required explicitly and must match `^[A-Za-z0-9][A-Za-z0-9._-]*$`, at most 64 characters. Archives are staged under `data/datasets/` and atomically moved to `data/datasets/<name>/` after validation; metadata is stored as `metadata.json` alongside the class folders. Existing names are not overwritten. ZIP entries are checked for traversal, absolute paths, Windows drive paths, symbolic links, encryption, oversized expansion, and invalid image content. A one-folder archive wrapper is supported. Image metadata records class names/counts, observed width-by-height counts per class, total images, expanded image bytes, archive SHA-256, and creation time. The configured limits are 256 MiB for upload, 1 GiB expanded image bytes, 50,000 archive entries, and 100 million pixels per image.

The `upload_dataset` action resolves metadata only for an already uploaded dataset. If the SLM proposes a dataset name not present in the upload store, chat returns a clarification asking for the user-named dataset upload; no dataset is inferred or defaulted. Dataset names remain required by the action schema. No CIFAR dataset is used as a default.

Validation results: **121 passed** in the required repository/master suite; **46 passed** in the backend suite, including generated-image fixture tests for folder and ZIP uploads, required-name and name-pattern validation, duplicate names, corrupt image data, path traversal/absolute/drive paths, symlink rejection, archive/expanded-size limits, metadata retrieval, and the upload-action name flow. `compileall` passed. A scan of the backend and master Python code found no CIFAR-10 default references. Fixture images exercise implementation behavior only; no real user dataset has been uploaded and no hardware result is claimed for M2.

## M3 - Pre-processing library v1 - PASS

Implemented typed image validation, resize, `standard`/`minmax`/`none` normalization, seeded stratified train/validation/test splitting, stratified and Dirichlet worker sharding, and class-balance reporting. The approved `preprocess_dataset` action now checks each worker is idle, uploads its shard through the authenticated training API, then reads worker metadata back and verifies counts and SHA-256. Worker dataset storage defaults to `/var/lib/dfl-slm-dtn/datasets`; a narrowly scoped systemd drop-in adds only that directory to `ReadWritePaths`. The install script provisions the same owner, mode, drop-in, and configurable dataset root for fresh installs.

For the real M3 hardware check, the local official CIFAR-10 archive was converted to 2,000 class-folder PNGs (200 per class) and uploaded under the explicit name `cifar_subset_demo`. The approved action produced 32x32 images with standard normalization and a seeded stratified split:

| Split | Rows | Per-class rows |
|---|---:|---:|
| Train | 1,600 | 160 |
| Validation | 200 | 20 |
| Test | 200 | 20 |

| Worker | Train rows | Per-class rows | PC SHA-256 | Jetson SHA-256 |
|---|---:|---:|---|---|
| worker1 | 540 | 54 | `b3e690adca01deb7400119c20e825e07fe13574d1116bc838258d66ede6df4f1` | `b3e690adca01deb7400119c20e825e07fe13574d1116bc838258d66ede6df4f1` |
| worker2 | 530 | 53 | `9d24ac9f228a16e661558b08b439322a95b5a7f16ee32201e34d36f1d693f5e0` | `9d24ac9f228a16e661558b08b439322a95b5a7f16ee32201e34d36f1d693f5e0` |
| worker3 | 530 | 53 | `1faa62abf4e2a373035ef983573091e8a5d624addcbc87c06d8ede2e69fedb3e` | `1faa62abf4e2a373035ef983573091e8a5d624addcbc87c06d8ede2e69fedb3e` |

The worker row total is 1,600, matching the training split. Reconstructed deterministic shard indices were pairwise disjoint and covered the complete training split; worker class counts sum to 160 for each source class. All three authenticated status endpoints were idle before installation, and each actual endpoint write probe installed and read back 4 rows (2 per probe class) before only its `writetest_probe` directory was removed. Services remained active after installation.

Validation after M3: **123 passed** in `python -m pytest --import-mode=importlib -q tests master\tests`; **53 passed** in `python -m pytest --import-mode=importlib -q dtn\backend\tests`. The backend run reports one Starlette/httpx deprecation warning. The three Jetson `worker/training_service.py` files and the dataset-root drop-ins were updated one worker at a time with backups.

## M4 - Dataset-derived initial model and NAS - PASS

Implemented dataset-derived deterministic CNN initialization from the uploaded dataset metadata and preprocessing dimensions. The architecture and NPZ weights are saved under the dataset's PC-side model directory, constrained to 12,000 parameters, sent to idle workers sequentially, and verified by architecture and weight SHA-256. Worker installation validates the archive, architecture fields, weight names/shapes, finite values, and parameter count; repeated identical installs are idempotent while different installs are rejected.

The NAS action now requires an explicit dataset name to run. The master coordinator sends that name to each worker; each worker evaluates the matching installed `train.npz` shard, carries its own class names and image shape, and does not fall back to the legacy CIFAR shard. Missing dataset names return a clarification result. The CIFAR-specific Phase 6 functions remain separate for existing experiment history and are not used by this dataset NAS action.

Local validation: **42 passed** across the focused NAS, worker API, master coordinator, and master action tests. The full required suites then passed: **130 passed** in `python -m pytest --import-mode=importlib -q tests master\tests`; **56 passed** in `dtn\backend\tests` (one Starlette/httpx deprecation warning). These tests use generated shards and fake worker/search responses where appropriate.

Hardware update (2026-10-28): the first minimum-budget dataset NAS approval failed because the master was running older code. The backend history recorded `422: The SLM service rejected the request`; the last 60 lines of the master journal did not include that request. PC/master hashes proved that `master/app.py`, `master/training.py`, `common/nas.py`, and `common/slm_action_schema.json` were stale. The master was deployed using `deploy/deploy_master_slm.ps1`, with timestamped backups, and the topology was restored to `bidirectional_ring` through ASK_HUMAN approval. `/interpret` returned HTTP 200, and all production source/data hashes installed by the deployment wrapper matched the PC.

The real ASK_HUMAN `create_initial_model` decision `a6d564bf-eb07-4b54-a4ea-8be2bfd30fc9` installed the model for `cifar_subset_demo`. PC and all three workers reported architecture SHA-256 `56c833efbf361003338143ad5547b7231b14db1904da2178a43c31587914ef9a`, weight SHA-256 `cc7f678dc5995da6d27c31749706d0206b0259c4a46558885ef2d731f3bb9a83`, and 898 parameters.

The first approved dataset NAS group `dataset-nas-1793171089-3bcbb3` completed in 23.039 seconds on the 2,000-image `cifar_subset_demo`, but validation scores were only 7.27%-14.55% across candidates. Each worker trained on 420-430 samples for one epoch/four batches and validated on 110 samples (11 per class); Phase 6 uses 13,330 training and 3,330 validation samples per worker, one epoch, and about 105 training batches. Worker validation-data hashes all differed; workers 2 and 3 happened to use the same local index positions, but their data hashes differed. The identical 12.73% winning score was 14/110 on each worker and does not indicate identical validation data. The subset sample count/steps were too small for a meaningful architecture ranking.

The approval helper was fixed to unwrap the real controller/backend/master envelope and tested with success and failure fixtures. A terminal-state test confirms sampler cleanup waits for the NAS group to become terminal. The rerun `dataset-nas-1793171517-09d364` completed through ASK_HUMAN in 20.792 seconds, selecting `b2_c16_k5` (7,802 parameters). Its candidate accuracies were:

| Candidate | worker1 | worker2 | worker3 |
|---|---:|---:|---:|
| `b1_c16_k3` | 12.73% | 9.09% | 12.73% |
| `b1_c16_k5` | 8.18% | 11.82% | 10.91% |
| `b1_c8_k3` | 10.00% | 10.00% | 10.00% |
| `b1_c8_k5` | 10.00% | 10.91% | 10.00% |
| `b2_c16_k3` | 8.18% | 7.27% | 14.55% |
| `b2_c16_k5` | 12.73% | 12.73% | 12.73% |
| `b2_c8_k3` | 10.91% | 10.00% | 10.00% |
| `b2_c8_k5` | 10.00% | 10.00% | 10.00% |

Full-interval tegrastats peaks for that rerun: master 2,196 MB RAM / 813 MB swap (38 samples), worker1 5,263 MB / 513 MB (38), worker2 5,352 MB / 64 MB (38), and worker3 5,244 MB / 57 MB (38). This confirms the small-subset architecture ranking is not meaningful.

Per the requested larger-data follow-up, the real `cifar_10k_demo` upload contained 1,000 images per class. ASK_HUMAN upload and preprocessing decisions `2164b7c6-092b-46c6-9ac2-8abc8adf86dc` and `671457b7-9c87-4c69-92a1-a3fef92a0f0b` completed the PC preprocessing and worker shard installation. The split was 8,000 train / 1,000 validation / 1,000 test. Worker training shards were 2,670 / 2,670 / 2,660 rows, with per-class counts 267 / 267 / 266, respectively; rows summed to 8,000 and row hashes showed no pairwise overlap. PC and Jetson SHA-256 values matched:

| Worker | Rows | PC and Jetson shard SHA-256 |
|---|---:|---|
| worker1 | 2,670 | `2dd444e3abdfa5abb13ddfb86ddbb08d325417a54db05bf1bd81600c804f5317` |
| worker2 | 2,670 | `f4059ce9fec8e513fe07eb2b03ca3b43893d204779a0ed82efe1e675f2b889a5` |
| worker3 | 2,660 | `43eb82f969a6c415ac9eee9a75c46b7ef392c92b6acc10062010626b706c0b76` |

The one approved 300-second-budget dataset NAS on this larger dataset (decision `53e303a5-4ff0-4c58-a168-b688ba7f28b6`, group `dataset-nas-1793171849-24a747`) completed in 23.95 seconds. Each worker trained on 2,130-2,140 samples for one epoch/17 batches and validated on 530 samples. The selected candidate was `b2_c16_k5` (7,802 parameters), scoring 21.70% / 20.75% / 17.17% on worker1/2/3 (mean 19.87%, versus 10% random-class chance). Candidate scores varied from 9.81%-21.70%; the selected candidate led on all workers. The best validation accuracy is below the illustrative 25% threshold, so this is evidence above chance with candidate differentiation, but remains a one-epoch preliminary architecture signal, not a fully trained performance claim.

Full-interval tegrastats peaks for the larger search: master 2,196 MB RAM / 813 MB swap (37 samples), worker1 5,427 MB / 513 MB (37), worker2 5,376 MB / 122 MB (37), and worker3 5,478 MB / 57 MB (38). Worker NAS subprocess peak RSS was 1,629.5 MB, below its 2,048 MB cap. The controller and all workers returned idle after completion. **M4 - PASS**: deterministic initial-model hashes matched on all three workers (898 parameters), dataset-specific NAS completed through ASK_HUMAN for both the original and larger named datasets, real candidate metrics and measured device telemetry were recorded, and the larger dataset provides above-chance validation with distinct candidate scores. The result is preliminary because each candidate is trained for one epoch.

## M5 - Feedback-driven training loops - PASS

The real acceptance used `cifar_10k_demo` with `bidirectional_ring`, 3 rounds x 2 epochs per run, in ASK_HUMAN and FULL_ACCESS. The 10k subset results are not comparable with Phase 4-6.

| Mode / iteration | Accuracy | Macro-F1 | Communication | Duration | Peak RAM / swap (MB): master; worker1; worker2; worker3 |
|---|---:|---:|---:|---:|---|
| ASK_HUMAN 1 | 0.171698 | 0.104211 | 37,024 B | 5.11 s | 2208/813; 5488/513; 5428/74; 5462/66 |
| ASK_HUMAN 2 | 0.171698 | 0.104211 | 37,046 B | 5.05 s | 3627/813; 5495/513; 5454/74; 5474/84 |
| FULL_ACCESS 1 | 0.171698 | 0.104211 | 37,044 B | 7.52 s | 3429/813; 5495/513; 5458/74; 5414/84 |
| FULL_ACCESS 2 | 0.171698 | 0.104211 | 37,056 B | 5.38 s | 3429/813; 5550/513; 5533/136; 5452/84 |

Both real SLM feedback suggestions in the completed acceptance were schema-valid (**2/2**); deterministic fallback was not used. The persisted ASK_HUMAN decision `19ae45ba-19e3-4573-8e3f-e34e47493232` records `source=slm`, passed validation before dispatch, and has `created`/`approved` history events. FULL_ACCESS suggestions execute directly and are not persisted as decision-history rows; the accepted runner output recorded the second valid SLM suggestion. Final count: **SLM valid 2 of 2; fallback used 0 times**. The fallback flag is OFF in the currently running backend. A real global STOP test halted an active loop, blocked a state-changing dispatch with HTTP 423 until reset, and left the backend in ASK_HUMAN with STOP inactive. Backend tests cover rejected suggestions, retry, and `needs_human`; the live acceptance did not produce an invalid suggestion.

## M6 - React UI - PASS

The production frontend build passed (`npm run build`), ESLint passed (`npm run lint`), and the frontend test suite passed (**11 tests across 6 files**). The repository had no lint command/configuration; ESLint was added with TypeScript and React Hooks rules. Two existing effects that initiate asynchronous API refreshes have narrowly scoped suppressions for the hooks rule that otherwise treats those asynchronous updates as synchronous effect-state updates.

## Remaining milestones

## M7 - Reboot and failure recovery - FAIL (Step 3)

The initial M7 preflight was **BLOCKED**, not a failed recovery test: port 8000 refused connections before any hardware action. The prior process exit cause cannot be established. The last 40 stderr lines show only the later process startup and Alembic initialization; stdout is empty; no shutdown/crash trace or process-specific exit event was available. PID `101548`, started at 17:36:55, was subsequently found listening after its parent PowerShell had exited. The detached launch was then verified at startup and again five minutes later with the same PID `103684`. A Scheduled Task launch attempt exited with task result 1 without producing backend startup logs; that task was removed rather than retried.

The hidden detached launch is available as `.\scripts\start_backend.ps1` from the repository root. It runs the documented `python -m app.main` from the repository root with `PYTHONPATH=dtn\backend`, redirecting stdout/stderr to the backend data logs and placing no secrets on the command line. Backend health, required routes, ASK_HUMAN, and STOP-off state passed verification. Multiple SQLite backups were taken before restarts and the test; see the M7 database backup files in the final report.

Step 0 persistence inspection: master, Ollama, and all three worker services reported `enabled`; worker services use `Restart=on-failure`, while master and Ollama use `Restart=always`. Dataset shards and the installed M4 model live under `/var/lib/dfl-slm-dtn/datasets` and were confirmed present on workers. Worker1's CIFAR 10k shard SHA-256 matched the PC before and after reboot. The installed model was for `cifar_subset_demo`, not `cifar_10k_demo`; its architecture and weight hashes matched the PC before and after reboot. Autonomy mode, decisions, loops, and global STOP are SQLite-backed. The master coordinator and worker topology fields are runtime state; topology must be restored through the autonomy action after master restart.

| Test | Injected failure | Expected | Measured result | Recovery time | Result |
|---|---|---|---|---:|---|
| Step 0 persistence | None | Report boot enablement and durable/runtime state | Units enabled; dataset/model files on disk; SQLite state durable; topology runtime-only | — | PASS |
| Step 1 worker crash | SIGKILL worker1 during real 3x2 `cifar_10k_demo` training | Group fails with reason; loop pauses and history records it; worker recovers; fresh run works | Group `bidirectional_ring-1793177538-c59e94` failed with worker1 connection refused; loop paused and `loop_paused` event recorded; subsequent 3x2 group `bidirectional_ring-1793177557-a51ea0` completed | 5.59 s automatic recovery; 2.03 s after explicit restart; fresh run 11.71 s | PASS |
| Step 2 backend restart | Restart only port-8000 backend with pending decision, paused loop, expired decision, and STOP active | Paused loop and STOP reload; expired decision is not auto-approved | Loop `c49795e2-6a7b-4b15-9d61-55e05a05e54c` remained PAUSED; decision `269f2b49-aa7b-431f-a673-d140f536c672` reloaded as EXPIRED with no approval event; STOP reloaded true, then reset | Health returned at launcher startup | PASS |
| Step 3 worker reboot | Reboot idle worker1 | SSH/service/authenticated idle recover; hashes persist; short run succeeds | SSH 62.44 s, service active 62.73 s, authenticated idle 62.78 s; shard and M4 architecture/weight hashes persisted. The required post-reboot 1x1 run failed: `TrainingError: Worker request failed: worker1: Could not reach worker1 for /v1/results?run_id=bidirectional_ring-seed-2026-c9d1447a: timed out`. All workers returned idle and the loop was paused during cleanup. | 62.78 s for worker recovery; training did not recover | FAIL |
| Step 4 master reboot | Not injected | Ollama/master and PC backend recover; topology restored | Not run after Step 3 failure | — | Not run |
| Step 5 security check | Not run | Unauthorized requests rejected; current-token scan passes | Not run after Step 3 failure | — | Not run |

M7 is **FAIL** because Step 3's post-reboot training acceptance failed. The master reboot and security re-check were not run; the unexpected result stopped the sequence. No M8 or M9 work was started.

M8 and M9 acceptance criteria recovered from the project handoff:

- **M7:** real reboot/failure-recovery tests, with the cluster idle before each test, one node at a time, and backups, recovery/restore, and service verification.
- **M8:** centralized baseline on the same split/test set and `b1_c16_k5` architecture, with equal total training (10 rounds x 5 epochs) across three seeds; append real measured results to the Phase 6 report.
- **M9:** run the full required tests and `scripts/demo_e2e.py`; document setup, services, ports, and modes in README; add a results report with figures from saved real runs and limitations.

## How to start the backend

From the repository root in PowerShell, start the backend with the documented import path:

```powershell
Set-Location 'D:\Lab\MS Research Project\dfl-slm-dtn'
$env:PYTHONPATH = 'dtn\backend'
python -m app.main
```

`Settings.from_env()` loads the root `.env` into the backend process. Keep the `.env` file out of command arguments and logs. Redirect stdout and stderr to `dtn\backend\data\backend.stdout.log` and `dtn\backend\data\backend.stderr.log` when running the process detached.

For a hidden detached process that survives the launching PowerShell command, run `.\scripts\start_backend.ps1` from the repository root after backing up the SQLite database. It launches the same documented Python module from the root with `PYTHONPATH=dtn\backend`; logs are redirected to the two backend log files. The launcher refuses to start if port 8000 already has a listener.

Feedback suggestions are checked against the shared action schema. An invalid
suggestion is logged and retried once; if the retry is also invalid, the API
returns `needs_human` without dispatching an action. Deterministic rule fallback
is disabled by default and can be explicitly enabled with
`DTN_FEEDBACK_RULE_FALLBACK=true`. When enabled, fallback decisions are recorded
with `source=rule_fallback` evidence. Training round and epoch limits are taken
from the shared schema; the M5 hardware runner itself uses 3 rounds and 2 epochs.

## TASK I - Real MNIST dataset step - INCOMPLETE (FAIL)

The local Keras cache contained the real MNIST archive (60,000 training and
10,000 test images; archive SHA-256
`731c5ac602752760c8e48fbffcf8c3b850d9dc2a2aedcf2cc48468fc17b673d1`).
The fixed-seed upload contained 300 training images per class (3,000 total)
as grayscale PNGs in class folders. The upload ZIP SHA-256 was
`d5a769b766c443265db844866769b4b6e2fce3273e9152a7607c3cdbb92de324`.

The real upload was listed and processed through one ASK_HUMAN decision,
approved via the API. Metadata and the persisted preprocessing report show
1-channel 28x28 standard-normalized images; stratified split sizes were
2,400/300/300 (train/validation/test), with 240/30/30 samples per class.
Worker training shards each contained 800 rows (80 per class), their shard
indices were disjoint, and the worker-reported SHA-256 matched each PC shard:

| Worker | Rows | PC and Jetson SHA-256 |
|---|---:|---|
| worker1 | 800 | `d7097eb3253cc026b49da7779c027ba3c15c249a850814ec6bab7dff037ebd56` |
| worker2 | 800 | `4e4c77d86f1fc9c077c4bf3db840c20e0cf23397500467f1beebdf7ecda0143e` |
| worker3 | 800 | `c646390ae90d099612aeeb8e1ae6da7133be7b5712380b3612eaccf59680e999` |

All four negative upload tests were rejected with explicit errors: ZIP path
traversal (422), corrupt PNG (422), invalid dataset name (422), and duplicate
`mnist_demo` (409). No rejected test dataset remained on the PC or workers.
Persisted preprocessing duration was 2.381 seconds. During shard shipping,
tegrastats reported peak RAM / used swap (MiB): master 2,207 / 813; worker1
2,141 / 12; worker2 2,464 / 137; worker3 2,492 / 84.

The acceptance runner then raised `KeyError: 'shipping_duration_seconds'`
while formatting its final report. The successful backend path does not add
that field to the persisted report; the runner's PC-memory sampler had
collected samples, but its peak was not emitted or persisted before the
exception. Therefore the required PC peak-RAM evidence is unavailable and
this dataset step is **INCOMPLETE / FAIL**, despite the upload, preprocessing,
and shard installation being present and verified. Do not rerun or overwrite
the existing `mnist_demo` dataset as part of this acceptance. No initial model,
NAS, or training was started. The runner was corrected to treat the absent
shipping-duration field as optional and passed `py_compile`; this does not
recover the lost PC peak-memory measurement or change the acceptance result.

### M7 Step 3 timeout - read-only diagnosis

The recorded master failure for group
`bidirectional_ring-1793177958-9444d2` was a timeout reaching worker1's
`/v1/results` endpoint for run
`bidirectional_ring-seed-2026-c9d1447a`. The master result request has a
10-second timeout. Worker1 later logged HTTP 200 for its results request and
peer-weight receipt, followed by cancellation and
“Training was cancelled before peer exchange.” This establishes the proximate
failure as the master's results poll timing out and the run being cancelled;
the available logs do not determine whether compute delay, scheduling, or
transient transport caused the delay.

Worker code deployed for this task was backed up before installation:

- worker1: `/opt/dfl-slm-dtn/worker/training_service.py.bak-20261028181946`,
  `/opt/dfl-slm-dtn/worker/nas_search.py.bak-20261028181946`
- worker2: `/opt/dfl-slm-dtn/worker/training_service.py.bak-20261028181955`,
  `/opt/dfl-slm-dtn/worker/nas_search.py.bak-20261028181955`
- worker3: `/opt/dfl-slm-dtn/worker/training_service.py.bak-20261028182008`,
  `/opt/dfl-slm-dtn/worker/nas_search.py.bak-20261028182008`

## TASK J - Datasets frontend

Added a Datasets page that calls only authenticated PC-backend API routes. It
supports ZIP upload, metadata and class balance, preprocessing options from the
shared action schema, ASK_HUMAN approval/rejection/edit, FULL_ACCESS completion,
global STOP messaging/disabled controls, and PC-vs-worker shard SHA-256 display.
The backend adds one authenticated latest-preprocessing-report GET endpoint;
its response includes PC shard hashes computed from the persisted worker files.
The displayed 80/10/10 split is informational only and is not sent as an action
parameter.

Frontend checks: **17 tests passed**, lint passed, production build passed.
Backend suite: **76 tests passed**, with one existing Starlette/httpx deprecation
warning. A manual browser click-through remains for the user.

## MNIST follow-up - TASK K

### Backend lifecycle and stability

The 2026-10-29 read-only lifecycle investigation found backend startup lines in
`backend.stderr.log`, no Python traceback, and no `app.main` process. The Windows
System log contained User32 event 1074: `winlogon.exe` initiated a PC power-off
on behalf of `NT AUTHORITY\SYSTEM`. This supports a Windows power-off/restart,
not an application crash. It does not show that an agent terminal close caused
the loss. TaskScheduler Operational history was disabled, so the former task's
result-1 exception was not recoverable.

The `DFL Backend Coordinator` Scheduled Task now starts the hidden
`scripts/backend_task_runner.ps1` at user logon and is also manually startable.
Its working directory is the repository root; the runner sets
`PYTHONPATH=dtn\backend` and invokes `python -m app.main`. Task Scheduler is
configured to restart after failure every minute, up to five times. Start it
with `Start-ScheduledTask -TaskName 'DFL Backend Coordinator'`; stop it with
`Stop-ScheduledTask -TaskName 'DFL Backend Coordinator'`. A PC restart stops the
task until the next user logon. Health was polled every 30 seconds for **601
seconds (21 samples)**; every `/healthz` returned HTTP 200, the task stayed
Running, the preprocessing-report route remained in OpenAPI, and autonomy
remained ASK_HUMAN with global STOP inactive.

Integrity-checked SQLite backups taken before backend starts/restarts during
this recovery were `dtn/backend/data/dtn.sqlite3.bak-20261029163636` and
`dtn/backend/data/dtn.sqlite3.bak-20261029163834`.

### Results

All data below are from the real MNIST sample and Jetson cluster. These numbers
are not compared with CIFAR results.

| Step | Result | Measured evidence |
|---|---|---|
| 0. Read-only preflight | PASS | `mnist_demo`: 3,000 images; shape `[28,28,1]`; split 2,400/300/300; 800 training rows on each worker. ASK_HUMAN; STOP off; master and workers idle. |
| 1. `mnist_demo2` telemetry copy | PASS | Same fixed-seed 3,000-image sample; seed 42; PC system RAM used peak **14,467.0 MB** (8 samples); preprocessing **2.345 s**; shard shipping **0.416 s**. Split 2,400/300/300. Upload ZIP SHA-256 `d327d204283281b69f336f8af199d8654467803d80fa828499e94fee2aa4feb7`. `mnist_demo` was not re-uploaded or overwritten. |
| 2. Initial model | PASS | ASK_HUMAN decision `e6f9db96-2351-4b73-8d3d-088f49950cf8`; input `[28,28,1]`; **754 parameters**. PC and all workers had architecture SHA-256 `b66616d37317a211cd9dc2cb8d02b6213cfc292d049a2258dc7711d1e7f73bb2` and weights SHA-256 `d23e3d771bb065c347655ac932ab90cc3a8ce12ea01345c533ffca004be78d75`. |
| 3. Dataset NAS | PASS, preliminary | ASK_HUMAN decision `a73ff3f1-51b3-40bd-a73c-b94e98210e0f`; 120-second limit; search **17.931 s**; selected `b2_c16_k5`, **7,002 parameters**; each worker validated 160 samples per candidate. Candidate accuracies and tegrastats are below. The earlier completed run and this telemetry rerun selected the same architecture and had identical per-candidate accuracies; selected mean was **12.0833%** in both. This is only 2.08 percentage points over 10% chance and is **not clearly above chance** for this small validation set. |
| 4. Short training | PASS | ASK_HUMAN topology decision `77981c85-a005-4ff4-b09b-c7e8e418c069`; training decision `04efdde0-86c7-4589-b8db-6ec05cf0a346`; bidirectional_ring, 5 rounds x 2 epochs, seed 2026; final mean accuracy **10.00%**, macro-F1 **1.818%**, sent+received communication **90,228 bytes**, duration **10.811 s**. |
| 4b. NAS-architecture follow-up | UNSUPPORTED | `start_training` schema/handler has no architecture parameter. No architecture override was attempted or implied. |
| 5. SLM feedback iteration | PASS | Fallback `False`. One SLM suggestion valid out of one; source `slm`. Raw output proposed `start_training(dataset_name=mnist_demo, rounds=3, epochs=2)`; validator result `valid`. ASK_HUMAN decision `c0f9982f-7c4c-4721-8bbf-16668b61c25c` was approved and its 3x2 follow-up run completed. Accuracy **10.00%**, macro-F1 **1.818%**, communication **54,128 bytes**, duration **4.881 s**. |

`mnist_demo2` shard SHA-256 values (PC == worker): worker1
`d7097eb3253cc026b49da7779c027ba3c15c249a850814ec6bab7dff037ebd56`, worker2
`4e4c77d86f1fc9c077c4bf3db840c20e0cf23397500467f1beebdf7ecda0143e`, worker3
`c646390ae90d099612aeeb8e1ae6da7133be7b5712380b3612eaccf59680e999`.

The single raw SLM response was:

```json
{"action":{"action":"start_training","parameters":{"dataset_name":"mnist_demo","epochs":2,"rounds":3}},"metrics":{"latency_ms":52707.8,"model_latency_ms":52702.3},"reply":"start_training"}
```

Validator result was `valid`; source was `slm`; fallback remained disabled. The
pending decision reason was `start_training`, with evidence containing the
measured short-run summary: validation accuracy 0.10, macro-F1 0.01818,
communication 45,114 bytes for the five-round run, and `bidirectional_ring`
topology. The decision was approved and its history has `created` and
`approved` events.

NAS candidate validation accuracies (worker1 / worker2 / worker3):

| Candidate | Parameters | Accuracy per worker | Mean |
|---|---:|---|---:|
| `b1_c16_k3` | 330 | 10.00% / 10.00% / 10.00% | 10.00% |
| `b1_c16_k5` | 586 | 5.625% / 8.125% / 8.125% | 7.292% |
| `b1_c8_k3` | 170 | 10.00% / 10.00% / 10.00% | 10.00% |
| `b1_c8_k5` | 298 | 10.00% / 10.00% / 10.00% | 10.00% |
| `b2_c16_k3` | 2,650 | 10.625% / 13.125% / 10.625% | 11.458% |
| `b2_c16_k5` | 7,002 | 13.125% / 11.250% / 11.875% | 12.083% |
| `b2_c8_k3` | 754 | 8.125% / 11.250% / 10.625% | 10.000% |
| `b2_c8_k5` | 1,906 | 10.00% / 10.00% / 10.00% | 10.00% |

Full-interval tegrastats peaks (RAM / used swap, MB; sample count):

| Run | Master | Worker1 | Worker2 | Worker3 |
|---|---:|---:|---:|---:|
| NAS telemetry rerun | 2,233 / 761 (36) | 5,314 / 86 (36) | 5,145 / 114 (35) | 5,186 / 83 (35) |
| 5x2 training | 2,239 / 761 (20) | 5,119 / 86 (20) | 4,960 / 114 (20) | 5,004 / 83 (20) |
| SLM feedback and 3x2 follow-up | 3,585 / 761 (66) | 5,126 / 32 (66) | 4,968 / 114 (66) | 5,021 / 83 (65) |

Per-worker training metrics (final validation accuracy / macro-F1; sent / received
bytes):

| Run | Worker1 | Worker2 | Worker3 | Total sent + received |
|---|---|---|---|---:|
| 5x2 | 10.00% / 1.818%; 15,030 / 15,042 | 10.00% / 1.818%; 15,048 / 15,033 | 10.00% / 1.818%; 15,036 / 15,039 | 90,228 B |
| SLM 3x2 follow-up | 10.00% / 1.818%; 9,014 / 9,025 | 10.00% / 1.818%; 9,028 / 9,018 | 10.00% / 1.818%; 9,022 / 9,021 | 54,128 B |

The recovered NAS group's result parser initially failed after the approval
response omitted `result.data`. The group nevertheless completed and was
recovered read-only (group `dataset-nas-1793260918-14afda`, **23.518 s**). The
acceptance runner now reuses the tested M4 approval-envelope parser and polls
the master results endpoint for async groups until completed/failed, keeping
tegrastats active until terminal state. Three fixture tests cover approval with
data, async approval without data, and a failed group; all **3 passed**.

Decision history includes topology, 5x2 training, and SLM feedback creation and
approval. Final state after the runs was ASK_HUMAN, global STOP false, with zero
pending decisions and zero running loops. Result JSON files:
`data/mnist_demo2_telemetry.json`, `data/mnist_nas_telemetry_rerun.json`,
`data/mnist_task_k_training_feedback.json`, and
`data/task_k_mnist_results.json`.

Jetson backups for the grayscale NAS deployment:

- Each worker: `/opt/dfl-slm-dtn/worker/training_service.py`, `nas_search.py`,
  and `/opt/dfl-slm-dtn/common/nas.py`, each backed up with suffixes
  `20261029165702` (worker1), `20261029165735` (worker2), and `20261029165751`
  (worker3).
- Master deployment destinations under `/opt/dfl-slm-dtn/{master,common,topology_opt,data}`,
  plus `/etc/dfl-slm-master.env` and `/etc/systemd/system/dfl-slm-master.service`,
  backed up with suffix `20261029165819`.
