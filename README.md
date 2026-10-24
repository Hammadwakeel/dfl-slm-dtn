# DFL-SLM Digital Twin Network Project

This repository contains the completed Phase 0–5 DFL-SLM implementation: authenticated Jetson status services, the PC-hosted FastAPI/React coordinator, the master SLM, real CIFAR-10 training across three Jetson workers, and measured topology optimization.

## Phase 0 status

- PC: Windows 10 Pro, Python 3.10.11, 31.7 GiB RAM, about 215.9 GiB free on D:
- Master: `192.168.11.15`
- Workers: `192.168.11.123`, `192.168.11.14`, `192.168.11.13`
- SSH key-based batch access works to all four Jetsons.
- All four report Jetson Orin Nano Engineering Reference Developer Kit Super, Ubuntu 22.04.4, L4T R36.3, Python 3.10.12, CUDA 12.2, TensorFlow 2.15.0, and GPU enumeration through TensorFlow.
- The master has active Ollama 0.34.0 and `llama3.2:1b` (Q8_0, 1.3 GB).
- Jetsons use the distinct hostnames `dfl-master`, `dfl-worker1`, `dfl-worker2`, and `dfl-worker3`; each has a unique `/etc/machine-id`.
- Python 3.10 is retained on the PC and Jetsons as requested.
- The originally requested indoor CO2 CSV path was not supplied. Phase 4 uses the official CIFAR-10 dataset with disjoint stratified worker shards; no CO2 experiment results are claimed.

See [docs/hardware-discovery.md](./docs/hardware-discovery.md) for the hardware findings and recommendations, and [docs/hardware-probe-results/](./docs/hardware-probe-results/) for the raw output from each Jetson.

Phase 1 deployment, API details, live readings, authentication checks, and TCP connectivity results are recorded in [docs/phase1-network-device.md](./docs/phase1-network-device.md).

## Phase 2 status

- FastAPI coordinator on the PC at `0.0.0.0:8000`, using bearer-token authentication, SQLite/Alembic, authenticated device/history endpoints, and a WebSocket event stream.
- Background polling records live authenticated status from all four Jetsons; timeouts mark nodes offline.
- React/TypeScript/Vite dashboard on port `5173` shows the real device metrics and master/worker topology. The browser calls only the PC backend.
- Backend pytest, frontend Vitest, production build, live PC/device/peer TCP connectivity, and worker offline/restore checks passed.

See [docs/phase2-coordinator.md](./docs/phase2-coordinator.md) for setup, API usage, verification output, network results, and the security limitation that this phase uses trusted-LAN HTTP without TLS.

## Phase 3 status

- SLM master service on `dfl-master:5003`; the existing authenticated device-status agent remains on `dfl-master:5002`.
- Flask API uses a distinct `MASTER_SLM_API_TOKEN`. Ollama `llama3.2:1b` is constrained to the shared JSON action schema. Live status/log actions are available; Phase 4 adds validated worker-training controls and experiment results.
- The SLM Assistant page forwards chat through the FastAPI coordinator and displays the model reply, validated action, action result, latency, and measured system RAM change.

See [docs/phase3-slm-master.md](./docs/phase3-slm-master.md) for deployment, endpoints, action validation, real prompt results, tests, and memory/latency measurements.

## Phase 4 status

- Worker training APIs are installed on `dfl-worker1..3:5004`; the existing status services remain on port `5001`. The master coordinates on `dfl-master:5003`, leaving its status service on `5002`.
- Each Jetson received only its own stratified CIFAR-10 training shard. The held-out test split and full training set are on the master evaluator only.
- Real 10-round, 5-local-epoch, three-seed Ring and Mesh experiments completed with peer-to-peer weight exchange and held-out accuracy, loss, macro-F1, runtime, RAM, and communication metrics recorded per round.
- A matching centralized baseline completed on the master: mean held-out accuracy 57.06% across the three seeds, with peak process RSS of 3,445.7 MB.
- A worker-stop failure injection was detected by the coordinator, and the stopped service was restored successfully.

See [docs/phase4-worker-training.md](./docs/phase4-worker-training.md) for dataset inspection, measured smoke/full experiment results, direct peer-log evidence, deployment details, and tests.

## Phase 5 status

- The master and workers support directed Ring, bidirectional Ring, Mesh, chain, and star graphs. Ring remains available as an alias for directed Ring.
- Deterministic Dirichlet CIFAR-10 shards (α=0.3 and 0.5, seed 2026) are stored separately per worker; only each worker's own shard is deployed.
- The authenticated SLM `optimize_topology` action runs measured candidate sweeps. A configurable reward-based selector is compared with seeded-random and best-observed-accuracy baselines; no GNN or APG is claimed.
- The Topology page shows the peer graph, active topology, candidate reward breakdown, and selector/baseline comparison. Direct worker-to-worker exchanges were verified in packet captures.
- The full 30-run matrix (five topologies × two Dirichlet splits × three seeds; 10 rounds × 5 epochs) completed in about 58.6 minutes. The reward selector chose bidirectional Ring at α=0.5; the best mean held-out accuracy was 47.283%.

See [docs/phase5-topology-optimization.md](./docs/phase5-topology-optimization.md) for packet counts, reward configuration, pilot timing, full-matrix metrics, and baseline comparison.

## Project layout

```text
dfl-slm-dtn/
├── README.md
├── .env.example
├── .gitattributes
├── .gitignore
├── docs/
│   └── hardware-probe-results/
├── config/
├── common/
├── dtn/
├── master/
├── worker/
├── topology_opt/
├── slm_finetune/
├── deploy/
├── scripts/
└── tests/
```

## Running the read-only Jetson probe

From Windows PowerShell, use `cmd.exe` redirection so the LF-formatted script is sent as file bytes instead of being converted to Windows line endings:

```powershell
$script = (Resolve-Path -LiteralPath '.\scripts\probe_device.sh').Path
$command = 'ssh -o BatchMode=yes hammad@192.168.11.15 "bash -s" < "' + $script + '"'
& $env:ComSpec /d /c $command
if ($LASTEXITCODE -ne 0) { throw "Probe failed with exit code $LASTEXITCODE" }
```

Repeat with each worker IP to collect another hardware report.

## Phase 1 operations

Deploy or refresh the authenticated status service from PowerShell:

```powershell
.\deploy\deploy_jetsons.ps1
```

Run unit tests and check that authenticated status requests succeed and unauthenticated requests receive HTTP 401:

```powershell
python -m unittest discover -v
python -m scripts.check_phase1
```

Hardware and runtime version decisions are recorded in [docs/setup-notes.md](./docs/setup-notes.md). Model fine-tuning has not been tested; see the memory limitations in the hardware discovery report.
