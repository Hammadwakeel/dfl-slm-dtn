# DFL-SLM Digital Twin Network Project

This repository contains Phase 0 hardware discovery and the Phase 1 authenticated Jetson status service. The DTN, SLM orchestration, and worker training application layers have not yet been built.

## Phase 0 status

- PC: Windows 10 Pro, Python 3.10.11, 31.7 GiB RAM, about 215.9 GiB free on D:
- Master: `192.168.11.15`
- Workers: `192.168.11.123`, `192.168.11.14`, `192.168.11.13`
- SSH key-based batch access works to all four Jetsons.
- All four report Jetson Orin Nano Engineering Reference Developer Kit Super, Ubuntu 22.04.4, L4T R36.3, Python 3.10.12, CUDA 12.2, TensorFlow 2.15.0, and GPU enumeration through TensorFlow.
- The master has active Ollama 0.34.0 and `llama3.2:1b` (Q8_0, 1.3 GB).
- Jetsons use the distinct hostnames `dfl-master`, `dfl-worker1`, `dfl-worker2`, and `dfl-worker3`; each has a unique `/etc/machine-id`.
- Python 3.10 is retained on the PC and Jetsons as requested.
- The indoor CO2 dataset has not been provided.

See [docs/hardware-discovery.md](./docs/hardware-discovery.md) for the hardware findings and recommendations, and [docs/hardware-probe-results/](./docs/hardware-probe-results/) for the raw output from each Jetson.

Phase 1 deployment, API details, live readings, authentication checks, and TCP connectivity results are recorded in [docs/phase1-network-device.md](./docs/phase1-network-device.md).

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
