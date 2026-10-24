# DFL-SLM Digital Twin Network Project

This repository is the initial scaffold for the DFL-SLM Digital Twin Network project. Phase 0 hardware discovery is complete. No application code or device configuration has been deployed.

## Phase 0 status

- PC: Windows 10 Pro, Python 3.10.11, 31.7 GiB RAM, about 215.9 GiB free on D:
- Master: `192.168.11.15`
- Workers: `192.168.11.123`, `192.168.11.14`, `192.168.11.13`
- SSH key-based batch access works to all four Jetsons.
- All four report Jetson Orin Nano Engineering Reference Developer Kit Super, Ubuntu 22.04.4, L4T R36.3, Python 3.10.12, CUDA 12.2, TensorFlow 2.15.0, and GPU enumeration through TensorFlow.
- The master has active Ollama 0.34.0 and `llama3.2:1b` (Q8_0, 1.3 GB).
- All four Jetsons have distinct device-tree hardware serials, but the same hostname and `/etc/machine-id`; correct these before Phase 1 deployment.
- The indoor CO2 dataset has not been provided.

See [docs/hardware-discovery.md](./docs/hardware-discovery.md) for the hardware findings and recommendations, and [docs/hardware-probe-results/](./docs/hardware-probe-results/) for the raw output from each Jetson.

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

## Before Phase 1

1. Assign each Jetson a unique hostname and regenerate its cloned `/etc/machine-id`.
2. Verify TCP connectivity for the intended service ports from the PC and between workers; ICMP from the Jetsons to the PC failed during discovery.
3. Run a small, real TensorFlow training smoke test.
4. Supply the indoor CO2 dataset before beginning the federated-training phase.

Hardware and runtime version decisions are recorded in [docs/setup-notes.md](./docs/setup-notes.md). Model fine-tuning has not been tested; see the memory limitations in the hardware discovery report.
