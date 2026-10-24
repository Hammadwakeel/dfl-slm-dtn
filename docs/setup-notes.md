# Hardware and setup notes

## Confirmed roles

- PC: Windows 10 Pro, `192.168.11.12`, Python 3.10.11
- Master Jetson: `dfl-master`, `192.168.11.15`
- Worker 1: `dfl-worker1`, `192.168.11.123`
- Worker 2: `dfl-worker2`, `192.168.11.14`
- Worker 3: `dfl-worker3`, `192.168.11.13`
- SSH user: `hammad`

Batch-mode SSH and passwordless sudo work for all four Jetsons. Python 3.10 is retained on the PC and Jetsons as requested.

## Hardware discovery

The hardware probe ran successfully on all four devices. Each is reported by the device tree as an NVIDIA Jetson Orin Nano Engineering Reference Developer Kit Super. All report Ubuntu 22.04.4 LTS, L4T R36.3, Python 3.10.12, CUDA 12.2, cuDNN 8.9.4.25, and TensorRT 8.6.2.3.

The devices have distinct device-tree hardware serials and Wi-Fi MAC addresses. At Phase 0 discovery they had identical `/etc/machine-id` and hostname values, consistent with cloned OS images. Before Phase 1 deployment, unique hostnames and machine IDs were configured. The final values and TCP checks are in [phase1-network-device.md](./phase1-network-device.md).

Each device has 7.4 GiB total RAM, with approximately 5.0–5.3 GiB available at Phase 0 probe time and 3.7 GiB swap. Root storage is approximately 234 GiB, with 130–139 GiB available. TensorFlow 2.15.0 imports on all devices and exposes a GPU device. This verifies the installed runtime and GPU enumeration, not a complete training run.

Raw outputs are in [`hardware-probe-results/`](./hardware-probe-results/).

## Version recommendations

| Device(s) | Observed | Recommendation |
|---|---|---|
| PC | Windows 10 Pro, Python 3.10.11 | Keep Python 3.10 for the PC DTN backend. |
| Master Jetson | Python 3.10.12; TensorFlow 2.15.0; Ollama 0.34.0 active; local `llama3.2:1b`, Q8_0, 1.3 GB | Keep Ollama and the existing Q8_0 model for initial testing; cap context at 4,096 tokens while measuring unified-memory use. |
| Workers 1–3 | Python 3.10.12; TensorFlow 2.15.0; GPU device exposed by TensorFlow | Keep Python 3.10.12 and TensorFlow 2.15.0; do not upgrade before a real training smoke test. |

TensorFlow startup emitted duplicate cuDNN/cuFFT/cuBLAS registration errors and NUMA notices, but imported successfully and enumerated a GPU on each device. Run a real training smoke test before relying on it.

## Risks and limitations

- The master has 7.4 GiB of shared system memory. The existing 1.3 GB Q8_0 weights are modest, but long context/KV cache and concurrent services consume the same memory. A 4,096-token context is a conservative initial setting, not a benchmark result.
- On-device SLM fine-tuning is not proven feasible. Measure peak memory with the intended adapter method, sequence length, and batch size before attempting it.
- `jtop` is absent; `tegrastats` is available and provided real RAM, CPU, GPU frequency, temperature, and power readings.
- The indoor CO2 dataset is not yet available.
- The provided work report describes earlier preliminary results but explicitly says aggregation was emulated and RAM/GPU/energy data were not collected. Those results are background only and do not replace real-hardware acceptance evidence.

## Phase 1 update

The authenticated status service is installed and enabled on the master and all workers. Live readings, authentication, and direct TCP reachability were verified; per-node API tokens are held only in the ignored local `.env` and the device `/etc/dfl-slm-node.env` files. See [phase1-network-device.md](./phase1-network-device.md).

## Probe script portability

The probe script uses Bash. When sending it from Windows PowerShell, use `cmd.exe` file redirection to preserve LF line endings. The documented command is in [README.md](../README.md).
