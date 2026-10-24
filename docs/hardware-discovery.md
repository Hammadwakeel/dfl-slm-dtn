# Phase 0 hardware discovery report

## Result

All four Jetsons accepted batch-mode SSH and completed `scripts/probe_device.sh` with exit code 0. Full raw output is preserved in [`hardware-probe-results/`](./hardware-probe-results/).

| Role | IP | Device-tree serial distinct? | L4T | Python | TensorFlow | RAM available at probe | Root disk free |
|---|---|---:|---|---|---|---:|---:|
| Master | `192.168.11.15` | Yes | R36.3 | 3.10.12 | 2.15.0, GPU enumerated | 5.3 GiB | 133 GiB |
| Worker 1 | `192.168.11.123` | Yes | R36.3 | 3.10.12 | 2.15.0, GPU enumerated | 5.0 GiB | 130 GiB |
| Worker 2 | `192.168.11.14` | Yes | R36.3 | 3.10.12 | 2.15.0, GPU enumerated | 5.2 GiB | 137 GiB |
| Worker 3 | `192.168.11.13` | Yes | R36.3 | 3.10.12 | 2.15.0, GPU enumerated | 5.0 GiB | 139 GiB |

All report Ubuntu 22.04.4 LTS, arm64, CUDA 12.2, cuDNN 8.9.4.25, TensorRT 8.6.2.3, 7.4 GiB total RAM, and 3.7 GiB swap.

At Phase 0, the devices had distinct hardware serials and distinct active Wi-Fi MAC addresses, supporting that they are separate physical boards, but shared the hostname `hammad-desktop` and `/etc/machine-id` `5dbfb12414a3456d9014d88183e338b1`. Before Phase 1, the hostnames and machine IDs were changed to unique values; see [phase1-network-device.md](./phase1-network-device.md).

## Software inventory and recommendations

- **Workers:** Keep Python 3.10.12 and TensorFlow 2.15.0, which are already installed and expose a TensorFlow GPU device on this L4T R36.3 image. Do not upgrade before a real training smoke test.
- **Master:** Keep the installed Ollama 0.34.0. It is active and has `llama3.2:1b` locally, with 1.2B parameters, 1.3 GB model size, and Q8_0 quantization. Use the existing Q8_0 model for initial integration and cap context at 4,096 tokens while measuring memory.
- **PC:** The observed PC is Windows 10 Pro with Python 3.10.11, 31.7 GiB RAM, and 215.9 GiB free on D:. Keep Python 3.10 for the DTN backend, as requested.
- **Fine-tuning:** The master has 7.4 GiB total shared memory, with about 5.3 GiB available while idle. This is not sufficient evidence to claim LoRA/QLoRA feasibility. Measure peak memory with the exact training configuration in Phase 7 before attempting it.

TensorFlow startup emitted duplicate cuDNN/cuFFT/cuBLAS registration errors and NUMA notices. Despite these messages, import completed and TensorFlow enumerated a GPU on each Jetson; no model-training workload was run during Phase 0.

## Network observations

Each Jetson could ping all three other Jetson IPs. All Jetsons reported `PING_FAIL` for the PC IP `192.168.11.12`; the PC's prior TCP/22 reachability checks to the Jetsons succeeded. ICMP failure did not imply application TCP was blocked: Phase 1 tests confirmed TCP reachability in both directions and across all worker pairs.

## Recommended next phase

Phase 0 discovery and Phase 1 device status deployment are complete. Before Phase 4, a real two-epoch TensorFlow 2.15 GPU training smoke test passed on each worker. The originally requested CO2 CSV path was not supplied; Phase 4 uses the verified CIFAR-10 archive and reports its split and measured training results in [phase4-worker-training.md](./phase4-worker-training.md).

## Prior work reference

The provided detailed work report describes earlier DTN/NAS/topology work and preliminary indoor CO2 results (about 3,907 ppm² fixed-model MSE and 2,868 ppm² NAS MSE). The report explicitly says aggregation was emulated and no RAM, GPU, or energy data were collected. These are background claims only; they do not satisfy this project's real peer-to-peer exchange or hardware-results acceptance criteria and were not used as probe evidence. The report also describes a Django DTN, whereas the current project specification calls for FastAPI and React.
