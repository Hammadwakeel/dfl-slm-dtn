# Phase 0 setup notes

## Confirmed roles

- PC: Windows 10 Pro, `192.168.11.12`
- Master Jetson: `192.168.11.15`
- Worker 1: `192.168.11.123`
- Worker 2: `192.168.11.14`
- Worker 3: `192.168.11.13`
- SSH user: `hammad`

Batch-mode SSH works for all four Jetsons.

## Hardware discovery

The hardware probe ran successfully on all four devices. Each is reported by the device tree as an NVIDIA Jetson Orin Nano Engineering Reference Developer Kit Super. All report Ubuntu 22.04.4 LTS, L4T R36.3, Python 3.10.12, CUDA 12.2, cuDNN 8.9.4.25, and TensorRT 8.6.2.3.

The devices have four distinct device-tree hardware serials and different active Wi-Fi MAC addresses, supporting that the four IPs are separate physical units. However, all four have the same `/etc/machine-id` and hostname (`hammad-desktop`). This indicates the OS images were cloned without regenerating their machine IDs. Before Phase 1 services are deployed, give each device a unique hostname and regenerate each machine ID; do this as a controlled device setup step.

Each device has 7.4 GiB total RAM, with approximately 5.0–5.3 GiB available at probe time and 3.7 GiB swap. Root storage is approximately 234 GiB, with 130–139 GiB available. TensorFlow 2.15.0 imports on all devices and exposes a GPU device. This verifies the installed runtime and GPU enumeration, not a complete training run.

The probe can ping all Jetsons, but the Jetsons cannot ping the PC at `192.168.11.12`. This may be an ICMP/firewall policy; verify the actual application TCP ports during Phase 1.

Raw outputs are in [`hardware-probe-results/`](./hardware-probe-results/).

## Version recommendations

| Device(s) | Observed | Recommendation for the initial implementation |
|---|---|---|
| PC | Windows 10 Pro, Python 3.10.11 | Use Python 3.12 for the PC DTN backend as specified in the project stack. Python 3.12 is not yet verified as installed on this PC. |
| Master Jetson | Python 3.10.12; TensorFlow 2.15.0; Ollama 0.34.0 active; local `llama3.2:1b` model, Q8_0, 1.3 GB | Keep Ollama 0.34.0 and the existing Q8_0 model for initial testing; cap the model context to 4,096 tokens during integration and watch unified-memory use. |
| Workers 1–3 | Python 3.10.12; TensorFlow 2.15.0; GPU device exposed by TensorFlow | Keep the installed Python 3.10.12 and TensorFlow 2.15.0; do not upgrade until the real training test passes on this JetPack image. |

All Jetsons report L4T R36.3 (JetPack 6.0 GA), CUDA 12.2, cuDNN 8.9.4.25, and TensorRT 8.6.2.3. TensorFlow startup emitted duplicate cuDNN/cuFFT/cuBLAS registration errors and NUMA notices, but imported successfully and enumerated a GPU on all devices. Run a small real training smoke test before deciding whether to change the image or TensorFlow build.

## Risks and limitations

- The master has only 7.4 GiB of shared system memory. Although the existing 1.3 GB Q8_0 weights are modest, long context/KV cache and concurrent services consume the same memory. The 4,096-token context recommendation is a conservative starting point, not a benchmark result.
- On-device SLM fine-tuning is not proven feasible. Do not start it until Phase 7 measures peak memory using the intended adapter method, sequence length, and batch size. The observed free memory is not evidence that training will fit.
- The OS machine IDs are duplicated across four devices. Correct these, and assign unique hostnames, before collecting persistent service telemetry or deploying systemd units.
- `jtop` is absent; `tegrastats` is available and provided live RAM, CPU, GPU frequency, temperature, and power readings.
- Ollama is active and the model is present only on the master. Workers do not have Ollama installed, as expected.
- The indoor CO2 dataset is not yet available.

## Probe script portability

The probe script uses Bash. When sending its content to a Jetson from Windows PowerShell, normalize Windows CRLF line endings to LF before piping it to `bash -s`; the final probe logs were captured using byte-preserving SSH input and exited successfully.
