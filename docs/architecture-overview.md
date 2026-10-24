# Architecture overview

Phases 0–4 are implemented on the four Jetsons and the PC. The remaining planned phases are:

5. Topology optimization
6. NAS / model selection
7. Synthetic data and fine-tuning
8. Hardening and documentation

## Target architecture

- PC hosts the DTN coordinator (FastAPI + React frontend)
- Master Jetson hosts the SLM service
- Three Jetson workers each train locally and exchange peer-to-peer weights over the LAN
- All communication should be authenticated and config-driven
- No hard-coded IPs, secrets, or tokens in git

## Phase 0 findings

SSH access and hardware probes are complete. The four boards have distinct device-tree serials, unique hostnames and machine IDs, and authenticated live status agents. TCP reachability passed for PC-to-device, device-to-PC, and worker-to-worker paths. TensorFlow 2.15 GPU training was verified on each Jetson. Phase 4 adds authenticated worker training on port 5004, direct Ring/Mesh weight exchange, and a real CIFAR-10 dashboard; its dataset inventory and experimental results are in [phase4-worker-training.md](./phase4-worker-training.md). See [hardware-discovery.md](./hardware-discovery.md) and [phase1-network-device.md](./phase1-network-device.md) for the earlier hardware and network evidence.
