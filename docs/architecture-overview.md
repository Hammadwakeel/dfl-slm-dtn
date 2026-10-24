# Architecture overview

This repository currently contains the project skeleton and completed Phase 0 hardware discovery. The application will be built in the following phases:

1. Discovery and setup
2. Network and device layer
3. DTN coordinator
4. SLM master
5. Worker training and DFL
6. Topology optimization
7. NAS / model selection
8. Synthetic data and fine-tuning
9. Hardening and documentation

## Target architecture

- PC hosts the DTN coordinator (FastAPI + React frontend)
- Master Jetson hosts the SLM service
- Three Jetson workers each train locally and exchange peer-to-peer weights over the LAN
- All communication should be authenticated and config-driven
- No hard-coded IPs, secrets, or tokens in git

## Phase 0 findings

SSH access and the hardware probes are complete. The four boards have distinct device-tree serials, but the Jetsons share a cloned hostname and machine ID. Before Phase 1, assign unique hostnames and regenerate each machine ID, then test service-port reachability and run a real TensorFlow training smoke test. See [hardware-discovery.md](./hardware-discovery.md) for details.
