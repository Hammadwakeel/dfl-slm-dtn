# Phase 1: network and device layer

## Deployed status API

The same Python 3.10 standard-library service runs on every physical Jetson:

- Master `dfl-master` (`192.168.11.15`): TCP 5002
- Worker 1 `dfl-worker1` (`192.168.11.123`): TCP 5001
- Worker 2 `dfl-worker2` (`192.168.11.14`): TCP 5001
- Worker 3 `dfl-worker3` (`192.168.11.13`): TCP 5001

The systemd unit is enabled at boot as `dfl-slm-status.service`. It runs as the unprivileged `hammad` user from `/opt/dfl-slm-dtn`. Per-node secrets are kept in the ignored PC `.env` file and installed on each device at `/etc/dfl-slm-node.env` with mode `0600`; they are not printed by the deployment script or committed to git.

Routes:

- `GET /healthz`: minimal unauthenticated process liveness only; returns `{"status":"ok"}`.
- `GET /v1/status`: requires the node-specific bearer credential in the `Authorization` header. Missing or incorrect tokens return HTTP 401.

The authenticated status payload contains node ID, role, hostname, sample timestamp, CPU/GPU utilization, real RAM byte counters, real thermal-zone readings, and metric sources. `tegrastats` supplies live CPU/GPU readings; CPU falls back to `/proc/stat` if needed. Missing GPU readings remain `null` rather than being fabricated. This agent reports the current node; it does not start training or perform model exchange.

The shared `common.api_client.ApiClient` obtains status over HTTP with a bearer token. Phase 1 HTTP is intended only for the trusted LAN; bearer tokens are not encrypted in transit. Use a protected network and add TLS before extending access beyond that LAN.

## Deployment and verification

From PowerShell at the repository root, with batch-mode SSH and passwordless sudo configured:

```powershell
.\deploy\deploy_jetsons.ps1
python -m unittest discover -v
python -m scripts.check_phase1
```

The deploy script creates missing per-node random tokens in the ignored `.env`, stages only the shared agent, service unit, and node-specific environment file to each Jetson, installs `/opt` and `/etc` files with constrained permissions, then enables and starts the systemd service. It does not alter hostnames, machine IDs, firewall rules, or Ollama.

The integration check verifies from the PC that all four live status endpoints respond using their individual credentials and reject both missing and incorrect credentials with HTTP 401. It prints device readings but never prints tokens.

## Observed TCP connectivity

Verified after deployment:

- PC `192.168.11.12` to master `192.168.11.15:5002`: open.
- PC `192.168.11.12` to each worker `:5001`: open.
- Every directed worker-to-worker TCP connection to peer `:5001`: open (six of six).
- Master and all three workers to a temporary listener on PC `192.168.11.12:8000`: open (four of four).

Windows Firewall profiles were enabled with a default inbound-block policy, but the real inbound TCP probes to the temporary listener succeeded from all four Jetsons. **No additional Windows Firewall rule is needed for TCP/8000 on the tested LAN path.** Recheck the production backend after it binds to `192.168.11.12:8000`, and scope any later rule to the DTN port and private LAN if network policy changes.

All four Jetsons now report the distinct configured hostnames and distinct `/etc/machine-id` values:

| Node | Hostname | `/etc/machine-id` |
|---|---|---|
| Master | `dfl-master` | `0d73902445f94427bec8ddb819d71ed4` |
| Worker 1 | `dfl-worker1` | `3377b0a960d5477da71e4565c5032e8f` |
| Worker 2 | `dfl-worker2` | `72b07e78fb9a480f997e4771aa5380d0` |
| Worker 3 | `dfl-worker3` | `2cbbc51775124bfca0e8f0371a940e66` |

The temporary PC listener was stopped after the inbound connectivity check.

## Service logs and operator commands

Run these commands on a Jetson:

```bash
sudo systemctl status dfl-slm-status.service
sudo journalctl -u dfl-slm-status.service -f
sudo systemctl restart dfl-slm-status.service
```

The API is bound to `0.0.0.0` so peers and the PC can reach it. Keep the Jetsons on the trusted LAN and do not expose these ports to the public Internet.
