# Phase 2: DTN Coordinator

## Delivered

The Windows PC hosts the coordinator API at `0.0.0.0:8000` and the Vite development dashboard at `0.0.0.0:5173`. The browser communicates only with the FastAPI coordinator; the coordinator polls the four authenticated Jetson `/v1/status` endpoints using the shared `common.api_client.ApiClient`.

The FastAPI app factory loads settings from the root `.env` and the physical node registry in `config/nodes.yaml`. `DTN_API_TOKEN` protects the REST API with bearer authentication. The API exposes device list/detail and bounded status-history endpoints, an authenticated WebSocket status stream, `/healthz`, and OpenAPI/Swagger documentation at `/docs`. Alembic applies the SQLite schema from `dtn/backend/migrations`; local readings are stored in the ignored `dtn/backend/data/dtn.sqlite3` database. The poller samples every three seconds by default, records real CPU, GPU, memory, and temperature data, and marks failed devices offline after a request timeout.

The React/TypeScript dashboard includes live device cards, cluster counts, a coordinator/worker topology view, and planned-workspace pages for Experiments, Model Studio, Architecture Search, and SLM Assistant. The coordinator token is entered by the operator and kept in browser `sessionStorage` for that tab session. REST requests use the bearer header; the WebSocket receives the token in its first JSON auth frame rather than the URL.

## Configuration and startup

Use Python 3.10 on the PC. Copy `.env.example` to `.env` only if the root `.env` is not already configured; preserve the existing Jetson credentials. Set a random `DTN_API_TOKEN` of at least 32 characters and ensure `DTN_ALLOWED_ORIGINS` contains only the exact frontend origins in use (the default allows `http://localhost:5173` and `http://192.168.11.12:5173`). Do not commit `.env`.

Install backend dependencies from the repository root:

```powershell
Set-Location 'D:\Lab\MS Research Project\dfl-slm-dtn\dtn\backend'
python -m pip install -e '.[test]'
```

Start the backend from the repository root in one PowerShell window. Alembic migrates the local database during app startup:

```powershell
Set-Location 'D:\Lab\MS Research Project\dfl-slm-dtn'
$env:PYTHONPATH = 'dtn\backend'
python -m app.main
```

Start the frontend in another PowerShell window:

```powershell
Set-Location 'D:\Lab\MS Research Project\dfl-slm-dtn\dtn\frontend'
npm install
npm run dev
```

Open `http://192.168.11.12:5173`, enter the configured coordinator API token, and confirm the four physical node cards report live metrics. API docs are at `http://192.168.11.12:8000/docs`; the public liveness endpoint is `http://192.168.11.12:8000/healthz`.

## API surface

- `GET /api/auth/check`, `GET /api/devices`, `GET /api/devices/{node_id}`, and `GET /api/devices/{node_id}/history` require `Authorization: Bearer <DTN_API_TOKEN>`.
- `GET /ws/devices` is authenticated with a first-frame JSON message `{"type":"auth","token":"..."}`. The server sends a `device_snapshot` after authentication and `device_status` events as real node readings arrive or a device transitions offline/online.
- `/docs` and `/openapi.json` document the REST API. Do not put tokens in URLs or logs.

This implementation uses HTTP on the trusted lab LAN, not TLS. Keep the PC and Jetsons on the protected network; bearer tokens and status readings are not encrypted in transit. Add TLS before extending access beyond the LAN.

## Verification performed

Environment versions: Python `3.10.11`, Node.js `v24.19.0`, npm `11.17.0`.

Tests and build:

```text
Backend: 13 passed
Frontend: 3 passed
Frontend production build: passed
npm audit: found 0 vulnerabilities
```

The live coordinator returned authenticated readings from all four Jetsons (CPU/GPU in percent, RAM in bytes, and maximum temperature shown):

```text
{"cpu_percent": 0.17, "gpu_percent": 0.0, "hostname": "dfl-master", "last_seen": "2026-10-24T10:00:19.882469", "max_temperature_c": 45.09, "node_id": "master", "online": true, "ram_total_bytes": 7990542336, "ram_used_bytes": 1511567360}
{"cpu_percent": 0.0, "gpu_percent": 0.0, "hostname": "dfl-worker1", "last_seen": "2026-10-24T10:00:19.856573", "max_temperature_c": 47.22, "node_id": "worker1", "online": true, "ram_total_bytes": 7990542336, "ram_used_bytes": 1489899520}
{"cpu_percent": 0.0, "gpu_percent": 0.0, "hostname": "dfl-worker2", "last_seen": "2026-10-24T10:00:19.889472", "max_temperature_c": 43.56, "node_id": "worker2", "online": true, "ram_total_bytes": 7990534144, "ram_used_bytes": 1503916032}
{"cpu_percent": 0.0, "gpu_percent": 0.0, "hostname": "dfl-worker3", "last_seen": "2026-10-24T10:00:19.876482", "max_temperature_c": 46.34, "node_id": "worker3", "online": true, "ram_total_bytes": 7990546432, "ram_used_bytes": 1483051008}
```

Follow-up polling confirmed the `last_seen` timestamp advanced for every node. TCP checks passed from the PC to all four Jetson API ports, from each Jetson to the actual PC coordinator on TCP/8000, and between all six directed worker-to-worker pairs on TCP/5001. **No Windows Firewall rule was needed on the tested LAN path.** Earlier Phase 1 connectivity details are in [phase1-network-device.md](./phase1-network-device.md).

For the outage test, only `dfl-worker1`'s `dfl-slm-status.service` was stopped. The WebSocket delivered `online=false`, and authenticated REST returned `online=False` with a connection-refused last error. The service was immediately restarted and verified `active`; the WebSocket delivered `online=true`, and REST confirmed `online=True` with hostname `dfl-worker1`.

The backend poller regression test also verifies repeated polling and graceful shutdown. It protects against the Python 3.10 `asyncio.TimeoutError` distinction that previously terminated the background loop after its first interval.

## Tests

```powershell
Set-Location 'D:\Lab\MS Research Project\dfl-slm-dtn\dtn\backend'
python -m pytest -q

Set-Location 'D:\Lab\MS Research Project\dfl-slm-dtn\dtn\frontend'
npm test
npm run build
npm audit
```
