import json
import threading
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from fastapi.testclient import TestClient
import yaml

from app.core.config import _load_devices
from app.db.models import Device, DeviceStatus
from app.services.cluster import cluster_nodes, sync_training_registry


def test_health_is_public(client: TestClient) -> None:
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_real_inventory_loads_master_and_five_workers(monkeypatch) -> None:
    repo_root = Path(__file__).resolve().parents[3]
    registry = yaml.safe_load(
        (repo_root / "config" / "nodes.yaml").read_text(encoding="utf-8")
    )
    for node_id in ["master", *(f"worker{index}" for index in range(1, 6))]:
        monkeypatch.setenv(f"{node_id.upper()}_API_TOKEN", f"{node_id}-token-" + "x" * 32)

    nodes = _load_devices(registry)

    assert [node.node_id for node in nodes] == [
        "master",
        "worker1",
        "worker2",
        "worker3",
        "worker4",
        "worker5",
    ]
    assert nodes[-1].hostname == "dflworker5"
    assert nodes[-1].ip == "192.168.11.76"
    assert nodes[-1].role == "worker"
    assert nodes[-1].training_port == 5004


def test_api_rejects_missing_and_invalid_tokens(client: TestClient) -> None:
    assert client.get("/api/devices").status_code == 401
    assert client.get(
        "/api/devices",
        headers={"Authorization": "Bearer incorrect-token"},
    ).status_code == 401


def test_device_registry_requires_token_and_returns_configured_nodes(
    client: TestClient,
    auth_headers: dict[str, str],
) -> None:
    response = client.get("/api/devices", headers=auth_headers)
    assert response.status_code == 200
    devices = response.json()
    assert [device["node_id"] for device in devices] == [
        "master",
        "worker1",
        "worker2",
        "worker3",
    ]
    assert all(device["online"] is False for device in devices)


def test_training_sync_marks_configured_workers_ready_even_when_not_managed(
    client: TestClient,
    auth_headers: dict[str, str],
    monkeypatch,
) -> None:
    import app.services.cluster

    monkeypatch.setattr(
        app.services.cluster,
        "sync_training_workers",
        lambda _settings, workers: {
            "configured_workers": [
                worker.node_id for worker in workers if worker.role == "worker"
            ]
        },
    )

    assert sync_training_registry(
        client.app.state.settings,
        client.app.state.session_factory,
    ) is None

    response = client.get("/api/devices", headers=auth_headers)
    workers = [device for device in response.json() if device["role"] == "worker"]
    assert all(device["training_ready"] is True for device in workers)
    assert all(device["training_error"] is None for device in workers)


def test_worker_memory_cleanup_uses_configured_workers_only(
    client: TestClient,
    auth_headers: dict[str, str],
    monkeypatch,
) -> None:
    calls = []

    def cleanup(workers, *, timeout_seconds):
        calls.append((workers, timeout_seconds))
        return {
            "complete": True,
            "phase": "cleanup",
            "message": "Memory cleanup completed on all workers.",
            "workers": [],
        }

    monkeypatch.setattr(
        "app.routers.devices.cleanup_worker_memory_cluster",
        cleanup,
    )
    unauthorized = client.post("/api/devices/memory/cleanup")
    response = client.post(
        "/api/devices/memory/cleanup",
        headers=auth_headers,
    )

    assert unauthorized.status_code == 401
    assert response.status_code == 200
    assert [worker.node_id for worker in calls[0][0]] == ["worker1", "worker2", "worker3"]
    assert calls[0][1] == 30


def test_register_reachable_device_and_remove_it(
    client: TestClient,
    auth_headers: dict[str, str],
) -> None:
    device_id = "local-probe"
    device_token = "test-local-device-token-that-is-long-enough"
    payload = {
        "node_id": device_id,
        "hostname": "Local probe",
        "ip": "127.0.0.1",
        "port": 0,
        "training_port": 5004,
        "api_token": device_token,
    }

    class StatusHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            if self.path != "/v1/status":
                self.send_error(404)
                return
            if self.headers.get("Authorization") != f"Bearer {device_token}":
                self.send_error(401)
                return
            body = json.dumps(
                {
                    "node_id": device_id,
                    "role": "worker",
                    "hostname": "Local probe",
                    "sampled_at": datetime.now(timezone.utc).isoformat(),
                    "cpu_utilization_percent": 5.0,
                    "gpu_utilization_percent": 0.0,
                    "memory": {
                        "total_bytes": 1000,
                        "available_bytes": 700,
                        "used_bytes": 300,
                    },
                    "temperatures_celsius": {"cpu": 38.0},
                }
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, _format: str, *_args: object) -> None:
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), StatusHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    payload["port"] = server.server_address[1]
    try:
        response = client.post(
            "/api/devices",
            json=payload,
            headers=auth_headers,
        )
        assert response.status_code == 201
        device = response.json()
        assert device["node_id"] == device_id
        assert device["online"] is True
        assert device["is_managed"] is True
        assert device["training_port"] == 5004
        assert device["training_ready"] is False
        assert device["training_error"]
        assert "api_token" not in device

        listed = client.get("/api/devices", headers=auth_headers)
        assert any(item["node_id"] == device_id and item["online"] for item in listed.json())
        duplicate = client.post("/api/devices", json=payload, headers=auth_headers)
        assert duplicate.status_code == 409

        updated = client.patch(
            f"/api/devices/{device_id}",
            json={"hostname": "Renamed local probe"},
            headers=auth_headers,
        )
        assert updated.status_code == 200
        assert updated.json()["hostname"] == "Renamed local probe"
        assert "api_token" not in updated.json()

        static_edit = client.patch(
            "/api/devices/worker1",
            json={"hostname": "Renamed worker"},
            headers=auth_headers,
        )
        assert static_edit.status_code == 409
        empty_edit = client.patch(
            f"/api/devices/{device_id}",
            json={"hostname": None},
            headers=auth_headers,
        )
        assert empty_edit.status_code == 422

        removed = client.delete(f"/api/devices/{device_id}", headers=auth_headers)
        assert removed.status_code == 204
        assert client.get(
            f"/api/devices/{device_id}",
            headers=auth_headers,
        ).status_code == 404
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_cluster_nodes_includes_database_registered_workers(
    client: TestClient,
    settings,
) -> None:
    with client.app.state.session_factory.begin() as session:
        session.add(
            Device(
                node_id="worker5",
                role="worker",
                hostname="Dynamic worker 5",
                ip="192.168.11.75",
                port=5001,
                training_port=5004,
                is_managed=True,
                api_token="worker5-token-" + "x" * 32,
                online=True,
            )
        )

    nodes = cluster_nodes(settings, client.app.state.session_factory)
    worker = next(node for node in nodes if node.node_id == "worker5")

    assert worker.is_managed is True
    assert worker.training_url == "http://192.168.11.75:5004"


def test_device_registration_rejects_public_address_and_protects_static_nodes(
    client: TestClient,
    auth_headers: dict[str, str],
) -> None:
    public_ip = client.post(
        "/api/devices",
        json={
            "node_id": "public-host",
            "hostname": "Public host",
            "ip": "8.8.8.8",
            "port": 5001,
            "api_token": "test-local-device-token-that-is-long-enough",
        },
        headers=auth_headers,
    )
    assert public_ip.status_code == 422
    static_removal = client.delete("/api/devices/worker1", headers=auth_headers)
    assert static_removal.status_code == 409


def test_missing_device_returns_404(
    client: TestClient,
    auth_headers: dict[str, str],
) -> None:
    response = client.get("/api/devices/unknown", headers=auth_headers)
    assert response.status_code == 404


def test_history_rejects_invalid_limit(
    client: TestClient,
    auth_headers: dict[str, str],
) -> None:
    response = client.get(
        "/api/devices/master/history?limit=0",
        headers=auth_headers,
    )
    assert response.status_code == 422


def test_history_returns_persisted_readings_with_utc_timestamps(
    client: TestClient,
    auth_headers: dict[str, str],
) -> None:
    sampled_at = datetime.now(timezone.utc) - timedelta(minutes=1)
    with client.app.state.session_factory() as session:
        session.add(
            DeviceStatus(
                node_id="master",
                sampled_at=sampled_at,
                cpu_utilization_percent=12.5,
                gpu_utilization_percent=4.0,
                memory={"total_bytes": 100, "available_bytes": 70, "used_bytes": 30},
                temperatures_celsius={"cpu": 41.5},
                payload={"node_id": "master"},
            )
        )
        device = session.get(Device, "master")
        assert device is not None
        device.last_seen = sampled_at
        session.commit()

    history = client.get(
        "/api/devices/master/history",
        headers=auth_headers,
    )
    device = client.get("/api/devices/master", headers=auth_headers)
    assert history.status_code == 200
    assert len(history.json()) == 1
    assert history.json()[0]["sampled_at"].endswith(("Z", "+00:00"))
    assert device.json()["last_seen"].endswith(("Z", "+00:00"))


def test_openapi_declares_bearer_auth(client: TestClient) -> None:
    response = client.get("/api/openapi.json")
    assert response.status_code == 200
    schema = response.json()
    assert "HTTPBearer" in schema["components"]["securitySchemes"]
    assert "/api/devices" in schema["paths"]
    assert "/api/supervisor/runs" in schema["paths"]


def test_cors_allows_only_configured_frontend_origins(client: TestClient) -> None:
    allowed = client.options(
        "/api/devices",
        headers={
            "Origin": "http://localhost:5173",
            "Access-Control-Request-Method": "GET",
        },
    )
    blocked = client.options(
        "/api/devices",
        headers={
            "Origin": "http://untrusted.example",
            "Access-Control-Request-Method": "GET",
        },
    )
    assert allowed.headers["access-control-allow-origin"] == "http://localhost:5173"
    assert "access-control-allow-origin" not in blocked.headers
    delete_allowed = client.options(
        "/api/devices/lab-worker-1",
        headers={
            "Origin": "http://localhost:5173",
            "Access-Control-Request-Method": "DELETE",
        },
    )
    assert "DELETE" in delete_allowed.headers["access-control-allow-methods"]
    patch_allowed = client.options(
        "/api/devices/lab-worker-1",
        headers={
            "Origin": "http://localhost:5173",
            "Access-Control-Request-Method": "PATCH",
        },
    )
    assert "PATCH" in patch_allowed.headers["access-control-allow-methods"]
