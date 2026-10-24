from datetime import datetime, timedelta, timezone

from fastapi.testclient import TestClient

from app.db.models import Device, DeviceStatus


def test_health_is_public(client: TestClient) -> None:
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


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
