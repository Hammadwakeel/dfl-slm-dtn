from app.core.config import DeviceConfig
from app.services import worker_operations


def _workers() -> tuple[DeviceConfig, ...]:
    return tuple(
        DeviceConfig(
            node_id=f"worker{index}",
            hostname=f"worker-{index}",
            ip=f"192.0.2.{index}",
            role="worker",
            port=5001,
            token=f"worker-token-{index}-long-enough",
            training_port=5004,
        )
        for index in range(1, 4)
    )


def test_process_snapshot_keeps_worker_order_and_reports_unavailable_worker(monkeypatch):
    def request(worker, path, *, method="GET", timeout_seconds=10.0):
        if worker.node_id == "worker2":
            raise worker_operations.WorkerOperationError("worker2 is offline")
        return {
            "node_id": worker.node_id,
            "sampled_at": "2026-11-09T00:00:00Z",
            "memory_available_mb": 512.0,
            "nas_active": False,
            "training": {},
            "processes": [],
        }

    monkeypatch.setattr(worker_operations, "_request_worker", request)

    result = worker_operations.get_worker_processes(_workers())

    assert [worker["node_id"] for worker in result["workers"]] == [
        "worker1",
        "worker2",
        "worker3",
    ]
    assert result["workers"][0]["memory_available_mb"] == 512.0
    assert result["workers"][1]["error"] == "worker2 is offline"


def test_control_worker_uses_only_supported_management_paths(monkeypatch):
    requested = []
    monkeypatch.setattr(
        worker_operations,
        "_request_worker",
        lambda worker, path, **kwargs: requested.append((worker.node_id, path, kwargs["method"])) or {"accepted": True},
    )

    assert worker_operations.control_worker(_workers()[0], "stop_training") == {
        "accepted": True
    }
    assert worker_operations.control_worker(_workers()[0], "stop_nas") == {
        "accepted": True
    }
    assert worker_operations.control_worker(_workers()[0], "reboot") == {
        "accepted": True
    }
    assert requested == [
        ("worker1", "/v1/cancel", "POST"),
        ("worker1", "/v1/nas/cancel", "POST"),
        ("worker1", "/v1/system/reboot", "POST"),
    ]


def test_authenticated_routes_expose_worker_processes_and_reboot(
    client,
    auth_headers,
    monkeypatch,
):
    from app.routers import devices

    def process_report(workers):
        assert [worker.node_id for worker in workers] == [
            "worker1",
            "worker2",
            "worker3",
        ]
        return {"workers": [{"node_id": "worker1", "processes": []}]}

    monkeypatch.setattr(devices, "get_worker_processes", process_report)
    monkeypatch.setattr(
        devices,
        "control_worker",
        lambda worker, action: (
            {"status": "accepted", "action": "reboot"}
            if action == "reboot"
            else {"accepted": True, "node_id": worker.node_id, "action": action}
        ),
    )

    processes = client.get("/api/devices/workers/processes", headers=auth_headers)
    reboot = client.post("/api/devices/worker1/reboot", headers=auth_headers)
    stop_nas = client.post("/api/devices/worker1/stop-nas", headers=auth_headers)
    unauthenticated = client.get("/api/devices/workers/processes")

    assert processes.status_code == 200
    assert processes.json()["workers"][0]["node_id"] == "worker1"
    assert reboot.status_code == 200
    assert reboot.json() == {
        "status": "accepted",
        "action": "reboot",
    }
    assert stop_nas.status_code == 200
    assert stop_nas.json()["action"] == "stop_nas"
    assert unauthenticated.status_code == 401
