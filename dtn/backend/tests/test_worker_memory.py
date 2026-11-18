from app.core.config import DeviceConfig
from app.services import worker_memory


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


def test_memory_cleanup_does_not_start_if_any_worker_fails_preflight(monkeypatch):
    cleaned = []

    def check(worker, *, timeout_seconds):
        if worker.node_id == "worker2":
            raise worker_memory.WorkerShardError("worker2 has a training job running")
        return {"node_id": worker.node_id}

    monkeypatch.setattr(worker_memory, "get_worker_training_status", check)
    monkeypatch.setattr(
        worker_memory,
        "_cleanup_worker",
        lambda worker, timeout_seconds: cleaned.append(worker.node_id),
    )

    result = worker_memory.cleanup_worker_memory_cluster(_workers())

    assert result["complete"] is False
    assert result["phase"] == "preflight"
    assert "worker2 has a training job running" in result["workers"][1]["error"]
    assert all(not worker["success"] for worker in result["workers"])
    assert cleaned == []


def test_memory_cleanup_reports_per_worker_partial_failures(monkeypatch):
    monkeypatch.setattr(
        worker_memory,
        "get_worker_training_status",
        lambda worker, *, timeout_seconds: {"node_id": worker.node_id},
    )

    def cleanup(worker, timeout_seconds, recycle=False):
        if worker.node_id == "worker3":
            return {
                "node_id": worker.node_id,
                "success": False,
                "cleaned_runs": 2,
                "process_memory_cleaned": True,
                "memory_available_before_mb": 100.0,
                "memory_available_after_mb": 150.0,
                "process_rss_before_mb": 75.0,
                "process_rss_after_mb": 40.0,
                "os_page_cache_dropped": False,
                "error": "cache helper is unavailable",
            }
        return {
            "node_id": worker.node_id,
            "success": True,
            "cleaned_runs": 2,
            "process_memory_cleaned": True,
            "memory_available_before_mb": 100.0,
            "memory_available_after_mb": 150.0,
            "process_rss_before_mb": 75.0,
            "process_rss_after_mb": 40.0,
            "os_page_cache_dropped": True,
        }

    monkeypatch.setattr(worker_memory, "_cleanup_worker", cleanup)

    result = worker_memory.cleanup_worker_memory_cluster(_workers())

    assert result["complete"] is False
    assert result["phase"] == "cleanup"
    assert [worker["node_id"] for worker in result["workers"]] == [
        "worker1",
        "worker2",
        "worker3",
    ]
    assert result["workers"][0]["memory_available_after_mb"] == 150.0
    assert result["workers"][2]["success"] is False
    assert result["workers"][2]["error"] == "cache helper is unavailable"
    assert result["workers"][2]["process_memory_cleaned"] is True
