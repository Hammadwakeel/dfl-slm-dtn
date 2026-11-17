import base64
import json
import threading

import pytest

from worker.training_service import (
    Peer,
    WorkerConfig,
    WorkerTrainer,
    topology_neighbours,
    topology_senders,
)


def test_worker4_config_accepts_all_three_peer_nodes(monkeypatch) -> None:
    peers = [
        {
            "node_id": f"worker{index}",
            "base_url": f"http://127.0.0.1:{5003 + index}",
            "token": f"worker{index}-token-" + "x" * 32,
        }
        for index in range(1, 4)
    ]
    monkeypatch.setenv("DFL_NODE_ID", "worker4")
    monkeypatch.setenv("DFL_API_TOKEN", "worker4-token-" + "x" * 32)
    monkeypatch.setenv(
        "DFL_PEERS_JSON_B64",
        base64.b64encode(json.dumps(peers).encode("utf-8")).decode("ascii"),
    )
    monkeypatch.delenv("DFL_PEER_TIMEOUT_SECONDS", raising=False)

    config = WorkerConfig.from_env()

    assert config.node_id == "worker4"
    assert {peer.node_id for peer in config.peers} == {"worker1", "worker2", "worker3"}
    assert config.peer_timeout_seconds == 600


def test_four_worker_topology_has_correct_ring_and_mesh_neighbours() -> None:
    worker_ids = ("worker1", "worker2", "worker3", "worker4")

    assert topology_neighbours("worker4", worker_ids, "directed_ring") == ("worker1",)
    assert topology_senders("worker1", worker_ids, "directed_ring") == ("worker4",)
    assert topology_neighbours("worker1", worker_ids, "bidirectional_ring") == (
        "worker2",
        "worker4",
    )
    assert topology_neighbours("worker4", worker_ids, "mesh") == (
        "worker1",
        "worker2",
        "worker3",
    )


def test_worker_runtime_peer_configuration_accepts_a_new_sequential_worker() -> None:
    config = WorkerConfig(
        node_id="worker4",
        api_token="worker4-token-" + "x" * 32,
        port=5004,
        data_path="unused.npz",
        peers=(),
        log_path="unused.log",
    )
    trainer = WorkerTrainer.__new__(WorkerTrainer)
    trainer.config = config
    trainer._condition = threading.Condition()
    trainer._active_run = None
    trainer._scheduled_run = None
    trainer._peer_map = {}
    peers = tuple(
        Peer(
            node_id=f"worker{index}",
            base_url=f"http://127.0.0.1:{5004 + index}",
            token=f"worker{index}-token-" + "x" * 32,
        )
        for index in (1, 2, 3, 5)
    )

    result = trainer.configure_peers(peers)

    assert result["configured_peers"] == ["worker1", "worker2", "worker3", "worker5"]
    assert trainer.config.peers == peers
    assert set(trainer._peer_map) == {"worker1", "worker2", "worker3", "worker5"}


def test_worker_runtime_peer_configuration_rejects_changes_during_training() -> None:
    config = WorkerConfig(
        node_id="worker4",
        api_token="worker4-token-" + "x" * 32,
        port=5004,
        data_path="unused.npz",
        peers=(),
        log_path="unused.log",
    )
    trainer = WorkerTrainer.__new__(WorkerTrainer)
    trainer.config = config
    trainer._condition = threading.Condition()
    trainer._active_run = "run-1"
    trainer._scheduled_run = None
    trainer._peer_map = {}
    peers = tuple(
        Peer(
            node_id=f"worker{index}",
            base_url=f"http://127.0.0.1:{5004 + index}",
            token=f"worker{index}-token-" + "x" * 32,
        )
        for index in (1, 2, 3)
    )

    with pytest.raises(RuntimeError, match="training is active"):
        trainer.configure_peers(peers)
