import json
import threading
from http.server import ThreadingHTTPServer
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import numpy as np
import pytest

from worker.training_service import (
    Peer,
    WorkerConfig,
    WorkerTrainer,
    make_handler,
    mesh_neighbours,
    ring_neighbours,
    serialize_weights,
    topology_neighbours,
    topology_senders,
    weighted_average_weights,
)


def test_ring_and_mesh_neighbour_sets_are_distinct():
    worker_ids = ("worker1", "worker2", "worker3")

    assert ring_neighbours("worker1", worker_ids) == ("worker2",)
    assert ring_neighbours("worker3", worker_ids) == ("worker1",)
    assert mesh_neighbours("worker1", worker_ids) == ("worker2", "worker3")


def test_all_candidate_topologies_have_expected_directed_adjacency():
    worker_ids = ("worker1", "worker2", "worker3")

    assert topology_neighbours("worker1", worker_ids, "directed_ring") == ("worker2",)
    assert topology_senders("worker2", worker_ids, "directed_ring") == ("worker1",)
    assert topology_neighbours("worker1", worker_ids, "bidirectional_ring") == (
        "worker2",
        "worker3",
    )
    assert topology_neighbours("worker2", worker_ids, "mesh") == ("worker1", "worker3")
    assert topology_neighbours("worker2", worker_ids, "chain") == ("worker1", "worker3")
    assert topology_neighbours("worker1", worker_ids, "star") == ("worker2", "worker3")
    assert topology_neighbours("worker3", worker_ids, "star") == ("worker1",)


def test_weighted_average_uses_sample_counts():
    averaged = weighted_average_weights(
        [
            ([np.array([1.0, 2.0], dtype=np.float32)], 2),
            ([np.array([4.0, 5.0], dtype=np.float32)], 1),
        ]
    )

    np.testing.assert_allclose(averaged[0], np.array([2.0, 3.0], dtype=np.float32))


def test_weighted_average_rejects_mismatched_shapes():
    with pytest.raises(ValueError, match="shape"):
        weighted_average_weights(
            [
                ([np.zeros((2,), dtype=np.float32)], 1),
                ([np.zeros((3,), dtype=np.float32)], 1),
            ]
        )


def test_peer_weight_endpoint_requires_peer_token_and_stores_payload(tmp_path):
    shard_path = tmp_path / "worker1.npz"
    np.savez(
        shard_path,
        images=np.zeros((3, 32, 32, 3), dtype=np.uint8),
        labels=np.array([0, 1, 2]),
    )
    token = "a" * 48
    peer_token = "b" * 48
    config = WorkerConfig(
        node_id="worker1",
        api_token=token,
        port=5004,
        data_path=str(shard_path),
        peers=(
            Peer("worker2", "http://127.0.0.1:5005", peer_token),
            Peer("worker3", "http://127.0.0.1:5006", "c" * 48),
        ),
        log_path=str(tmp_path / "worker.jsonl"),
    )
    trainer = WorkerTrainer(config)
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(config, trainer))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        payload = serialize_weights(
            [
                np.ones((3, 3, 3, 8), dtype=np.float32),
                np.ones((8,), dtype=np.float32),
                np.ones((512, 10), dtype=np.float32),
                np.ones((10,), dtype=np.float32),
            ]
        )
        headers = {
            "Authorization": f"Bearer {peer_token}",
            "Content-Type": "application/octet-stream",
            "X-Run-ID": "test-run",
            "X-Round": "0",
            "X-Sender-Node": "worker2",
            "X-Sample-Count": "3",
        }
        request = Request(
            f"http://127.0.0.1:{server.server_port}/v1/weights",
            data=payload,
            headers=headers,
            method="POST",
        )
        with urlopen(request, timeout=3) as response:
            assert response.status == 202
        assert trainer._incoming[("test-run", 0)]["worker2"][1] == 3

        bad_request = Request(
            f"http://127.0.0.1:{server.server_port}/v1/weights",
            data=payload,
            headers={**headers, "Authorization": "Bearer " + "x" * 48},
            method="POST",
        )
        with pytest.raises(HTTPError) as error:
            urlopen(bad_request, timeout=3)
        assert error.value.code == 401
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)
