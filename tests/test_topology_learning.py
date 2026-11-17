import numpy as np
import pytest

from topology_opt.learning import TopologyLearningError, _strongly_connected, learn_senders, measurement_matrices

WORKERS = [f"worker{index}" for index in range(1, 6)]


def probe_row(bad_sender: str = "worker5", slow_sender: str = "worker4") -> dict:
    row = {}
    for receiver in WORKERS:
        others = [worker for worker in WORKERS if worker != receiver]
        row[receiver] = {
            # One sender's model generalises badly to every peer.
            "peer_losses": {sender: (3.0 if sender == bad_sender else 0.4) for sender in others},
            # One sender always arrives late.
            "peer_arrival_seconds": {sender: (40.0 if sender == slow_sender else 1.0) for sender in others},
            "class_histogram": [10, 5] if receiver in {"worker1", "worker2"} else [5, 10],
        }
    return row


def test_learned_topology_avoids_bad_and_slow_senders_and_stays_connected() -> None:
    senders, diagnostics = learn_senders(WORKERS, probe_row())

    assert set(senders) == set(WORKERS)
    assert all(senders[receiver] and receiver not in senders[receiver] for receiver in WORKERS)
    receivers_of_bad = [receiver for receiver, chosen in senders.items() if "worker5" in chosen]
    receivers_of_good = [receiver for receiver, chosen in senders.items() if "worker2" in chosen]
    assert len(receivers_of_bad) <= len(receivers_of_good)
    assert diagnostics["edges"] < diagnostics["mesh_edges"]
    n = len(WORKERS)
    adjacency = np.zeros((n, n), dtype=np.int8)
    for receiver, chosen in senders.items():
        for sender in chosen:
            adjacency[WORKERS.index(receiver), WORKERS.index(sender)] = 1
    assert _strongly_connected(adjacency)


def test_missing_measurements_count_as_the_worst_link() -> None:
    row = probe_row()
    del row["worker1"]["peer_losses"]["worker2"]
    loss, wait, theta = measurement_matrices(WORKERS, row)
    assert loss[0, 1] == pytest.approx(3.0)
    assert np.allclose(theta.sum(axis=1), 1.0)
    assert np.all(np.diag(loss) == 0) and np.all(np.diag(wait) == 0)


def test_probe_without_class_histograms_is_rejected() -> None:
    row = probe_row()
    row["worker3"]["class_histogram"] = []
    with pytest.raises(TopologyLearningError):
        learn_senders(WORKERS, row)


def test_strong_connectivity_check() -> None:
    ring = np.zeros((3, 3), dtype=np.int8)
    for receiver in range(3):
        ring[receiver, (receiver - 1) % 3] = 1
    assert _strongly_connected(ring)
    star_in = np.zeros((3, 3), dtype=np.int8)
    star_in[0, 1] = star_in[0, 2] = star_in[1, 0] = star_in[2, 0] = 1
    assert _strongly_connected(star_in)
    one_way = np.zeros((3, 3), dtype=np.int8)
    one_way[1, 0] = one_way[2, 0] = one_way[0, 1] = 1
    assert not _strongly_connected(one_way)
