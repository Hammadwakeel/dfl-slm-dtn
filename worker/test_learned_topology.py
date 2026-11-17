from types import SimpleNamespace

import numpy as np
import pytest

from worker.training_service import WorkerTrainer, _cross_entropy, _validate_learned_senders

WORKERS = ("worker1", "worker2", "worker3", "worker4")


def test_learned_senders_are_validated() -> None:
    assert _validate_learned_senders({"worker1": ["worker3", "worker2"]}) == {"worker1": ("worker2", "worker3")}
    for bad in (None, {}, {"worker1": []}, {"worker1": ["worker1"]}, {"worker1": ["worker2", "worker2"]}):
        with pytest.raises(ValueError):
            _validate_learned_senders(bad)


def test_learned_graph_gives_each_worker_its_senders_and_receivers() -> None:
    trainer = SimpleNamespace(
        _learned_senders={
            "worker1": ("worker4",),
            "worker2": ("worker1",),
            "worker3": ("worker1", "worker2"),
            "worker4": ("worker3",),
        }
    )
    senders = lambda node: WorkerTrainer._round_senders(trainer, node, WORKERS, "learned")  # noqa: E731
    neighbours = lambda node: WorkerTrainer._round_neighbours(trainer, node, WORKERS, "learned")  # noqa: E731
    assert senders("worker3") == ("worker1", "worker2")
    assert neighbours("worker1") == ("worker2", "worker3")
    assert neighbours("worker4") == ("worker1",)
    # Named topologies keep their existing behaviour.
    assert WorkerTrainer._round_senders(trainer, "worker1", WORKERS, "directed_ring") == ("worker4",)
    trainer._learned_senders = {"worker1": ("worker9",)}
    with pytest.raises(ValueError):
        WorkerTrainer._round_senders(trainer, "worker1", WORKERS, "learned")


def test_cross_entropy_matches_the_definition() -> None:
    logits = np.array([[2.0, 0.0], [0.0, 3.0]])
    labels = np.array([0, 1])
    expected = -np.mean([np.log(np.exp(2) / (np.exp(2) + 1)), np.log(np.exp(3) / (np.exp(3) + 1))])
    assert _cross_entropy(labels, logits) == pytest.approx(expected)
