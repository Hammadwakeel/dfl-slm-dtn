import pytest
import numpy as np
import json

from common.nas import (
    ARCHITECTURE_SPACE,
    FIXED_ARCHITECTURE,
    MAX_PARAMETERS,
    architecture_parameters,
    architecture_spec,
    select_architecture,
)
from worker import nas_search
from worker.nas_search import _split_indices, search_cifar10_architectures


def test_search_space_is_bounded_and_contains_fixed_architecture() -> None:
    assert len(ARCHITECTURE_SPACE) == 8
    assert FIXED_ARCHITECTURE in {row["id"] for row in ARCHITECTURE_SPACE}
    assert max(architecture_parameters(row["id"]) for row in ARCHITECTURE_SPACE) == 11_466
    assert max(architecture_parameters(row["id"]) for row in ARCHITECTURE_SPACE) < MAX_PARAMETERS


def test_architecture_validation_rejects_modified_or_unknown_candidate() -> None:
    with pytest.raises(ValueError):
        architecture_spec("b4_c128_k9")
    with pytest.raises(ValueError):
        architecture_spec({"id": "b1_c8_k3", "blocks": 2, "channels": 8, "kernel_size": 3})


def test_selector_uses_mean_worker_validation_accuracy_with_simple_tie_break() -> None:
    scores = []
    for architecture in ARCHITECTURE_SPACE:
        accuracy = 0.7 if architecture["id"] == "b1_c8_k3" else 0.6
        for _ in range(3):
            scores.append({
                "architecture": architecture,
                "parameter_count": architecture_parameters(architecture["id"]),
                "validation_accuracy": accuracy,
                "elapsed_seconds": 2.0,
                "peak_rss_mb": 1024.0,
            })
    selected = select_architecture(scores)
    assert selected["architecture"]["id"] == "b1_c8_k3"
    assert selected["mean_validation_accuracy"] == pytest.approx(0.7)


def test_selector_rejects_missing_worker_candidate_results() -> None:
    candidate = ARCHITECTURE_SPACE[0]
    with pytest.raises(ValueError):
        select_architecture([{
            "architecture": candidate,
            "parameter_count": architecture_parameters(candidate["id"]),
            "validation_accuracy": 0.5,
            "elapsed_seconds": 2.0,
            "peak_rss_mb": 1024.0,
        }])


def test_worker_validation_split_is_disjoint_stratified_and_repeatable() -> None:
    labels = np.repeat(np.arange(10), 100)
    training, validation = _split_indices(labels, 2026)
    repeated_training, repeated_validation = _split_indices(labels, 2026)

    assert np.array_equal(training, repeated_training)
    assert np.array_equal(validation, repeated_validation)
    assert set(training).isdisjoint(set(validation))
    assert len(training) + len(validation) == len(labels)
    assert len(validation) == 200
    assert all(np.count_nonzero(labels[validation] == label) == 20 for label in range(10))


def test_worker_search_rejects_time_limit_outside_hard_bounds() -> None:
    with pytest.raises(ValueError, match="Search time limit"):
        search_cifar10_architectures(
            np.zeros((10, 32, 32, 3), dtype=np.uint8),
            np.zeros((10,), dtype=np.int64),
            seed=1,
            time_limit_seconds=301,
            node_id="worker1",
            logger=lambda *args, **kwargs: None,
        )


def test_isolated_search_stream_events_are_logged_and_result_is_returned(monkeypatch) -> None:
    candidate = {
        "architecture": ARCHITECTURE_SPACE[0],
        "parameter_count": architecture_parameters(ARCHITECTURE_SPACE[0]["id"]),
        "validation_accuracy": 0.5,
        "elapsed_seconds": 2.0,
        "peak_rss_mb": 1024.0,
    }
    output = "\n".join(
        (
            json.dumps({"type": "nas_candidate_evaluated", **candidate}),
            json.dumps({"type": "nas_search_result", "data": {"candidates": [candidate]}}),
        )
    )
    called = {}

    def run(command, **kwargs):
        called.update(command=command, kwargs=kwargs)
        return nas_search.subprocess.CompletedProcess(command, 0, output, "")

    monkeypatch.setenv("DFL_API_TOKEN", "do-not-forward")
    monkeypatch.setattr(nas_search.subprocess, "run", run)
    logged = []
    result = nas_search.search_cifar10_architectures_isolated(
        "/worker/train.npz",
        seed=2026,
        time_limit_seconds=60,
        node_id="worker1",
        logger=lambda name, **details: logged.append((name, details)),
    )

    assert result == {"candidates": [candidate]}
    assert logged == [("nas_candidate_evaluated", candidate)]
    assert called["kwargs"]["timeout"] == 90
    assert "DFL_API_TOKEN" not in called["kwargs"]["env"]
