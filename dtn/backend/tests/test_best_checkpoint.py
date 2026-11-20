"""The supervisor run's best-model checkpoint: save from a worker, install on all."""

import base64
import io
from types import SimpleNamespace

import numpy as np
import pytest

from app.services import best_checkpoint
from app.services.initial_models import InitialModelError

WEIGHTS = [np.arange(6, dtype=np.float32).reshape(2, 3), np.ones(3, dtype=np.float32)]


def _encoded(weights) -> str:
    buffer = io.BytesIO()
    np.savez(buffer, **{f"weight_{index}": value for index, value in enumerate(weights)})
    return base64.b64encode(buffer.getvalue()).decode()


def _workers(*names):
    return [SimpleNamespace(node_id=name, role="worker") for name in names]


@pytest.fixture
def cluster(tmp_path, monkeypatch):
    models = {
        "worker1": {"run_id": "r2", "saved_at": 2, "validation_accuracy": 0.90, "architecture": {"id": "w1"}, "weights_b64": _encoded(WEIGHTS)},
        "worker2": {"run_id": "r2", "saved_at": 2, "validation_accuracy": 0.80, "architecture": {"id": "w2"}, "weights_b64": _encoded(WEIGHTS)},
    }
    installs = []
    monkeypatch.setattr(best_checkpoint, "get_worker_latest_model", lambda worker, dataset: models[worker.node_id])

    def distribute(dataset_store, workers, dataset_name, **kwargs):
        installs.append({"dataset_name": dataset_name, **kwargs})
        return {"architecture_sha256": "a", "weights_sha256": "w"}

    monkeypatch.setattr(best_checkpoint, "create_and_distribute_initial_model", distribute)
    return SimpleNamespace(store=SimpleNamespace(root=tmp_path), models=models, installs=installs)


def test_the_preferred_worker_model_is_saved_and_installed_with_its_weights(cluster) -> None:
    workers = _workers("worker1", "worker2")
    saved = best_checkpoint.save_best_checkpoint(cluster.store, workers, "mnist", checkpoint_id="run-1", source_worker="worker2")
    assert saved["saved"] is True and saved["source_worker"] == "worker2"
    # The worker trains something newer afterwards; the checkpoint keeps the saved model.
    cluster.models["worker2"]["architecture"] = {"id": "newer"}

    installed = best_checkpoint.install_best_checkpoint(cluster.store, workers, "mnist", checkpoint_id="run-1", seed=3)

    assert installed["installed"] is True and installed["source_worker"] == "worker2"
    install = cluster.installs[-1]
    assert install["architecture"] == {"id": "w2"} and install["replace_existing"] is True and install["seed"] == 3
    assert all(np.array_equal(a, b) for a, b in zip(install["initial_weights"], WEIGHTS))


def test_a_checkpoint_of_another_run_or_none_is_never_installed(cluster) -> None:
    workers = _workers("worker1", "worker2")
    with pytest.raises(InitialModelError, match="No best checkpoint"):
        best_checkpoint.install_best_checkpoint(cluster.store, workers, "mnist", checkpoint_id="run-1")
    best_checkpoint.save_best_checkpoint(cluster.store, workers, "mnist", checkpoint_id="run-1")
    with pytest.raises(InitialModelError, match="another run"):
        best_checkpoint.install_best_checkpoint(cluster.store, workers, "mnist", checkpoint_id="run-2")
    assert cluster.installs == []


def test_invalid_checkpoint_ids_and_weights_are_rejected(cluster) -> None:
    workers = _workers("worker1", "worker2")
    with pytest.raises(InitialModelError, match="checkpoint_id"):
        best_checkpoint.save_best_checkpoint(cluster.store, workers, "mnist", checkpoint_id="../x")
    cluster.models["worker1"]["weights_b64"] = base64.b64encode(b"not an npz").decode()
    with pytest.raises(InitialModelError, match="unreadable"):
        best_checkpoint.save_best_checkpoint(cluster.store, workers, "mnist", checkpoint_id="run-1", source_worker="worker1")
    assert not (cluster.store.root / "mnist" / "models" / best_checkpoint.BEST_CHECKPOINT_FILE).exists()
