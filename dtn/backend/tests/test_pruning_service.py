import pytest

from app.services.initial_models import InitialModelError
from app.services.pruning import select_source_model


def test_source_model_is_the_best_worker_of_the_newest_run() -> None:
    models = {
        "worker1": {"run_id": "old", "saved_at": 10, "validation_accuracy": 0.99},
        "worker2": {"run_id": "new", "saved_at": 20, "validation_accuracy": 0.91},
        "worker3": {"run_id": "new", "saved_at": 21, "validation_accuracy": 0.94},
        "worker4": None,
    }
    node, model = select_source_model(models)
    assert node == "worker3" and model["validation_accuracy"] == 0.94


def test_source_model_requires_a_trained_worker() -> None:
    with pytest.raises(InitialModelError, match="train first"):
        select_source_model({"worker1": None})


def test_restore_source_model_redeploys_the_saved_unpruned_weights(tmp_path, monkeypatch) -> None:
    import base64
    import io
    import json
    from types import SimpleNamespace

    import numpy as np

    from app.services import pruning

    weights = [np.arange(6, dtype=np.float32).reshape(2, 3), np.ones(3, dtype=np.float32)]
    buffer = io.BytesIO()
    np.savez(buffer, **{f"weight_{index}": value for index, value in enumerate(weights)})
    store = SimpleNamespace(root=tmp_path)
    path = pruning._source_path(store, "mnist")
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({
        "architecture": {"id": "nas-x"},
        "weights_b64": base64.b64encode(buffer.getvalue()).decode(),
        "source_worker": "worker2",
    }))
    calls = {}

    def fake_distribute(dataset_store, workers, dataset_name, **kwargs):
        calls.update(kwargs, dataset_name=dataset_name)
        return {"architecture_sha256": "a", "weights_sha256": "w"}

    monkeypatch.setattr(pruning, "create_and_distribute_initial_model", fake_distribute)
    report = pruning.restore_source_model(store, [], "mnist")

    assert report["restored"] is True and report["source_worker"] == "worker2"
    assert calls["architecture"] == {"id": "nas-x"} and calls["replace_existing"] is True
    assert all(np.array_equal(a, b) for a, b in zip(calls["initial_weights"], weights))


def test_restore_without_a_saved_source_is_rejected(tmp_path) -> None:
    from types import SimpleNamespace

    from app.services import pruning
    from app.services.initial_models import InitialModelError

    with pytest.raises(InitialModelError):
        pruning.restore_source_model(SimpleNamespace(root=tmp_path), [], "mnist")
