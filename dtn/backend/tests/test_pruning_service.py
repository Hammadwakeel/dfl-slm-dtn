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
