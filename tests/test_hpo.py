import pytest

from common.hpo import hpo_candidates, select_best_hyperparameters
from common.model_design import validate_model_design

BASE = {
    "model_type": "custom",
    "id": "nas-d2-w32x64-k3-conv-h64-drop20",
    "dataset_name": "mnist",
    "input_shape": [28, 28, 1],
    "classes": [str(index) for index in range(10)],
    "num_classes": 10,
    "parameter_count": 1,
    "normalization": {"method": "minmax"},
    "layers": [
        {"type": "conv2d", "filters": 8, "kernel_size": 3, "padding": "same", "activation": "relu"},
        {"type": "global_average_pooling2d"},
        {"type": "dense", "units": 10, "activation": "linear"},
    ],
    "training": {"optimizer": "adam", "learning_rate": 0.001, "batch_size": 32},
}


def test_mnist_hpo_grid_varies_only_training_settings() -> None:
    candidates = hpo_candidates(BASE, input_shape=(28, 28, 1), num_classes=10)

    assert len(candidates) == 6
    assert len({candidate["id"] for candidate in candidates}) == 6
    assert {candidate["training"]["learning_rate"] for candidate in candidates} == {0.0003, 0.001, 0.003}
    assert {candidate["training"]["batch_size"] for candidate in candidates} == {32, 64}
    _, parameters = validate_model_design(BASE, (28, 28, 1), 10)
    for candidate in candidates:
        assert candidate["layers"] == BASE["layers"]
        assert candidate["parameter_count"] == parameters
        # Bookkeeping keys are dropped so workers validate a clean design.
        assert "dataset_name" not in candidate and "normalization" not in candidate
    assert "hpo-adam-lr3e-4-bs32" in {candidate["id"] for candidate in candidates}


def test_memory_constrained_hpo_keeps_batch_size() -> None:
    base = {**BASE, "training": {"optimizer": "adam", "learning_rate": 0.001, "batch_size": 8}}
    candidates = hpo_candidates(base, input_shape=(28, 28, 1), num_classes=10, memory_constrained=True)

    assert len(candidates) == 3
    assert {candidate["training"]["batch_size"] for candidate in candidates} == {8}


def test_select_best_hyperparameters_prefers_accuracy_then_f1() -> None:
    rows = [
        {"architecture": {"id": "a"}, "validation_accuracy": 0.9, "validation_macro_f1": 0.8, "elapsed_seconds": 5},
        {"architecture": {"id": "b"}, "validation_accuracy": 0.92, "validation_macro_f1": 0.7, "elapsed_seconds": 9},
        {"architecture": {"id": "c"}, "validation_accuracy": 0.92, "validation_macro_f1": 0.75, "elapsed_seconds": 9},
    ]
    assert select_best_hyperparameters(rows)["architecture"]["id"] == "c"
    with pytest.raises(ValueError):
        select_best_hyperparameters([])
