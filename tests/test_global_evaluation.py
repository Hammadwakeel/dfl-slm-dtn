import numpy as np
import pytest

from common.global_evaluation import classification_metrics, summarize


def test_metrics_expose_a_model_that_ignores_classes() -> None:
    labels = np.array([0, 0, 1, 1, 2, 2])
    predictions = np.array([0, 0, 1, 1, 1, 1])  # class 2 never predicted

    metrics = classification_metrics(labels, predictions, 3)

    assert metrics["accuracy"] == pytest.approx(4 / 6, abs=1e-6)
    assert metrics["per_class_recall"] == [1.0, 1.0, 0.0]
    # F1: class0 = 1, class1 = 2*0.5*1/1.5, class2 = 0
    assert metrics["macro_f1"] == pytest.approx((1 + 2 / 3 + 0) / 3, abs=1e-6)
    assert metrics["samples"] == 6


def test_metrics_reject_mismatched_inputs() -> None:
    with pytest.raises(ValueError):
        classification_metrics(np.array([0, 1]), np.array([0]), 2)


def test_summary_ignores_failed_workers() -> None:
    summary = summarize({
        "worker1": {"accuracy": 0.5, "macro_f1": 0.4},
        "worker2": {"accuracy": 0.7, "macro_f1": 0.6},
        "worker3": {"error": "bad weights"},
    })
    assert summary["workers_evaluated"] == 2
    assert summary["mean_accuracy"] == pytest.approx(0.6)
    assert summary["std_accuracy"] == pytest.approx(0.1)
    assert summarize({})["workers_evaluated"] == 0
