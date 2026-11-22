import numpy as np

from common.cifar_model import classification_metrics
from scripts.prepare_cifar10 import _dirichlet_indices, _stratified_indices


def test_stratified_split_is_deterministic_disjoint_and_covers_every_class():
    labels = np.repeat(np.arange(10), 12)
    first = _stratified_indices(labels, seed=2026)
    second = _stratified_indices(labels, seed=2026)

    assert all(np.array_equal(left, right) for left, right in zip(first, second))
    assert sorted(np.concatenate(first).tolist()) == list(range(len(labels)))
    for shard in first:
        assert set(labels[shard]) == set(range(10))
        assert np.unique(shard).size == shard.size


def test_dirichlet_split_is_deterministic_and_disjoint():
    labels = np.repeat(np.arange(10), 100)
    first = _dirichlet_indices(labels, seed=7, alpha=0.5)
    second = _dirichlet_indices(labels, seed=7, alpha=0.5)

    assert all(np.array_equal(left, right) for left, right in zip(first, second))
    assert sorted(np.concatenate(first).tolist()) == list(range(len(labels)))


def test_classification_metrics_report_classification_scores_and_confusion_matrix():
    labels = np.array([0, 1, 2, 3])
    logits = np.full((4, 10), -4.0)
    logits[np.arange(4), labels] = 4.0

    metrics = classification_metrics(labels, logits)

    assert metrics["accuracy"] == 1.0
    assert metrics["macro_precision"] == 0.4
    assert metrics["macro_recall"] == 0.4
    assert metrics["macro_f1"] == 0.4
    assert metrics["confusion_matrix_counts"][0][0] == 1
    assert metrics["confusion_matrix_row_normalized_percent"][0][0] == 100.0
    assert metrics["loss"] < 0.01


def test_classification_metrics_support_more_than_ten_classes():
    labels = np.array([0, 61, 30, 30])
    logits = np.zeros((4, 62))
    logits[np.arange(4), [0, 61, 30, 5]] = 5.0
    metrics = classification_metrics(labels, logits)
    assert metrics["accuracy"] == 0.75
    assert len(metrics["confusion_matrix_counts"]) == 62
    assert len(metrics["per_class_f1"]) == 62
