import numpy as np
import pytest

from topology_opt.sharding import (
    constrained_dirichlet_indices,
    iid_stratified_indices,
    stratified_local_validation_split,
)


@pytest.fixture
def mnist_train_labels() -> np.ndarray:
    return np.repeat(np.arange(10, dtype=np.int64), 240)


def _assert_partition(shards: list[np.ndarray]) -> None:
    assert len(shards) == 8
    assert all(len(indices) == 300 for indices in shards)
    merged = np.concatenate(shards)
    assert merged.size == 2400
    assert np.array_equal(np.sort(merged), np.arange(2400))


def test_iid_shards_are_eight_equal_stratified_train_only_shards(
    mnist_train_labels: np.ndarray,
) -> None:
    shards = iid_stratified_indices(mnist_train_labels, seed=2026)
    _assert_partition(shards)
    for indices in shards:
        assert np.array_equal(np.bincount(mnist_train_labels[indices], minlength=10), np.full(10, 30))


def test_constrained_dirichlet_shards_respect_node_and_class_capacities(
    mnist_train_labels: np.ndarray,
) -> None:
    shards = constrained_dirichlet_indices(mnist_train_labels, alpha=0.1, seed=2026)
    _assert_partition(shards)
    class_counts = np.asarray(
        [np.bincount(mnist_train_labels[indices], minlength=10) for indices in shards]
    )
    assert np.array_equal(class_counts.sum(axis=0), np.full(10, 240))
    assert np.array_equal(class_counts.sum(axis=1), np.full(8, 300))
    assert not np.all(class_counts == 30)


def test_constrained_dirichlet_sharding_is_deterministic(
    mnist_train_labels: np.ndarray,
) -> None:
    first = constrained_dirichlet_indices(mnist_train_labels, alpha=0.1, seed=2026)
    second = constrained_dirichlet_indices(mnist_train_labels, alpha=0.1, seed=2026)
    assert all(np.array_equal(left, right) for left, right in zip(first, second))


@pytest.mark.parametrize("sharder", [iid_stratified_indices, constrained_dirichlet_indices])
def test_each_node_has_240_train_and_60_disjoint_local_validation_rows(
    mnist_train_labels: np.ndarray,
    sharder,
) -> None:
    shards = sharder(mnist_train_labels, seed=2026)
    for node_index, indices in enumerate(shards):
        local_labels = mnist_train_labels[indices]
        split = stratified_local_validation_split(local_labels, seed=17 + node_index)
        assert len(split.train_indices) == 240
        assert len(split.validation_indices) == 60
        assert not np.intersect1d(split.train_indices, split.validation_indices).size
        assert set(split.train_indices) | set(split.validation_indices) == set(range(300))


def test_sharding_rejects_non_train_row_count(mnist_train_labels: np.ndarray) -> None:
    with pytest.raises(ValueError, match="exactly 2,400"):
        iid_stratified_indices(mnist_train_labels[:-1])