"""Deterministic eight-node MNIST sharding over TRAIN labels only."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


NODE_COUNT = 8
SHARD_ROWS = 300
LOCAL_VALIDATION_ROWS = 60
SHARDING_SEED = 2026
MNIST_CLASS_COUNT = 10
MNIST_TRAIN_ROWS_PER_CLASS = 240


@dataclass(frozen=True)
class LocalSplit:
    train_indices: np.ndarray
    validation_indices: np.ndarray
    train_class_counts: tuple[int, ...]
    validation_class_counts: tuple[int, ...]


def _validate_train_labels(labels: np.ndarray, node_count: int, shard_rows: int) -> np.ndarray:
    values = np.asarray(labels, dtype=np.int64).reshape(-1)
    if node_count != NODE_COUNT or shard_rows != SHARD_ROWS:
        raise ValueError("Phase B sharding is specified for eight 300-row MNIST nodes")
    if values.shape != (node_count * shard_rows,):
        raise ValueError("TRAIN labels must contain exactly 2,400 rows")
    counts = np.bincount(values, minlength=MNIST_CLASS_COUNT)
    if values.min() != 0 or values.max() != MNIST_CLASS_COUNT - 1:
        raise ValueError("MNIST TRAIN labels must be integers 0..9")
    if not np.all(counts == MNIST_TRAIN_ROWS_PER_CLASS):
        raise ValueError("Expected exactly 240 MNIST TRAIN rows per class")
    return values


def iid_stratified_indices(
    labels: np.ndarray,
    *,
    seed: int = SHARDING_SEED,
    node_count: int = NODE_COUNT,
    shard_rows: int = SHARD_ROWS,
) -> list[np.ndarray]:
    """Split each class evenly across nodes using only the supplied TRAIN labels."""
    values = _validate_train_labels(labels, node_count, shard_rows)
    rng = np.random.default_rng(seed)
    shards: list[list[int]] = [[] for _ in range(node_count)]
    for label in range(MNIST_CLASS_COUNT):
        indices = np.flatnonzero(values == label)
        rng.shuffle(indices)
        for node_index, node_indices in enumerate(np.array_split(indices, node_count)):
            shards[node_index].extend(int(index) for index in node_indices)
    result = [np.asarray(sorted(indices), dtype=np.int64) for indices in shards]
    _validate_partition(result, values.size, shard_rows)
    return result


def constrained_dirichlet_indices(
    labels: np.ndarray,
    *,
    alpha: float = 0.1,
    seed: int = SHARDING_SEED,
    node_count: int = NODE_COUNT,
    shard_rows: int = SHARD_ROWS,
) -> list[np.ndarray]:
    """Draw per-class Dirichlet proportions, allocating under exact node capacities.

    For each class, 240 TRAIN indices are assigned one at a time according to a
    Dirichlet(alpha) draw, renormalized over nodes that still have row capacity.
    This constrained procedure preserves both 240 rows/class and 300 rows/node.
    """
    values = _validate_train_labels(labels, node_count, shard_rows)
    if not np.isfinite(alpha) or alpha <= 0:
        raise ValueError("Dirichlet alpha must be positive and finite")
    rng = np.random.default_rng(seed)
    row_capacity = np.full(node_count, shard_rows, dtype=np.int64)
    class_allocations = np.zeros((node_count, MNIST_CLASS_COUNT), dtype=np.int64)
    class_indices_by_node: list[list[int]] = [[] for _ in range(node_count)]

    for label in range(MNIST_CLASS_COUNT):
        proportions = rng.dirichlet(np.full(node_count, alpha, dtype=np.float64))
        class_indices = np.flatnonzero(values == label)
        rng.shuffle(class_indices)
        assignments: list[list[int]] = [[] for _ in range(node_count)]
        for index in class_indices:
            eligible = np.flatnonzero(row_capacity > 0)
            if eligible.size == 0:
                raise ValueError("Constrained Dirichlet allocation exhausted node capacities")
            probabilities = proportions[eligible]
            probabilities = probabilities / probabilities.sum()
            node_index = int(rng.choice(eligible, p=probabilities))
            assignments[node_index].append(int(index))
            row_capacity[node_index] -= 1
            class_allocations[node_index, label] += 1
        for node_index, node_indices in enumerate(assignments):
            class_indices_by_node[node_index].extend(node_indices)

    if not np.all(row_capacity == 0):
        raise ValueError(f"Constrained Dirichlet row capacities were not satisfied: {row_capacity.tolist()}")
    if not np.all(class_allocations.sum(axis=0) == MNIST_TRAIN_ROWS_PER_CLASS):
        raise ValueError("Constrained Dirichlet class capacities were not satisfied")
    result = [np.asarray(sorted(indices), dtype=np.int64) for indices in class_indices_by_node]
    _validate_partition(result, values.size, shard_rows)
    return result


def stratified_local_validation_split(
    shard_labels: np.ndarray,
    *,
    validation_rows: int = LOCAL_VALIDATION_ROWS,
    seed: int,
) -> LocalSplit:
    """Hold out exactly 60 rows, proportionally stratified within one node shard."""
    labels = np.asarray(shard_labels, dtype=np.int64).reshape(-1)
    if labels.shape != (SHARD_ROWS,):
        raise ValueError("Each local shard must contain exactly 300 TRAIN rows")
    if validation_rows != LOCAL_VALIDATION_ROWS:
        raise ValueError("Phase B local validation is specified as exactly 60 rows")
    if labels.min() < 0 or labels.max() >= MNIST_CLASS_COUNT:
        raise ValueError("Local MNIST labels must be integers 0..9")

    classes, counts = np.unique(labels, return_counts=True)
    validation_capacity = counts - 1
    if validation_capacity.sum() < validation_rows:
        raise ValueError("Constrained local validation cannot preserve at least one TRAIN row per class")
    ideal = validation_rows * counts.astype(np.float64) / labels.size
    allocation = np.minimum(np.floor(ideal).astype(np.int64), validation_capacity)
    remaining = validation_rows - int(allocation.sum())
    while remaining:
        eligible = np.flatnonzero(allocation < validation_capacity)
        if eligible.size == 0:
            raise ValueError("Could not allocate exactly 60 local validation rows")
        residual = ideal[eligible] - allocation[eligible]
        selected = int(eligible[int(np.argmax(residual))])
        allocation[selected] += 1
        remaining -= 1

    rng = np.random.default_rng(seed)
    train_indices: list[int] = []
    validation_indices: list[int] = []
    validation_counts = np.zeros(MNIST_CLASS_COUNT, dtype=np.int64)
    train_counts = np.zeros(MNIST_CLASS_COUNT, dtype=np.int64)
    for class_index, label in enumerate(classes):
        indices = np.flatnonzero(labels == label)
        rng.shuffle(indices)
        count = int(allocation[class_index])
        validation_indices.extend(int(value) for value in indices[:count])
        train_indices.extend(int(value) for value in indices[count:])
        validation_counts[int(label)] = count
        train_counts[int(label)] = len(indices) - count

    train_result = np.asarray(sorted(train_indices), dtype=np.int64)
    validation_result = np.asarray(sorted(validation_indices), dtype=np.int64)
    if len(train_result) != SHARD_ROWS - validation_rows or len(validation_result) != validation_rows:
        raise AssertionError("Local train/validation sizes do not match the fixed split")
    if np.intersect1d(train_result, validation_result).size:
        raise AssertionError("Local training and validation rows overlap")
    return LocalSplit(
        train_indices=train_result,
        validation_indices=validation_result,
        train_class_counts=tuple(int(value) for value in train_counts),
        validation_class_counts=tuple(int(value) for value in validation_counts),
    )


def _validate_partition(shards: list[np.ndarray], total_rows: int, shard_rows: int) -> None:
    if len(shards) != NODE_COUNT or any(len(shard) != shard_rows for shard in shards):
        raise ValueError("Each of eight nodes must receive exactly 300 TRAIN rows")
    concatenated = np.concatenate(shards)
    if not np.array_equal(np.sort(concatenated), np.arange(total_rows, dtype=np.int64)):
        raise ValueError("Shards must be disjoint and cover the supplied TRAIN rows exactly")