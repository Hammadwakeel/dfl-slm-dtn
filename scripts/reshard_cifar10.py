"""Repartition existing disjoint CIFAR-10 worker shards for a larger cluster."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_SHARDS = (
    ("stratified", REPO_ROOT / "data" / "worker_shards"),
    ("dirichlet_a03", REPO_ROOT / "data" / "worker_shards" / "dirichlet_a03"),
    ("dirichlet_a05", REPO_ROOT / "data" / "worker_shards" / "dirichlet_a05"),
)


def _load_source_shards(source_root: Path) -> tuple[np.ndarray, np.ndarray]:
    shards = []
    for worker_index in range(1, 4):
        shard_path = source_root / f"worker{worker_index}.npz"
        with np.load(shard_path, allow_pickle=False) as archive:
            shards.append(
                (
                    archive["images"],
                    archive["labels"],
                    archive["train_indices"],
                )
            )
    images = np.concatenate([shard[0] for shard in shards])
    labels = np.concatenate([shard[1] for shard in shards])
    indices = np.concatenate([shard[2] for shard in shards])
    if (
        len(np.unique(indices)) != len(indices)
        or not np.array_equal(np.sort(indices), np.arange(len(indices)))
        or len(indices) != 50_000
    ):
        raise ValueError(f"{source_root} does not contain a complete disjoint CIFAR-10 train split")
    order = np.argsort(indices)
    return images[order], labels[order]


def _partition(
    labels: np.ndarray,
    *,
    worker_count: int,
    seed: int,
    alpha: float | None,
) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    shards: list[list[np.ndarray]] = [[] for _ in range(worker_count)]
    for label in np.unique(labels):
        indices = rng.permutation(np.flatnonzero(labels == label))
        if alpha is None:
            groups = np.array_split(indices, worker_count)
        else:
            counts = rng.multinomial(
                len(indices),
                rng.dirichlet(np.full(worker_count, alpha)),
            )
            groups = np.split(indices, np.cumsum(counts)[:-1])
        for worker_index, group in enumerate(groups):
            shards[worker_index].append(group)
    return [
        np.sort(np.concatenate(groups)).astype(np.int64)
        for groups in shards
    ]


def reshard_cifar10(
    output_root: Path,
    *,
    worker_count: int = 5,
    seed: int = 2026,
) -> dict[str, dict[str, int]]:
    if worker_count < 3:
        raise ValueError("At least three workers are required")
    reports = {}
    for name, source_root in SOURCE_SHARDS:
        images, labels = _load_source_shards(source_root)
        alpha = {"dirichlet_a03": 0.3, "dirichlet_a05": 0.5}.get(name)
        index_groups = _partition(
            labels,
            worker_count=worker_count,
            seed=seed,
            alpha=alpha,
        )
        flattened = np.concatenate(index_groups)
        if len(np.unique(flattened)) != len(labels) or not np.array_equal(
            np.sort(flattened),
            np.arange(len(labels)),
        ):
            raise RuntimeError(f"{name} repartition does not preserve the training split")
        destination = output_root / name
        destination.mkdir(parents=True, exist_ok=True)
        reports[name] = {}
        for worker_index, indices in enumerate(index_groups, 1):
            worker_id = f"worker{worker_index}"
            np.savez_compressed(
                destination / f"{worker_id}.npz",
                images=images[indices],
                labels=labels[indices],
                train_indices=indices,
            )
            reports[name][worker_id] = int(len(indices))
    return reports


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--worker-count", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    print(
        json.dumps(
            reshard_cifar10(
                args.output_root,
                worker_count=args.worker_count,
                seed=args.seed,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
