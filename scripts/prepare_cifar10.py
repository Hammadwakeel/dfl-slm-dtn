"""Prepare official CIFAR-10 Python batches into local evaluation and worker files."""

from __future__ import annotations

import argparse
import json
import pickle
import tarfile
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parents[1]
CLASS_NAMES = (
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)
TRAIN_BATCHES = tuple(f"cifar-10-batches-py/data_batch_{index}" for index in range(1, 6))
TEST_BATCH = "cifar-10-batches-py/test_batch"


def _read_batch(archive: tarfile.TarFile, member_name: str) -> tuple[np.ndarray, np.ndarray]:
    member = archive.extractfile(member_name)
    if member is None:
        raise ValueError(f"Missing CIFAR-10 archive member: {member_name}")
    payload: dict[bytes, Any] = pickle.load(member, encoding="bytes")
    images = np.asarray(payload[b"data"], dtype=np.uint8).reshape(-1, 3, 32, 32)
    labels = np.asarray(payload[b"labels"], dtype=np.int64)
    return images.transpose(0, 2, 3, 1), labels


def load_cifar10(archive_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with tarfile.open(archive_path, mode="r:gz") as archive:
        train = [_read_batch(archive, member) for member in TRAIN_BATCHES]
        test_images, test_labels = _read_batch(archive, TEST_BATCH)
    train_images = np.concatenate([batch[0] for batch in train])
    train_labels = np.concatenate([batch[1] for batch in train])
    if train_images.shape != (50_000, 32, 32, 3) or test_images.shape != (10_000, 32, 32, 3):
        raise ValueError("CIFAR-10 archive had unexpected image counts or dimensions")
    if train_labels.min() != 0 or train_labels.max() != 9:
        raise ValueError("CIFAR-10 labels are outside the expected 0-9 range")
    return train_images, train_labels, test_images, test_labels


def _stratified_indices(labels: np.ndarray, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    shards: list[list[np.ndarray]] = [[], [], []]
    for label in range(10):
        class_indices = np.flatnonzero(labels == label)
        rng.shuffle(class_indices)
        for worker_index, indices in enumerate(np.array_split(class_indices, 3)):
            shards[worker_index].append(indices)
    return [np.concatenate(worker_parts) for worker_parts in shards]


def _dirichlet_indices(labels: np.ndarray, seed: int, alpha: float) -> list[np.ndarray]:
    if alpha <= 0:
        raise ValueError("Dirichlet alpha must be positive")
    rng = np.random.default_rng(seed)
    shards: list[list[np.ndarray]] = [[], [], []]
    for label in range(10):
        class_indices = np.flatnonzero(labels == label)
        rng.shuffle(class_indices)
        proportions = rng.dirichlet(np.full(3, alpha))
        split_points = np.cumsum(np.floor(proportions[:-1] * len(class_indices)).astype(int))
        for worker_index, indices in enumerate(np.split(class_indices, split_points)):
            shards[worker_index].append(indices)
    return [np.concatenate(worker_parts) for worker_parts in shards]


def _save_grid(images: np.ndarray, labels: np.ndarray, output_path: Path) -> None:
    columns = 5
    rows = len(CLASS_NAMES)
    tile_width = 144
    image_size = 128
    label_height = 26
    margin = 16
    canvas = Image.new(
        "RGB",
        (margin * 2 + columns * tile_width, margin * 2 + rows * (image_size + label_height + 8) + 40),
        "#101923",
    )
    draw = ImageDraw.Draw(canvas)
    draw.text((margin, 10), "CIFAR-10 training examples (5 per class)", fill="#edf4fa")
    font = ImageFont.load_default()
    for label, class_name in enumerate(CLASS_NAMES):
        chosen = images[np.flatnonzero(labels == label)[:columns]]
        row_top = margin + 30 + label * (image_size + label_height + 8)
        for column, image_array in enumerate(chosen):
            left = margin + column * tile_width
            image = Image.fromarray(image_array, mode="RGB").resize(
                (image_size, image_size),
                Image.Resampling.NEAREST,
            )
            canvas.paste(image, (left, row_top))
            draw.text((left, row_top + image_size + 5), class_name, font=font, fill="#c7d7e6")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, format="PNG", optimize=True)


def prepare(
    archive_path: Path,
    data_root: Path,
    docs_root: Path,
    seed: int,
    split_mode: str = "stratified",
    dirichlet_alpha: float = 0.5,
) -> dict[str, Any]:
    train_images, train_labels, test_images, test_labels = load_cifar10(archive_path)
    index_groups = (
        _stratified_indices(train_labels, seed)
        if split_mode == "stratified"
        else _dirichlet_indices(train_labels, seed, dirichlet_alpha)
    )
    shard_root = data_root / "worker_shards"
    shard_root.mkdir(parents=True, exist_ok=True)
    raw_root = data_root / "raw"
    raw_root.mkdir(parents=True, exist_ok=True)
    shard_summaries = {}
    for worker_index, indices in enumerate(index_groups, 1):
        shard_path = shard_root / f"worker{worker_index}.npz"
        np.savez_compressed(
            shard_path,
            images=train_images[indices],
            labels=train_labels[indices],
            train_indices=indices,
        )
        shard_summaries[f"worker{worker_index}"] = {
            "shape": list(train_images[indices].shape),
            "rows": int(len(indices)),
            "class_counts": {
                CLASS_NAMES[label]: int(count)
                for label, count in sorted(Counter(train_labels[indices]).items())
            },
            "file_bytes": shard_path.stat().st_size,
        }

    train_path = raw_root / "cifar10_train.npz"
    np.savez_compressed(train_path, images=train_images, labels=train_labels)
    test_path = raw_root / "cifar10_test.npz"
    np.savez_compressed(test_path, images=test_images, labels=test_labels)
    _save_grid(train_images, train_labels, docs_root / "cifar10_sample_grid.png")

    summary = {
        "dataset": "CIFAR-10 Python version",
        "seed": seed,
        "split_mode": split_mode,
        "dirichlet_alpha": dirichlet_alpha if split_mode == "dirichlet" else None,
        "train_shape": list(train_images.shape),
        "train_rows": int(len(train_labels)),
        "train_class_counts": {
            CLASS_NAMES[label]: int(count)
            for label, count in sorted(Counter(train_labels).items())
        },
        "test_shape": list(test_images.shape),
        "test_rows": int(len(test_labels)),
        "test_class_counts": {
            CLASS_NAMES[label]: int(count)
            for label, count in sorted(Counter(test_labels).items())
        },
        "pixel_range": [int(train_images.min()), int(train_images.max())],
        "worker_shards": shard_summaries,
        "train_file": str(train_path.relative_to(REPO_ROOT)),
        "test_file": str(test_path.relative_to(REPO_ROOT)),
        "sample_grid": str((docs_root / "cifar10_sample_grid.png").relative_to(REPO_ROOT)),
    }
    metadata_path = data_root / "cifar10_metadata.json"
    metadata_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--archive",
        type=Path,
        default=REPO_ROOT / "data" / "raw" / "cifar-10-python.tar.gz",
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--split", choices=("stratified", "dirichlet"), default="stratified")
    parser.add_argument("--alpha", type=float, default=0.5)
    args = parser.parse_args()
    summary = prepare(
        args.archive,
        REPO_ROOT / "data",
        REPO_ROOT / "docs",
        args.seed,
        args.split,
        args.alpha,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
