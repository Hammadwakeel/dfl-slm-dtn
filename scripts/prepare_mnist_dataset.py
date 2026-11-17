"""Prepare the official MNIST train/test archive as a fixed local image dataset."""

import argparse
import hashlib
import json
import os
import shutil
import tempfile
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MNIST_URL = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"
MAX_ARCHIVE_BYTES = 20 * 1024 * 1024
DATASET_NAME = "mnist"
SPLIT_POLICY = {"type": "official_mnist_npz"}


def _download_archive(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    try:
        with urllib.request.urlopen(MNIST_URL, timeout=60) as response:
            length = response.headers.get("Content-Length")
            if length is not None and int(length) > MAX_ARCHIVE_BYTES:
                raise ValueError("MNIST archive exceeds the download size limit")
            total = 0
            with temporary_path.open("wb") as output:
                while chunk := response.read(1024 * 1024):
                    total += len(chunk)
                    if total > MAX_ARCHIVE_BYTES:
                        raise ValueError("MNIST archive exceeds the download size limit")
                    output.write(chunk)
        if total == 0:
            raise ValueError("MNIST archive download was empty")
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _load_archive(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    try:
        with np.load(path, allow_pickle=False) as archive:
            train_images = np.asarray(archive["x_train"])
            train_labels = np.asarray(archive["y_train"])
            test_images = np.asarray(archive["x_test"])
            test_labels = np.asarray(archive["y_test"])
    except (OSError, ValueError, KeyError) as error:
        raise ValueError(f"MNIST archive is invalid: {error}") from error

    if (
        train_images.shape != (60_000, 28, 28)
        or test_images.shape != (10_000, 28, 28)
        or train_labels.shape != (60_000,)
        or test_labels.shape != (10_000,)
        or train_labels.dtype.kind not in {"i", "u"}
        or test_labels.dtype.kind not in {"i", "u"}
        or not np.isin(train_labels, np.arange(10)).all()
        or not np.isin(test_labels, np.arange(10)).all()
    ):
        raise ValueError("MNIST archive does not contain the expected 60k/10k split")
    if np.bincount(train_labels, minlength=10).min() == 0:
        raise ValueError("MNIST training split is missing a class")
    if np.bincount(test_labels, minlength=10).min() == 0:
        raise ValueError("MNIST test split is missing a class")
    return train_images, train_labels, test_images, test_labels


def prepare_dataset(archive_path: Path, dataset_root: Path) -> dict[str, object]:
    dataset_root.mkdir(parents=True, exist_ok=True)
    target = dataset_root / DATASET_NAME
    if target.exists():
        raise FileExistsError(
            f"{target} already exists; refusing to replace existing dataset data"
        )
    if not archive_path.is_file():
        _download_archive(archive_path)
    archive_digest = hashlib.sha256(archive_path.read_bytes()).hexdigest()
    _, train_labels, _, test_labels = _load_archive(archive_path)
    class_counts = Counter(
        str(label) for label in np.concatenate((train_labels, test_labels)).tolist()
    )

    with tempfile.TemporaryDirectory(prefix=".mnist-", dir=dataset_root) as temporary:
        staging = Path(temporary) / DATASET_NAME
        staging.mkdir()
        shutil.copyfile(archive_path, staging / "mnist.npz")

        metadata = {
            "dataset_name": DATASET_NAME,
            "classes": [str(label) for label in range(10)],
            "class_counts": dict(sorted(class_counts.items())),
            "image_sizes": {
                name: {"28x28": count} for name, count in sorted(class_counts.items())
            },
            "image_channels": {
                name: {"1": count} for name, count in sorted(class_counts.items())
            },
            "image_count": 70_000,
            "expanded_image_bytes": 70_000 * 28 * 28,
            "source": "official MNIST",
            "source_url": MNIST_URL,
            "source_archive_sha256": archive_digest,
            "source_split_policy": SPLIT_POLICY,
            "source_split_counts": {"train": 60_000, "test": 10_000},
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        (staging / "metadata.json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        os.rename(staging, target)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--archive",
        type=Path,
        default=ROOT / "data" / "raw" / "mnist.npz",
        help="Local official MNIST NPZ cache path",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=ROOT / "data" / "datasets",
        help="Dataset catalog root",
    )
    arguments = parser.parse_args()
    metadata = prepare_dataset(arguments.archive, arguments.dataset_root)
    print(
        json.dumps(
            {
                "dataset_name": metadata["dataset_name"],
                "image_count": metadata["image_count"],
                "source_split_counts": metadata["source_split_counts"],
                "source_archive_sha256": metadata["source_archive_sha256"],
                "dataset_path": str(arguments.dataset_root / DATASET_NAME),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
