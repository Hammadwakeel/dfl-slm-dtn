"""Download and register official CIFAR-10 for coordinator preprocessing."""

import argparse
import hashlib
import json
import os
import sys
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import urlopen

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.prepare_cifar10 import CLASS_NAMES, load_cifar10  # noqa: E402

CIFAR10_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
CIFAR10_MD5 = "c58f30108f718f92721af3b95e74349a"
MAX_ARCHIVE_BYTES = 256 * 1024 * 1024
DATASET_NAME = "cifar10"
SPLIT_POLICY = {"type": "official_cifar10_npz"}


def _file_md5(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download_archive(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    try:
        with urlopen(CIFAR10_URL, timeout=60) as response:
            length = response.headers.get("Content-Length")
            if length is not None and int(length) > MAX_ARCHIVE_BYTES:
                raise ValueError("CIFAR-10 archive exceeds the download size limit")
            total = 0
            with temporary_path.open("wb") as output:
                while chunk := response.read(1024 * 1024):
                    total += len(chunk)
                    if total > MAX_ARCHIVE_BYTES:
                        raise ValueError("CIFAR-10 archive exceeds the download size limit")
                    output.write(chunk)
        if total == 0:
            raise ValueError("CIFAR-10 archive download was empty")
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def prepare_dataset(archive_path: Path, dataset_root: Path) -> dict[str, object]:
    dataset_root.mkdir(parents=True, exist_ok=True)
    target = dataset_root / DATASET_NAME
    if target.exists():
        raise FileExistsError(
            f"{target} already exists; refusing to replace existing dataset data"
        )
    if not archive_path.is_file():
        _download_archive(archive_path)
    if archive_path.stat().st_size > MAX_ARCHIVE_BYTES:
        raise ValueError("CIFAR-10 archive exceeds the download size limit")
    archive_md5 = _file_md5(archive_path)
    if archive_md5 != CIFAR10_MD5:
        raise ValueError("CIFAR-10 archive checksum does not match the official archive")
    train_images, train_labels, test_images, test_labels = load_cifar10(archive_path)
    all_labels = np.concatenate((train_labels, test_labels))
    class_counts = Counter(CLASS_NAMES[int(label)] for label in all_labels)
    ordered_counts = {name: int(class_counts[name]) for name in CLASS_NAMES}

    with tempfile.TemporaryDirectory(prefix=".cifar10-", dir=dataset_root) as temporary:
        staging = Path(temporary) / DATASET_NAME
        staging.mkdir()
        with (staging / "cifar10.npz").open("wb") as output:
            np.savez_compressed(
                output,
                x_train=train_images,
                y_train=train_labels,
                x_test=test_images,
                y_test=test_labels,
            )
        metadata: dict[str, object] = {
            "dataset_name": DATASET_NAME,
            "classes": list(CLASS_NAMES),
            "class_counts": ordered_counts,
            "image_sizes": {name: {"32x32": count} for name, count in ordered_counts.items()},
            "image_channels": {name: {"3": count} for name, count in ordered_counts.items()},
            "image_count": 60_000,
            "expanded_image_bytes": 60_000 * 32 * 32 * 3,
            "source": "official CIFAR-10",
            "source_url": CIFAR10_URL,
            "source_archive_md5": archive_md5,
            "source_split_policy": SPLIT_POLICY,
            "source_split_counts": {"train": 50_000, "test": 10_000},
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
        default=REPO_ROOT / "data" / "raw" / "cifar-10-python.tar.gz",
        help="Local official CIFAR-10 archive cache path",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=REPO_ROOT / "data" / "datasets",
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
                "source_archive_md5": metadata["source_archive_md5"],
                "dataset_path": str(arguments.dataset_root / DATASET_NAME),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
