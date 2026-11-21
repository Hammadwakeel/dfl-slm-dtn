"""Prepare Fashion-MNIST and FEMNIST as fixed local NPZ datasets (like official MNIST).

    python scripts/prepare_extra_datasets.py --dataset fashion_mnist
    python scripts/prepare_extra_datasets.py --dataset femnist      # needs pyarrow

Fashion-MNIST: the official 60k/10k train/test split.

FEMNIST: the LEAF federated EMNIST data (62 classes: digits, upper- and lower-case
letters, 28x28 grayscale) as published by Flower (``flwrlabs/femnist``). The full set
has about 815k images, too many for eight Jetson workers, so a fixed seeded subset of
70,000 images is drawn and split 60,000 / 10,000 stratified by class. LEAF stores dark
strokes on a white background; images are inverted to MNIST polarity (white strokes).
Workers still receive Dirichlet label shards, so the writer IDs are kept only in the
metadata summary.

Each dataset gets ``models/baseline_architecture.json``: the same plain CNN for both
(two 3x3 conv blocks with batch normalization, global pooling, a 128-unit dense head),
the starting model of every supervisor run.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
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
FASHION_BASE = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/"
FASHION_FILES = {
    "x_train": "train-images-idx3-ubyte.gz",
    "y_train": "train-labels-idx1-ubyte.gz",
    "x_test": "t10k-images-idx3-ubyte.gz",
    "y_test": "t10k-labels-idx1-ubyte.gz",
}
FASHION_CLASSES = ["t_shirt", "trouser", "pullover", "dress", "coat", "sandal", "shirt", "sneaker", "bag", "ankle_boot"]
FEMNIST_URL = "https://huggingface.co/datasets/flwrlabs/femnist/resolve/refs%2Fconvert%2Fparquet/default/train/0000.parquet"
FEMNIST_CLASSES = [str(d) for d in range(10)] + [chr(c) for c in range(ord("A"), ord("Z") + 1)] + [chr(c) for c in range(ord("a"), ord("z") + 1)]
FEMNIST_SUBSET = 70_000
FEMNIST_TEST = 10_000
FEMNIST_SEED = 0
MAX_DOWNLOAD_BYTES = 400 * 1024 * 1024


def _download(url: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        with urllib.request.urlopen(url, timeout=120) as response, temporary.open("wb") as output:
            total = 0
            while chunk := response.read(1024 * 1024):
                total += len(chunk)
                if total > MAX_DOWNLOAD_BYTES:
                    raise ValueError(f"{url} exceeds the download size limit")
                output.write(chunk)
        if total == 0:
            raise ValueError(f"{url} download was empty")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _read_idx(path: Path) -> np.ndarray:
    data = gzip.decompress(path.read_bytes())
    magic = int.from_bytes(data[:4], "big")
    dims = magic & 0xFF
    shape = tuple(int.from_bytes(data[4 + 4 * i : 8 + 4 * i], "big") for i in range(dims))
    return np.frombuffer(data, dtype=np.uint8, offset=4 + 4 * dims).reshape(shape)


def load_fashion(raw: Path) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    arrays, digests = {}, {}
    for key, name in FASHION_FILES.items():
        path = raw / "fashion_mnist" / name
        if not path.is_file():
            _download(FASHION_BASE + name, path)
        digests[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        arrays[key] = _read_idx(path)
    arrays["y_train"] = arrays["y_train"].astype(np.int64)
    arrays["y_test"] = arrays["y_test"].astype(np.int64)
    if arrays["x_train"].shape != (60_000, 28, 28) or arrays["x_test"].shape != (10_000, 28, 28):
        raise ValueError("Fashion-MNIST does not have the official 60k/10k shapes")
    return arrays, digests


def load_femnist(raw: Path) -> tuple[dict[str, np.ndarray], dict[str, str], dict[str, int]]:
    import pyarrow.parquet as pq  # only FEMNIST needs it
    from PIL import Image

    path = raw / "femnist" / "femnist_train.parquet"
    if not path.is_file():
        _download(FEMNIST_URL, path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    table = pq.read_table(path, columns=["image", "writer_id", "character"])
    labels_all = np.asarray(table.column("character").to_pylist(), dtype=np.int64)
    if labels_all.min() < 0 or labels_all.max() >= len(FEMNIST_CLASSES):
        raise ValueError("FEMNIST labels are outside 0..61")
    rng = np.random.default_rng(FEMNIST_SEED)
    chosen = np.sort(rng.choice(labels_all.size, size=FEMNIST_SUBSET, replace=False))
    images_col = table.column("image")
    writers_col = table.column("writer_id")
    images = np.empty((FEMNIST_SUBSET, 28, 28), dtype=np.uint8)
    writers = []
    for out_index, row in enumerate(chosen.tolist()):
        cell = images_col[row].as_py()
        raw_bytes = cell["bytes"] if isinstance(cell, dict) else cell
        with Image.open(io.BytesIO(raw_bytes)) as image:
            pixels = np.asarray(image.convert("L"), dtype=np.uint8)
        if pixels.shape != (28, 28):
            raise ValueError(f"FEMNIST image {row} is {pixels.shape}, expected 28x28")
        images[out_index] = pixels
        writers.append(writers_col[row].as_py())
    labels = labels_all[chosen]
    if images.mean() > 127:  # LEAF polarity: dark strokes on white
        images = 255 - images
    # stratified 60k / 10k split of the subset
    test_mask = np.zeros(FEMNIST_SUBSET, dtype=bool)
    for label in np.unique(labels):
        members = rng.permutation(np.flatnonzero(labels == label))
        count = int(round(len(members) * FEMNIST_TEST / FEMNIST_SUBSET))
        test_mask[members[:count]] = True
    arrays = {
        "x_train": images[~test_mask],
        "y_train": labels[~test_mask],
        "x_test": images[test_mask],
        "y_test": labels[test_mask],
    }
    stats = {"source_images": int(labels_all.size), "writers_in_subset": len(set(writers)), "subset_images": FEMNIST_SUBSET}
    return arrays, {"femnist_train.parquet": digest}, stats


def baseline_architecture(dataset: str, classes: list[str]) -> dict[str, object]:
    conv = lambda filters: {"activation": "relu", "filters": filters, "kernel_size": 3, "padding": "same", "strides": 1, "type": "conv2d"}
    return {
        "classes": classes,
        "dataset_name": dataset,
        "initialization_seed": 42,
        "input_shape": [28, 28, 1],
        "layers": [
            conv(32), {"type": "batch_normalization"}, {"pool_size": 2, "type": "max_pooling2d"},
            conv(64), {"type": "batch_normalization"}, {"pool_size": 2, "type": "max_pooling2d"},
            {"type": "global_average_pooling2d"},
            {"activation": "relu", "type": "dense", "units": 128},
            {"rate": 0.2, "type": "dropout"},
            {"activation": "linear", "type": "dense", "units": len(classes)},
        ],
        "model_type": "custom",
        "num_classes": len(classes),
        "training": {"batch_size": 32, "learning_rate": 0.001, "optimizer": "adam"},
    }


def prepare(dataset: str, dataset_root: Path, raw: Path) -> dict[str, object]:
    target = dataset_root / dataset
    if target.exists():
        raise FileExistsError(f"{target} already exists; refusing to replace existing dataset data")
    extra: dict[str, object] = {}
    if dataset == "fashion_mnist":
        arrays, digests = load_fashion(raw)
        classes, source, url = FASHION_CLASSES, "official Fashion-MNIST", FASHION_BASE
    else:
        arrays, digests, extra = load_femnist(raw)
        classes, source, url = FEMNIST_CLASSES, "LEAF FEMNIST (flwrlabs/femnist), seeded 70k subset", FEMNIST_URL
    for key in ("y_train", "y_test"):
        if np.bincount(arrays[key], minlength=len(classes)).min() == 0:
            raise ValueError(f"{dataset} {key} is missing a class")
    labels = np.concatenate((arrays["y_train"], arrays["y_test"]))
    counts = Counter(classes[int(label)] for label in labels.tolist())
    n_train, n_test = int(arrays["y_train"].size), int(arrays["y_test"].size)
    with tempfile.TemporaryDirectory(prefix=f".{dataset}-", dir=dataset_root) as temporary:
        staging = Path(temporary) / dataset
        (staging / "models").mkdir(parents=True)
        np.savez_compressed(staging / f"{dataset}.npz", **arrays)
        metadata = {
            "dataset_name": dataset,
            "classes": classes,
            "class_counts": {name: int(counts.get(name, 0)) for name in classes},
            "image_sizes": {name: {"28x28": int(counts.get(name, 0))} for name in classes},
            "image_channels": {name: {"1": int(counts.get(name, 0))} for name in classes},
            "image_count": n_train + n_test,
            "expanded_image_bytes": (n_train + n_test) * 28 * 28,
            "source": source,
            "source_url": url,
            "source_sha256": digests,
            "source_split_policy": {"type": f"official_{dataset}_npz"},
            "source_split_counts": {"train": n_train, "test": n_test},
            "created_at": datetime.now(timezone.utc).isoformat(),
            **({"source_details": extra} if extra else {}),
        }
        (staging / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        (staging / "models" / "baseline_architecture.json").write_text(
            json.dumps(baseline_architecture(dataset, classes), indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.rename(staging, target)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", choices=("fashion_mnist", "femnist"), required=True)
    parser.add_argument("--dataset-root", type=Path, default=ROOT / "data" / "datasets")
    parser.add_argument("--raw-root", type=Path, default=ROOT / "data" / "raw")
    args = parser.parse_args()
    metadata = prepare(args.dataset, args.dataset_root.resolve(), args.raw_root.resolve())
    print(json.dumps({k: metadata[k] for k in ("dataset_name", "image_count", "source_split_counts", "source")}, indent=2))


if __name__ == "__main__":
    main()
