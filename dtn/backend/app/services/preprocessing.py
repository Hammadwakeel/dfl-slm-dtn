"""Image validation, preprocessing, splitting, and worker-shard preparation."""

import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image, UnidentifiedImageError

SUPPORTED_EXTENSIONS = frozenset(
    {".bmp", ".gif", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
)
MAX_IMAGE_PIXELS = 100_000_000
MAX_PROCESSED_BYTES = 1024 * 1024 * 1024
DIRICHLET_ALPHAS = {
    "dirichlet_a01": 0.1,
    "dirichlet_a03": 0.3,
    "dirichlet_a05": 0.5,
    "dirichlet_a07": 0.7,
}
SPLIT_STRATEGIES = frozenset({"stratified", *DIRICHLET_ALPHAS})


class PreprocessingError(ValueError):
    pass


def validate_images(dataset_path: Path) -> dict[str, Any]:
    dataset_path = dataset_path.resolve()
    metadata_path = dataset_path / "metadata.json"
    if not metadata_path.is_file():
        raise PreprocessingError(f"Dataset metadata is missing: {metadata_path}")
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise PreprocessingError("Dataset metadata cannot be read") from error
    classes = metadata.get("classes")
    if not isinstance(classes, list) or not classes or any(
        not isinstance(name, str) or not name for name in classes
    ):
        raise PreprocessingError("Dataset metadata must contain a non-empty classes list")
    if len(set(classes)) != len(classes):
        raise PreprocessingError("Dataset metadata contains duplicate class names")
    split_policy = metadata.get("source_split_policy")
    if split_policy is not None:
        if split_policy == {"type": "official_mnist_npz"}:
            raise PreprocessingError(
                "Official MNIST is stored as NPZ arrays and does not contain image folders"
            )
        if split_policy == {"type": "official_cifar10_npz"}:
            raise PreprocessingError(
                "Official CIFAR-10 is stored as NPZ arrays and does not contain image folders"
            )
        raise PreprocessingError("Dataset source split policy is unsupported")

    images: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    class_counts = {class_name: 0 for class_name in classes}
    for label, class_name in enumerate(classes):
        if Path(class_name).name != class_name or class_name in {".", ".."}:
            raise PreprocessingError(f"Dataset contains an unsafe class name: {class_name}")
        class_path = dataset_path / class_name
        if class_path.is_symlink() or not class_path.is_dir():
            raise PreprocessingError(f"Dataset class folder is missing or unsafe: {class_name}")
        if class_path.resolve().parent != dataset_path:
            raise PreprocessingError(f"Dataset class folder escapes dataset root: {class_name}")
        for image_path in sorted(class_path.iterdir(), key=lambda item: item.name.casefold()):
            if image_path.suffix.casefold() not in SUPPORTED_EXTENSIONS:
                continue
            if image_path.is_symlink() or not image_path.is_file():
                errors.append({"path": str(image_path), "error": "not a regular image file"})
                continue
            try:
                with Image.open(image_path) as image:
                    width, height = image.size
                    mode = image.mode
                    channels = len(image.getbands())
                    if width < 1 or height < 1 or width * height > MAX_IMAGE_PIXELS:
                        raise PreprocessingError("image dimensions exceed the supported limit")
                    image.verify()
                images.append(
                    {
                        "path": image_path,
                        "label": label,
                        "class_name": class_name,
                        "width": width,
                        "height": height,
                        "mode": mode,
                        "channels": channels,
                    }
                )
                class_counts[class_name] += 1
            except (
                OSError,
                ValueError,
                UnidentifiedImageError,
                Image.DecompressionBombError,
            ) as error:
                errors.append({"path": str(image_path), "error": str(error)})

    if not images:
        raise PreprocessingError("Dataset contains no readable supported images")
    return {
        "classes": classes,
        "images": images,
        "image_count": len(images),
        "class_counts": class_counts,
        "invalid_images": errors,
    }


def resize_images(
    validation: dict[str, Any],
    target_size: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    height, width = target_size
    if not 1 <= height <= 2048 or not 1 <= width <= 2048:
        raise PreprocessingError("Target image height and width must be between 1 and 2048")
    records = validation.get("images")
    if not isinstance(records, list) or not records:
        raise PreprocessingError("Validated image records are required")
    grayscale_modes = {"1", "L", "I", "F"}
    channels = (
        1
        if all(record.get("mode") in grayscale_modes for record in records)
        else 3
    )
    output_bytes = len(records) * height * width * channels * 4
    if output_bytes > MAX_PROCESSED_BYTES:
        raise PreprocessingError(
            f"Resized dataset would require {output_bytes} bytes; limit is {MAX_PROCESSED_BYTES}"
        )
    images = np.empty((len(records), height, width, channels), dtype=np.uint8)
    labels = np.empty(len(records), dtype=np.int64)
    for index, record in enumerate(records):
        try:
            with Image.open(record["path"]) as image:
                resized = image.convert("L" if channels == 1 else "RGB").resize(
                    (width, height),
                    resample=Image.Resampling.BILINEAR,
                )
                pixels = np.asarray(resized, dtype=np.uint8)
                images[index] = pixels[..., np.newaxis] if channels == 1 else pixels
            labels[index] = int(record["label"])
        except (OSError, ValueError, KeyError, TypeError) as error:
            raise PreprocessingError(
                f"Could not resize validated image {record.get('path', '<unknown>')}: {error}"
            ) from error
    return images, labels


def normalize_images(
    images: np.ndarray,
    method: str,
    training_indices: Sequence[int],
) -> tuple[np.ndarray, dict[str, Any]]:
    if method not in {"standard", "minmax", "none"}:
        raise PreprocessingError("Normalization must be standard, minmax, or none")
    if images.ndim != 4 or images.shape[-1] not in {1, 3} or images.size == 0:
        raise PreprocessingError("Images must be a non-empty N x H x W x C array with C=1 or C=3")
    if not np.issubdtype(images.dtype, np.number):
        raise PreprocessingError("Images must contain numeric values")
    if not np.isfinite(images).all():
        raise PreprocessingError("Images contain non-finite values")
    if images.min() < 0 or images.max() > 255:
        raise PreprocessingError("Image values must be between 0 and 255")
    if method == "none":
        return images.copy(), {"method": method}
    if len(training_indices) == 0:
        raise PreprocessingError("At least one training image is required for normalization")
    if min(training_indices) < 0 or max(training_indices) >= len(images):
        raise PreprocessingError("Training indices are outside the image array")

    values = images.astype(np.float32) / 255.0
    if method == "minmax":
        return values, {"method": method, "minimum": 0.0, "maximum": 1.0}
    training_values = values[np.asarray(training_indices, dtype=np.int64)]
    mean = training_values.mean(axis=(0, 1, 2), dtype=np.float64)
    standard_deviation = training_values.std(axis=(0, 1, 2), dtype=np.float64)
    standard_deviation[standard_deviation == 0] = 1.0
    values -= mean.astype(np.float32)
    values /= standard_deviation.astype(np.float32)
    return values, {
        "method": method,
        "mean": mean.tolist(),
        "standard_deviation": standard_deviation.tolist(),
    }


def stratified_split(
    labels: Sequence[int],
    *,
    validation_fraction: float = 0.1,
    test_fraction: float = 0.1,
    seed: int = 42,
) -> dict[str, list[int]]:
    if not 0 < validation_fraction < 1 or not 0 <= test_fraction < 1:
        raise PreprocessingError("Validation fraction must be in (0, 1) and test fraction in [0, 1)")
    if validation_fraction + test_fraction >= 1:
        raise PreprocessingError("Validation and test fractions must sum to less than 1")
    labels_array = np.asarray(labels)
    if labels_array.ndim != 1 or labels_array.size == 0:
        raise PreprocessingError("Labels must be a non-empty one-dimensional sequence")
    if not np.issubdtype(labels_array.dtype, np.integer):
        raise PreprocessingError("Labels must be integers")
    if labels_array.min() < 0:
        raise PreprocessingError("Labels must be non-negative")

    rng = np.random.default_rng(seed)
    splits: dict[str, list[int]] = {"train": [], "validation": [], "test": []}
    for label in np.unique(labels_array):
        indices = np.flatnonzero(labels_array == label)
        if len(indices) < 3:
            raise PreprocessingError(
                f"Class label {int(label)} needs at least 3 images for stratified splitting"
            )
        shuffled = rng.permutation(indices)
        validation_count = max(1, int(round(len(indices) * validation_fraction)))
        test_count = (
            max(1, int(round(len(indices) * test_fraction)))
            if test_fraction > 0
            else 0
        )
        if validation_count + test_count >= len(indices):
            raise PreprocessingError(
                f"Class label {int(label)} is too small for the requested split fractions"
            )
        splits["validation"].extend(int(index) for index in shuffled[:validation_count])
        splits["test"].extend(
            int(index) for index in shuffled[validation_count : validation_count + test_count]
        )
        splits["train"].extend(
            int(index) for index in shuffled[validation_count + test_count :]
        )
    for indices in splits.values():
        indices.sort()
    return splits


def shard_training_indices(
    labels: Sequence[int],
    *,
    strategy: str = "stratified",
    seed: int = 42,
    worker_count: int = 3,
) -> list[list[int]]:
    if strategy not in SPLIT_STRATEGIES:
        raise PreprocessingError(f"Unsupported worker shard strategy: {strategy}")
    if worker_count < 3:
        raise PreprocessingError("At least three workers are required for worker sharding")
    labels_array = np.asarray(labels)
    if labels_array.ndim != 1 or labels_array.size < 3:
        raise PreprocessingError("At least three training labels are required for worker sharding")
    if not np.issubdtype(labels_array.dtype, np.integer) or labels_array.min() < 0:
        raise PreprocessingError("Worker shard labels must be non-negative integers")
    rng = np.random.default_rng(seed)
    shards: list[list[int]] = [[] for _ in range(worker_count)]
    alpha = DIRICHLET_ALPHAS.get(strategy)
    for label in np.unique(labels_array):
        indices = rng.permutation(np.flatnonzero(labels_array == label))
        if alpha is None:
            groups = np.array_split(indices, worker_count)
        else:
            proportions = rng.dirichlet(np.full(worker_count, alpha))
            counts = rng.multinomial(len(indices), proportions)
            boundaries = np.cumsum(counts)[:-1]
            groups = np.split(indices, boundaries)
        for worker_index, group in enumerate(groups):
            shards[worker_index].extend(int(index) for index in group)

    if any(not shard for shard in shards):
        largest = max(range(worker_count), key=lambda index: len(shards[index]))
        for empty in (index for index, shard in enumerate(shards) if not shard):
            shards[empty].append(shards[largest].pop())
    for shard in shards:
        shard.sort()
    if sum(map(len, shards)) != len(labels_array):
        raise RuntimeError("Worker shard split did not preserve all training rows")
    return shards


def class_balance_report(
    labels: Sequence[int],
    class_names: Sequence[str],
) -> dict[str, int]:
    labels_array = np.asarray(labels)
    if labels_array.ndim != 1:
        raise PreprocessingError("Labels must be one-dimensional")
    if (
        not np.issubdtype(labels_array.dtype, np.integer)
        or any(int(label) < 0 or int(label) >= len(class_names) for label in labels_array)
    ):
        raise PreprocessingError("A label has no corresponding class name")
    counts = np.bincount(labels_array.astype(np.int64), minlength=len(class_names))
    return {name: int(counts[index]) for index, name in enumerate(class_names)}


def _write_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary_path.open("wb") as output:
            np.savez_compressed(output, **arrays)
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _load_official_mnist(
    dataset_path: Path,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    metadata_path = dataset_path / "metadata.json"
    archive_path = dataset_path / "mnist.npz"
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        with np.load(archive_path, allow_pickle=False) as archive:
            train_images = np.asarray(archive["x_train"])
            train_labels = np.asarray(archive["y_train"])
            test_images = np.asarray(archive["x_test"])
            test_labels = np.asarray(archive["y_test"])
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        raise PreprocessingError("Official MNIST archive or metadata is invalid") from error

    classes = [str(label) for label in range(10)]
    if (
        metadata.get("dataset_name") != "mnist"
        or metadata.get("classes") != classes
        or metadata.get("source_split_policy") != {"type": "official_mnist_npz"}
        or train_images.shape != (60_000, 28, 28)
        or test_images.shape != (10_000, 28, 28)
        or train_labels.shape != (60_000,)
        or test_labels.shape != (10_000,)
        or train_images.dtype != np.uint8
        or test_images.dtype != np.uint8
        or not np.isin(train_labels, np.arange(10)).all()
        or not np.isin(test_labels, np.arange(10)).all()
    ):
        raise PreprocessingError("Official MNIST does not match the expected 60k/10k format")

    images = np.concatenate((train_images, test_images), axis=0)[..., np.newaxis]
    labels = np.concatenate((train_labels, test_labels)).astype(np.int64, copy=False)
    validation = {
        "classes": classes,
        "image_count": int(labels.size),
        "class_counts": class_balance_report(labels, classes),
        "invalid_images": [],
    }
    official_split = {
        "train": np.arange(train_labels.size, dtype=np.int64),
        "test": np.arange(train_labels.size, labels.size, dtype=np.int64),
    }
    return validation | {"official_split": official_split}, images, labels


def _load_official_cifar10(
    dataset_path: Path,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    metadata_path = dataset_path / "metadata.json"
    archive_path = dataset_path / "cifar10.npz"
    classes = [
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
    ]
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        with np.load(archive_path, allow_pickle=False) as archive:
            train_images = np.asarray(archive["x_train"])
            train_labels = np.asarray(archive["y_train"])
            test_images = np.asarray(archive["x_test"])
            test_labels = np.asarray(archive["y_test"])
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        raise PreprocessingError("Official CIFAR-10 archive or metadata is invalid") from error

    if (
        metadata.get("dataset_name") != "cifar10"
        or metadata.get("classes") != classes
        or metadata.get("source_split_policy") != {"type": "official_cifar10_npz"}
        or train_images.shape != (50_000, 32, 32, 3)
        or test_images.shape != (10_000, 32, 32, 3)
        or train_labels.shape != (50_000,)
        or test_labels.shape != (10_000,)
        or train_images.dtype != np.uint8
        or test_images.dtype != np.uint8
        or not np.isin(train_labels, np.arange(10)).all()
        or not np.isin(test_labels, np.arange(10)).all()
    ):
        raise PreprocessingError("Official CIFAR-10 does not match the expected 50k/10k format")

    images = np.concatenate((train_images, test_images), axis=0)
    labels = np.concatenate((train_labels, test_labels)).astype(np.int64, copy=False)
    validation = {
        "classes": classes,
        "image_count": int(labels.size),
        "class_counts": class_balance_report(labels, classes),
        "invalid_images": [],
    }
    official_split = {
        "train": np.arange(train_labels.size, dtype=np.int64),
        "test": np.arange(train_labels.size, labels.size, dtype=np.int64),
    }
    return validation | {"official_split": official_split}, images, labels


def preprocess_image_dataset(
    dataset_path: Path,
    *,
    normalization: str = "standard",
    split_strategy: str = "stratified",
    target_size: tuple[int, int] = (32, 32),
    validation_fraction: float = 0.1,
    test_fraction: float = 0.1,
    seed: int = 42,
    worker_ids: Sequence[str] = ("worker1", "worker2", "worker3"),
) -> dict[str, Any]:
    if len(worker_ids) < 3 or len(set(worker_ids)) != len(worker_ids):
        raise PreprocessingError("At least three distinct worker IDs are required")
    dataset_path = dataset_path.resolve()
    try:
        metadata = json.loads((dataset_path / "metadata.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise PreprocessingError("Dataset metadata cannot be read") from error
    source_split_policy = metadata.get("source_split_policy")
    if source_split_policy == {"type": "official_mnist_npz"}:
        validation, images, labels = _load_official_mnist(dataset_path)
        if target_size != (28, 28):
            raise PreprocessingError("The fixed official MNIST dataset uses 28x28 images")
    elif source_split_policy == {"type": "official_cifar10_npz"}:
        validation, images, labels = _load_official_cifar10(dataset_path)
        if target_size != (32, 32):
            raise PreprocessingError("The fixed official CIFAR-10 dataset uses 32x32 images")
    else:
        validation = validate_images(dataset_path)
        images, labels = resize_images(validation, target_size)
    if validation["invalid_images"]:
        example = validation["invalid_images"][0]
        raise PreprocessingError(
            f"{len(validation['invalid_images'])} corrupt or unsafe images found; "
            f"first: {example['path']}: {example['error']}"
        )
    class_names = validation["classes"]
    official_split = validation.get("official_split")
    if official_split is not None:
        official_train = np.asarray(official_split["train"], dtype=np.int64)
        official_test = np.asarray(official_split["test"], dtype=np.int64)
        train_validation = stratified_split(
            labels[official_train],
            validation_fraction=validation_fraction,
            test_fraction=0,
            seed=seed,
        )
        splits = {
            "train": official_train[np.asarray(train_validation["train"], dtype=np.int64)].tolist(),
            "validation": official_train[
                np.asarray(train_validation["validation"], dtype=np.int64)
            ].tolist(),
            "test": official_test.tolist(),
        }
    else:
        splits = stratified_split(
            labels,
            validation_fraction=validation_fraction,
            test_fraction=test_fraction,
            seed=seed,
        )
    normalized_images, normalization_report = normalize_images(
        images,
        normalization,
        splits["train"],
    )
    train_indices = np.asarray(splits["train"], dtype=np.int64)
    shard_indices = shard_training_indices(
        labels[train_indices],
        strategy=split_strategy,
        seed=seed,
        worker_count=len(worker_ids),
    )
    split_indices_flat = [
        index for indices in splits.values() for index in indices
    ]
    worker_indices_flat = [index for indices in shard_indices for index in indices]
    if (
        len(split_indices_flat) != validation["image_count"]
        or len(set(split_indices_flat)) != validation["image_count"]
        or set(split_indices_flat) != set(range(validation["image_count"]))
    ):
        raise RuntimeError("Preprocessing splits overlap or omit source images")
    if (
        len(worker_indices_flat) != len(train_indices)
        or len(set(worker_indices_flat)) != len(train_indices)
        or set(worker_indices_flat) != set(range(len(train_indices)))
    ):
        raise RuntimeError("Worker shards overlap or omit training images")
    run_path = dataset_path / "processed" / f"run-{uuid.uuid4().hex}"
    run_path.mkdir(parents=True, exist_ok=False)

    split_class_counts: dict[str, dict[str, int]] = {}
    for split_name, indices in splits.items():
        split_indices = np.asarray(indices, dtype=np.int64)
        split_class_counts[split_name] = class_balance_report(
            labels[split_indices],
            class_names,
        )
        _write_npz(
            run_path / f"{split_name}.npz",
            images=normalized_images[split_indices],
            labels=labels[split_indices],
            class_names=np.asarray(class_names),
        )

    worker_reports: dict[str, dict[str, Any]] = {}
    for worker_id, local_indices in zip(worker_ids, shard_indices, strict=True):
        global_indices = train_indices[np.asarray(local_indices, dtype=np.int64)]
        shard_path = run_path / "workers" / f"{worker_id}.npz"
        _write_npz(
            shard_path,
            images=normalized_images[global_indices],
            labels=labels[global_indices],
            class_names=np.asarray(class_names),
        )
        worker_reports[worker_id] = {
            "rows": int(len(global_indices)),
            "class_counts": class_balance_report(labels[global_indices], class_names),
            "shard_path": str(shard_path),
        }

    report: dict[str, Any] = {
        "dataset_name": dataset_path.name,
        "classes": class_names,
        "image_count": validation["image_count"],
        "source_class_counts": validation["class_counts"],
        "target_size": {"height": target_size[0], "width": target_size[1]},
        "channels": int(normalized_images.shape[-1]),
        "image_shape": [int(value) for value in normalized_images.shape[1:]],
        "normalization": normalization_report,
        "split_strategy": split_strategy,
        "split_method": (
            f"official_{dataset_path.name}_train_validation_test"
            if official_split is not None
            else "seeded_stratified"
        ),
        "official_test_set_preserved": official_split is not None,
        "split_sizes": {name: len(indices) for name, indices in splits.items()},
        "split_indices_disjoint": True,
        "split_class_counts": split_class_counts,
        "worker_shards_disjoint": True,
        "workers": worker_reports,
        "seed": seed,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "output_path": str(run_path),
    }
    report_path = run_path / "preprocessing_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
