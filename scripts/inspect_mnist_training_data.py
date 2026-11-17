"""Inspect real MNIST worker shards and export labeled samples/contact sheets."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = ROOT / "data" / "datasets" / "mnist_demo"
OUTPUT_ROOT = ROOT / "data" / "mnist_training_inspection"
RESULT_PATH = ROOT / "data" / "mnist_training_inspection.json"


def _stats(array: np.ndarray) -> dict[str, float]:
    values = np.asarray(array, dtype=np.float64)
    return {
        "min": float(values.min()),
        "max": float(values.max()),
        "mean": float(values.mean()),
        "std": float(values.std()),
    }


def _write_samples(
    worker_id: str,
    images: np.ndarray,
    labels: np.ndarray,
    raw_images: np.ndarray,
) -> list[str]:
    worker_dir = OUTPUT_ROOT / worker_id
    worker_dir.mkdir(parents=True, exist_ok=True)
    selected: list[tuple[int, int, Image.Image]] = []
    for label in range(10):
        indices = np.flatnonzero(labels == label)
        if indices.size == 0:
            raise RuntimeError(f"{worker_id} has no examples for label {label}")
        index = int(indices[0])
        pixels = np.clip(np.rint(raw_images[index, :, :, 0]), 0, 255).astype(np.uint8)
        image = Image.fromarray(pixels, mode="L")
        filename = f"{worker_id}_label_{label}_sample_00.png"
        image.save(worker_dir / filename)
        selected.append((label, index, image))

    cell_width = 140
    cell_height = 126
    sheet = Image.new("RGB", (5 * cell_width, 2 * cell_height), "white")
    draw = ImageDraw.Draw(sheet)
    for position, (label, _, image) in enumerate(selected):
        column = position % 5
        row = position // 5
        x = column * cell_width
        y = row * cell_height
        enlarged = image.resize((112, 112), resample=Image.Resampling.NEAREST)
        sheet.paste(enlarged.convert("RGB"), (x + 14, y))
        draw.text((x + 14, y + 113), f"label {label}", fill="black")
    contact_path = worker_dir / f"{worker_id}_contact_sheet.png"
    sheet.save(contact_path)
    return [str(worker_dir / f"{worker_id}_label_{label}_sample_00.png") for label in range(10)] + [
        str(contact_path)
    ]


def main() -> None:
    report_paths = sorted(
        (DATASET_ROOT / "processed").glob("run-*/preprocessing_report.json"),
        key=lambda path: path.stat().st_mtime_ns,
    )
    if not report_paths:
        raise FileNotFoundError("No MNIST preprocessing report found")
    report = json.loads(report_paths[-1].read_text(encoding="utf-8"))
    normalization = report.get("normalization", {})
    if normalization.get("method") != "standard":
        raise RuntimeError("Expected the recorded standard normalization")
    mean = np.asarray(normalization.get("mean"), dtype=np.float64).reshape((1, 1, 1, -1))
    std = np.asarray(normalization.get("standard_deviation"), dtype=np.float64).reshape((1, 1, 1, -1))
    if mean.size != 1 or std.size != 1:
        raise RuntimeError("MNIST normalization metadata must contain one channel statistic")

    results: dict[str, object] = {
        "dataset_name": "mnist_demo",
        "report_path": str(report_paths[-1]),
        "normalization": normalization,
        "workers": {},
    }
    for worker_id in ("worker1", "worker2", "worker3"):
        shard_path = DATASET_ROOT / "processed" / report_paths[-1].parent.name / "workers" / f"{worker_id}.npz"
        if not shard_path.is_file():
            raise FileNotFoundError(f"Missing worker shard: {worker_id}")
        with np.load(shard_path, allow_pickle=False) as shard:
            images = np.asarray(shard["images"])
            labels = np.asarray(shard["labels"])
            classes = np.asarray(shard["class_names"]).tolist()
        if images.ndim != 4 or images.shape[1:] != (28, 28, 1):
            raise RuntimeError(f"{worker_id} shard shape is unexpected: {images.shape}")
        if labels.ndim != 1 or labels.size != images.shape[0]:
            raise RuntimeError(f"{worker_id} image/label lengths do not match")
        unique_labels, counts = np.unique(labels, return_counts=True)
        if unique_labels.tolist() != list(range(10)):
            raise RuntimeError(f"{worker_id} label IDs are incomplete or out of order")

        raw_images = (images.astype(np.float64) * std + mean) * 255.0
        sample_paths = _write_samples(worker_id, images, labels, raw_images)
        row: dict[str, object] = {
            "shard_path": str(shard_path),
            "image_shape": list(images.shape),
            "image_dtype_after_normalization": str(images.dtype),
            "label_shape": list(labels.shape),
            "label_dtype": str(labels.dtype),
            "class_names": classes,
            "label_distribution": {
                str(int(label)): int(count) for label, count in zip(unique_labels, counts, strict=True)
            },
            "pixel_before_normalization_reconstructed": _stats(raw_images),
            "pixel_after_normalization": _stats(images),
            "sample_pngs": sample_paths,
        }
        results["workers"][worker_id] = row
        print(
            f"{worker_id}: shape={row['image_shape']}; image_dtype={images.dtype}; "
            f"label_dtype={labels.dtype}; labels={row['label_distribution']}; "
            f"before={row['pixel_before_normalization_reconstructed']}; "
            f"after={row['pixel_after_normalization']}"
        )
        print(f"{worker_id}: sample_pngs={json.dumps(sample_paths)}")

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"diagnostic_json={RESULT_PATH}")


if __name__ == "__main__":
    main()
