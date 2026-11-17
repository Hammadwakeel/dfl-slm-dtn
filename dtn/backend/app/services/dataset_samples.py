"""Small, deterministic previews for the supported official datasets."""

import base64
import io
from typing import Any

import numpy as np
from PIL import Image

from app.services.datasets import DatasetError, DatasetStore

SAMPLE_ARCHIVES = {
    "mnist": {
        "filename": "mnist.npz",
        "policy": {"type": "official_mnist_npz"},
        "image_shape": (60_000, 28, 28),
    },
    "cifar10": {
        "filename": "cifar10.npz",
        "policy": {"type": "official_cifar10_npz"},
        "image_shape": (50_000, 32, 32, 3),
    },
}


def get_dataset_samples(store: DatasetStore, dataset_name: str) -> list[dict[str, Any]]:
    metadata = store.get_metadata(dataset_name)
    archive_config = SAMPLE_ARCHIVES.get(dataset_name)
    classes = metadata.get("classes")
    if (
        archive_config is None
        or metadata.get("dataset_name") != dataset_name
        or metadata.get("source_split_policy") != archive_config["policy"]
        or not isinstance(classes, list)
        or len(classes) != 10
        or any(not isinstance(name, str) for name in classes)
    ):
        raise DatasetError(f"Official sample data is not available for {dataset_name}", 404)

    archive_path = store.root / dataset_name / str(archive_config["filename"])
    try:
        with np.load(archive_path, allow_pickle=False) as archive:
            images = np.asarray(archive["x_train"])
            labels = np.asarray(archive["y_train"])
    except (OSError, ValueError, KeyError) as error:
        raise DatasetError(f"Official sample data is invalid for {dataset_name}") from error

    expected_shape = archive_config["image_shape"]
    if (
        images.shape != expected_shape
        or images.dtype != np.uint8
        or labels.shape != (expected_shape[0],)
        or not np.issubdtype(labels.dtype, np.integer)
        or not np.isin(labels, np.arange(10)).all()
    ):
        raise DatasetError(f"Official sample data is invalid for {dataset_name}")

    first_indices: dict[int, int] = {}
    for index, label in enumerate(labels):
        first_indices.setdefault(int(label), index)
        if len(first_indices) == 10:
            break
    if len(first_indices) != 10:
        raise DatasetError(f"Official sample data is incomplete for {dataset_name}")

    samples = []
    for label, index in sorted(first_indices.items()):
        output = io.BytesIO()
        Image.fromarray(images[index]).save(output, format="PNG", optimize=True)
        samples.append(
            {
                "label": classes[label],
                "image_data": f"data:image/png;base64,{base64.b64encode(output.getvalue()).decode('ascii')}",
            }
        )
    return samples
