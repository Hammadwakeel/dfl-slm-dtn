import io
import json
import threading
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import pytest
from PIL import Image

from app.core.config import DeviceConfig
from app.services.dataset_pipeline import preprocess_and_ship
from app.services.datasets import DatasetStore
from app.services import preprocessing
from app.services.preprocessing import (
    PreprocessingError,
    class_balance_report,
    normalize_images,
    preprocess_image_dataset,
    resize_images,
    shard_training_indices,
    stratified_split,
    validate_images,
)


def _image_bytes(color: tuple[int, int, int]) -> bytes:
    output = io.BytesIO()
    Image.new("RGB", (9, 7), color).save(output, format="PNG")
    return output.getvalue()


def _make_dataset(root, per_class: int = 10):
    root.mkdir()
    classes = ["cat", "dog"]
    for class_index, class_name in enumerate(classes):
        class_path = root / class_name
        class_path.mkdir()
        for index in range(per_class):
            Image.new(
                "RGB",
                (9, 7),
                (class_index * 100, index * 10, 20),
            ).save(class_path / f"{index}.png")
    (root / "metadata.json").write_text(
        json.dumps({"classes": classes}),
        encoding="utf-8",
    )
    return root


def test_validate_resize_normalize_and_split_images(tmp_path):
    dataset_path = _make_dataset(tmp_path / "images")
    validation = validate_images(dataset_path)

    assert validation["image_count"] == 20
    assert validation["class_counts"] == {"cat": 10, "dog": 10}
    images, labels = resize_images(validation, (5, 6))
    assert images.shape == (20, 5, 6, 3)
    assert labels.shape == (20,)

    splits = stratified_split(labels, validation_fraction=0.2, test_fraction=0.2, seed=7)
    assert {name: len(indices) for name, indices in splits.items()} == {
        "train": 12,
        "validation": 4,
        "test": 4,
    }
    all_indices = [index for indices in splits.values() for index in indices]
    assert sorted(all_indices) == list(range(20))
    assert not (set(splits["train"]) & set(splits["test"]))

    normalized, report = normalize_images(images, "standard", splits["train"])
    assert normalized.shape == images.shape
    assert normalized.dtype == np.float32
    assert report["method"] == "standard"

    minmax, minmax_report = normalize_images(images, "minmax", splits["train"])
    assert float(minmax.min()) >= 0.0
    assert float(minmax.max()) <= 1.0
    assert minmax_report["maximum"] == 1.0

    unchanged, none_report = normalize_images(images, "none", splits["train"])
    np.testing.assert_array_equal(unchanged, images)
    assert none_report == {"method": "none"}


def test_grayscale_mnist_shape_is_preserved_in_preprocessing_metadata(tmp_path):
    dataset_path = tmp_path / "mnist"
    dataset_path.mkdir()
    for label in range(10):
        class_path = dataset_path / str(label)
        class_path.mkdir()
        for index in range(3):
            Image.new("L", (28, 28), color=label * 20 + index).save(
                class_path / f"{index}.png"
            )
    (dataset_path / "metadata.json").write_text(
        json.dumps({"classes": [str(label) for label in range(10)]}),
        encoding="utf-8",
    )

    report = preprocess_image_dataset(
        dataset_path,
        target_size=(28, 28),
        normalization="standard",
        seed=42,
    )

    assert report["image_count"] == 30
    assert report["target_size"] == {"height": 28, "width": 28}
    assert report["channels"] == 1
    assert report["image_shape"] == [28, 28, 1]
    assert report["split_sizes"] == {"train": 10, "validation": 10, "test": 10}
    with np.load(Path(report["output_path"]) / "train.npz", allow_pickle=False) as archive:
        assert archive["images"].shape[1:] == (28, 28, 1)


def test_official_mnist_test_split_is_preserved(tmp_path):
    dataset_path = tmp_path / "mnist"
    dataset_path.mkdir()
    train_images = np.zeros((60_000, 28, 28), dtype=np.uint8)
    test_images = np.full((10_000, 28, 28), 255, dtype=np.uint8)
    train_labels = np.tile(np.arange(10, dtype=np.uint8), 6_000)
    test_labels = np.tile(np.arange(10, dtype=np.uint8), 1_000)
    np.savez_compressed(
        dataset_path / "mnist.npz",
        x_train=train_images,
        y_train=train_labels,
        x_test=test_images,
        y_test=test_labels,
    )
    (dataset_path / "metadata.json").write_text(
        json.dumps(
            {
                "dataset_name": "mnist",
                "classes": [str(label) for label in range(10)],
                "source_split_policy": {"type": "official_mnist_npz"},
            }
        ),
        encoding="utf-8",
    )

    report = preprocess_image_dataset(
        dataset_path,
        target_size=(28, 28),
        validation_fraction=0.1,
        seed=42,
    )

    assert report["split_method"] == "official_mnist_train_validation_test"
    assert report["official_test_set_preserved"] is True
    assert report["split_sizes"] == {"train": 54_000, "validation": 6_000, "test": 10_000}
    with np.load(Path(report["output_path"]) / "test.npz", allow_pickle=False) as archive:
        np.testing.assert_array_equal(archive["labels"], test_labels)
        assert np.all(archive["images"] == 1)


def test_official_cifar10_test_split_is_preserved(tmp_path, monkeypatch):
    dataset_path = tmp_path / "cifar10"
    dataset_path.mkdir()
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
    (dataset_path / "metadata.json").write_text(
        json.dumps(
            {
                "dataset_name": "cifar10",
                "classes": classes,
                "source_split_policy": {"type": "official_cifar10_npz"},
            }
        ),
        encoding="utf-8",
    )
    train_labels = np.tile(np.arange(10, dtype=np.int64), 10)
    test_labels = np.tile(np.arange(10, dtype=np.int64), 2)
    images = np.zeros((120, 32, 32, 3), dtype=np.uint8)
    images[100:] = 255
    labels = np.concatenate((train_labels, test_labels))
    validation = {
        "classes": classes,
        "image_count": len(labels),
        "class_counts": class_balance_report(labels, classes),
        "invalid_images": [],
        "official_split": {
            "train": np.arange(100, dtype=np.int64),
            "test": np.arange(100, 120, dtype=np.int64),
        },
    }
    monkeypatch.setattr(
        preprocessing,
        "_load_official_cifar10",
        lambda _path: (validation, images, labels),
    )

    report = preprocess_image_dataset(
        dataset_path,
        target_size=(32, 32),
        validation_fraction=0.1,
        seed=42,
    )

    assert report["split_method"] == "official_cifar10_train_validation_test"
    assert report["official_test_set_preserved"] is True
    assert report["target_size"] == {"height": 32, "width": 32}
    assert report["channels"] == 3
    assert report["split_sizes"] == {"train": 90, "validation": 10, "test": 20}
    with np.load(Path(report["output_path"]) / "test.npz", allow_pickle=False) as archive:
        np.testing.assert_array_equal(archive["labels"], test_labels)
        assert np.all(archive["images"] == 1)


@pytest.mark.parametrize(
    "strategy",
    [
        "stratified",
        "dirichlet_a01",
        "dirichlet_a03",
        "dirichlet_a05",
        "dirichlet_a07",
    ],
)
def test_worker_sharding_preserves_rows_and_reports_class_balance(strategy):
    labels = np.repeat(np.arange(3, dtype=np.int64), 15)
    shards = shard_training_indices(
        labels,
        strategy=strategy,
        seed=13,
        worker_count=4,
    )

    flattened = [index for shard in shards for index in shard]
    assert sorted(flattened) == list(range(len(labels)))
    assert len(shards) == 4
    assert all(shard for shard in shards)
    assert class_balance_report(labels[shards[0]], ["a", "b", "c"]) == {
        "a": Counter(labels[shards[0]])[0],
        "b": Counter(labels[shards[0]])[1],
        "c": Counter(labels[shards[0]])[2],
    }


def test_preprocessing_rejects_corrupt_images_and_small_strata(tmp_path):
    dataset_path = _make_dataset(tmp_path / "images", per_class=3)
    (dataset_path / "cat" / "broken.png").write_bytes(b"not an image")

    validation = validate_images(dataset_path)
    assert len(validation["invalid_images"]) == 1
    with pytest.raises(PreprocessingError, match="corrupt or unsafe images"):
        preprocess_image_dataset(dataset_path)

    with pytest.raises(PreprocessingError, match="at least 3 images"):
        stratified_split([0, 0, 1, 1, 1])


class _WorkerFixtureHandler(BaseHTTPRequestHandler):
    node_id = ""
    token = ""
    installed: dict[str, dict] = {}
    installed_methods: dict[str, str] = {}

    def log_message(self, _format, *_args):
        return

    def _authorized(self) -> bool:
        return self.headers.get("Authorization") == f"Bearer {self.token}"

    def _send_json(self, status: int, value: dict) -> None:
        payload = json.dumps(value).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):
        if not self._authorized():
            self._send_json(401, {"error": "unauthorized"})
            return
        if self.path == "/v1/status":
            self._send_json(
                200,
                {
                    "node_id": self.node_id,
                    "active_run_id": None,
                    "scheduled_run_id": None,
                },
            )
            return
        if self.path.startswith("/v1/datasets/"):
            metadata = self.installed.get(self.node_id)
            if metadata is None:
                self._send_json(404, {"error": "dataset_not_found"})
                return
            self._send_json(200, metadata)
            return
        else:
            self._send_json(404, {"error": "not_found"})

    def do_POST(self):
        if not self._authorized():
            self._send_json(401, {"error": "unauthorized"})
            return
        self.installed_methods[self.node_id] = self.command
        dataset_name = urlparse(self.path).path.split("/")[-2]
        payload = self.rfile.read(int(self.headers["Content-Length"]))
        with np.load(io.BytesIO(payload), allow_pickle=False) as shard:
            labels = shard["labels"]
            names = [str(name) for name in shard["class_names"].tolist()]
        counts = {
            names[int(label)]: int(count)
            for label, count in zip(*np.unique(labels, return_counts=True))
        }
        result = {
            "node_id": self.node_id,
            "dataset_name": dataset_name,
            "rows": int(len(labels)),
            "class_counts": counts,
            "shard_sha256": "fixture-hash",
        }
        self.installed[self.node_id] = result
        self._send_json(200, result)

    def do_PUT(self):
        self.do_POST()


@pytest.mark.parametrize("replace_existing", [False, True])
def test_preprocess_action_pipeline_checks_and_installs_all_worker_shards(
    tmp_path,
    replace_existing,
):
    store = DatasetStore(tmp_path / "datasets")
    files = []
    for class_name, base_color in (("cat", (40, 0, 0)), ("dog", (0, 40, 0))):
        for index in range(10):
            files.append(
                (
                    f"{class_name}/{index}.png",
                    _image_bytes((base_color[0], base_color[1], index)),
                )
            )
    store.upload_folder("user-images", files)

    workers = []
    servers = []
    threads = []
    _WorkerFixtureHandler.installed = {}
    _WorkerFixtureHandler.installed_methods = {}
    for index in range(4):
        token = f"worker-fixture-token-{index}-long-enough"
        handler = type(
            f"WorkerFixtureHandler{index}",
            (_WorkerFixtureHandler,),
            {"node_id": f"worker{index + 1}", "token": token},
        )
        server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        servers.append(server)
        threads.append(thread)
        workers.append(
            DeviceConfig(
                node_id=f"worker{index + 1}",
                hostname=f"worker{index + 1}",
                ip="127.0.0.1",
                role="worker",
                port=5001,
                token=token,
                training_port=server.server_port,
            )
        )
    try:
        report = preprocess_and_ship(
            store,
            workers,
            "user-images",
            normalization="minmax",
            split_strategy="stratified",
            replace_existing=replace_existing,
        )
        assert report["split_sizes"] == {"train": 16, "validation": 2, "test": 2}
        assert report["shipping_duration_seconds"] >= 0
        assert set(report["uploaded_workers"]) == {
            "worker1",
            "worker2",
            "worker3",
            "worker4",
        }
        expected_method = "PUT" if replace_existing else "POST"
        assert set(_WorkerFixtureHandler.installed_methods.values()) == {expected_method}
        for node_id, expected in report["workers"].items():
            installed = _WorkerFixtureHandler.installed[node_id]
            assert installed["rows"] == expected["rows"]
            assert installed["class_counts"] == {
                name: count
                for name, count in expected["class_counts"].items()
                if count > 0
            }
    finally:
        for server in servers:
            server.shutdown()
            server.server_close()
        for thread in threads:
            thread.join(timeout=3)
