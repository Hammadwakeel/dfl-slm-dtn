import hashlib
import io
import json
import shutil
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import zipfile

import numpy as np
import pytest
from PIL import Image

from app.core.config import DeviceConfig
from app.services.datasets import DatasetStore
from app.services import initial_models
from app.services.initial_models import (
    InitialModelError,
    _make_bundle,
    build_initial_model,
    create_and_distribute_initial_model,
    delete_initial_model,
    get_initial_model_report,
)
from app.services.worker_shards import WorkerShardError


def _image_bytes(color: tuple[int, int, int]) -> bytes:
    output = io.BytesIO()
    Image.new("RGB", (4, 4), color).save(output, format="PNG")
    return output.getvalue()


class _InitialModelHandler(BaseHTTPRequestHandler):
    node_id = ""
    token = ""
    dataset_root = None
    post_calls = 0

    def log_message(self, _format, *_args):
        return

    def _send_json(self, status: int, value: dict) -> None:
        payload = json.dumps(value).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _authorized(self) -> bool:
        return self.headers.get("Authorization") == f"Bearer {self.token}"

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
        if self.path.endswith("/initial-model"):
            metadata = (
                self.dataset_root
                / "models"
                / "initial_model"
                / "metadata.json"
            )
            if not metadata.is_file():
                self._send_json(404, {"error": "model_not_found"})
                return
            self._send_json(200, json.loads(metadata.read_text(encoding="utf-8")))
            return
        self._send_json(404, {"error": "not_found"})

    def do_POST(self):
        if not self._authorized():
            self._send_json(401, {"error": "unauthorized"})
            return
        type(self).post_calls += 1
        payload = self.rfile.read(int(self.headers["Content-Length"]))
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            architecture_bytes = archive.read("architecture.json")
            architecture = json.loads(architecture_bytes)
            weights = archive.read("initial_weights.npz")
        with np.load(io.BytesIO(weights), allow_pickle=False) as arrays:
            parameter_count = sum(arrays[name].size for name in arrays.files)
        model_path = self.dataset_root / "models" / "initial_model"
        metadata = {
            "node_id": self.node_id,
            "dataset_name": architecture["dataset_name"],
            "parameter_count": int(parameter_count),
            "architecture_sha256": hashlib.sha256(architecture_bytes).hexdigest(),
            "weights_sha256": hashlib.sha256(weights).hexdigest(),
        }
        if model_path.exists():
            existing = json.loads((model_path / "metadata.json").read_text(encoding="utf-8"))
            if all(
                existing.get(key) == value
                for key, value in metadata.items()
                if key != "node_id"
            ):
                self._send_json(200, metadata)
            else:
                self._send_json(409, {"error": "different_model_installed"})
            return
        model_path.mkdir(parents=True)
        (model_path / "architecture.json").write_bytes(architecture_bytes)
        (model_path / "initial_weights.npz").write_bytes(weights)
        (model_path / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
        self._send_json(200, metadata)

    def do_DELETE(self):
        if not self._authorized():
            self._send_json(401, {"error": "unauthorized"})
            return
        model_path = self.dataset_root / "models" / "initial_model"
        if model_path.exists():
            shutil.rmtree(model_path)
        self.send_response(204)
        self.end_headers()


def test_build_initial_model_is_deterministic_and_under_parameter_cap():
    kwargs = {
        "dataset_name": "sample-dataset",
        "input_shape": (32, 32, 3),
        "classes": ["cat", "dog", "bird"],
        "seed": 12,
    }
    architecture, weights = build_initial_model(**kwargs)
    repeat_architecture, repeat_weights = build_initial_model(**kwargs)

    assert architecture == repeat_architecture
    # b1_c8_k3: conv 3*3*3*8 + 8 = 224, classifier 8*3 + 3 = 27.
    assert architecture["parameter_count"] == 251
    assert architecture["parameter_count"] <= 12_000
    for name in weights:
        np.testing.assert_array_equal(weights[name], repeat_weights[name])


def test_build_initial_model_supports_single_channel_mnist():
    architecture, weights = build_initial_model(
        dataset_name="mnist-demo",
        input_shape=(28, 28, 1),
        classes=[str(label) for label in range(10)],
        seed=42,
    )

    assert architecture["input_shape"] == [28, 28, 1]
    assert architecture["weight_shapes"]["conv1_kernel"] == [3, 3, 1, 8]
    assert weights["conv1_kernel"].shape == (3, 3, 1, 8)
    assert architecture["parameter_count"] <= 12_000


def test_build_initial_model_uses_the_selected_architecture():
    design = {"id": "b1_c16_k5", "blocks": 1, "channels": 16, "kernel_size": 5}
    architecture, weights = build_initial_model(
        dataset_name="mnist",
        input_shape=(28, 28, 1),
        classes=[str(label) for label in range(10)],
        seed=7,
        architecture=design,
    )

    assert architecture["design"] == design
    assert architecture["parameter_count"] == 586
    assert architecture["weight_shapes"]["conv1_kernel"] == [5, 5, 1, 16]
    assert "conv2_kernel" not in weights
    assert weights["classifier_kernel"].shape == (16, 10)


def test_build_initial_model_accepts_custom_keras_design():
    architecture, weights = build_initial_model(
        dataset_name="mnist",
        input_shape=(28, 28, 1),
        classes=[str(label) for label in range(10)],
        seed=42,
        architecture={
            "model_type": "custom",
            "layers": [
                {"type": "conv2d", "filters": 8, "kernel_size": 3},
                {"type": "global_average_pooling2d"},
                {"type": "dense", "units": 10, "activation": "linear"},
            ],
        },
    )

    assert architecture["model_type"] == "custom"
    assert architecture["parameter_count"] == 170
    assert architecture["initialization_seed"] == 42
    assert weights == {}


@pytest.mark.parametrize("model_type", ["custom", "pretrained_deit"])
def test_weightless_initial_model_bundles_are_closed_valid_archives(model_type):
    bundle = _make_bundle({"model_type": model_type}, {})

    with zipfile.ZipFile(io.BytesIO(bundle)) as archive:
        assert archive.testzip() is None
        assert archive.namelist() == ["architecture.json"]
        assert json.loads(archive.read("architecture.json"))["model_type"] == model_type


def test_weighted_initial_model_bundle_contains_weights():
    bundle = _make_bundle(
        {"model_type": "small_cnn"},
        {"weights": np.array([1.0], dtype=np.float32)},
    )

    with zipfile.ZipFile(io.BytesIO(bundle)) as archive:
        assert archive.testzip() is None
        assert archive.namelist() == ["architecture.json", "initial_weights.npz"]


def test_initial_model_rejects_invalid_input_and_parameter_overflow():
    with pytest.raises(InitialModelError, match="one-channel grayscale"):
        build_initial_model(
            dataset_name="sample-dataset",
            input_shape=(32, 32, 2),
            classes=["cat", "dog"],
        )
    with pytest.raises(InitialModelError, match="more than"):
        build_initial_model(
            dataset_name="sample-dataset",
            input_shape=(32, 32, 3),
            classes=["cat", "dog"],
            architecture={
                "model_type": "custom",
                "layers": [
                    {"type": "flatten"},
                    {"type": "dense", "units": 8192},
                    {"type": "dense", "units": 8192},
                    {"type": "dense", "units": 2, "activation": "linear"},
                ],
            },
        )


def test_initial_model_report_rejects_non_object_architecture(tmp_path):
    store = DatasetStore(tmp_path)
    store.upload_folder(
        "sample-dataset",
        [
            ("cat/cat.png", _image_bytes((255, 0, 0))),
            ("dog/dog.png", _image_bytes((0, 0, 255))),
        ],
    )
    model_path = tmp_path / "sample-dataset" / "models" / "initial_model"
    model_path.mkdir(parents=True)
    (model_path / "architecture.json").write_text("[]", encoding="utf-8")
    (model_path / "metadata.json").write_text("{}", encoding="utf-8")

    with pytest.raises(InitialModelError, match="metadata or hashes"):
        get_initial_model_report(store, "sample-dataset")


def test_create_and_distribute_initial_model_from_preprocessed_dataset(tmp_path, monkeypatch):
    dataset_root = tmp_path / "datasets"
    store = DatasetStore(dataset_root)
    store.upload_folder(
        "sample-dataset",
        [
            ("cat/cat.png", _image_bytes((255, 0, 0))),
            ("dog/dog.png", _image_bytes((0, 0, 255))),
        ],
    )
    processed = dataset_root / "sample-dataset" / "processed" / "run-fixture"
    processed.mkdir(parents=True)
    (processed / "preprocessing_report.json").write_text(
        json.dumps(
            {
                "target_size": {"height": 16, "width": 12},
                "channels": 1,
                "image_shape": [16, 12, 1],
            }
        ),
        encoding="utf-8",
    )

    servers = []
    threads = []
    workers = []
    handlers = []
    try:
        for index in range(3):
            worker_root = tmp_path / f"worker{index + 1}"
            worker_root.mkdir()
            worker_dataset_root = worker_root / "sample-dataset"
            worker_dataset_root.mkdir()
            handler = type(
                f"InitialModelHandler{index + 1}",
                (_InitialModelHandler,),
                {
                    "node_id": f"worker{index + 1}",
                    "token": f"worker-token-{index}-long-enough",
                    "dataset_root": worker_dataset_root,
                    "post_calls": 0,
                },
            )
            server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            servers.append(server)
            threads.append(thread)
            handlers.append(handler)
            workers.append(
                DeviceConfig(
                    node_id=f"worker{index + 1}",
                    hostname=f"worker{index + 1}",
                    ip="127.0.0.1",
                    role="worker",
                    port=5001,
                    token=handler.token,
                    training_port=server.server_port,
                )
            )

        report = create_and_distribute_initial_model(
            store,
            workers,
            "sample-dataset",
            seed=21,
        )
        assert report["parameter_count"] == report["architecture"]["parameter_count"]
        assert report["architecture"]["input_shape"] == [16, 12, 1]
        assert set(report["workers"]) == {"worker1", "worker2", "worker3"}
        assert len({item["weights_sha256"] for item in report["workers"].values()}) == 1
        assert all(
            item["weights_sha256"] == report["weights_sha256"]
            for item in report["workers"].values()
        )
        assert all(
            item["architecture_sha256"] == report["architecture_sha256"]
            for item in report["workers"].values()
        )
        assert [handler.post_calls for handler in handlers] == [1, 1, 1]
        model_path = dataset_root / "sample-dataset" / "models" / "initial_model"
        assert (model_path / "architecture.json").is_file()
        assert (model_path / "initial_weights.npz").is_file()
        with np.load(model_path / "initial_weights.npz", allow_pickle=False) as arrays:
            assert sum(arrays[name].size for name in arrays.files) == report["parameter_count"]
        persisted = get_initial_model_report(store, "sample-dataset")
        assert persisted is not None
        assert persisted["architecture"]["design"] == report["architecture"]["design"]
        missing_model_path = (
            tmp_path / "worker3" / "sample-dataset" / "models" / "initial_model"
        )
        shutil.rmtree(missing_model_path)
        repaired = create_and_distribute_initial_model(
            store,
            workers,
            "sample-dataset",
            seed=21,
        )
        assert repaired["reused_existing_model"] is True
        assert repaired["architecture_sha256"] == report["architecture_sha256"]
        assert repaired["weights_sha256"] == report["weights_sha256"]
        assert all(
            item["architecture_sha256"] == report["architecture_sha256"]
            for item in repaired["workers"].values()
        )
        assert all(
            item["weights_sha256"] == report["weights_sha256"]
            for item in repaired["workers"].values()
        )
        assert [handler.post_calls for handler in handlers] == [1, 1, 2]
        # A distribution that stopped part-way can leave one worker on another model.
        real_present = initial_models.get_worker_dataset_initial_model_if_present
        stale_workers = {"worker2"}

        def stale_on_worker2(worker, name):
            found = real_present(worker, name)
            if found is not None and worker.node_id in stale_workers:
                return {**found, "architecture_sha256": "stale-model"}
            return found

        removed = []
        real_delete = initial_models.delete_worker_initial_model

        def recording_delete(worker, name, **kwargs):
            removed.append(worker.node_id)
            stale_workers.discard(worker.node_id)
            return real_delete(worker, name, **kwargs)

        monkeypatch.setattr(initial_models, "get_worker_dataset_initial_model_if_present", stale_on_worker2)
        monkeypatch.setattr(initial_models, "delete_worker_initial_model", recording_delete)
        with pytest.raises(WorkerShardError, match="different initial model"):
            create_and_distribute_initial_model(store, workers, "sample-dataset", seed=21)
        assert removed == []  # without replace_existing a mismatch is never overwritten
        healed = create_and_distribute_initial_model(
            store, workers, "sample-dataset", seed=21, replace_existing=True
        )
        assert removed == ["worker2"]
        assert healed["architecture_sha256"] == report["architecture_sha256"]
        assert [handler.post_calls for handler in handlers] == [1, 2, 2]
        monkeypatch.undo()
        # restore_baseline reuses only an untrained initialization from the same seed.
        same_seed = create_and_distribute_initial_model(
            store, workers, "sample-dataset", seed=21, restore_baseline=True
        )
        assert same_seed["reused_existing_model"] is True
        other_seed = create_and_distribute_initial_model(
            store, workers, "sample-dataset", seed=22, restore_baseline=True
        )
        assert other_seed["reused_existing_model"] is False
        assert other_seed["weights_sha256"] != report["weights_sha256"]
        metadata_path = model_path / "metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        assert metadata["initialization"] == {"trained": False, "seed": 22}
        # A run ends by installing its trained best model; the next run must not reuse it.
        metadata["initialization"] = {"trained": True, "seed": None}
        metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
        after_checkpoint = create_and_distribute_initial_model(
            store, workers, "sample-dataset", seed=22, restore_baseline=True
        )
        assert after_checkpoint["reused_existing_model"] is False
        assert after_checkpoint["weights_sha256"] == other_seed["weights_sha256"]
        deleted = delete_initial_model(store, workers, "sample-dataset")
        assert deleted["deleted"] is True
        assert deleted["workers"] == ["worker1", "worker2", "worker3"]
        assert not model_path.exists()
        assert all(
            not (
                tmp_path / worker.node_id / "sample-dataset" / "models" / "initial_model"
            ).exists()
            for worker in workers
        )
    finally:
        for server in servers:
            server.shutdown()
            server.server_close()
        for thread in threads:
            thread.join(timeout=3)


def test_initial_model_deletion_reports_partial_worker_failure(tmp_path, monkeypatch):
    store = DatasetStore(tmp_path / "datasets")
    model_path = store.root / "sample-dataset" / "models" / "initial_model"
    model_path.mkdir(parents=True)
    workers = [
        DeviceConfig(
            node_id=f"worker{index}",
            hostname=f"worker{index}",
            ip="127.0.0.1",
            role="worker",
            port=5000 + index,
            token=f"worker-token-{index}-long-enough",
            training_port=58000 + index,
        )
        for index in range(1, 4)
    ]
    monkeypatch.setattr(initial_models, "get_worker_training_status", lambda _worker: {})

    def delete_worker(worker, _dataset_name):
        if worker.node_id == "worker2":
            raise WorkerShardError("worker2 is unreachable")

    monkeypatch.setattr(initial_models, "delete_worker_initial_model", delete_worker)

    with pytest.raises(
        InitialModelError,
        match="workers already cleared: worker1",
    ):
        delete_initial_model(store, workers, "sample-dataset")

    assert model_path.is_dir()


def test_report_accepts_a_pruned_custom_model_with_stored_trained_weights(tmp_path):
    store = DatasetStore(tmp_path)
    store.upload_folder(
        "sample-dataset",
        [("cat/cat.png", _image_bytes((255, 0, 0))), ("dog/dog.png", _image_bytes((0, 0, 255)))],
    )
    model_path = tmp_path / "sample-dataset" / "models" / "initial_model"
    model_path.mkdir(parents=True)
    architecture = {
        "model_type": "custom",
        "dataset_name": "sample-dataset",
        "input_shape": [4, 4, 3],
        "classes": ["cat", "dog"],
        "num_classes": 2,
        "parameter_count": 98,
        "layers": [{"type": "flatten"}, {"type": "dense", "units": 2, "activation": "linear"}],
    }
    architecture_bytes = json.dumps(architecture).encode("utf-8")
    (model_path / "architecture.json").write_bytes(architecture_bytes)
    (model_path / "initial_weights.npz").write_bytes(b"trained-weights")
    (model_path / "metadata.json").write_text(
        json.dumps({
            "dataset_name": "sample-dataset",
            "architecture_sha256": hashlib.sha256(architecture_bytes).hexdigest(),
            "weights_sha256": "a" * 64,
            "parameter_count": 98,
        }),
        encoding="utf-8",
    )

    report = get_initial_model_report(store, "sample-dataset")

    assert report is not None and report["architecture"]["model_type"] == "custom"
