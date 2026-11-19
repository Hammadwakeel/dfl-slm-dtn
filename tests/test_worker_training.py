import json
import hashlib
import io
import sys
import threading
import zipfile
from http.server import ThreadingHTTPServer
from types import SimpleNamespace
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import numpy as np
import pytest

from worker import training_service
from worker.training_service import (
    Peer,
    WorkerConfig,
    WorkerTrainer,
    _load_installed_dataset_initial_model,
    deserialize_weights,
    install_dataset_shard,
    install_global_validation,
    install_initial_model,
    make_handler,
    mesh_neighbours,
    ring_neighbours,
    serialize_weights,
    topology_neighbours,
    topology_senders,
)


def _make_initial_model_bundle(
    dataset_name: str,
    *,
    input_shape: tuple[int, int, int] = (8, 8, 1),
    class_names: tuple[str, ...] = ("cat", "dog"),
    training: dict | None = None,
) -> bytes:
    import tensorflow as tf

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=input_shape, dtype=tf.float32),
            tf.keras.layers.Conv2D(1, (3, 3), padding="same", activation="relu"),
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(len(class_names), activation="softmax"),
        ],
        name=f"dataset_{dataset_name}_initial",
    )
    model(np.zeros((1, *input_shape), dtype=np.float32), training=False)
    weights = model.get_weights()
    weight_shapes = {
        "conv1_kernel": list(weights[0].shape),
        "conv1_bias": list(weights[1].shape),
        "classifier_kernel": list(weights[2].shape),
        "classifier_bias": list(weights[3].shape),
    }
    architecture = {
        "model_type": "small_cnn",
        "dataset_name": dataset_name,
        "input_shape": list(input_shape),
        "classes": list(class_names),
        "num_classes": len(class_names),
        "parameter_count": int(sum(np.prod(array.shape) for array in weights)),
        "weight_shapes": weight_shapes,
        "layers": [
            {
                "type": "conv2d",
                "filters": 1,
                "kernel_size": [3, 3],
                "padding": "same",
                "activation": "relu",
            },
            {"type": "global_average_pooling2d"},
            {
                "type": "dense",
                "units": len(class_names),
                "activation": "softmax",
            },
        ],
    }
    if training is not None:
        architecture["training"] = training
    weight_buffer = io.BytesIO()
    np.savez_compressed(
        weight_buffer,
        conv1_kernel=weights[0],
        conv1_bias=weights[1],
        classifier_kernel=weights[2],
        classifier_bias=weights[3],
    )
    bundle = io.BytesIO()
    with zipfile.ZipFile(bundle, "w", compression=zipfile.ZIP_DEFLATED) as zipped:
        zipped.writestr("architecture.json", json.dumps(architecture).encode("utf-8"))
        zipped.writestr("initial_weights.npz", weight_buffer.getvalue())
    return bundle.getvalue()


def _installed_bundle_trainer(tmp_path, training):
    dataset_name = "settings-run"
    dataset_path = tmp_path / "datasets" / dataset_name
    dataset_path.mkdir(parents=True)
    labels = np.tile(np.arange(2, dtype=np.int64), 10)
    images = np.linspace(0.0, 1.0, 20 * 8 * 8, dtype=np.float32).reshape(20, 8, 8, 1)
    np.savez_compressed(
        dataset_path / "train.npz", images=images, labels=labels, class_names=np.asarray(("cat", "dog"))
    )
    shard_path = tmp_path / "default_shard.npz"
    np.savez(shard_path, images=np.zeros((1, 32, 32, 3), dtype=np.uint8), labels=np.array([0]))
    config = WorkerConfig(
        node_id="worker1",
        api_token="a" * 48,
        port=5004,
        data_path=str(shard_path),
        dataset_root=str(tmp_path / "datasets"),
        peers=(),
        log_path=str(tmp_path / "worker.jsonl"),
    )
    install_initial_model(config, dataset_name, _make_initial_model_bundle(dataset_name, training=training))
    return WorkerTrainer(config), dataset_name


def test_installed_model_trains_with_its_own_settings_and_run_augmentation_overrides(tmp_path):
    training = {"optimizer": "adam", "learning_rate": 0.003, "batch_size": 4, "augmentation": "flip_shift"}
    trainer, dataset_name = _installed_bundle_trainer(tmp_path, training)
    trainer.configure(rounds=1, epochs=1, topology="ring", dataset_name=dataset_name)
    trainer._create_run_model("own-settings", 7)
    audit = trainer._run_model_audits["own-settings"]
    assert audit["learning_rate"] == pytest.approx(0.003)
    assert audit["batch_size"] == 4
    assert audit["augmentation"] == "flip_shift"

    trainer.configure(rounds=1, epochs=1, topology="ring", dataset_name=dataset_name, augmentation="none")
    trainer._create_run_model("override", 7)
    assert trainer._run_model_audits["override"]["augmentation"] == "none"
    with pytest.raises(ValueError, match="augmentation"):
        trainer.configure(rounds=1, epochs=1, topology="ring", dataset_name=dataset_name, augmentation="mixup")


def test_installed_model_without_settings_keeps_worker_defaults(tmp_path):
    trainer, dataset_name = _installed_bundle_trainer(tmp_path, None)
    trainer.configure(rounds=1, epochs=1, topology="ring", dataset_name=dataset_name, augmentation="flip_shift")
    trainer._create_run_model("defaults", 7)
    audit = trainer._run_model_audits["defaults"]
    assert audit["learning_rate"] == pytest.approx(0.01)
    assert audit["batch_size"] == 32
    assert audit["augmentation"] == "flip_shift"


def test_ring_and_mesh_neighbour_sets_are_distinct():
    worker_ids = ("worker1", "worker2", "worker3")

    assert ring_neighbours("worker1", worker_ids) == ("worker2",)
    assert ring_neighbours("worker3", worker_ids) == ("worker1",)
    assert mesh_neighbours("worker1", worker_ids) == ("worker2", "worker3")


def test_all_candidate_topologies_have_expected_directed_adjacency():
    worker_ids = ("worker1", "worker2", "worker3")

    assert topology_neighbours("worker1", worker_ids, "directed_ring") == ("worker2",)
    assert topology_senders("worker2", worker_ids, "directed_ring") == ("worker1",)
    assert topology_neighbours("worker1", worker_ids, "bidirectional_ring") == (
        "worker2",
        "worker3",
    )
    assert topology_neighbours("worker2", worker_ids, "mesh") == ("worker1", "worker3")
    assert topology_neighbours("worker2", worker_ids, "chain") == ("worker1", "worker3")
    assert topology_neighbours("worker1", worker_ids, "star") == ("worker2", "worker3")
    assert topology_neighbours("worker3", worker_ids, "star") == ("worker1",)


def test_weight_serialization_accepts_individual_arrays_larger_than_four_megabytes():
    weights = np.random.default_rng(42).random(1_100_000, dtype=np.float32)

    decoded = deserialize_weights(serialize_weights([weights]))

    np.testing.assert_array_equal(decoded[0], weights)


def test_worker_configure_loads_only_the_named_preprocessed_dataset(tmp_path):
    base_shard = tmp_path / "base.npz"
    np.savez(
        base_shard,
        images=np.zeros((4, 32, 32, 3), dtype=np.uint8),
        labels=np.array([0, 1, 0, 1]),
    )
    dataset_root = tmp_path / "datasets"
    train_shard = dataset_root / "cifar_10k_demo" / "train.npz"
    train_shard.parent.mkdir(parents=True)
    np.savez_compressed(
        train_shard,
        images=np.full((6, 8, 8, 3), 0.25, dtype=np.float32),
        labels=np.array([0, 0, 0, 1, 1, 1]),
        class_names=np.array(["cat", "dog"]),
    )
    config = WorkerConfig(
        node_id="worker1",
        api_token="a" * 48,
        port=5004,
        data_path=str(base_shard),
        dataset_root=str(dataset_root),
        peers=(),
        log_path=str(tmp_path / "worker.jsonl"),
    )
    trainer = WorkerTrainer(config)

    configured = trainer.configure(
        rounds=1,
        epochs=1,
        topology="bidirectional_ring",
        dataset_name="cifar_10k_demo",
    )

    assert configured["dataset_name"] == "cifar_10k_demo"
    assert configured["sample_count"] == 6
    assert trainer.status()["dataset_name"] == "cifar_10k_demo"
    assert trainer.dataset_split == "dataset:cifar_10k_demo"
    assert trainer.images.dtype == np.float32
    assert trainer.images.shape == (6, 8, 8, 3)
    assert trainer._dataset_class_names == ["cat", "dog"]


def test_cleanup_run_releases_model_and_cached_round_state(tmp_path, monkeypatch):
    shard_path = tmp_path / "base.npz"
    np.savez(
        shard_path,
        images=np.zeros((4, 32, 32, 3), dtype=np.uint8),
        labels=np.array([0, 1, 0, 1]),
    )
    config = WorkerConfig(
        node_id="worker1",
        api_token="a" * 48,
        port=5004,
        data_path=str(shard_path),
        peers=(),
        log_path=str(tmp_path / "worker.jsonl"),
    )
    trainer = WorkerTrainer(config)
    cleared_sessions = []
    tensorflow = SimpleNamespace(
        keras=SimpleNamespace(
            backend=SimpleNamespace(clear_session=lambda: cleared_sessions.append(True))
        )
    )
    monkeypatch.setitem(sys.modules, "tensorflow", tensorflow)
    monkeypatch.setattr(training_service, "_trim_process_heap", lambda: None)
    trainer._models["run-1"] = object()
    trainer._run_architectures["run-1"] = {}
    trainer._run_model_audits["run-1"] = {}
    trainer._completed_rounds["run-1"] = 0
    trainer._round_weights[("run-1", 0)] = [np.ones(2)]
    trainer._round_sample_counts[("run-1", 0)] = 2
    trainer._results["run-1"] = [{}]
    trainer._run_errors["run-1"] = "old error"
    trainer._incoming[("run-1", 0)] = {}

    result = trainer.cleanup_run("run-1")

    assert result == {"run_id": "run-1", "cleaned": True}
    assert cleared_sessions == [True]
    assert trainer._models == {}
    assert trainer._run_architectures == {}
    assert trainer._run_model_audits == {}
    assert trainer._completed_rounds == {}
    assert trainer._round_weights == {}
    assert trainer._round_sample_counts == {}
    assert trainer._results == {}
    assert trainer._run_errors == {}
    assert trainer._incoming == {}


def test_cleanup_memory_clears_all_run_caches_and_reports_memory(tmp_path, monkeypatch):
    shard_path = tmp_path / "base.npz"
    np.savez(
        shard_path,
        images=np.zeros((4, 32, 32, 3), dtype=np.uint8),
        labels=np.array([0, 1, 0, 1]),
    )
    config = WorkerConfig(
        node_id="worker1",
        api_token="a" * 48,
        port=5004,
        data_path=str(shard_path),
        peers=(),
        log_path=str(tmp_path / "worker.jsonl"),
    )
    trainer = WorkerTrainer(config)
    trainer._run_architectures["run-1"] = {}
    trainer._results["run-1"] = [{}]
    trainer._incoming[("run-2", 0)] = {}
    trainer._round_weights[("run-3", 0)] = [np.ones(2)]
    cache_drops = []
    memory_values = iter((500.0, 620.0))
    rss_values = iter((200.0, 120.0))
    monkeypatch.setattr(trainer, "_drop_os_page_cache", lambda: cache_drops.append(True))
    monkeypatch.setattr(training_service, "_memory_available_mb", lambda: next(memory_values))
    monkeypatch.setattr(training_service, "_resident_memory_mb", lambda: next(rss_values))
    monkeypatch.setattr(training_service, "_trim_process_heap", lambda: None)

    result = trainer.cleanup_memory(threading.Lock())

    assert result == {
        "node_id": "worker1",
        "cleaned_runs": 3,
        "memory_available_before_mb": 500.0,
        "memory_available_after_mb": 620.0,
        "process_rss_before_mb": 200.0,
        "process_rss_after_mb": 120.0,
        "process_memory_cleaned": True,
        "os_page_cache_dropped": True,
        "recycle_scheduled": False,
    }
    assert cache_drops == [True]
    assert trainer._run_architectures == {}
    assert trainer._results == {}
    assert trainer._incoming == {}
    assert trainer._round_weights == {}
    assert trainer.images.shape == (4, 32, 32, 3)


def test_cleanup_with_recycle_restarts_only_a_bloated_idle_service(tmp_path, monkeypatch):
    shard_path = tmp_path / "base.npz"
    np.savez(shard_path, images=np.zeros((4, 32, 32, 3), dtype=np.uint8), labels=np.array([0, 1, 0, 1]))
    config = WorkerConfig(
        node_id="worker1",
        api_token="a" * 48,
        port=5004,
        data_path=str(shard_path),
        peers=(),
        log_path=str(tmp_path / "worker.jsonl"),
    )
    trainer = WorkerTrainer(config)
    timers = []

    class FakeTimer:
        def __init__(self, delay, function):
            timers.append((delay, function))
            self.daemon = False

        def start(self):
            pass

    monkeypatch.setattr(trainer, "_drop_os_page_cache", lambda: None)
    monkeypatch.setattr(training_service, "_memory_available_mb", lambda: 3000.0)
    monkeypatch.setattr(training_service, "_trim_process_heap", lambda: None)
    monkeypatch.setattr(training_service.threading, "Timer", FakeTimer)

    # A lean service is never recycled, even when asked.
    monkeypatch.setattr(training_service, "_resident_memory_mb", lambda: 900.0)
    assert trainer.cleanup_memory(threading.Lock(), recycle=True)["recycle_scheduled"] is False
    assert not timers and trainer._reboot_requested is False

    # A bloated one is recycled only when the caller asks for it.
    monkeypatch.setattr(training_service, "_resident_memory_mb", lambda: 3900.0)
    assert trainer.cleanup_memory(threading.Lock())["recycle_scheduled"] is False
    assert not timers
    assert trainer.cleanup_memory(threading.Lock(), recycle=True)["recycle_scheduled"] is True
    assert timers and timers[0][1] == trainer._exit_for_recycle
    # New work is refused until the process is replaced.
    assert trainer._reboot_requested is True


def test_cleanup_memory_clears_loaded_tensorflow_session_without_tracked_models(tmp_path, monkeypatch):
    shard_path = tmp_path / "base.npz"
    np.savez(
        shard_path,
        images=np.zeros((2, 32, 32, 3), dtype=np.uint8),
        labels=np.array([0, 1]),
    )
    trainer = WorkerTrainer(
        WorkerConfig(
            node_id="worker1",
            api_token="a" * 48,
            port=5004,
            data_path=str(shard_path),
            peers=(),
            log_path=str(tmp_path / "worker.jsonl"),
        )
    )
    session_cleared = []
    tensorflow = SimpleNamespace(
        keras=SimpleNamespace(
            backend=SimpleNamespace(clear_session=lambda: session_cleared.append(True)),
        ),
    )
    monkeypatch.setitem(sys.modules, "tensorflow", tensorflow)
    monkeypatch.setattr(trainer, "_drop_os_page_cache", lambda: None)
    monkeypatch.setattr(training_service, "_memory_available_mb", lambda: 500.0)
    monkeypatch.setattr(training_service, "_resident_memory_mb", lambda: 100.0)
    monkeypatch.setattr(training_service, "_trim_process_heap", lambda: None)

    trainer.cleanup_memory(threading.Lock())

    assert session_cleared == [True]


def test_cleanup_memory_refuses_training_and_active_nas(tmp_path, monkeypatch):
    shard_path = tmp_path / "base.npz"
    np.savez(
        shard_path,
        images=np.zeros((2, 32, 32, 3), dtype=np.uint8),
        labels=np.array([0, 1]),
    )
    trainer = WorkerTrainer(
        WorkerConfig(
            node_id="worker1",
            api_token="a" * 48,
            port=5004,
            data_path=str(shard_path),
            peers=(),
            log_path=str(tmp_path / "worker.jsonl"),
        )
    )
    cache_drops = []
    monkeypatch.setattr(trainer, "_drop_os_page_cache", lambda: cache_drops.append(True))
    trainer._active_run = "run-1"
    with pytest.raises(RuntimeError, match="training is active"):
        trainer.cleanup_memory(threading.Lock())
    trainer._active_run = None

    locked_nas = threading.Lock()
    locked_nas.acquire()
    try:
        with pytest.raises(RuntimeError, match="NAS search is active"):
            trainer.cleanup_memory(locked_nas)
    finally:
        locked_nas.release()
    assert cache_drops == []


def test_cleanup_memory_releases_models_when_page_cache_helper_fails(tmp_path, monkeypatch):
    shard_path = tmp_path / "base.npz"
    np.savez(
        shard_path,
        images=np.zeros((2, 32, 32, 3), dtype=np.uint8),
        labels=np.array([0, 1]),
    )
    trainer = WorkerTrainer(
        WorkerConfig(
            node_id="worker1",
            api_token="a" * 48,
            port=5004,
            data_path=str(shard_path),
            peers=(),
            log_path=str(tmp_path / "worker.jsonl"),
        )
    )
    trainer._results["run-1"] = [{}]

    def fail_cache_drop():
        raise RuntimeError("cache helper is unavailable")

    memory_values = iter((500.0, 600.0))
    rss_values = iter((200.0, 100.0))
    monkeypatch.setattr(trainer, "_drop_os_page_cache", fail_cache_drop)
    monkeypatch.setattr(training_service, "_memory_available_mb", lambda: next(memory_values))
    monkeypatch.setattr(training_service, "_resident_memory_mb", lambda: next(rss_values))
    monkeypatch.setattr(training_service, "_trim_process_heap", lambda: None)

    result = trainer.cleanup_memory(threading.Lock())

    assert result["process_memory_cleaned"] is True
    assert result["os_page_cache_dropped"] is False
    assert result["error"] == "cache helper is unavailable"
    assert result["cleaned_runs"] == 1
    assert trainer._results == {}


def test_worker_memory_cleanup_endpoint_returns_measured_result(tmp_path, monkeypatch):
    shard_path = tmp_path / "base.npz"
    np.savez(
        shard_path,
        images=np.zeros((2, 32, 32, 3), dtype=np.uint8),
        labels=np.array([0, 1]),
    )
    config = WorkerConfig(
        node_id="worker1",
        api_token="a" * 48,
        port=5004,
        data_path=str(shard_path),
        peers=(),
        log_path=str(tmp_path / "worker.jsonl"),
    )
    trainer = WorkerTrainer(config)
    trainer._results["run-1"] = [{}]
    monkeypatch.setattr(trainer, "_drop_os_page_cache", lambda: None)
    monkeypatch.setattr(training_service, "_memory_available_mb", lambda: 500.0)
    monkeypatch.setattr(training_service, "_resident_memory_mb", lambda: 200.0)
    monkeypatch.setattr(training_service, "_trim_process_heap", lambda: None)
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(config, trainer))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        request = Request(
            f"http://127.0.0.1:{server.server_port}/v1/memory/cleanup",
            data=b"",
            headers={"Authorization": f"Bearer {config.api_token}"},
            method="POST",
        )
        with urlopen(request, timeout=3) as response:
            result = json.loads(response.read().decode("utf-8"))
        assert result["node_id"] == "worker1"
        assert result["cleaned_runs"] == 1
        assert result["process_memory_cleaned"] is True
        assert result["os_page_cache_dropped"] is True
        assert trainer._results == {}
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_process_snapshot_endpoint_requires_worker_auth_and_reports_status(tmp_path, monkeypatch):
    shard_path = tmp_path / "base.npz"
    np.savez(
        shard_path,
        images=np.zeros((2, 32, 32, 3), dtype=np.uint8),
        labels=np.array([0, 1]),
    )
    config = WorkerConfig(
        node_id="worker1",
        api_token="a" * 48,
        port=5004,
        data_path=str(shard_path),
        peers=(),
        log_path=str(tmp_path / "worker.jsonl"),
    )
    trainer = WorkerTrainer(config)
    monkeypatch.setattr(
        training_service,
        "_process_snapshot",
        lambda: {
            "sampled_at": "2026-11-09T00:00:00+00:00",
            "processes": [{"pid": 42, "name": "python", "rss_mb": 100.0}],
            "memory_available_mb": 500.0,
        },
    )
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(config, trainer))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        request = Request(f"http://127.0.0.1:{server.server_port}/v1/processes")
        with pytest.raises(HTTPError) as unauthorized:
            urlopen(request, timeout=3)
        assert unauthorized.value.code == 401

        request = Request(
            f"http://127.0.0.1:{server.server_port}/v1/processes",
            headers={"Authorization": f"Bearer {config.api_token}"},
        )
        with urlopen(request, timeout=3) as response:
            result = json.loads(response.read().decode("utf-8"))
        assert result["node_id"] == "worker1"
        assert result["memory_available_mb"] == 500.0
        assert result["processes"][0]["pid"] == 42
        assert result["training"]["active_run_id"] is None
        assert result["nas_active"] is False
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_reboot_request_is_rejected_while_training_is_active(tmp_path, monkeypatch):
    shard_path = tmp_path / "base.npz"
    np.savez(
        shard_path,
        images=np.zeros((2, 32, 32, 3), dtype=np.uint8),
        labels=np.array([0, 1]),
    )
    config = WorkerConfig(
        node_id="worker1",
        api_token="a" * 48,
        port=5004,
        data_path=str(shard_path),
        peers=(),
        log_path=str(tmp_path / "worker.jsonl"),
    )
    trainer = WorkerTrainer(config)
    trainer._scheduled_run = "run-1"
    requests = []
    monkeypatch.setattr(
        training_service,
        "urlopen",
        lambda *args, **kwargs: requests.append(args),
    )
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(config, trainer))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        request = Request(
            f"http://127.0.0.1:{server.server_port}/v1/system/reboot",
            data=b"",
            headers={"Authorization": f"Bearer {config.api_token}"},
            method="POST",
        )
        with pytest.raises(HTTPError) as conflict:
            urlopen(request, timeout=3)
        assert conflict.value.code == 409
        assert requests == []
        assert trainer._reboot_requested is False
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_idle_worker_reboot_is_accepted_and_blocks_new_training(tmp_path, monkeypatch):
    shard_path = tmp_path / "base.npz"
    np.savez(
        shard_path,
        images=np.zeros((2, 32, 32, 3), dtype=np.uint8),
        labels=np.array([0, 1]),
    )
    config = WorkerConfig(
        node_id="worker1",
        api_token="a" * 48,
        port=5004,
        data_path=str(shard_path),
        peers=(),
        log_path=str(tmp_path / "worker.jsonl"),
    )
    trainer = WorkerTrainer(config)
    requested = []

    class HelperResponse(io.BytesIO):
        status = 202

    def request_helper(request, *, timeout):
        requested.append((request.full_url, timeout))
        return HelperResponse(b'{"status":"accepted","action":"reboot"}')

    monkeypatch.setattr(training_service, "urlopen", request_helper)
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(config, trainer))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        request = Request(
            f"http://127.0.0.1:{server.server_port}/v1/system/reboot",
            data=b"",
            headers={"Authorization": f"Bearer {config.api_token}"},
            method="POST",
        )
        with urlopen(request, timeout=3) as response:
            result = json.loads(response.read().decode("utf-8"))
            assert response.status == 202
        assert result == {"status": "accepted", "action": "reboot"}
        assert requested == [("http://127.0.0.1:5005/reboot", 5)]
        assert trainer._reboot_requested is True
        with pytest.raises(RuntimeError, match="preparing to reboot"):
            trainer.start_run("new-run", seed=1, topology="ring")
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_initial_model_bundle_loads_and_predictions_match_reference(tmp_path):
    import tensorflow as tf

    dataset_name = "user-images"
    dataset_root = tmp_path / "datasets"
    dataset_root.mkdir()
    (dataset_root / dataset_name).mkdir()
    shard_path = tmp_path / "default_shard.npz"
    np.savez(
        shard_path,
        images=np.zeros((1, 32, 32, 3), dtype=np.uint8),
        labels=np.array([0], dtype=np.int64),
    )
    config = WorkerConfig(
        node_id="worker1",
        api_token="a" * 48,
        port=5004,
        data_path=str(shard_path),
        dataset_root=str(dataset_root),
        peers=(),
        log_path=str(tmp_path / "worker.jsonl"),
    )
    bundle_bytes = _make_initial_model_bundle(dataset_name)
    install_initial_model(config, dataset_name, bundle_bytes)
    images = np.linspace(0.0, 1.0, 128, dtype=np.float32).reshape(2, 8, 8, 1)
    reference_model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(8, 8, 1), dtype=tf.float32),
            tf.keras.layers.Conv2D(1, (3, 3), padding="same", activation="relu"),
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(2, activation="softmax"),
        ]
    )
    reference_model(np.zeros((1, 8, 8, 1), dtype=np.float32), training=False)
    with zipfile.ZipFile(io.BytesIO(bundle_bytes), "r") as zipped:
        json.loads(zipped.read("architecture.json").decode("utf-8"))
        with np.load(io.BytesIO(zipped.read("initial_weights.npz")), allow_pickle=False) as archive:
            reference_model.set_weights([
                np.asarray(archive["conv1_kernel"]),
                np.asarray(archive["conv1_bias"]),
                np.asarray(archive["classifier_kernel"]),
                np.asarray(archive["classifier_bias"]),
            ])
    loaded_model, _ = _load_installed_dataset_initial_model(
        config,
        dataset_name,
        input_shape=(8, 8, 1),
        class_names=["cat", "dog"],
    )
    np.testing.assert_allclose(
        loaded_model.predict(images, verbose=0),
        reference_model.predict(images, verbose=0),
        atol=1e-6,
    )


def test_run_round_uses_bundle_and_records_training_audit(tmp_path, monkeypatch):
    import tensorflow as tf
    import worker.nas_search as nas_search

    dataset_name = "mnist-run-round"
    dataset_root = tmp_path / "datasets"
    dataset_path = dataset_root / dataset_name
    dataset_path.mkdir(parents=True)
    class_names = tuple(str(label) for label in range(10))
    labels = np.tile(np.arange(10, dtype=np.int64), 2)
    images = np.linspace(0.0, 1.0, 20 * 8 * 8, dtype=np.float32).reshape(20, 8, 8, 1)
    np.savez_compressed(
        dataset_path / "train.npz",
        images=images,
        labels=labels,
        class_names=np.asarray(class_names),
    )
    shard_path = tmp_path / "default_shard.npz"
    np.savez(
        shard_path,
        images=np.zeros((1, 32, 32, 3), dtype=np.uint8),
        labels=np.array([0], dtype=np.int64),
    )
    config = WorkerConfig(
        node_id="worker1",
        api_token="a" * 48,
        port=5004,
        data_path=str(shard_path),
        dataset_root=str(dataset_root),
        peers=(),
        log_path=str(tmp_path / "worker.jsonl"),
    )
    bundle_bytes = _make_initial_model_bundle(
        dataset_name,
        input_shape=(8, 8, 1),
        class_names=class_names,
    )
    installed = install_initial_model(config, dataset_name, bundle_bytes)
    trainer = WorkerTrainer(config)
    trainer.configure(
        rounds=1,
        epochs=1,
        topology="ring",
        dataset_name=dataset_name,
    )
    _, validation_indices = nas_search._split_indices(labels, seed=7)
    validation_images = images[validation_indices]

    reference_model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(8, 8, 1), dtype=tf.float32),
            tf.keras.layers.Conv2D(1, (3, 3), padding="same", activation="relu"),
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(10, activation="softmax"),
        ]
    )
    reference_model(np.zeros((1, 8, 8, 1), dtype=np.float32), training=False)
    with zipfile.ZipFile(io.BytesIO(bundle_bytes), "r") as zipped:
        with np.load(io.BytesIO(zipped.read("initial_weights.npz")), allow_pickle=False) as archive:
            reference_model.set_weights(
                [
                    np.asarray(archive["conv1_kernel"]),
                    np.asarray(archive["conv1_bias"]),
                    np.asarray(archive["classifier_kernel"]),
                    np.asarray(archive["classifier_bias"]),
                ]
            )
    expected_predictions = reference_model.predict(validation_images, verbose=0)
    observed: dict[str, object] = {}
    original_create = trainer._create_run_model

    def capture_run_model(run_id, seed):
        model = original_create(run_id, seed)
        observed["predictions_before_fit"] = model.predict(validation_images, verbose=0)
        observed["learning_rate"] = float(
            tf.keras.backend.get_value(model.optimizer.learning_rate)
        )
        observed["initial_model_loaded"] = trainer._run_model_audits[run_id][
            "initial_model_loaded"
        ]
        original_fit = model.fit

        def capture_fit(*args, **kwargs):
            observed["batch_size"] = kwargs.get("batch_size")
            callbacks = kwargs.get("callbacks", [])

            class ObserveProgress(tf.keras.callbacks.Callback):
                def on_epoch_begin(self, epoch, logs=None):
                    observed["progress"] = trainer.results("bundle-run")["progress"]

            kwargs["callbacks"] = [*callbacks, ObserveProgress()]
            return original_fit(*args, **kwargs)

        monkeypatch.setattr(model, "fit", capture_fit)
        return model

    def reject_fresh_builder(**kwargs):
        raise AssertionError("Installed-bundle run_round called the fresh dataset builder")

    monkeypatch.setattr(trainer, "_create_run_model", capture_run_model)
    monkeypatch.setattr(nas_search, "_build_dataset_candidate", reject_fresh_builder)
    monkeypatch.setattr(training_service, "topology_neighbours", lambda *args: ())
    monkeypatch.setattr(training_service, "topology_senders", lambda *args: ())
    monkeypatch.setattr(training_service, "_memory_available_kb", lambda: 1)
    monkeypatch.setattr(training_service, "_resident_memory_mb", lambda: 1.0)
    monkeypatch.setattr(training_service, "_peak_process_rss_mb", lambda: 1.0)
    trainer._incoming[("bundle-run", 0)] = {}

    result = trainer.run_round("bundle-run", 0, 1, 7, "ring")

    np.testing.assert_allclose(
        observed["predictions_before_fit"],
        expected_predictions,
        atol=1e-6,
    )
    assert observed["initial_model_loaded"] is True
    assert observed["learning_rate"] == pytest.approx(0.01)
    assert observed["batch_size"] == 32
    assert result["optimizer"] == "Adam"
    assert result["learning_rate"] == pytest.approx(0.01)
    assert result["batch_size"] == 32
    assert result["initial_model_loaded"] is True
    assert result["architecture_sha256"] == installed["architecture_sha256"]
    assert result["weights_sha256"] == installed["weights_sha256"]
    assert result["local_epochs"] == 1
    assert result["sample_count"] == 10
    assert observed["progress"] == {
        "run_id": "bundle-run",
        "round": 1,
        "epoch": 1,
        "epochs": 1,
        "phase": "local_training",
    }
    assert trainer.results("bundle-run")["progress"] is None


def test_custom_initial_model_is_built_and_persisted_without_embedded_weights(
    tmp_path,
    monkeypatch,
):
    class FakeModel:
        def count_params(self):
            return 170

        def get_weights(self):
            return [np.arange(170, dtype=np.float32)]

    dataset_root = tmp_path / "datasets"
    dataset_path = dataset_root / "mnist"
    dataset_path.mkdir(parents=True)
    np.savez_compressed(
        dataset_path / "train.npz",
        images=np.zeros((2, 28, 28, 1), dtype=np.float32),
        labels=np.array([0, 1], dtype=np.int64),
        class_names=np.asarray([str(index) for index in range(10)]),
    )
    config = WorkerConfig(
        node_id="worker1",
        api_token="a" * 48,
        port=5004,
        data_path=str(tmp_path / "unused.npz"),
        dataset_root=str(dataset_root),
        peers=(),
        log_path=str(tmp_path / "worker.jsonl"),
    )
    architecture = {
        "model_type": "custom",
        "dataset_name": "mnist",
        "input_shape": [28, 28, 1],
        "classes": [str(index) for index in range(10)],
        "num_classes": 10,
        "parameter_count": 170,
        "initialization_seed": 42,
        "layers": [
            {"type": "conv2d", "filters": 8, "kernel_size": 3},
            {"type": "global_average_pooling2d"},
            {"type": "dense", "units": 10, "activation": "linear"},
        ],
    }
    bundle = io.BytesIO()
    with zipfile.ZipFile(bundle, "w", compression=zipfile.ZIP_DEFLATED) as zipped:
        zipped.writestr("architecture.json", json.dumps(architecture).encode("utf-8"))
    monkeypatch.setattr(
        training_service,
        "build_dataset_model",
        lambda **_kwargs: FakeModel(),
    )

    metadata = install_initial_model(config, "mnist", bundle.getvalue())

    weights_path = dataset_path / "models" / "initial_model" / "initial_weights.npz"
    assert weights_path.is_file()
    assert metadata["parameter_count"] == 170
    with np.load(weights_path, allow_pickle=False) as arrays:
        assert arrays["weight_0"].shape == (170,)


def test_worker_rejects_present_but_broken_dataset_bundle_without_fallback(
    tmp_path,
    monkeypatch,
):
    dataset_name = "user-images"
    dataset_root = tmp_path / "datasets"
    dataset_root.mkdir()
    (dataset_root / dataset_name).mkdir()
    shard_path = tmp_path / "default_shard.npz"
    np.savez(
        shard_path,
        images=np.zeros((4, 32, 32, 3), dtype=np.uint8),
        labels=np.array([0, 1, 0, 1], dtype=np.int64),
    )
    config = WorkerConfig(
        node_id="worker1",
        api_token="a" * 48,
        port=5004,
        data_path=str(shard_path),
        dataset_root=str(dataset_root),
        peers=(),
        log_path=str(tmp_path / "worker.jsonl"),
    )
    bundle_bytes = _make_initial_model_bundle(dataset_name)
    install_initial_model(config, dataset_name, bundle_bytes)
    metadata_path = dataset_root / dataset_name / "models" / "initial_model" / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["architecture_sha256"] = "bad" * 16
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    trainer = WorkerTrainer(config)
    trainer.dataset_name = dataset_name
    trainer.images = np.zeros((4, 8, 8, 1), dtype=np.float32)
    trainer._dataset_class_names = ["cat", "dog"]
    monkeypatch.setattr(training_service, "topology_neighbours", lambda *args: ())
    monkeypatch.setattr(training_service, "topology_senders", lambda *args: ())
    with pytest.raises(ValueError, match="hash mismatch|malformed|incomplete"):
        trainer.run_round("broken-bundle-run", 0, 1, 7, "ring")


def test_worker_uses_legacy_builder_when_no_dataset_bundle_is_present(tmp_path, monkeypatch):
    original_builder = training_service.build_cifar10_model
    builder_calls = []

    def record_builder(seed, architecture):
        builder_calls.append((seed, architecture))
        return original_builder(seed, architecture)

    monkeypatch.setattr(training_service, "build_cifar10_model", record_builder)
    shard_path = tmp_path / "default_shard.npz"
    np.savez(
        shard_path,
        images=np.zeros((4, 32, 32, 3), dtype=np.uint8),
        labels=np.array([0, 1, 0, 1], dtype=np.int64),
    )
    config = WorkerConfig(
        node_id="worker1",
        api_token="a" * 48,
        port=5004,
        data_path=str(shard_path),
        dataset_root=str(tmp_path / "datasets"),
        peers=(),
        log_path=str(tmp_path / "worker.jsonl"),
    )
    trainer = WorkerTrainer(config)
    model = trainer._create_run_model("legacy-run", seed=9)
    assert trainer._run_architectures["legacy-run"] == trainer._architecture
    assert builder_calls == [(9, trainer._architecture)]
    assert model is not None
    assert model.output_shape == (None, 10)


def test_worker_uses_dataset_builder_when_named_bundle_is_absent(tmp_path, monkeypatch):
    import worker.nas_search as nas_search

    dataset_name = "user-images"
    dataset_root = tmp_path / "datasets"
    dataset_path = dataset_root / dataset_name
    dataset_path.mkdir(parents=True)
    labels = np.tile(np.arange(10, dtype=np.int64), 2)
    np.savez_compressed(
        dataset_path / "train.npz",
        images=np.zeros((20, 8, 8, 1), dtype=np.float32),
        labels=labels,
        class_names=np.array([str(label) for label in range(10)]),
    )
    shard_path = tmp_path / "default_shard.npz"
    np.savez(
        shard_path,
        images=np.zeros((4, 32, 32, 3), dtype=np.uint8),
        labels=np.array([0, 1, 0, 1], dtype=np.int64),
    )
    config = WorkerConfig(
        node_id="worker1",
        api_token="a" * 48,
        port=5004,
        data_path=str(shard_path),
        dataset_root=str(dataset_root),
        peers=(),
        log_path=str(tmp_path / "worker.jsonl"),
    )
    trainer = WorkerTrainer(config)
    trainer.configure(
        rounds=1,
        epochs=1,
        topology="ring",
        dataset_name=dataset_name,
    )
    builder_calls = []
    original_builder = nas_search._build_dataset_candidate

    def record_builder(**kwargs):
        builder_calls.append(kwargs)
        return original_builder(**kwargs)

    monkeypatch.setattr(nas_search, "_build_dataset_candidate", record_builder)
    monkeypatch.setattr(training_service, "topology_neighbours", lambda *args: ())
    monkeypatch.setattr(training_service, "topology_senders", lambda *args: ())
    monkeypatch.setattr(training_service, "_memory_available_kb", lambda: 1)
    monkeypatch.setattr(training_service, "_resident_memory_mb", lambda: 1.0)
    monkeypatch.setattr(training_service, "_peak_process_rss_mb", lambda: 1.0)
    trainer._incoming[("named-dataset-run", 0)] = {}

    result = trainer.run_round("named-dataset-run", 0, 1, 11, "ring")

    assert builder_calls == [
        {
            "input_shape": (8, 8, 1),
            "num_classes": 10,
            "architecture": trainer._architecture,
            "seed": 11,
        }
    ]
    assert result["initial_model_loaded"] is False
    assert result["learning_rate"] == pytest.approx(0.01)
    assert result["batch_size"] == 32
    assert result["architecture_sha256"] is None
    assert result["weights_sha256"] is None
    assert 0.0 <= result["validation_macro_precision"] <= 1.0
    assert 0.0 <= result["validation_macro_recall"] <= 1.0
    assert sum(map(sum, result["validation_confusion_matrix_counts"])) == (
        result["validation_sample_count"]
    )
    assert result["peer_wait_seconds"] >= 0.0


def test_worker_legacy_cifar_round_uses_configured_batch_size(tmp_path, monkeypatch):
    shard_path = tmp_path / "default_shard.npz"
    np.savez(
        shard_path,
        images=np.zeros((4, 32, 32, 3), dtype=np.uint8),
        labels=np.array([0, 1, 0, 1], dtype=np.int64),
    )
    config = WorkerConfig(
        node_id="worker1",
        api_token="a" * 48,
        port=5004,
        data_path=str(shard_path),
        dataset_root=str(tmp_path / "datasets"),
        peers=(),
        log_path=str(tmp_path / "worker.jsonl"),
    )
    trainer = WorkerTrainer(config)
    captured_fit = {}

    class Model:
        def fit(self, *args, **kwargs):
            captured_fit.update(kwargs)
            return type("FitHistory", (), {"history": {"loss": [0.5], "accuracy": [0.5]}})()

        def get_weights(self):
            return [np.zeros((1,), dtype=np.float32)]

        def set_weights(self, weights):
            pass

    monkeypatch.setattr(training_service, "build_cifar10_model", lambda *args: Model())

    def compile_legacy_model(model):
        model.optimizer = SimpleNamespace(learning_rate=0.001)

    monkeypatch.setattr(training_service, "compile_cifar10_model", compile_legacy_model)
    monkeypatch.setattr(training_service, "topology_neighbours", lambda *args: ())
    monkeypatch.setattr(training_service, "topology_senders", lambda *args: ())
    monkeypatch.setattr(training_service, "_memory_available_kb", lambda: 1)
    monkeypatch.setattr(training_service, "_resident_memory_mb", lambda: 1.0)
    monkeypatch.setattr(training_service, "_peak_process_rss_mb", lambda: 1.0)
    monkeypatch.setattr(training_service, "_trim_process_heap", lambda: None)
    monkeypatch.setitem(
        sys.modules,
        "tensorflow",
        SimpleNamespace(
            keras=SimpleNamespace(
                backend=SimpleNamespace(get_value=lambda value: value),
                callbacks=SimpleNamespace(Callback=object),
            )
        ),
    )
    trainer._incoming[("legacy-cifar-run", 0)] = {}

    trainer.run_round("legacy-cifar-run", 0, 1, 7, "ring")

    assert captured_fit["batch_size"] == 32


def test_worker_defaults_to_dataset_lr_0_01_and_batch_32(tmp_path, monkeypatch):
    import tensorflow as tf

    shard_path = tmp_path / "default_shard.npz"
    np.savez(
        shard_path,
        images=np.zeros((4, 32, 32, 3), dtype=np.uint8),
        labels=np.array([0, 1, 0, 1], dtype=np.int64),
    )
    dataset_root = tmp_path / "datasets"
    dataset_path = dataset_root / "mnist-demo"
    dataset_path.mkdir(parents=True)
    np.savez_compressed(
        dataset_path / "train.npz",
        images=np.zeros((20, 8, 8, 1), dtype=np.float32),
        labels=np.tile(np.arange(10, dtype=np.int64), 2),
        class_names=np.array([str(label) for label in range(10)]),
    )
    config = WorkerConfig(
        node_id="worker1",
        api_token="a" * 48,
        port=5004,
        data_path=str(shard_path),
        dataset_root=str(dataset_root),
        peers=(),
        log_path=str(tmp_path / "worker.jsonl"),
    )
    trainer = WorkerTrainer(config)
    trainer.configure(
        rounds=1,
        epochs=1,
        topology="ring",
        dataset_name="mnist-demo",
    )
    assert config.dataset_learning_rate == 0.01
    assert config.dataset_batch_size == 32
    assert trainer.status()["training_config"]["dataset_learning_rate"] == 0.01
    assert trainer.status()["training_config"]["dataset_batch_size"] == 32

    model = trainer._create_run_model("defaults-run", seed=7)
    assert float(tf.keras.backend.get_value(model.optimizer.learning_rate)) == pytest.approx(0.01)
    fit_arguments = {}

    def capture_fit(*args, **kwargs):
        fit_arguments.update(kwargs)
        return type("FitHistory", (), {"history": {"loss": [0.5], "accuracy": [0.5]}})()

    monkeypatch.setattr(model, "fit", capture_fit)
    monkeypatch.setattr(training_service, "topology_neighbours", lambda *args: ())
    monkeypatch.setattr(training_service, "topology_senders", lambda *args: ())
    monkeypatch.setattr(training_service, "_memory_available_kb", lambda: 1)
    monkeypatch.setattr(training_service, "_resident_memory_mb", lambda: 1.0)
    monkeypatch.setattr(training_service, "_peak_process_rss_mb", lambda: 1.0)
    trainer._incoming[("defaults-run", 0)] = {}
    trainer.run_round("defaults-run", 0, 1, 7, "ring")
    assert fit_arguments["batch_size"] == 32

    custom_architecture = {
        "model_type": "custom",
        "layers": [
            {"type": "flatten"},
            {"type": "dense", "units": 10, "activation": "linear"},
        ],
        "training": {
            "optimizer": "adamw",
            "learning_rate": 0.0005,
            "batch_size": 64,
        },
    }
    trainer.configure(
        rounds=1,
        epochs=1,
        topology="ring",
        dataset_name="mnist-demo",
        architecture=custom_architecture,
    )
    custom_model = trainer._create_run_model("custom-run", seed=7)
    assert float(tf.keras.backend.get_value(custom_model.optimizer.learning_rate)) == pytest.approx(0.0005)
    custom_fit_arguments = {}

    def capture_custom_fit(*args, **kwargs):
        custom_fit_arguments.update(kwargs)
        return type("FitHistory", (), {"history": {"loss": [0.5], "accuracy": [0.5]}})()

    monkeypatch.setattr(custom_model, "fit", capture_custom_fit)
    trainer._incoming[("custom-run", 0)] = {}
    trainer.run_round("custom-run", 0, 1, 7, "ring")
    assert custom_fit_arguments["batch_size"] == 64
    assert trainer._results["custom-run"][0]["optimizer"] == "AdamW"


def test_peer_weight_endpoint_requires_peer_token_and_stores_payload(tmp_path):
    shard_path = tmp_path / "worker1.npz"
    np.savez(
        shard_path,
        images=np.zeros((3, 32, 32, 3), dtype=np.uint8),
        labels=np.array([0, 1, 2]),
    )
    token = "a" * 48
    peer_token = "b" * 48
    config = WorkerConfig(
        node_id="worker1",
        api_token=token,
        port=5004,
        data_path=str(shard_path),
        dataset_root=str(tmp_path / "datasets"),
        peers=(
            Peer("worker2", "http://127.0.0.1:5005", peer_token),
            Peer("worker3", "http://127.0.0.1:5006", "c" * 48),
        ),
        log_path=str(tmp_path / "worker.jsonl"),
    )
    trainer = WorkerTrainer(config)
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(config, trainer))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        payload = serialize_weights(
            [
                np.ones((3, 3, 3, 8), dtype=np.float32),
                np.ones((8,), dtype=np.float32),
                np.ones((512, 10), dtype=np.float32),
                np.ones((10,), dtype=np.float32),
            ]
        )
        headers = {
            "Authorization": f"Bearer {peer_token}",
            "Content-Type": "application/octet-stream",
            "X-Run-ID": "test-run",
            "X-Round": "0",
            "X-Sender-Node": "worker2",
            "X-Sample-Count": "3",
        }
        request = Request(
            f"http://127.0.0.1:{server.server_port}/v1/weights",
            data=payload,
            headers=headers,
            method="POST",
        )
        with urlopen(request, timeout=3) as response:
            assert response.status == 202
        assert trainer._incoming[("test-run", 0)]["worker2"][1] == 3

        bad_request = Request(
            f"http://127.0.0.1:{server.server_port}/v1/weights",
            data=payload,
            headers={**headers, "Authorization": "Bearer " + "x" * 48},
            method="POST",
        )
        with pytest.raises(HTTPError) as error:
            urlopen(bad_request, timeout=3)
        assert error.value.code == 401

        cleanup_request = Request(
            f"http://127.0.0.1:{server.server_port}/v1/runs/test-run",
            headers={"Authorization": "Bearer " + token},
            method="DELETE",
        )
        with urlopen(cleanup_request, timeout=3) as response:
            cleanup_result = json.loads(response.read())
            assert response.status == 200
        assert cleanup_result == {"run_id": "test-run", "cleaned": True}
        assert trainer._incoming == {}
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)


def test_authenticated_dataset_shard_endpoint_stores_and_reports_real_counts(tmp_path):
    from io import BytesIO

    shard_path = tmp_path / "default_shard.npz"
    np.savez(
        shard_path,
        images=np.zeros((3, 32, 32, 3), dtype=np.uint8),
        labels=np.array([0, 1, 2]),
    )
    token = "d" * 48
    config = WorkerConfig(
        node_id="worker1",
        api_token=token,
        port=5004,
        data_path=str(shard_path),
        dataset_root=str(tmp_path / "datasets"),
        peers=(
            Peer("worker2", "http://127.0.0.1:5005", "e" * 48),
            Peer("worker3", "http://127.0.0.1:5006", "f" * 48),
        ),
        log_path=str(tmp_path / "worker.jsonl"),
    )
    trainer = WorkerTrainer(config)
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(config, trainer))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        output = BytesIO()
        images = np.zeros((4, 28, 28, 1), dtype=np.float32)
        images[2:] = 1.0
        np.savez_compressed(
            output,
            images=images,
            labels=np.array([0, 0, 1, 1], dtype=np.int64),
            class_names=np.array(["cat", "dog"]),
        )
        payload = output.getvalue()
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/octet-stream",
        }
        request = Request(
            f"http://127.0.0.1:{server.server_port}/v1/datasets/user-images/shard",
            data=payload,
            headers=headers,
            method="POST",
        )
        with urlopen(request, timeout=3) as response:
            result = json.loads(response.read())
            assert response.status == 200
        assert result["node_id"] == "worker1"
        assert result["rows"] == 4
        assert result["shape"] == [4, 28, 28, 1]
        assert result["image_size"] == [28, 28]
        assert result["channels"] == 1
        assert result["class_counts"] == {"cat": 2, "dog": 2}
        assert (
            tmp_path / "datasets" / "user-images" / "train.npz"
        ).read_bytes() == payload

        get_request = Request(
            f"http://127.0.0.1:{server.server_port}/v1/datasets/user-images",
            headers=headers,
        )
        with urlopen(get_request, timeout=3) as response:
            assert json.loads(response.read())["rows"] == 4

        architecture = {
            "model_type": "small_cnn",
            "dataset_name": "user-images",
            "input_shape": [28, 28, 1],
            "classes": ["cat", "dog"],
            "num_classes": 2,
            "parameter_count": 14,
            "weight_shapes": {
                "conv_kernel": [3, 3, 1, 1],
                "conv_bias": [1],
                "classifier_kernel": [1, 2],
                "classifier_bias": [2],
            },
        }
        weights_buffer = BytesIO()
        np.savez_compressed(
            weights_buffer,
            conv_kernel=np.zeros((3, 3, 1, 1), dtype=np.float32),
            conv_bias=np.zeros((1,), dtype=np.float32),
            classifier_kernel=np.zeros((1, 2), dtype=np.float32),
            classifier_bias=np.zeros((2,), dtype=np.float32),
        )
        architecture_bytes = json.dumps(architecture).encode("utf-8")
        model_bundle = BytesIO()
        with zipfile.ZipFile(model_bundle, "w", compression=zipfile.ZIP_DEFLATED) as zipped:
            zipped.writestr("architecture.json", architecture_bytes)
            zipped.writestr("initial_weights.npz", weights_buffer.getvalue())
        model_payload = model_bundle.getvalue()
        model_request = Request(
            f"http://127.0.0.1:{server.server_port}/v1/datasets/user-images/initial-model",
            data=model_payload,
            headers=headers,
            method="POST",
        )
        with urlopen(model_request, timeout=3) as response:
            installed_model = json.loads(response.read())
            assert response.status == 200
        assert installed_model["parameter_count"] == 14
        assert installed_model["weights_sha256"] == hashlib.sha256(
            weights_buffer.getvalue()
        ).hexdigest()

        model_get = Request(
            f"http://127.0.0.1:{server.server_port}/v1/datasets/user-images/initial-model",
            headers=headers,
        )
        with urlopen(model_get, timeout=3) as response:
            assert json.loads(response.read()) == installed_model

        duplicate_model = Request(
            f"http://127.0.0.1:{server.server_port}/v1/datasets/user-images/initial-model",
            data=model_payload,
            headers=headers,
            method="POST",
        )
        with urlopen(duplicate_model, timeout=3) as response:
            assert json.loads(response.read()) == installed_model

        delete_model = Request(
            f"http://127.0.0.1:{server.server_port}/v1/datasets/user-images/initial-model",
            headers=headers,
            method="DELETE",
        )
        with urlopen(delete_model, timeout=3) as response:
            assert response.status == 204
        model_directory = tmp_path / "datasets" / "user-images" / "models" / "initial_model"
        assert not model_directory.exists()

        with urlopen(model_request, timeout=3) as response:
            assert response.status == 200

        with pytest.raises(ValueError, match="valid ZIP archive"):
            install_initial_model(
                config,
                "user-images",
                b"not a model bundle",
            )

        duplicate = Request(
            f"http://127.0.0.1:{server.server_port}/v1/datasets/user-images/shard",
            data=payload,
            headers=headers,
            method="POST",
        )
        with urlopen(duplicate, timeout=3) as response:
            assert response.status == 200
            assert json.loads(response.read())["rows"] == 4

        with pytest.raises(ValueError, match="Invalid dataset name"):
            install_dataset_shard(config, "../unsafe", payload)
        with pytest.raises(ValueError, match="valid NPZ archive"):
            install_dataset_shard(config, "invalid-npz", b"not an NPZ")

        altered = BytesIO()
        changed_images = images.copy()
        changed_images[0] = 0.5
        np.savez_compressed(
            altered,
            images=changed_images,
            labels=np.array([0, 0, 1, 1], dtype=np.int64),
            class_names=np.array(["cat", "dog"]),
        )
        changed_request = Request(
            f"http://127.0.0.1:{server.server_port}/v1/datasets/user-images/shard",
            data=altered.getvalue(),
            headers=headers,
            method="POST",
        )
        with pytest.raises(HTTPError) as changed_error:
            urlopen(changed_request, timeout=3)
        assert changed_error.value.code == 409
        marker = model_directory / "preserve-on-dataset-replacement.txt"
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text("keep initial-model files", encoding="utf-8")
        replacement_payload = altered.getvalue()
        replace_request = Request(
            f"http://127.0.0.1:{server.server_port}/v1/datasets/user-images/shard",
            data=replacement_payload,
            headers=headers,
            method="PUT",
        )
        with urlopen(replace_request, timeout=3) as response:
            replacement = json.loads(response.read())
            assert response.status == 200
        assert replacement["shard_sha256"] == hashlib.sha256(replacement_payload).hexdigest()
        assert (
            tmp_path / "datasets" / "user-images" / "train.npz"
        ).read_bytes() == replacement_payload
        assert marker.read_text(encoding="utf-8") == "keep initial-model files"

        unauthorized = Request(
            f"http://127.0.0.1:{server.server_port}/v1/datasets/user-images/shard",
            data=payload,
            headers={"Authorization": "Bearer " + "x" * 48},
            method="POST",
        )
        with pytest.raises(HTTPError) as unauthorized_error:
            urlopen(unauthorized, timeout=3)
        assert unauthorized_error.value.code == 401

        trainer._active_run = "busy"
        busy = Request(
            f"http://127.0.0.1:{server.server_port}/v1/datasets/other-dataset/shard",
            data=payload,
            headers=headers,
            method="PUT",
        )
        with pytest.raises(HTTPError) as busy_error:
            urlopen(busy, timeout=3)
        assert busy_error.value.code == 409

        busy_delete = Request(
            f"http://127.0.0.1:{server.server_port}/v1/datasets/user-images/initial-model",
            headers=headers,
            method="DELETE",
        )
        with pytest.raises(HTTPError) as busy_delete_error:
            urlopen(busy_delete, timeout=3)
        assert busy_delete_error.value.code == 409
        assert model_directory.is_dir()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)


def test_initial_model_endpoint_returns_service_unavailable_for_import_errors(
    tmp_path,
    monkeypatch,
):
    shard_path = tmp_path / "worker-train.npz"
    np.savez(
        shard_path,
        images=np.zeros((2, 32, 32, 3), dtype=np.uint8),
        labels=np.array([0, 1]),
    )
    config = WorkerConfig(
        node_id="worker1",
        api_token="q" * 48,
        port=5004,
        data_path=str(shard_path),
        dataset_root=str(tmp_path / "datasets"),
        peers=(
            Peer("worker2", "http://127.0.0.1:5005", "r" * 48),
            Peer("worker3", "http://127.0.0.1:5006", "s" * 48),
        ),
        log_path=str(tmp_path / "worker.jsonl"),
    )
    trainer = WorkerTrainer(config)

    def fail_install(*_args, **_kwargs):
        raise ImportError("Keras cannot be imported")

    monkeypatch.setattr(training_service, "install_initial_model", fail_install)
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(config, trainer))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        request = Request(
            f"http://127.0.0.1:{server.server_port}/v1/datasets/mnist/initial-model",
            data=b"model",
            headers={"Authorization": "Bearer " + config.api_token},
            method="POST",
        )
        with pytest.raises(HTTPError) as error:
            urlopen(request, timeout=3)
        assert error.value.code == 503
        assert json.loads(error.value.read()) == {
            "error": "Model-building dependency unavailable: Keras cannot be imported"
        }

        with urlopen(
            Request(
                f"http://127.0.0.1:{server.server_port}/healthz",
                headers={"Authorization": "Bearer " + config.api_token},
            ),
            timeout=3,
        ) as response:
            assert response.status == 200
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)


def test_nas_search_uses_the_named_dataset_shard(tmp_path, monkeypatch):
    root = tmp_path / "datasets"
    root.mkdir()
    payload_buffer = io.BytesIO()
    np.savez_compressed(
        payload_buffer,
        images=np.zeros((4, 8, 6, 3), dtype=np.float32),
        labels=np.array([0, 0, 1, 1], dtype=np.int64),
        class_names=np.array(["cat", "dog"]),
    )
    shard_payload = payload_buffer.getvalue()
    token = "n" * 48
    config = WorkerConfig(
        node_id="worker1",
        api_token=token,
        port=5004,
        data_path=str(tmp_path / "cifar-train.npz"),
        dataset_root=str(root),
        peers=(
            Peer("worker2", "http://127.0.0.1:5005", "o" * 48),
            Peer("worker3", "http://127.0.0.1:5006", "p" * 48),
        ),
        log_path=str(tmp_path / "worker.jsonl"),
    )
    np.savez(
        config.data_path,
        images=np.zeros((4, 32, 32, 3), dtype=np.uint8),
        labels=np.array([0, 1, 0, 1], dtype=np.int64),
    )
    install_dataset_shard(config, "cifar10", shard_payload)
    trainer = WorkerTrainer(config)
    calls = {}

    def fake_search(images, labels, class_names, **kwargs):
        calls.update(
            image_shape=images.shape,
            labels=labels.tolist(),
            class_names=class_names,
            kwargs=kwargs,
        )
        return {
            "node_id": kwargs["node_id"],
            "dataset_classes": class_names,
            "candidates": [
                {"architecture": architecture}
                for architecture in (kwargs.get("architectures") or [])
            ],
        }

    monkeypatch.setattr(
        training_service,
        "search_dataset_architectures",
        fake_search,
    )
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(config, trainer))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        request = Request(
            f"http://127.0.0.1:{server.server_port}/v1/nas/search",
            data=json.dumps(
                {
                    "dataset_name": "cifar10",
                    "seed": 12,
                    "time_limit_seconds": 30,
                }
            ).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urlopen(request, timeout=3) as response:
            result = json.loads(response.read())
        assert result["dataset_name"] == "cifar10"
        assert result["dataset_samples"] == 4
        assert calls["image_shape"] == (4, 8, 6, 3)
        assert calls["labels"] == [0, 0, 1, 1]
        assert calls["class_names"] == ["cat", "dog"]
        assert calls["kwargs"]["time_limit_seconds"] == 30
        assert calls["kwargs"]["reuse_tensorflow_runtime"] is True
        assert calls["kwargs"]["memory_constrained"] is True

        smoke_request = Request(
            f"http://127.0.0.1:{server.server_port}/v1/nas/search",
            data=json.dumps(
                {
                    "dataset_name": "cifar10",
                    "seed": 12,
                    "time_limit_seconds": 90,
                    "epochs": 3,
                    "smoke_test": True,
                }
            ).encode("utf-8"),
            headers={
                "Authorization": "Bearer " + token,
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urlopen(smoke_request, timeout=3) as response:
            smoke_result = json.loads(response.read())
        assert smoke_result["smoke_test"] is True
        assert smoke_result["search_method"] == "single_candidate_smoke_test"
        assert calls["kwargs"]["epochs"] == 1
        assert calls["kwargs"]["architectures"][0]["blocks"] == 2
        assert calls["kwargs"]["architectures"][0]["attention"] is False
        assert len(smoke_result["candidates"]) == 1

        missing_name = Request(
            f"http://127.0.0.1:{server.server_port}/v1/nas/search",
            data=json.dumps({"time_limit_seconds": 30}).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with pytest.raises(HTTPError) as error:
            urlopen(missing_name, timeout=3)
        assert error.value.code == 400
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)


def test_worker_config_reads_dataset_root_from_environment(monkeypatch, tmp_path):
    from worker.training_service import DEFAULT_DATASET_ROOT

    monkeypatch.setenv("DFL_NODE_ID", "worker1")
    monkeypatch.setenv("DFL_API_TOKEN", "a" * 48)
    monkeypatch.setenv(
        "DFL_PEERS_JSON_B64",
        "W3sibm9kZV9pZCI6IndvcmtlcjIiLCJiYXNlX3VybCI6Imh0dHA6Ly8xMjcuMC4wLjE6NTAwNSIsInRva2VuIjoiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmIifSx7Im5vZGVfaWQiOiJ3b3JrZXIzIiwiYmFzZV91cmwiOiJodHRwOi8vMTI3LjAuMC4xOjUwMDYiLCJ0b2tlbiI6ImNjY2NjY2NjY2NjY2NjY2NjY2NjY2NjY2NjY2NjY2NjY2NjY2NjY2MifV0=",
    )
    monkeypatch.delenv("DFL_TRAINING_DATASET_ROOT", raising=False)
    assert WorkerConfig.from_env().dataset_root == DEFAULT_DATASET_ROOT

    custom_root = str(tmp_path / "worker-datasets")
    monkeypatch.setenv("DFL_TRAINING_DATASET_ROOT", custom_root)
    assert WorkerConfig.from_env().dataset_root == custom_root


def test_worker_keeps_bounded_round_weights_and_one_split_per_run(tmp_path, monkeypatch):
    import worker.nas_search as nas_search

    dataset_name = "user-images"
    dataset_path = tmp_path / "datasets" / dataset_name
    dataset_path.mkdir(parents=True)
    np.savez_compressed(
        dataset_path / "train.npz",
        images=np.zeros((20, 8, 8, 1), dtype=np.float32),
        labels=np.tile(np.arange(10, dtype=np.int64), 2),
        class_names=np.array([str(label) for label in range(10)]),
    )
    shard_path = tmp_path / "default_shard.npz"
    np.savez(shard_path, images=np.zeros((4, 32, 32, 3), dtype=np.uint8), labels=np.array([0, 1, 0, 1]))
    trainer = WorkerTrainer(
        WorkerConfig(
            node_id="worker1",
            api_token="a" * 48,
            port=5004,
            data_path=str(shard_path),
            dataset_root=str(tmp_path / "datasets"),
            peers=(),
            log_path=str(tmp_path / "worker.jsonl"),
        )
    )
    trainer.configure(rounds=3, epochs=1, topology="ring", dataset_name=dataset_name)
    split_calls = []
    original_split = nas_search._split_indices

    def record_split(labels, seed):
        split_calls.append(seed)
        return original_split(labels, seed)

    monkeypatch.setattr(nas_search, "_split_indices", record_split)
    monkeypatch.setattr(training_service, "topology_neighbours", lambda *args: ())
    monkeypatch.setattr(training_service, "topology_senders", lambda *args: ())
    monkeypatch.setattr(training_service, "_memory_available_kb", lambda: 1)
    monkeypatch.setattr(training_service, "_resident_memory_mb", lambda: 1.0)
    monkeypatch.setattr(training_service, "_peak_process_rss_mb", lambda: 1.0)

    for round_index in range(3):
        trainer._incoming[("bounded-run", round_index)] = {}
        trainer.run_round("bounded-run", round_index, 1, 5, "ring")

    assert split_calls == [5]
    assert sorted(key[1] for key in trainer._round_weights if key[0] == "bounded-run") == [1, 2]
    with pytest.raises(KeyError):
        trainer.weights("bounded-run", 0)
    payload, _sample_count = trainer.weights("bounded-run", 2)
    assert payload

    trainer.cleanup_run("bounded-run")
    assert "bounded-run" not in trainer._run_splits
    assert not any(key[0] == "bounded-run" for key in trainer._round_weights)


def _npz_bytes(images, labels, class_names):
    buffer = io.BytesIO()
    np.savez_compressed(buffer, images=images, labels=labels, class_names=np.array(class_names))
    return buffer.getvalue()


def _dataset_worker_config(tmp_path, dataset_root):
    shard_path = tmp_path / "default_shard.npz"
    np.savez(
        shard_path,
        images=np.zeros((4, 32, 32, 3), dtype=np.uint8),
        labels=np.array([0, 1, 0, 1], dtype=np.int64),
    )
    return WorkerConfig(
        node_id="worker1",
        api_token="a" * 48,
        port=5004,
        data_path=str(shard_path),
        dataset_root=str(dataset_root),
        peers=(),
        log_path=str(tmp_path / "worker.jsonl"),
    )


def test_global_validation_install_requires_a_matching_shard(tmp_path):
    root = tmp_path / "datasets"
    root.mkdir()
    config = _dataset_worker_config(tmp_path, root)
    classes = [str(label) for label in range(3)]
    split = _npz_bytes(
        np.zeros((6, 8, 8, 1), dtype=np.float32), np.tile(np.arange(3), 2), classes
    )
    with pytest.raises(ValueError, match="training shard before"):
        install_global_validation(config, "digits", split)

    install_dataset_shard(
        config,
        "digits",
        _npz_bytes(np.zeros((9, 8, 8, 1), dtype=np.float32), np.tile(np.arange(3), 3), classes),
    )
    with pytest.raises(ValueError, match="classes do not match"):
        install_global_validation(
            config,
            "digits",
            _npz_bytes(np.zeros((6, 8, 8, 1), dtype=np.float32), np.tile(np.arange(3), 2), ["a", "b", "c"]),
        )
    with pytest.raises(ValueError, match="image shape"):
        install_global_validation(
            config,
            "digits",
            _npz_bytes(np.zeros((6, 4, 4, 1), dtype=np.float32), np.tile(np.arange(3), 2), classes),
        )

    result = install_global_validation(config, "digits", split)
    assert result["rows"] == 6
    assert result["class_counts"] == {"0": 2, "1": 2, "2": 2}
    assert (root / "digits" / training_service.GLOBAL_VALIDATION_FILE).read_bytes() == split


def test_dataset_round_reports_global_validation_metrics(tmp_path, monkeypatch):
    dataset_name = "user-images"
    root = tmp_path / "datasets"
    root.mkdir()
    config = _dataset_worker_config(tmp_path, root)
    classes = [str(label) for label in range(10)]
    rng = np.random.default_rng(0)
    # Non-IID local shard (classes 0-4 only); the shared split holds all ten classes.
    install_dataset_shard(
        config,
        dataset_name,
        _npz_bytes(
            rng.normal(size=(20, 8, 8, 1)).astype(np.float32),
            np.tile(np.arange(5, dtype=np.int64), 4),
            classes,
        ),
    )
    install_global_validation(
        config,
        dataset_name,
        _npz_bytes(
            rng.normal(size=(30, 8, 8, 1)).astype(np.float32),
            np.tile(np.arange(10, dtype=np.int64), 3),
            classes,
        ),
    )
    trainer = WorkerTrainer(config)
    trainer.configure(rounds=1, epochs=1, topology="ring", dataset_name=dataset_name)
    monkeypatch.setattr(training_service, "topology_neighbours", lambda *args: ())
    monkeypatch.setattr(training_service, "topology_senders", lambda *args: ())
    monkeypatch.setattr(training_service, "_memory_available_kb", lambda: 1)
    monkeypatch.setattr(training_service, "_resident_memory_mb", lambda: 1.0)
    monkeypatch.setattr(training_service, "_peak_process_rss_mb", lambda: 1.0)
    trainer._incoming[("global-run", 0)] = {}

    result = trainer.run_round("global-run", 0, 1, 11, "ring")

    assert result["global_validation_sample_count"] == 30
    assert 0.0 <= result["global_validation_accuracy"] <= 1.0
    assert 0.0 <= result["global_validation_macro_f1"] <= 1.0
    assert len(result["global_validation_per_class_recall"]) == 10


def test_round_without_global_validation_reports_no_global_metrics(tmp_path, monkeypatch):
    root = tmp_path / "datasets"
    root.mkdir()
    config = _dataset_worker_config(tmp_path, root)
    classes = [str(label) for label in range(10)]
    install_dataset_shard(
        config,
        "plain",
        _npz_bytes(np.zeros((20, 8, 8, 1), dtype=np.float32), np.tile(np.arange(10, dtype=np.int64), 2), classes),
    )
    trainer = WorkerTrainer(config)
    trainer.configure(rounds=1, epochs=1, topology="ring", dataset_name="plain")
    monkeypatch.setattr(training_service, "topology_neighbours", lambda *args: ())
    monkeypatch.setattr(training_service, "topology_senders", lambda *args: ())
    monkeypatch.setattr(training_service, "_memory_available_kb", lambda: 1)
    monkeypatch.setattr(training_service, "_resident_memory_mb", lambda: 1.0)
    monkeypatch.setattr(training_service, "_peak_process_rss_mb", lambda: 1.0)
    trainer._incoming[("plain-run", 0)] = {}

    result = trainer.run_round("plain-run", 0, 1, 11, "ring")

    assert "global_validation_accuracy" not in result


def test_fresh_model_builds_the_configured_design_instead_of_the_installed_one(tmp_path):
    trainer, dataset_name = _installed_bundle_trainer(tmp_path, None)
    design = {
        "id": "nas-finalist",
        "model_type": "custom",
        "layers": [
            {"type": "conv2d", "filters": 4, "kernel_size": 3},
            {"type": "global_average_pooling2d"},
            {"type": "dense", "units": 2, "activation": "linear"},
        ],
        "training": {"optimizer": "adam", "learning_rate": 0.002, "batch_size": 8},
    }
    trainer.configure(
        rounds=1, epochs=1, topology="ring", dataset_name=dataset_name, architecture=design, fresh_model=True
    )
    model = trainer._create_run_model("finalist", 7)
    audit = trainer._run_model_audits["finalist"]
    assert audit["initial_model_loaded"] is False
    assert audit["learning_rate"] == pytest.approx(0.002)
    assert model.count_params() == 4 * 9 + 4 + 4 * 2 + 2

    trainer.configure(rounds=1, epochs=1, topology="ring", dataset_name=dataset_name)
    trainer._create_run_model("installed", 7)
    assert trainer._run_model_audits["installed"]["initial_model_loaded"] is True
    with pytest.raises(ValueError, match="fresh_model"):
        trainer.configure(rounds=1, epochs=1, topology="ring", dataset_name=dataset_name, fresh_model="yes")


def test_peer_send_retries_transient_failures_but_not_client_errors(monkeypatch):
    from urllib.error import HTTPError
    from urllib.request import Request

    attempts = []

    class Accepted:
        status = 202

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    def flaky(request, timeout):
        attempts.append(timeout)
        if len(attempts) < 3:
            raise ConnectionResetError(104, "Connection reset by peer")
        return Accepted()

    monkeypatch.setattr(training_service, "urlopen", flaky)
    monkeypatch.setattr(training_service.time, "sleep", lambda _seconds: None)
    request = Request("http://127.0.0.1:1/v1/weights", data=b"x", method="POST")
    assert training_service._send_with_retry(request, timeout=5, peer_id="worker3") == 3

    def rejected(request, timeout):
        attempts.append("rejected")
        raise HTTPError(request.full_url, 400, "bad signature", {}, None)

    attempts.clear()
    monkeypatch.setattr(training_service, "urlopen", rejected)
    with pytest.raises(HTTPError):
        training_service._send_with_retry(request, timeout=5, peer_id="worker3")
    assert attempts == ["rejected"]

    def always_reset(request, timeout):
        raise ConnectionResetError(104, "Connection reset by peer")

    monkeypatch.setattr(training_service, "urlopen", always_reset)
    with pytest.raises(ConnectionResetError):
        training_service._send_with_retry(request, timeout=5, peer_id="worker3")


def test_identical_peer_reupload_is_acknowledged_and_a_different_one_rejected(tmp_path):
    shard_path = tmp_path / "default_shard.npz"
    np.savez(shard_path, images=np.zeros((1, 32, 32, 3), dtype=np.uint8), labels=np.array([0]))
    trainer = WorkerTrainer(
        WorkerConfig(
            node_id="worker1",
            api_token="a" * 48,
            port=5004,
            data_path=str(shard_path),
            peers=(Peer("worker2", "http://127.0.0.1:5005", "b" * 48),),
            log_path=str(tmp_path / "worker.jsonl"),
        )
    )
    first = serialize_weights([np.ones((2, 2), dtype=np.float32)])
    other = serialize_weights([np.zeros((2, 2), dtype=np.float32)])
    assert trainer.accept_peer_weights("retry-run", 0, "worker2", 10, first) is True
    assert trainer.accept_peer_weights("retry-run", 0, "worker2", 10, first) is True
    with pytest.raises(ValueError, match="Duplicate"):
        trainer.accept_peer_weights("retry-run", 0, "worker2", 10, other)


def test_global_validation_labels_are_flattened(tmp_path):
    from worker.training_service import GLOBAL_VALIDATION_FILE, load_global_validation

    np.savez(
        tmp_path / GLOBAL_VALIDATION_FILE,
        images=np.zeros((3, 4, 4, 1), dtype=np.float32),
        labels=np.array([[0], [1], [2]], dtype=np.int64),
    )
    images, labels = load_global_validation(tmp_path)
    # (N, 1) labels would broadcast against (N,) predictions into an N x N comparison.
    assert labels.shape == (3,)
    assert images.shape == (3, 4, 4, 1)
