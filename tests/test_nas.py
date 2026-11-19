import sys
import threading
from types import SimpleNamespace

import pytest
import numpy as np
import json

from common.dataset_nas import (
    CIFAR_INITIAL_POPULATION_SIZE,
    MEMORY_CONSTRAINED_MAX_PARAMETERS,
    MAX_DATASET_NAS_PARAMETERS,
    aggregate_worker_candidates,
    initial_population,
    mutate_population,
    pareto_front,
)
from common.model_design import validate_model_design
from common.nas import (
    ARCHITECTURE_SPACE,
    FIXED_ARCHITECTURE,
    MAX_PARAMETERS,
    architecture_parameters,
    architecture_spec,
    dataset_architecture_parameters,
    select_dataset_architecture,
    select_architecture,
)
from worker import nas_search
from worker.nas_search import (
    _split_indices,
    load_dataset_shard,
    search_cifar10_architectures,
)


def test_search_space_is_bounded_and_contains_fixed_architecture() -> None:
    assert len(ARCHITECTURE_SPACE) == 8
    assert FIXED_ARCHITECTURE in {row["id"] for row in ARCHITECTURE_SPACE}
    assert max(architecture_parameters(row["id"]) for row in ARCHITECTURE_SPACE) == 11_466
    assert max(architecture_parameters(row["id"]) for row in ARCHITECTURE_SPACE) < MAX_PARAMETERS


def test_dataset_nas_generates_deterministic_large_candidate_populations() -> None:
    population = initial_population(
        seed=17,
        input_shape=(28, 28, 1),
        num_classes=10,
    )
    repeated = initial_population(
        seed=17,
        input_shape=(28, 28, 1),
        num_classes=10,
    )

    assert [candidate["id"] for candidate in population] == [
        candidate["id"] for candidate in repeated
    ]
    assert len(population) == 8
    assert max(candidate["parameter_count"] for candidate in population) > 12_000
    assert all(
        candidate["parameter_count"] <= MAX_DATASET_NAS_PARAMETERS
        and candidate["model_type"] == "custom"
        and candidate["blocks"] <= 3
        and max(candidate["channel_schedule"]) <= 64
        and candidate["kernel_size"] <= 5
        and candidate["head_units"] <= 64
        for candidate in population
    )
    assert {candidate["attention"] for candidate in population} == {False, True}

    attention_candidate = next(candidate for candidate in population if candidate["attention"])
    normalized, parameter_count = validate_model_design(
        attention_candidate,
        (28, 28, 1),
        10,
    )
    assert parameter_count == attention_candidate["parameter_count"]
    layer_types = [layer["type"] for layer in normalized["layers"]]
    attention_index = layer_types.index("patch_embedding")
    assert layer_types[attention_index:attention_index + 3] == [
        "patch_embedding",
        "transformer_encoder",
        "global_average_pooling1d",
    ]
    assert layer_types[-1] == "dense"


def test_cifar_dataset_nas_uses_a_memory_constrained_architecture_population() -> None:
    initial = initial_population(
        seed=17,
        input_shape=(32, 32, 3),
        num_classes=10,
        size=CIFAR_INITIAL_POPULATION_SIZE,
        memory_constrained=True,
    )
    evolved = mutate_population(
        initial[:2],
        seed=19,
        input_shape=(32, 32, 3),
        num_classes=10,
        size=2,
        memory_constrained=True,
    )

    assert len(initial) == CIFAR_INITIAL_POPULATION_SIZE == 6
    assert len(evolved) == 4
    assert len({candidate["id"] for candidate in evolved}) == 4
    for candidate in [*initial, *evolved]:
        assert 2 <= candidate["blocks"] <= 3
        assert max(candidate["channel_schedule"]) <= 64
        assert candidate["kernel_size"] == 3 and candidate["attention"] is False
        assert candidate["parameter_count"] <= MEMORY_CONSTRAINED_MAX_PARAMETERS
    # The space now reaches models far larger than the former ~3k-parameter CNNs.
    assert max(candidate["parameter_count"] for candidate in initial) > 20_000


def test_dataset_nas_mutation_preserves_parents_and_adds_unique_offspring() -> None:
    initial = initial_population(
        seed=17,
        input_shape=(28, 28, 1),
        num_classes=10,
    )
    parents = initial[:2]
    evolved = mutate_population(
        parents,
        seed=20,
        input_shape=(28, 28, 1),
        num_classes=10,
        size=2,
    )

    assert len(evolved) == 4
    assert evolved[:2] == parents
    assert len({candidate["id"] for candidate in evolved}) == 4
    assert all(
        candidate["parameter_count"] <= MAX_DATASET_NAS_PARAMETERS
        and candidate["blocks"] <= 3
        and max(candidate["channel_schedule"]) <= 64
        and candidate["kernel_size"] <= 5
        and candidate["head_units"] <= 64
        for candidate in evolved
    )
    simple_parent = next(candidate for candidate in initial if not candidate["attention"])
    attention_mutation = mutate_population(
        [simple_parent],
        seed=5,
        input_shape=(28, 28, 1),
        num_classes=10,
        size=1,
    )
    assert attention_mutation[1]["attention"] is True


def test_dataset_nas_aggregates_only_complete_shared_candidate_scores() -> None:
    population = initial_population(
        seed=17,
        input_shape=(28, 28, 1),
        num_classes=10,
        size=2,
    )
    workers = [
        {
            "node_id": f"worker{worker_index}",
            "candidates": [
                {
                    "node_id": f"worker{worker_index}",
                    "architecture": architecture,
                    "parameter_count": architecture["parameter_count"],
                    "validation_accuracy": 0.8 + worker_index / 100,
                    "validation_macro_precision": 0.7,
                    "validation_macro_recall": 0.6,
                    "validation_macro_f1": 0.65,
                    "elapsed_seconds": 2.0,
                    "peak_rss_mb": 1200 + worker_index,
                    "model_summary": "Model: custom_mnist_model",
                    "validation_confusion_matrix": [[worker_index, 0], [0, 2]],
                }
                for architecture in population
            ],
        }
        for worker_index in range(1, 4)
    ]

    candidates = aggregate_worker_candidates(workers, worker_count=3)

    assert len(candidates) == 2
    assert candidates[0]["mean_validation_accuracy"] == pytest.approx(0.82)
    assert candidates[0]["worker_accuracies"] == pytest.approx([0.81, 0.82, 0.83])
    assert candidates[0]["model_summary"] == "Model: custom_mnist_model"
    assert candidates[0]["validation_confusion_matrix"] == [[6, 0], [0, 6]]
    with pytest.raises(ValueError, match="missing results from a worker"):
        aggregate_worker_candidates(workers[:2], worker_count=3)


def test_dataset_nas_pareto_front_keeps_accuracy_and_efficiency_tradeoffs() -> None:
    candidates = [
        {
            "architecture": {"id": "accurate"},
            "validation_accuracy": 0.95,
            "validation_macro_f1": 0.94,
            "parameter_count": 50_000,
            "elapsed_seconds": 12,
            "peak_rss_mb": 1500,
        },
        {
            "architecture": {"id": "small"},
            "validation_accuracy": 0.90,
            "validation_macro_f1": 0.89,
            "parameter_count": 5_000,
            "elapsed_seconds": 5,
            "peak_rss_mb": 800,
        },
        {
            "architecture": {"id": "dominated"},
            "validation_accuracy": 0.80,
            "validation_macro_f1": 0.79,
            "parameter_count": 60_000,
            "elapsed_seconds": 15,
            "peak_rss_mb": 1800,
        },
    ]

    assert {row["architecture"]["id"] for row in pareto_front(candidates)} == {
        "accurate",
        "small",
    }


def test_dataset_nas_enables_gpu_memory_growth_without_changing_configured_devices() -> None:
    devices = ["gpu0", "gpu1"]
    memory_growth = {"gpu0": True, "gpu1": False}
    changed = []

    def set_memory_growth(device, enabled):
        memory_growth[device] = enabled
        changed.append((device, enabled))

    tf = SimpleNamespace(
        config=SimpleNamespace(
            experimental=SimpleNamespace(
                get_memory_growth=lambda device: memory_growth[device],
                set_memory_growth=set_memory_growth,
            )
        )
    )

    nas_search._enable_gpu_memory_growth(tf, devices)

    assert changed == [("gpu1", True)]


def test_architecture_validation_rejects_modified_or_unknown_candidate() -> None:
    with pytest.raises(ValueError):
        architecture_spec("b4_c128_k9")
    with pytest.raises(ValueError):
        architecture_spec({"id": "b1_c8_k3", "blocks": 2, "channels": 8, "kernel_size": 3})


def test_selector_uses_mean_worker_validation_accuracy_with_simple_tie_break() -> None:
    scores = []
    for architecture in ARCHITECTURE_SPACE:
        accuracy = 0.7 if architecture["id"] == "b1_c8_k3" else 0.6
        for _ in range(3):
            scores.append({
                "architecture": architecture,
                "parameter_count": architecture_parameters(architecture["id"]),
                "validation_accuracy": accuracy,
                "elapsed_seconds": 2.0,
                "peak_rss_mb": 1024.0,
            })
    selected = select_architecture(scores)
    assert selected["architecture"]["id"] == "b1_c8_k3"
    assert selected["mean_validation_accuracy"] == pytest.approx(0.7)


def test_selector_rejects_missing_worker_candidate_results() -> None:
    candidate = ARCHITECTURE_SPACE[0]
    with pytest.raises(ValueError):
        select_architecture([{
            "architecture": candidate,
            "parameter_count": architecture_parameters(candidate["id"]),
            "validation_accuracy": 0.5,
            "elapsed_seconds": 2.0,
            "peak_rss_mb": 1024.0,
        }])


def test_worker_validation_split_is_disjoint_stratified_and_repeatable() -> None:
    labels = np.repeat(np.arange(10), 100)
    training, validation = _split_indices(labels, 2026)
    repeated_training, repeated_validation = _split_indices(labels, 2026)

    assert np.array_equal(training, repeated_training)
    assert np.array_equal(validation, repeated_validation)
    assert set(training).isdisjoint(set(validation))
    assert len(training) + len(validation) == len(labels)
    assert len(validation) == 200
    assert all(np.count_nonzero(labels[validation] == label) == 20 for label in range(10))


def test_worker_search_rejects_time_limit_outside_hard_bounds() -> None:
    with pytest.raises(ValueError, match="Search time limit"):
        search_cifar10_architectures(
            np.zeros((10, 32, 32, 3), dtype=np.uint8),
            np.zeros((10,), dtype=np.int64),
            seed=1,
            time_limit_seconds=301,
            node_id="worker1",
            logger=lambda *args, **kwargs: None,
        )


def test_isolated_search_stream_events_are_logged_and_result_is_returned(monkeypatch) -> None:
    candidate = {
        "architecture": ARCHITECTURE_SPACE[0],
        "parameter_count": architecture_parameters(ARCHITECTURE_SPACE[0]["id"]),
        "validation_accuracy": 0.5,
        "elapsed_seconds": 2.0,
        "peak_rss_mb": 1024.0,
    }
    output = "\n".join(
        (
            json.dumps({"type": "nas_candidate_evaluated", **candidate}),
            json.dumps({"type": "nas_search_result", "data": {"candidates": [candidate]}}),
        )
    )
    called = {}

    def run(command, **kwargs):
        called.update(command=command, kwargs=kwargs)
        return nas_search.subprocess.CompletedProcess(command, 0, output, "")

    monkeypatch.setenv("DFL_API_TOKEN", "do-not-forward")
    monkeypatch.setattr(nas_search.subprocess, "run", run)
    logged = []
    result = nas_search.search_cifar10_architectures_isolated(
        "/worker/train.npz",
        seed=2026,
        time_limit_seconds=60,
        node_id="worker1",
        logger=lambda name, **details: logged.append((name, details)),
    )

    assert result == {"candidates": [candidate]}
    assert logged == [("nas_candidate_evaluated", candidate)]
    assert called["kwargs"]["timeout"] == 90
    assert "DFL_API_TOKEN" not in called["kwargs"]["env"]


def test_isolated_dataset_search_limits_threads_and_logs_complete_failure_context(
    monkeypatch,
) -> None:
    stderr = "\n".join(
        [
            "Traceback (most recent call last):",
            '  File "worker/nas_search.py", line 307, in search_dataset_architectures',
            "    model.fit(training_data, epochs=1)",
            "RuntimeError: thread creation failed",
        ]
    )
    called = {}

    def run(actual_command, **kwargs):
        called.update(command=actual_command, kwargs=kwargs)
        return nas_search.subprocess.CompletedProcess(actual_command, 1, "", stderr)

    monkeypatch.setenv("DFL_API_TOKEN", "do-not-forward")
    monkeypatch.setenv("TF_NUM_INTRAOP_THREADS", "32")
    monkeypatch.setattr(nas_search.subprocess, "run", run)
    logged = []

    with pytest.raises(
        RuntimeError,
        match="Isolated dataset NAS search exited with code 1: RuntimeError: thread creation failed",
    ):
        nas_search.search_dataset_architectures_isolated(
            "/worker/mnist/train.npz",
            seed=2026,
            time_limit_seconds=60,
            node_id="worker1",
            logger=lambda name, **details: logged.append((name, details)),
        )

    environment = called["kwargs"]["env"]
    assert environment["TF_NUM_INTRAOP_THREADS"] == "1"
    assert environment["TF_NUM_INTEROP_THREADS"] == "1"
    assert environment["OMP_NUM_THREADS"] == "1"
    assert environment["OPENBLAS_NUM_THREADS"] == "1"
    assert environment["MKL_NUM_THREADS"] == "1"
    assert environment["NUMEXPR_NUM_THREADS"] == "1"
    assert "DFL_API_TOKEN" not in environment
    assert logged == [
        (
            "nas_search_failed",
            {
                "node_id": "worker1",
                "returncode": 1,
                "stderr_tail": stderr,
            },
        )
    ]


def test_isolated_dataset_search_terminates_on_cancellation(monkeypatch) -> None:
    event = threading.Event()
    event.set()
    terminated = []

    class Process:
        returncode = -15

        def __init__(self):
            self.communicate_calls = 0

        def communicate(self, *, timeout):
            self.communicate_calls += 1
            if self.communicate_calls == 1:
                raise nas_search.subprocess.TimeoutExpired("nas", timeout)
            return "", ""

        def terminate(self):
            terminated.append(True)

        def kill(self):
            raise AssertionError("Cancelled NAS subprocess should terminate gracefully")

    monkeypatch.setattr(nas_search.subprocess, "Popen", lambda *args, **kwargs: Process())

    with pytest.raises(RuntimeError, match="search was cancelled"):
        nas_search.search_dataset_architectures_isolated(
            "/worker/mnist/train.npz",
            seed=2026,
            time_limit_seconds=30,
            node_id="worker1",
            logger=lambda *_args, **_kwargs: None,
            cancel_event=event,
        )
    assert terminated == [True]


def test_dataset_selector_uses_dataset_class_count_and_worker_results() -> None:
    rows = []
    for spec in ARCHITECTURE_SPACE:
        for worker_index in range(1, 4):
            rows.append(
                {
                    "node_id": f"worker{worker_index}",
                    "architecture": spec,
                    "parameter_count": dataset_architecture_parameters(spec, 2),
                    "validation_accuracy": (
                        0.8 if spec["id"] == "b1_c8_k3" else 0.6
                    ),
                    "elapsed_seconds": 1.0,
                    "peak_rss_mb": 1000.0,
                }
            )
    selected = select_dataset_architecture(rows, num_classes=2)
    assert selected["architecture"]["id"] == "b1_c8_k3"
    assert selected["parameter_count"] == dataset_architecture_parameters(
        "b1_c8_k3", 2
    )
    assert selected["mean_validation_accuracy"] == pytest.approx(0.8)

    rows[0]["parameter_count"] += 1
    with pytest.raises(ValueError, match="invalid"):
        select_dataset_architecture(rows, num_classes=2)


def test_dataset_selector_rejects_missing_candidate_and_too_many_classes() -> None:
    rows = []
    for spec in ARCHITECTURE_SPACE:
        for worker_index in range(1, 4):
            rows.append(
                {
                    "node_id": f"worker{worker_index}",
                    "architecture": spec,
                    "parameter_count": dataset_architecture_parameters(spec, 2),
                    "validation_accuracy": 0.5,
                }
            )
    with pytest.raises(ValueError, match="exactly one"):
        select_dataset_architecture(rows[:-1], num_classes=2)
    with pytest.raises(ValueError, match="No NAS candidate"):
        select_dataset_architecture(rows, num_classes=1500)


def test_load_dataset_shard_preserves_user_class_names_and_rejects_bad_labels(tmp_path) -> None:
    shard_path = tmp_path / "train.npz"
    np.savez_compressed(
        shard_path,
        images=np.zeros((4, 28, 28, 1), dtype=np.float32),
        labels=np.array([0, 0, 1, 1], dtype=np.int64),
        class_names=np.array(["cat", "dog"]),
    )
    images, labels, classes = load_dataset_shard(shard_path)
    assert images.shape == (4, 28, 28, 1)
    assert labels.tolist() == [0, 0, 1, 1]
    assert classes == ["cat", "dog"]

    np.savez_compressed(
        shard_path,
        images=np.zeros((4, 28, 28, 1), dtype=np.float32),
        labels=np.array([0, 0, 1, 2], dtype=np.int64),
        class_names=np.array(["cat", "dog"]),
    )
    with pytest.raises(ValueError, match="class names"):
        load_dataset_shard(shard_path)


def test_dataset_nas_accepts_grayscale_until_gpu_precondition(monkeypatch) -> None:
    monkeypatch.setattr(nas_search, "_available_system_memory_mb", lambda: 1024.0)
    monkeypatch.setitem(
        sys.modules,
        "tensorflow",
        SimpleNamespace(config=SimpleNamespace(list_physical_devices=lambda _: [])),
    )
    with pytest.raises(RuntimeError, match="requires an enumerated TensorFlow GPU"):
        nas_search.search_dataset_architectures(
            np.zeros((4, 28, 28, 1), dtype=np.uint8),
            np.array([0, 0, 1, 1]),
            ["zero", "one"],
            seed=42,
            time_limit_seconds=120,
            node_id="worker1",
            logger=lambda *_args, **_kwargs: None,
            epochs=10,
        )


def test_dataset_nas_rejects_low_system_memory_before_loading_tensorflow(monkeypatch) -> None:
    monkeypatch.setattr(nas_search, "_available_system_memory_mb", lambda: 128.0)

    with pytest.raises(MemoryError, match="only 128.0 MiB is available"):
        nas_search.search_dataset_architectures(
            np.zeros((4, 28, 28, 1), dtype=np.uint8),
            np.array([0, 0, 1, 1]),
            ["zero", "one"],
            seed=42,
            time_limit_seconds=120,
            node_id="worker1",
            logger=lambda *_args, **_kwargs: None,
        )


def test_dataset_nas_bounds_training_and_validation_pipeline_memory() -> None:
    assert nas_search.dataset_nas_input_pipeline_limits(10_000, 2_000) == (8, 32, 16)
    assert nas_search.dataset_nas_input_pipeline_limits(8, 4) == (8, 8, 4)
    with pytest.raises(ValueError, match="non-empty"):
        nas_search.dataset_nas_input_pipeline_limits(0, 4)


def test_dataset_architecture_parameters_use_grayscale_input_channels() -> None:
    rgb_parameters = dataset_architecture_parameters("b1_c8_k3", 10)
    grayscale_parameters = dataset_architecture_parameters(
        "b1_c8_k3", 10, input_channels=1
    )
    assert rgb_parameters - grayscale_parameters == 3 * 3 * 2 * 8


def test_extended_population_has_vgg_anchor_and_stays_under_its_cap() -> None:
    from common.dataset_nas import (
        EXTENDED_INITIAL_POPULATION_SIZE,
        EXTENDED_MAX_PARAMETERS,
        EXTENDED_OFFSPRING,
        extended_initial_population,
        mutate_extended_population,
    )

    for seed in (17, 42, 2026):
        population = extended_initial_population(seed=seed, input_shape=(32, 32, 3), num_classes=10)
        ids = [candidate["id"] for candidate in population]
        assert len(ids) == len(set(ids)) == EXTENDED_INITIAL_POPULATION_SIZE
        assert ids[:2] == ["nas-d3-w32x64x128-k3-conv-h64-drop20-c2", "nas-d3-w32x64x64-k3-conv-h64-drop20"]
        for candidate in population:
            design, parameters = validate_model_design(candidate, (32, 32, 3), 10)
            assert parameters == candidate["parameter_count"] <= EXTENDED_MAX_PARAMETERS
            convs = sum(layer["type"] == "conv2d" for layer in design["layers"])
            assert convs == len(candidate["channel_schedule"]) * candidate.get("convs_per_block", 1)
        seed_design = {"id": "seed-baseline", "model_type": "custom", "layers": population[0]["layers"]}
        evolved = mutate_extended_population(
            [population[0], seed_design], seed=seed + 2, input_shape=(32, 32, 3), num_classes=10
        )
        assert evolved[:2] == [population[0], seed_design]  # an external seed is kept, not mutated
        assert len(evolved) == 2 + EXTENDED_OFFSPRING
        assert len({candidate["id"] for candidate in evolved}) == len(evolved)
    with pytest.raises(ValueError, match="searched parent"):
        mutate_extended_population([seed_design], seed=1, input_shape=(32, 32, 3), num_classes=10)


def test_single_conv_designs_keep_their_v1_form() -> None:
    from common.dataset_nas import build_dataset_architecture

    single = build_dataset_architecture(
        input_shape=(28, 28, 1), num_classes=10, channels=(32, 64), kernel_size=3,
        block_type="conv2d", head_units=64, dropout=0.2,
    )
    assert "convs_per_block" not in single and not single["id"].endswith("-c2")


def test_aggregation_reports_shared_split_accuracy_and_weakest_worker() -> None:
    architecture = initial_population(seed=3, input_shape=(28, 28, 1), num_classes=10, size=2)[0]
    rows = [
        {
            "node_id": f"worker{index}",
            "candidates": [
                {
                    "node_id": f"worker{index}",
                    "architecture": architecture,
                    "parameter_count": architecture["parameter_count"],
                    "validation_accuracy": 0.9,
                    "validation_macro_precision": 0.5,
                    "validation_macro_recall": 0.5,
                    "validation_macro_f1": 0.5,
                    "global_validation_accuracy": accuracy,
                    "global_validation_macro_f1": accuracy - 0.1,
                    "elapsed_seconds": 1.0,
                    "peak_rss_mb": 900.0,
                }
            ],
        }
        for index, accuracy in enumerate((0.4, 0.6), start=1)
    ]
    [candidate] = aggregate_worker_candidates(rows, worker_count=2)
    assert candidate["global_validation_accuracy"] == pytest.approx(0.5)
    assert candidate["global_validation_macro_f1"] == pytest.approx(0.4)
    assert candidate["global_validation_min_accuracy"] == pytest.approx(0.4)
    assert candidate["validation_accuracy"] == pytest.approx(0.9)  # local metric untouched here


@pytest.mark.parametrize("augmentation", ["none", "flip_shift"])
def test_dataset_search_scores_candidates_on_the_shared_split(monkeypatch, augmentation) -> None:
    import tensorflow as tf

    from common.dataset_nas import build_dataset_architecture

    monkeypatch.setattr(nas_search, "_available_system_memory_mb", lambda: 100_000.0)
    monkeypatch.setattr(tf.config, "list_physical_devices", lambda kind=None: [SimpleNamespace(name="GPU:0")])
    monkeypatch.setattr(nas_search, "_rss_mb", lambda: 500.0)  # /proc is Linux-only
    rng = np.random.default_rng(0)
    images = rng.random((40, 8, 8, 3), dtype=np.float32)
    labels = np.tile(np.arange(2, dtype=np.int64), 20)
    global_images = rng.random((10, 8, 8, 3), dtype=np.float32)
    global_labels = np.tile(np.arange(2, dtype=np.int64), 5)
    candidate = build_dataset_architecture(
        input_shape=(8, 8, 3), num_classes=2, channels=(16, 16), kernel_size=3,
        block_type="conv2d", head_units=0, dropout=0.0,
    )
    result = nas_search.search_dataset_architectures(
        images,
        labels,
        ["a", "b"],
        seed=5,
        time_limit_seconds=120,
        node_id="worker1",
        logger=lambda *_args, **_kwargs: None,
        architectures=[candidate],
        reuse_tensorflow_runtime=True,
        global_validation=(global_images, global_labels),
        augmentation=augmentation,
    )
    [row] = result["candidates"]
    assert result["augmentation"] == augmentation
    assert result["global_validation_samples"] == 10
    assert row["global_validation_samples"] == 10
    assert 0.0 <= row["global_validation_accuracy"] <= 1.0
    assert 0.0 <= row["global_validation_macro_f1"] <= 1.0
    with pytest.raises(ValueError, match="augmentation"):
        nas_search.search_dataset_architectures(
            images, labels, ["a", "b"], seed=5, time_limit_seconds=120, node_id="worker1",
            logger=lambda *_a, **_k: None, architectures=[candidate], augmentation="mixup",
        )


def test_split_training_order_mixes_classes_for_small_shuffle_buffers() -> None:
    from worker.nas_search import DATASET_NAS_SHUFFLE_BUFFER_SIZE, _split_indices

    labels = np.repeat(np.arange(10, dtype=np.int64), 200)  # stored class by class
    train, validation = _split_indices(labels, seed=2026)
    again_train, again_validation = _split_indices(labels, seed=2026)
    assert np.array_equal(train, again_train) and np.array_equal(validation, again_validation)
    assert set(train.tolist()).isdisjoint(validation.tolist())
    assert sorted([*train.tolist(), *validation.tolist()]) == list(range(labels.size))
    # Every window the search pipeline's shuffle buffer sees holds several classes.
    windows = [labels[train[start:start + DATASET_NAS_SHUFFLE_BUFFER_SIZE]] for start in range(0, train.size, DATASET_NAS_SHUFFLE_BUFFER_SIZE)]
    assert min(len(set(window.tolist())) for window in windows) >= 5
