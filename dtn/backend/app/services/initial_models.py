"""Dataset-derived small CNN initialization and authenticated worker deployment."""

import hashlib
import io
import json
import os
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from common.nas import FIXED_ARCHITECTURE, architecture_spec, dataset_architecture_parameters
from common.model_design import (
    MAX_MODEL_PARAMETERS,
    ModelDesignError,
    validate_model_design,
    validate_normalization,
)
from app.core.config import DeviceConfig
from app.services.datasets import DatasetError, DatasetStore, validate_dataset_name
from app.services.worker_shards import (
    WorkerShardError,
    get_worker_dataset_initial_model,
    get_worker_dataset_initial_model_if_present,
    get_worker_training_status,
    delete_worker_initial_model,
    install_worker_initial_model,
)

INITIAL_MODEL_DIRECTORY = "initial_model"


class InitialModelError(ValueError):
    pass



def _normalization_matches(stored: Any, current: Any, tolerance: float = 1e-3) -> bool:
    """Compare normalization settings, tolerating tiny drift in recomputed statistics.

    Re-preprocessing the same dataset recomputes mean/std and can differ in the
    5th decimal place; that must not block reuse of the deployed initial model.
    """
    if not isinstance(stored, dict) or not isinstance(current, dict):
        return stored == current
    if stored.keys() != current.keys() or stored.get("method") != current.get("method"):
        return False
    for key, value in stored.items():
        other = current[key]
        if isinstance(value, list) and isinstance(other, list):
            if len(value) != len(other) or any(
                not isinstance(a, (int, float)) or not isinstance(b, (int, float))
                or abs(a - b) > tolerance * max(1.0, abs(a))
                for a, b in zip(value, other)
            ):
                return False
        elif value != other:
            return False
    return True

def build_initial_model(
    *,
    dataset_name: str,
    input_shape: tuple[int, int, int],
    classes: Sequence[str],
    seed: int = 0,
    architecture: str | dict[str, Any] = FIXED_ARCHITECTURE,
    normalization: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    validate_dataset_name(dataset_name)
    if len(input_shape) != 3 or any(
        not isinstance(dimension, int) or dimension < 1 or dimension > 2048
        for dimension in input_shape
    ):
        raise InitialModelError("Model input_shape must contain three valid image dimensions")
    if input_shape[2] not in {1, 3}:
        raise InitialModelError("The initial CNN supports one-channel grayscale or three-channel RGB")
    if len(classes) < 2 or any(not isinstance(name, str) or not name for name in classes):
        raise InitialModelError("At least two non-empty dataset classes are required")
    if len(set(classes)) != len(classes):
        raise InitialModelError("Dataset class names must be unique")
    if not 0 <= seed <= 2**32 - 1:
        raise InitialModelError("Model initialization seed is out of range")

    if isinstance(architecture, dict) and architecture.get("model_type") in {
        "custom",
        "pretrained_deit",
    }:
        try:
            design, parameter_count = validate_model_design(
                architecture,
                input_shape,
                len(classes),
            )
            if design["model_type"] == "pretrained_deit":
                normalization = validate_normalization(
                    normalization or {"method": "minmax"},
                    input_shape[2],
                )
        except ModelDesignError as error:
            raise InitialModelError(str(error)) from error
        if parameter_count > MAX_MODEL_PARAMETERS:
            raise InitialModelError(
                f"Model has {parameter_count:,} parameters; cap is {MAX_MODEL_PARAMETERS:,}"
            )
        model_manifest = {
            **design,
            "dataset_name": dataset_name,
            "input_shape": list(input_shape),
            "classes": list(classes),
            "num_classes": len(classes),
            "parameter_count": parameter_count,
            "initialization_seed": seed,
            "normalization": normalization or {"method": "minmax"},
        }
        return model_manifest, {}

    try:
        design = architecture_spec(architecture)
        parameter_count = dataset_architecture_parameters(
            design,
            len(classes),
            input_channels=input_shape[2],
        )
    except ValueError as error:
        raise InitialModelError(str(error)) from error
    if parameter_count > MAX_MODEL_PARAMETERS:
        raise InitialModelError(
            f"Dataset-derived model has {parameter_count} parameters; cap is {MAX_MODEL_PARAMETERS}"
        )

    weight_shapes: dict[str, tuple[int, ...]] = {}
    previous_channels = input_shape[2]
    for index in range(1, design["blocks"] + 1):
        weight_shapes[f"conv{index}_kernel"] = (
            design["kernel_size"],
            design["kernel_size"],
            previous_channels,
            design["channels"],
        )
        weight_shapes[f"conv{index}_bias"] = (design["channels"],)
        previous_channels = design["channels"]
    weight_shapes["classifier_kernel"] = (design["channels"], len(classes))
    weight_shapes["classifier_bias"] = (len(classes),)
    actual_parameter_count = sum(int(np.prod(shape)) for shape in weight_shapes.values())
    if actual_parameter_count != parameter_count:
        raise InitialModelError("Generated model weight shapes do not match the selected architecture")
    if actual_parameter_count > MAX_MODEL_PARAMETERS:
        raise InitialModelError(
            f"Dataset-derived model has {actual_parameter_count} parameters; cap is {MAX_MODEL_PARAMETERS}"
        )
    parameter_count = actual_parameter_count

    rng = np.random.default_rng(seed)
    weights: dict[str, np.ndarray] = {}
    previous_channels = input_shape[2]
    for index in range(1, design["blocks"] + 1):
        kernel = weight_shapes[f"conv{index}_kernel"]
        fan_in = design["kernel_size"] ** 2 * previous_channels
        weights[f"conv{index}_kernel"] = rng.normal(
            0.0,
            np.sqrt(2.0 / fan_in),
            size=kernel,
        ).astype(np.float32)
        weights[f"conv{index}_bias"] = np.zeros(
            weight_shapes[f"conv{index}_bias"],
            dtype=np.float32,
        )
        previous_channels = design["channels"]
    weights["classifier_kernel"] = rng.normal(
        0.0,
        np.sqrt(2.0 / design["channels"]),
        size=weight_shapes["classifier_kernel"],
    ).astype(np.float32)
    weights["classifier_bias"] = np.zeros(
        weight_shapes["classifier_bias"],
        dtype=np.float32,
    )
    architecture: dict[str, Any] = {
        "model_type": "small_cnn",
        "dataset_name": dataset_name,
        "design": design,
        "input_shape": list(input_shape),
        "classes": list(classes),
        "num_classes": len(classes),
        "layers": [
            {
                "type": "conv2d",
                "filters": design["channels"],
                "kernel_size": [design["kernel_size"], design["kernel_size"]],
                "padding": "same",
                "activation": "relu",
            }
            for _ in range(design["blocks"])
        ]
        + [
            {"type": "global_average_pooling2d"},
            {"type": "dense", "units": len(classes), "activation": "linear"},
        ],
        "parameter_count": parameter_count,
        "weight_shapes": {name: list(shape) for name, shape in weight_shapes.items()},
        "initialization_seed": seed,
    }
    return architecture, weights


def _make_bundle(architecture: dict[str, Any], weights: dict[str, np.ndarray]) -> bytes:
    weight_buffer = io.BytesIO()
    np.savez_compressed(weight_buffer, **weights)
    bundle = io.BytesIO()
    with zipfile.ZipFile(bundle, "w", compression=zipfile.ZIP_DEFLATED) as zipped:
        zipped.writestr(
            "architecture.json",
            json.dumps(architecture, indent=2, sort_keys=True) + "\n",
        )
        if architecture["model_type"] not in {"custom", "pretrained_deit"}:
            zipped.writestr("initial_weights.npz", weight_buffer.getvalue())
    return bundle.getvalue()


def npz_bytes(weights: dict[str, np.ndarray]) -> bytes:
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **weights)
    return buffer.getvalue()


def create_and_distribute_initial_model(
    dataset_store: DatasetStore,
    workers: Sequence[DeviceConfig],
    dataset_name: str,
    *,
    seed: int = 0,
    architecture: str | dict[str, Any] | None = None,
) -> dict[str, Any]:
    validate_dataset_name(dataset_name)
    selected_workers = sorted(
        (worker for worker in workers if worker.role == "worker"),
        key=lambda worker: worker.node_id,
    )
    if (
        len(selected_workers) < 3
        or len({worker.node_id for worker in selected_workers}) != len(selected_workers)
    ):
        raise InitialModelError("At least three uniquely configured worker nodes are required")
    for worker in selected_workers:
        get_worker_training_status(worker)

    try:
        metadata = dataset_store.get_metadata(dataset_name)
    except DatasetError:
        raise
    dataset_path = (dataset_store.root / dataset_name).resolve()
    if dataset_path.parent != dataset_store.root.resolve() or not dataset_path.is_dir():
        raise InitialModelError(f"Dataset not found: {dataset_name}")
    report_paths = sorted(
        (dataset_path / "processed").glob("run-*/preprocessing_report.json"),
        key=lambda path: path.stat().st_mtime_ns,
    )
    if not report_paths:
        raise InitialModelError(
            f"Dataset has not been preprocessed; preprocess {dataset_name} before model creation"
        )
    try:
        preprocessing_report = json.loads(report_paths[-1].read_text(encoding="utf-8"))
        if "image_shape" in preprocessing_report:
            image_shape = preprocessing_report["image_shape"]
            if not isinstance(image_shape, list) or len(image_shape) != 3:
                raise ValueError("image_shape must contain height, width, and channels")
            input_shape = (
                int(image_shape[0]),
                int(image_shape[1]),
                int(image_shape[2]),
            )
        else:
            target_size = preprocessing_report["target_size"]
            input_shape = (
                int(target_size["height"]),
                int(target_size["width"]),
                int(preprocessing_report.get("channels", 3)),
            )
        classes = metadata["classes"]
        normalization = preprocessing_report.get("normalization", {"method": "minmax"})
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise InitialModelError("Dataset metadata or preprocessing report is invalid") from error

    local_model_path = dataset_path / "models" / INITIAL_MODEL_DIRECTORY
    existing_report = (
        get_initial_model_report(dataset_store, dataset_name)
        if local_model_path.exists() or local_model_path.is_symlink()
        else None
    )
    if existing_report is not None:
        if architecture is not None:
            raise InitialModelError(
                f"Initial model already exists for dataset {dataset_name}; "
                "delete it before selecting a different architecture"
            )
        architecture = existing_report["architecture"]
        if (
            architecture.get("input_shape") != list(input_shape)
            or architecture.get("classes") != list(classes)
            or architecture.get("dataset_name") != dataset_name
        ):
            raise InitialModelError(
                f"Existing initial model does not match the preprocessed {dataset_name} dataset"
            )
        if (
            architecture.get("model_type") in {"custom", "pretrained_deit"}
            and not _normalization_matches(
                architecture.get("normalization", {"method": "minmax"}), normalization
            )
        ):
            raise InitialModelError(
                f"Existing initial model normalization does not match the preprocessed {dataset_name} dataset"
            )
        architecture_bytes = (local_model_path / "architecture.json").read_bytes()
        weights_path = local_model_path / "initial_weights.npz"
        weights_bytes = weights_path.read_bytes() if weights_path.is_file() else None
        bundle_buffer = io.BytesIO()
        with zipfile.ZipFile(bundle_buffer, "w", compression=zipfile.ZIP_DEFLATED) as zipped:
            zipped.writestr("architecture.json", architecture_bytes)
            if weights_bytes is not None:
                zipped.writestr("initial_weights.npz", weights_bytes)
        bundle = bundle_buffer.getvalue()
        architecture_sha256 = existing_report["architecture_sha256"]
        reused_existing_model = True
    else:
        architecture, weights = build_initial_model(
            dataset_name=dataset_name,
            input_shape=input_shape,
            classes=classes,
            seed=seed,
            architecture=architecture or FIXED_ARCHITECTURE,
            normalization=normalization,
        )
        bundle = _make_bundle(architecture, weights)
        architecture_bytes = json.dumps(
            architecture, indent=2, sort_keys=True
        ).encode("utf-8") + b"\n"
        architecture_sha256 = hashlib.sha256(architecture_bytes).hexdigest()
        reused_existing_model = False

    deployed: dict[str, dict[str, Any]] = {}
    expected_weights_sha256: str | None = None
    if architecture["model_type"] not in {"custom", "pretrained_deit"}:
        with zipfile.ZipFile(io.BytesIO(bundle), "r") as model_bundle:
            expected_weights_sha256 = hashlib.sha256(
                model_bundle.read("initial_weights.npz")
            ).hexdigest()

    for worker in selected_workers:
        get_worker_training_status(worker)
        installed = get_worker_dataset_initial_model_if_present(worker, dataset_name)
        if installed is not None:
            if (
                installed.get("architecture_sha256") != architecture_sha256
                or installed.get("parameter_count") != architecture["parameter_count"]
            ):
                raise WorkerShardError(
                    f"{worker.node_id} already has a different initial model for {dataset_name}"
                )
            if (
                expected_weights_sha256 is not None
                and installed.get("weights_sha256") != expected_weights_sha256
            ):
                raise WorkerShardError(
                    f"{worker.node_id} initial model weights do not match the coordinator model"
                )
            verified = installed
        else:
            installed = install_worker_initial_model(worker, dataset_name, bundle)
            verified = get_worker_dataset_initial_model(worker, dataset_name)
        if architecture["model_type"] in {"custom", "pretrained_deit"}:
            worker_weights_hash = installed.get("weights_sha256")
            if not isinstance(worker_weights_hash, str) or verified.get("weights_sha256") != worker_weights_hash:
                raise WorkerShardError(
                    f"{worker.node_id} returned inconsistent model weight hashes"
                )
            if deployed and worker_weights_hash != next(iter(deployed.values()))["weights_sha256"]:
                raise WorkerShardError(
                    "Workers initialized different model weights; refusing distributed training"
                )
            installed["weights_sha256"] = worker_weights_hash
        else:
            if installed.get("weights_sha256") != expected_weights_sha256:
                raise WorkerShardError(
                    f"{worker.node_id} initial model weights do not match the PC model"
                )
        expected = {
            "parameter_count": architecture["parameter_count"],
            "architecture_sha256": architecture_sha256,
        }
        if architecture["model_type"] not in {"custom", "pretrained_deit"}:
            if expected_weights_sha256 is None:
                raise WorkerShardError("Coordinator model has no initialized weight bundle")
            expected["weights_sha256"] = expected_weights_sha256
        for key, value in expected.items():
            if installed.get(key) != value or verified.get(key) != value:
                raise WorkerShardError(
                    f"{worker.node_id} initial model {key} does not match the PC model"
                )
        deployed[worker.node_id] = verified

    weights_sha256 = deployed[selected_workers[0].node_id]["weights_sha256"]
    if not reused_existing_model:
        local_model_path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".initial-model-", dir=local_model_path.parent) as temp:
            staging = Path(temp) / INITIAL_MODEL_DIRECTORY
            staging.mkdir()
            (staging / "architecture.json").write_bytes(architecture_bytes)
            if weights:
                weights_payload = npz_bytes(weights)
                (staging / "initial_weights.npz").write_bytes(weights_payload)
            (staging / "metadata.json").write_text(
                json.dumps(
                    {
                        "dataset_name": dataset_name,
                        "parameter_count": architecture["parameter_count"],
                        "architecture_sha256": architecture_sha256,
                        "weights_sha256": weights_sha256,
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            os.rename(staging, local_model_path)

    return {
        "dataset_name": dataset_name,
        "architecture": architecture,
        "parameter_count": architecture["parameter_count"],
        "architecture_sha256": architecture_sha256,
        "weights_sha256": weights_sha256,
        "local_model_path": str(local_model_path),
        "workers": deployed,
        "reused_existing_model": reused_existing_model,
    }


def get_initial_model_report(
    dataset_store: DatasetStore,
    dataset_name: str,
) -> dict[str, Any] | None:
    validate_dataset_name(dataset_name)
    dataset_path = (dataset_store.root / dataset_name).resolve()
    if dataset_path.parent != dataset_store.root.resolve():
        raise InitialModelError("Initial model path is outside the dataset store")
    if not dataset_path.is_dir():
        raise DatasetError(f"Dataset not found: {dataset_name}", 404)
    model_path = dataset_path / "models" / INITIAL_MODEL_DIRECTORY
    if not model_path.exists():
        return None
    if model_path.is_symlink() or not model_path.is_dir():
        raise InitialModelError("Initial-model bundle path is not a regular directory")

    architecture_path = model_path / "architecture.json"
    weights_path = model_path / "initial_weights.npz"
    metadata_path = model_path / "metadata.json"
    if not architecture_path.is_file() or not metadata_path.is_file():
        raise InitialModelError("Initial-model bundle is incomplete")
    if architecture_path.is_symlink() or metadata_path.is_symlink():
        raise InitialModelError("Initial-model bundle cannot contain symbolic links")
    try:
        architecture_bytes = architecture_path.read_bytes()
        architecture = json.loads(architecture_bytes)
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        weights_bytes = weights_path.read_bytes() if weights_path.is_file() else None
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise InitialModelError("Initial-model bundle metadata is unreadable") from error
    if weights_path.is_symlink():
        raise InitialModelError("Initial-model bundle cannot contain symbolic links")
    if not isinstance(architecture, dict) or not isinstance(metadata, dict):
        raise InitialModelError("Initial-model bundle metadata or hashes do not match")
    if architecture.get("model_type") in {"custom", "pretrained_deit"}:
        weights_hash_valid = (
            weights_bytes is None
            and isinstance(metadata.get("weights_sha256"), str)
            and len(metadata["weights_sha256"]) == 64
            and all(character in "0123456789abcdef" for character in metadata["weights_sha256"])
        )
    else:
        weights_hash_valid = (
            weights_bytes is not None
            and metadata.get("weights_sha256") == hashlib.sha256(weights_bytes).hexdigest()
        )
    if (
        architecture.get("dataset_name") != dataset_name
        or metadata.get("dataset_name") != dataset_name
        or metadata.get("architecture_sha256") != hashlib.sha256(architecture_bytes).hexdigest()
        or not weights_hash_valid
    ):
        raise InitialModelError("Initial-model bundle metadata or hashes do not match")
    if architecture.get("model_type") in {"custom", "pretrained_deit"}:
        classes = architecture.get("classes")
        input_shape = architecture.get("input_shape")
        num_classes = architecture.get("num_classes")
        if (
            not isinstance(classes, list)
            or len(classes) < 2
            or any(not isinstance(name, str) or not name for name in classes)
            or len(set(classes)) != len(classes)
            or num_classes != len(classes)
        ):
            raise InitialModelError("Initial-model dataset metadata is invalid")
        try:
            normalized_design, expected_parameters = validate_model_design(
                architecture,
                input_shape,
                num_classes,
            )
            if normalized_design["model_type"] == "pretrained_deit":
                normalized_normalization = validate_normalization(
                    architecture.get("normalization"),
                    input_shape[2],
                )
                if architecture.get("normalization") != normalized_normalization:
                    raise ModelDesignError("Initial-model normalization metadata is not canonical")
        except (KeyError, TypeError, ValueError) as error:
            raise InitialModelError("Initial-model design is invalid") from error
        if (
            architecture.get("parameter_count") != expected_parameters
            or metadata.get("parameter_count") != expected_parameters
        ):
            raise InitialModelError("Initial-model parameter count does not match its design")
    design = architecture.get("design")
    if design is not None:
        try:
            selected_design = architecture_spec(design)
            expected_parameters = dataset_architecture_parameters(
                selected_design,
                int(architecture["num_classes"]),
                input_channels=int(architecture["input_shape"][2]),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise InitialModelError("Initial-model architecture specification is invalid") from error
        if (
            architecture.get("parameter_count") != expected_parameters
            or metadata.get("parameter_count") != expected_parameters
        ):
            raise InitialModelError("Initial-model parameter count does not match its architecture")
    return {
        "dataset_name": dataset_name,
        "architecture": architecture,
        "parameter_count": metadata.get("parameter_count"),
        "architecture_sha256": metadata["architecture_sha256"],
        "weights_sha256": metadata["weights_sha256"],
    }


def delete_initial_model(
    dataset_store: DatasetStore,
    workers: Sequence[DeviceConfig],
    dataset_name: str,
) -> dict[str, Any]:
    validate_dataset_name(dataset_name)
    dataset_directory = dataset_store.root / dataset_name
    dataset_path = dataset_directory.resolve()
    if (
        dataset_directory.is_symlink()
        or dataset_path.parent != dataset_store.root.resolve()
        or not dataset_path.is_dir()
    ):
        raise DatasetError(f"Dataset not found: {dataset_name}", 404)
    models_directory = dataset_path / "models"
    model_path = models_directory / INITIAL_MODEL_DIRECTORY
    if (
        models_directory.is_symlink()
        or model_path.is_symlink()
        or not model_path.is_dir()
    ):
        raise DatasetError(f"Initial model not found for {dataset_name}", 404)

    selected_workers = sorted(
        (worker for worker in workers if worker.role == "worker"),
        key=lambda worker: worker.node_id,
    )
    if (
        len(selected_workers) < 3
        or len({worker.node_id for worker in selected_workers}) != len(selected_workers)
    ):
        raise InitialModelError("At least three uniquely configured worker nodes are required")

    for worker in selected_workers:
        get_worker_training_status(worker)
    removed_workers: list[str] = []
    for worker in selected_workers:
        try:
            delete_worker_initial_model(worker, dataset_name)
        except WorkerShardError as error:
            removed_nodes = ", ".join(removed_workers) if removed_workers else "none"
            raise InitialModelError(
                f"Partial model deletion failed on {worker.node_id}: {error}. "
                f"The coordinator copy is retained; workers already cleared: {removed_nodes}. "
                "Correct the worker error and retry deletion."
            ) from error
        removed_workers.append(worker.node_id)
    shutil.rmtree(model_path)
    return {
        "dataset_name": dataset_name,
        "deleted": True,
        "workers": removed_workers,
    }
