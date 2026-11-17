"""PC-only MNIST central learnability sweep for two model paths and four settings."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf

from mnist_pc_training_check import (
    MODEL_ROOT,
    TRAINING_SEEDS,
    WORKER_ARCHITECTURE,
    _build_dataset_candidate,
    _build_from_initial_architecture,
    _compile,
    _dataset,
    _simulate_worker_schedule,
    _train,
)

ROOT = Path(__file__).resolve().parents[1]
SWEEP_PATH = ROOT / "data" / "mnist_pc_learnability_sweep.json"
CHECKPOINT_EPOCHS = (5, 20, 50, 100)
SETTINGS = (
    {"name": "lr_0.001_batch_128", "learning_rate": 0.001, "batch_size": 128},
    {"name": "lr_0.001_batch_32", "learning_rate": 0.001, "batch_size": 32},
    {"name": "lr_0.01_batch_32", "learning_rate": 0.01, "batch_size": 32},
    {"name": "lr_0.003_batch_64", "learning_rate": 0.003, "batch_size": 64},
)
MODEL_NAMES = (
    "initial_architecture_json_with_installed_weights",
    "worker_builder_b1_c8_k3_fresh_weights",
)
FEDERATED_SETTING = {"name": "lr_0.01_batch_32", "learning_rate": 0.01, "batch_size": 32}


def _save(results: dict[str, Any]) -> None:
    SWEEP_PATH.parent.mkdir(parents=True, exist_ok=True)
    SWEEP_PATH.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _run_federated_only() -> None:
    if not SWEEP_PATH.is_file():
        raise FileNotFoundError(f"Saved central sweep results are missing: {SWEEP_PATH}")
    results = json.loads(SWEEP_PATH.read_text(encoding="utf-8"))
    for model_name in MODEL_NAMES:
        settings = results.get("models", {}).get(model_name, {})
        for setting in SETTINGS:
            runs = settings.get(setting["name"])
            if not isinstance(runs, list) or len(runs) != 3:
                raise RuntimeError(
                    f"Saved central sweep is incomplete for {model_name}/{setting['name']}"
                )
    print(
        f"loaded_saved_central_results={SWEEP_PATH}; "
        "centralized_sweep_rerun=False"
    )
    _, _, _, _, per_worker = _dataset()
    architecture_path = MODEL_ROOT / "architecture.json"
    weights_path = MODEL_ROOT / "initial_weights.npz"
    architecture = json.loads(architecture_path.read_text(encoding="utf-8"))
    with np.load(weights_path, allow_pickle=False) as archive:
        initial_weights = [
            np.asarray(archive[key])
            for key in (
                "conv1_kernel",
                "conv1_bias",
                "conv2_kernel",
                "conv2_bias",
                "classifier_kernel",
                "classifier_bias",
            )
        ]
    variants = ((5, 2), (5, 10), (10, 5), (5, 20))
    outputs = []
    for model_kind in ("initial_model_json", "worker_builder"):
        for rounds, local_epochs in variants:
            result = _simulate_worker_schedule(
                per_worker,
                seed=2026,
                rounds=rounds,
                local_epochs=local_epochs,
                learning_rate=FEDERATED_SETTING["learning_rate"],
                batch_size=FEDERATED_SETTING["batch_size"],
                model_kind=model_kind,
                initial_architecture=(architecture if model_kind == "initial_model_json" else None),
                initial_weights=(initial_weights if model_kind == "initial_model_json" else None),
            )
            result["steps_per_worker_total"] = int(np.ceil(640 / FEDERATED_SETTING["batch_size"])) * rounds * local_epochs
            result["setting"] = dict(FEDERATED_SETTING)
            outputs.append(result)
            print(
                f"federated model={model_kind}; schedule={rounds}x{local_epochs}; "
                f"steps_per_worker={result['steps_per_worker_total']}; "
                f"mean_final_accuracy={result['mean_final_accuracy']:.4f}; "
                f"mean_final_macro_f1={result['mean_final_macro_f1']:.4f}; "
                f"per_worker={json.dumps(result['final_worker_metrics'], sort_keys=True)}"
            )
    results["federated_schedule_comparison"] = outputs
    results["federated_setting"] = dict(FEDERATED_SETTING)
    results["status"] = "PASS"
    _save(results)
    print(f"federated_sweep_json={SWEEP_PATH}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--federated-only", action="store_true")
    arguments = parser.parse_args()
    if arguments.federated_only:
        _run_federated_only()
        return
    train_images, train_labels, validation_images, validation_labels, per_worker = _dataset()
    architecture_path = MODEL_ROOT / "architecture.json"
    weights_path = MODEL_ROOT / "initial_weights.npz"
    if not architecture_path.is_file() or not weights_path.is_file():
        raise FileNotFoundError("PC initial-model architecture or weights are missing")
    architecture = json.loads(architecture_path.read_text(encoding="utf-8"))
    if architecture.get("input_shape") != [28, 28, 1] or architecture.get("num_classes") != 10:
        raise RuntimeError("Initial model JSON does not describe 1x28x28 MNIST")
    with np.load(weights_path, allow_pickle=False) as archive:
        initial_weights = [
            np.asarray(archive[key])
            for key in (
                "conv1_kernel",
                "conv1_bias",
                "conv2_kernel",
                "conv2_bias",
                "classifier_kernel",
                "classifier_bias",
            )
        ]

    results: dict[str, Any] = {
        "dataset_name": "mnist_demo",
        "train_samples": int(train_labels.size),
        "validation_samples": int(validation_labels.size),
        "training_seeds": list(TRAINING_SEEDS),
        "epochs": 100,
        "checkpoints": list(CHECKPOINT_EPOCHS),
        "models": {name: {} for name in MODEL_NAMES},
        "federated_schedules": [],
        "status": "RUNNING",
    }
    _save(results)
    for setting in SETTINGS:
        setting_name = setting["name"]
        learning_rate = float(setting["learning_rate"])
        batch_size = int(setting["batch_size"])
        central_steps = int(np.ceil(train_labels.size / batch_size))
        local_steps = int(np.ceil(640 / batch_size))
        print(
            f"setting={setting_name}; learning_rate={learning_rate}; batch_size={batch_size}; "
            f"central_steps_per_epoch={central_steps}; worker_steps_per_epoch={local_steps}"
        )
        for model_name in MODEL_NAMES:
            runs = []
            for seed in TRAINING_SEEDS:
                tf.keras.backend.clear_session()
                tf.keras.utils.set_random_seed(seed)
                if model_name == MODEL_NAMES[0]:
                    model = _build_from_initial_architecture(architecture)
                    model(np.zeros((1, 28, 28, 1), dtype=np.float32), training=False)
                    model.set_weights([np.array(weight, copy=True) for weight in initial_weights])
                else:
                    model = _build_dataset_candidate(
                        input_shape=(28, 28, 1),
                        num_classes=10,
                        architecture=WORKER_ARCHITECTURE,
                        seed=seed,
                    )
                run = _train(
                    model_name,
                    seed,
                    model,
                    train_images,
                    train_labels,
                    validation_images,
                    validation_labels,
                    epochs=100,
                    learning_rate=learning_rate,
                    batch_size=batch_size,
                    report_epochs=CHECKPOINT_EPOCHS,
                )
                run["setting"] = dict(setting)
                run["central_steps_per_epoch"] = central_steps
                run["worker_steps_per_epoch"] = local_steps
                runs.append(run)
                results["models"][model_name][setting_name] = runs
                _save(results)
                del model
                tf.keras.backend.clear_session()
    for model_name in MODEL_NAMES:
        for setting in SETTINGS:
            runs = results["models"][model_name][setting["name"]]
            means = {
                epoch: float(
                    np.mean(
                        [
                            next(row for row in run["epochs"] if row["epoch"] == epoch)["validation_accuracy"]
                            for run in runs
                        ]
                    )
                )
                for epoch in CHECKPOINT_EPOCHS
            }
            macro_f1_mean = float(np.mean([run["final_macro_f1"] for run in runs]))
            print(
                f"summary model={model_name}; setting={setting['name']}; "
                f"mean_val_accuracy={json.dumps(means, sort_keys=True)}; "
                f"mean_final_macro_f1={macro_f1_mean:.4f}"
            )

    worker_results = results["models"][MODEL_NAMES[1]]
    best_setting_name = max(
        worker_results,
        key=lambda setting_name: float(
            np.mean(
                [
                    run["epochs"][-1]["validation_accuracy"]
                    for run in worker_results[setting_name]
                ]
            )
        ),
    )
    best_runs = worker_results[best_setting_name]
    best_config = best_runs[0]["setting"]
    best_mean = float(
        np.mean([run["epochs"][-1]["validation_accuracy"] for run in best_runs])
    )
    print(
        f"best_worker_builder_setting={best_setting_name}; "
        f"mean_epoch_100_val_accuracy={best_mean:.4f}"
    )
    schedule_variants = ((5, 2), (5, 10), (10, 5), (5, 20))
    for rounds, local_epochs in schedule_variants:
        schedule = _simulate_worker_schedule(
            per_worker,
            seed=2026,
            rounds=rounds,
            local_epochs=local_epochs,
            learning_rate=float(best_config["learning_rate"]),
            batch_size=int(best_config["batch_size"]),
        )
        schedule["steps_per_epoch_per_worker"] = int(np.ceil(640 / int(best_config["batch_size"])))
        schedule["steps_per_worker_total"] = (
            schedule["steps_per_epoch_per_worker"] * rounds * local_epochs
        )
        schedule["best_central_worker_builder_setting"] = best_config
        results["federated_schedules"].append(schedule)
        _save(results)
        print(
            f"federated_schedule={rounds}x{local_epochs}; "
            f"steps_per_worker={schedule['steps_per_worker_total']}; "
            f"mean_final_accuracy={schedule['mean_final_accuracy']:.4f}; "
            f"per_worker={json.dumps(schedule['final_worker_metrics'], sort_keys=True)}"
        )
    results["status"] = "PASS"
    _save(results)
    print(f"learnability_sweep_json={SWEEP_PATH}")


if __name__ == "__main__":
    main()
