import hashlib
import json
from pathlib import Path

from fastapi.testclient import TestClient

from app.services.datasets import DatasetStore


def _write_metadata(root: Path, dataset_name: str) -> None:
    dataset_path = root / dataset_name
    dataset_path.mkdir(parents=True)
    (dataset_path / "metadata.json").write_text(
        json.dumps(
            {
                "dataset_name": dataset_name,
                "classes": [str(index) for index in range(10)],
                "class_counts": {str(index): 1 for index in range(10)},
                "image_sizes": {str(index): {"28x28": 1} for index in range(10)},
                "image_channels": {str(index): {"1": 1} for index in range(10)},
                "image_count": 10,
                "expanded_image_bytes": 100,
            }
        ),
        encoding="utf-8",
    )


def test_dataset_catalog_is_limited_to_official_mnist_and_cifar10(
    client: TestClient,
    auth_headers: dict[str, str],
    tmp_path: Path,
) -> None:
    root = tmp_path / "datasets"
    _write_metadata(root, "mnist")
    _write_metadata(root, "cifar10")
    _write_metadata(root, "another-dataset")
    client.app.state.datasets = DatasetStore(root)

    response = client.get("/api/datasets", headers=auth_headers)

    assert response.status_code == 200
    assert [item["dataset_name"] for item in response.json()] == ["cifar10", "mnist"]
    assert client.get("/api/datasets/mnist", headers=auth_headers).status_code == 200
    assert client.get("/api/datasets/cifar10", headers=auth_headers).status_code == 200
    assert client.get("/api/datasets/another-dataset", headers=auth_headers).status_code == 404


def test_dataset_samples_are_authenticated_and_return_preview_data(
    client: TestClient,
    auth_headers: dict[str, str],
    monkeypatch,
) -> None:
    from app.routers import datasets

    monkeypatch.setattr(
        datasets,
        "get_dataset_samples",
        lambda _store, dataset_name: [{"label": dataset_name, "image_data": "data:image/png;base64,test"}],
    )

    assert client.get("/api/datasets/cifar10/samples").status_code == 401
    response = client.get("/api/datasets/cifar10/samples", headers=auth_headers)

    assert response.status_code == 200
    assert response.json() == [{"label": "cifar10", "image_data": "data:image/png;base64,test"}]
    assert client.get("/api/datasets/other/samples", headers=auth_headers).status_code == 404


def test_initial_model_delete_is_authenticated_and_delegates_to_all_workers(
    client: TestClient,
    auth_headers: dict[str, str],
    monkeypatch,
) -> None:
    from app.routers import datasets

    calls = []

    def fake_delete(store, workers, dataset_name):
        calls.append((store, workers, dataset_name))

    monkeypatch.setattr(datasets, "delete_initial_model", fake_delete)
    assert client.delete("/api/datasets/mnist/initial-model").status_code == 401

    response = client.delete(
        "/api/datasets/mnist/initial-model",
        headers=auth_headers,
    )

    assert response.status_code == 204
    assert calls[0][0] is client.app.state.datasets
    assert calls[0][1] == client.app.state.settings.nodes
    assert calls[0][2] == "mnist"


def test_dataset_upload_routes_are_not_available(
    client: TestClient,
    auth_headers: dict[str, str],
) -> None:
    assert client.post("/api/datasets/upload", headers=auth_headers).status_code == 405
    assert client.post("/api/datasets/upload-folder", headers=auth_headers).status_code == 405


def test_dataset_endpoints_require_coordinator_authentication(client: TestClient) -> None:
    assert client.get("/api/datasets").status_code == 401
    assert client.get("/api/datasets/mnist").status_code == 401
    assert client.get("/api/datasets/cifar10").status_code == 401
    assert client.get("/api/datasets/another-dataset").status_code == 401


def test_latest_mnist_preprocessing_report_returns_worker4_shard_hash(
    client: TestClient,
    auth_headers: dict[str, str],
    tmp_path: Path,
) -> None:
    root = tmp_path / "datasets"
    _write_metadata(root, "mnist")
    report_path = root / "mnist" / "processed" / "run-latest" / "preprocessing_report.json"
    report_path.parent.mkdir(parents=True)
    shard_path = report_path.parent / "workers" / "worker4.npz"
    shard_path.parent.mkdir()
    shard_path.write_bytes(b"worker-shard")
    report_path.write_text(
        json.dumps(
            {
                "dataset_name": "mnist",
                "output_path": str(report_path.parent),
                "split_sizes": {"train": 2},
                "workers": {"worker4": {"rows": 2, "class_counts": {"0": 2}}},
            }
        ),
        encoding="utf-8",
    )
    client.app.state.datasets = DatasetStore(root)

    response = client.get("/api/datasets/mnist/preprocessing-report", headers=auth_headers)

    assert response.status_code == 200
    assert response.json()["dataset_name"] == "mnist"
    assert response.json()["workers"]["worker4"]["pc_shard_sha256"] == hashlib.sha256(
        b"worker-shard"
    ).hexdigest()
    assert client.get("/api/datasets/another-dataset/preprocessing-report", headers=auth_headers).status_code == 404
    assert client.get("/api/datasets/mnist/preprocessing-report").status_code == 401


def test_initial_model_endpoint_returns_the_verified_mnist_manifest(
    client: TestClient,
    auth_headers: dict[str, str],
    tmp_path: Path,
) -> None:
    root = tmp_path / "datasets"
    _write_metadata(root, "mnist")
    client.app.state.datasets = DatasetStore(root)
    assert client.get("/api/datasets/mnist/initial-model", headers=auth_headers).status_code == 404

    model_path = root / "mnist" / "models" / "initial_model"
    model_path.mkdir(parents=True)
    architecture_bytes = json.dumps(
        {
            "dataset_name": "mnist",
            "design": {"id": "b1_c16_k5", "blocks": 1, "channels": 16, "kernel_size": 5},
            "input_shape": [28, 28, 1],
            "classes": [str(index) for index in range(10)],
            "num_classes": 10,
            "parameter_count": 586,
        },
        sort_keys=True,
    ).encode("utf-8")
    weights_bytes = b"initial weights"
    (model_path / "architecture.json").write_bytes(architecture_bytes)
    (model_path / "initial_weights.npz").write_bytes(weights_bytes)
    (model_path / "metadata.json").write_text(
        json.dumps(
            {
                "dataset_name": "mnist",
                "parameter_count": 586,
                "architecture_sha256": hashlib.sha256(architecture_bytes).hexdigest(),
                "weights_sha256": hashlib.sha256(weights_bytes).hexdigest(),
            }
        ),
        encoding="utf-8",
    )

    response = client.get("/api/datasets/mnist/initial-model", headers=auth_headers)

    assert response.status_code == 200
    assert response.json()["architecture"]["design"]["id"] == "b1_c16_k5"
    assert response.json()["parameter_count"] == 586
    assert client.get("/api/datasets/another-dataset/initial-model", headers=auth_headers).status_code == 404


def test_model_creation_action_dispatches_the_chosen_architecture(
    client: TestClient,
    auth_headers: dict[str, str],
    monkeypatch,
) -> None:
    import app.main

    calls = []

    def create_model(
        dataset_store,
        workers,
        dataset_name,
        *,
        seed,
        architecture="b2_c8_k3",
        replace_existing=False,
        restore_baseline=False,
    ):
        calls.append((dataset_store, workers, dataset_name, seed, architecture))
        return {"dataset_name": dataset_name, "architecture": architecture}

    monkeypatch.setattr(app.main, "create_and_distribute_initial_model", create_model)
    monkeypatch.setattr(
        "app.routers.slm._forward",
        lambda *_args, **_kwargs: {"recorded": True},
    )
    architecture = {"id": "b1_c16_k5", "blocks": 1, "channels": 16, "kernel_size": 5}
    result = client.post(
        "/api/slm/command",
        json={
            "action": {
                "action": "create_initial_model",
                "parameters": {
                    "model_name": "mnist",
                    "architecture": architecture,
                    "seed": 42,
                },
            },
            "reason": "Build the user-designed MNIST model",
            "evidence": {"architecture": architecture},
        },
        headers=auth_headers,
    )
    assert result.json()["status"] == "executed"
    assert result.json()["result"]["data"] == {
        "dataset_name": "mnist",
        "architecture": architecture,
    }
    assert calls[0][2:] == ("mnist", 42, architecture)

    default_model = client.post(
        "/api/slm/command",
        json={
            "action": {
                "action": "create_initial_model",
                "parameters": {"model_name": "cifar10", "architecture": {}},
            },
            "reason": "Build the standard CIFAR-10 model",
        },
        headers=auth_headers,
    )
    assert default_model.status_code == 200
    assert calls[1][2:] == ("cifar10", 0, "b2_c8_k3")


def test_preprocess_action_dispatches_immediately_and_uses_mnist_parameters(
    client: TestClient,
    auth_headers: dict[str, str],
    monkeypatch,
) -> None:
    import app.main

    calls = []

    def preprocess(dataset_store, workers, dataset_name, **options):
        calls.append((dataset_store, workers, dataset_name, options))
        return {"dataset_name": dataset_name, "uploaded_workers": {}}

    monkeypatch.setattr(app.main, "preprocess_and_ship", preprocess)
    monkeypatch.setattr(
        "app.routers.slm._forward",
        lambda *_args, **_kwargs: {"recorded": True},
    )
    result = client.post(
        "/api/slm/command",
        json={
            "action": {
                "action": "preprocess_dataset",
                "parameters": {
                    "dataset_name": "mnist",
                    "normalization": "minmax",
                    "dataset_split": "dirichlet_a05",
                    "target_size": {"height": 28, "width": 28},
                    "replace_existing": True,
                },
            },
            "reason": "Preprocess the official MNIST dataset",
            "evidence": {"dataset_name": "mnist"},
        },
        headers=auth_headers,
    )
    assert result.json()["status"] == "executed"
    assert result.json()["result"]["data"] == {
        "dataset_name": "mnist",
        "uploaded_workers": {},
    }
    assert calls[0][2:] == (
        "mnist",
        {
            "normalization": "minmax",
            "split_strategy": "dirichlet_a05",
            "target_size": (28, 28),
            "seed": 42,
            "replace_existing": True,
        },
    )


def test_cifar_preprocessing_success_is_not_reported_as_failed_when_master_history_sync_is_rejected(
    client: TestClient,
    auth_headers: dict[str, str],
    monkeypatch,
) -> None:
    from fastapi import HTTPException

    import app.main
    from app.routers import slm

    calls = []

    def preprocess(dataset_store, workers, dataset_name, **options):
        calls.append((dataset_name, options))
        return {"dataset_name": dataset_name, "uploaded_workers": {"worker1": {}}}

    def reject_history(*_args, **_kwargs):
        raise HTTPException(status_code=422, detail="Master schema is outdated")

    monkeypatch.setattr(app.main, "preprocess_and_ship", preprocess)
    monkeypatch.setattr(
        app.main,
        "cluster_nodes",
        lambda _settings, _sessions: [],
    )
    monkeypatch.setattr(slm, "_forward", reject_history)

    response = client.post(
        "/api/autonomy/dispatch",
        json={
            "action": {
                "action": "preprocess_dataset",
                "parameters": {
                    "dataset_name": "cifar10",
                    "normalization": "standard",
                    "dataset_split": "stratified",
                    "target_size": {"height": 32, "width": 32},
                    "seed": 42,
                    "replace_existing": True,
                },
            },
            "reason": "Preprocess and ship official CIFAR-10",
            "evidence": {"dataset_name": "cifar10"},
        },
        headers=auth_headers,
    )

    assert response.status_code == 200
    assert response.json()["status"] == "executed"
    assert response.json()["result"]["data"]["dataset_name"] == "cifar10"
    assert "Do not repeat the action" in response.json()["history_warning"]
    assert calls == [
        (
            "cifar10",
            {
                "normalization": "standard",
                "split_strategy": "stratified",
                "target_size": (32, 32),
                "seed": 42,
                "replace_existing": True,
            },
        )
    ]


def test_slm_rejects_removed_dataset_upload_action(
    client: TestClient,
    auth_headers: dict[str, str],
) -> None:
    response = client.post(
        "/api/slm/command",
        json={
            "action": {
                "action": "upload_dataset",
                "parameters": {"dataset_name": "invented-dataset"},
            },
            "reason": "Upload a dataset",
            "evidence": {},
        },
        headers=auth_headers,
    )
    assert response.status_code == 422
