import hashlib
import io
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.services import dataset_pipeline
from app.services.datasets import DatasetStore
from app.services.preprocessing import PreprocessingError


def _write_validation(path: Path, labels: np.ndarray) -> None:
    np.savez_compressed(
        path,
        images=np.arange(labels.size * 4, dtype=np.float32).reshape(labels.size, 2, 2, 1),
        labels=labels,
        class_names=np.array(["a", "b", "c"]),
    )


def _workers(*node_ids: str):
    return [SimpleNamespace(node_id=node_id, role="worker") for node_id in node_ids]


def test_balanced_subset_takes_the_same_count_per_class_deterministically(tmp_path) -> None:
    path = tmp_path / "validation.npz"
    _write_validation(path, np.array([0] * 50 + [1] * 8 + [2] * 30, dtype=np.int64))

    first = dataset_pipeline.balanced_validation_subset(path, per_class=10)
    second = dataset_pipeline.balanced_validation_subset(path, per_class=10)

    assert first == second
    with np.load(io.BytesIO(first), allow_pickle=False) as subset:
        # The rarest class (8 samples) caps every class so the subset stays balanced.
        assert np.bincount(subset["labels"]).tolist() == [8, 8, 8]
        assert subset["class_names"].tolist() == ["a", "b", "c"]


def test_balanced_subset_rejects_a_split_missing_a_class(tmp_path) -> None:
    path = tmp_path / "validation.npz"
    _write_validation(path, np.array([0, 0, 1, 1], dtype=np.int64))
    with pytest.raises(PreprocessingError, match="missing at least one class"):
        dataset_pipeline.balanced_validation_subset(path)


def _make_run(root: Path, name: str, uploaded: dict[str, str], mtime: int) -> Path:
    run = root / "mnist" / "processed" / name
    run.mkdir(parents=True)
    _write_validation(run / "validation.npz", np.tile(np.arange(3, dtype=np.int64), 5))
    report = run / "preprocessing_report.json"
    report.write_text(
        json.dumps({"uploaded_workers": {node: {"shard_sha256": sha} for node, sha in uploaded.items()}}),
        encoding="utf-8",
    )
    os.utime(report, ns=(mtime, mtime))
    return run


def test_ship_uses_the_run_whose_shards_are_installed(tmp_path, monkeypatch) -> None:
    root = tmp_path / "datasets"
    installed_run = _make_run(root, "run-old", {"worker1": "s1", "worker2": "s2"}, 1_000)
    _make_run(root, "run-new", {"worker1": "x1", "worker2": "x2"}, 2_000)
    monkeypatch.setattr(
        dataset_pipeline,
        "get_worker_dataset_metadata",
        lambda worker, name: {"shard_sha256": {"worker1": "s1", "worker2": "s2"}[worker.node_id]},
    )
    uploads = []

    def fake_upload(worker, dataset_name, payload):
        uploads.append((worker.node_id, dataset_name, payload))
        return {"node_id": worker.node_id, "rows": 15, "sha256": hashlib.sha256(payload).hexdigest()}

    monkeypatch.setattr(dataset_pipeline, "upload_worker_global_validation", fake_upload)

    result = dataset_pipeline.ship_global_validation(
        DatasetStore(root), _workers("worker2", "worker1"), "mnist"
    )

    assert result["preprocessing_run"] == installed_run.name
    assert [node for node, *_ in uploads] == ["worker1", "worker2"]
    assert uploads[0][2] == uploads[1][2]
    assert result["sha256"] == hashlib.sha256(uploads[0][2]).hexdigest()


def test_ship_fails_when_no_run_matches_the_installed_shards(tmp_path, monkeypatch) -> None:
    root = tmp_path / "datasets"
    _make_run(root, "run-a", {"worker1": "other"}, 1_000)
    monkeypatch.setattr(dataset_pipeline, "get_worker_dataset_metadata", lambda worker, name: {"shard_sha256": "s1"})
    with pytest.raises(PreprocessingError, match="matches the shards installed"):
        dataset_pipeline.ship_global_validation(DatasetStore(root), _workers("worker1"), "mnist")


def test_global_validation_endpoint_is_authenticated_and_delegates(
    client: TestClient,
    auth_headers: dict[str, str],
    monkeypatch,
) -> None:
    from app.routers import datasets

    calls = []

    def fake_ship(store, workers, dataset_name):
        calls.append((store, workers, dataset_name))
        return {"dataset_name": dataset_name, "rows": 2000}

    monkeypatch.setattr(datasets, "ship_global_validation", fake_ship)
    assert client.post("/api/datasets/mnist/global-validation").status_code == 401

    response = client.post("/api/datasets/mnist/global-validation", headers=auth_headers)

    assert response.status_code == 200
    assert response.json() == {"dataset_name": "mnist", "rows": 2000}
    assert calls[0][0] is client.app.state.datasets
    assert calls[0][2] == "mnist"
