import os
from types import SimpleNamespace

import pytest

from app.services import global_evaluation as service


def test_latest_split_paths_use_the_newest_preprocessing_run(tmp_path) -> None:
    for index, name in enumerate(("run-old", "run-new")):
        run = tmp_path / "mnist" / "processed" / name
        run.mkdir(parents=True)
        for split in ("validation", "test"):
            (run / f"{split}.npz").write_bytes(b"x")
        report = run / "preprocessing_report.json"
        report.write_text("{}")
        os.utime(report, ns=(index * 10**9, index * 10**9))

    paths = service.latest_split_paths(SimpleNamespace(root=tmp_path), "mnist")

    assert paths["test"].parent.name == "run-new" and set(paths) == {"validation", "test"}


def test_missing_test_split_is_reported(tmp_path) -> None:
    run = tmp_path / "mnist" / "processed" / "run-a"
    run.mkdir(parents=True)
    (run / "preprocessing_report.json").write_text("{}")
    (run / "validation.npz").write_bytes(b"x")
    with pytest.raises(service.GlobalEvaluationError, match="test"):
        service.latest_split_paths(SimpleNamespace(root=tmp_path), "mnist")


def test_fetch_keeps_only_custom_models_with_weights(monkeypatch) -> None:
    workers = [SimpleNamespace(node_id=f"worker{index}", role="worker") for index in (2, 1, 3)]
    models = {
        "worker1": {"architecture": {"layers": []}, "weights_b64": "AA==", "run_id": "r1", "round": 9},
        "worker2": None,
        "worker3": {"architecture": {"layers": []}, "weights_b64": "AA==", "run_id": "r0", "round": 9},
    }
    monkeypatch.setattr(service, "get_worker_latest_model", lambda worker, name: models[worker.node_id])

    assert [item["node_id"] for item in service.fetch_worker_models(workers, "mnist")] == ["worker1", "worker3"]
    assert [item["node_id"] for item in service.fetch_worker_models(workers, "mnist", run_id="r1")] == ["worker1"]


def test_evaluation_without_an_environment_is_an_error(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("DTN_EVAL_PYTHON", str(tmp_path / "missing-python"))
    with pytest.raises(service.GlobalEvaluationError, match="DTN_EVAL_PYTHON"):
        service.evaluate_newest_models(SimpleNamespace(root=tmp_path), [], "mnist")
