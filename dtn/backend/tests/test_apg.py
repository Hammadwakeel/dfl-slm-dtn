import json
import threading
import time
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

from app.services.apg import NODES, build_verdict


def _edge_rows(topology: str, round_number: int) -> list[list[str]]:
    if topology == "mesh":
        return [[receiver, source] for receiver in NODES for source in NODES if receiver != source]
    if topology == "ring":
        return [
            [receiver, NODES[(index - 1) % len(NODES)]]
            for index, receiver in enumerate(NODES)
        ]
    edges = [[receiver, source] for receiver in NODES for source in NODES if receiver != source]
    if round_number >= 2:
        edges.remove(["node_0", "node_1"])
        edges.remove(["node_1", "node_0"])
    return edges


def _run(topology: str, seed: int, accuracy: float) -> dict[str, Any]:
    rounds = []
    for round_number in range(1, 4):
        matrix = [
            [None if row == column else float(row + column + round_number) for column in range(8)]
            for row in range(8)
        ]
        rounds.append(
            {
                "round": round_number,
                "source_label": "SIMULATED",
                "mean_validation_accuracy": 0.5 + 0.1 * round_number if topology == "apg" else 0.5,
                "mean_validation_macro_f1": 0.4,
                "edges_used": _edge_rows(topology, round_number),
                "edges_changed_vs_previous_round": None if round_number == 1 else (2 if topology == "apg" and round_number == 2 else 0),
                "per_node": {},
                "apg_inputs": {
                    "loss_matrix_normalized": matrix,
                    "wait_matrix_normalized": matrix,
                    "theta_train_only": [[float(round_number)] * 10 for _ in NODES],
                },
            }
        )
    return {
        "run_id": f"{topology}-{seed}",
        "simulation_mode": "all_simulated",
        "source_label": "SIMULATED",
        "data_setting": "iid_stratified",
        "topology": topology,
        "seed": seed,
        "rounds": rounds,
        "final_test": {"summary": {"accuracy": accuracy}},
    }


def _successful_runner(**kwargs: Any) -> dict[str, Any]:
    job = kwargs["job"]
    progress = kwargs["progress_callback"]
    for round_number in range(1, 31):
        if kwargs["should_stop"]():
            raise InterruptedError("Global STOP requested")
        progress(round_number, 30)
    run = _run(job.topology, job.seed, 0.8)
    run["run_id"] = job.job_id
    run["data_setting"] = job.data_setting
    run["rounds"] = run["rounds"] * 10
    run["final_test"] = {
        "split": "TEST",
        "source": "SIMULATED",
        "summary": {
            "accuracy": 0.8,
            "macro_f1": 0.7,
            "macro_precision": 0.7,
            "macro_recall": 0.7,
        },
        "confusion_matrix_counts": [[0] * 10 for _ in range(10)],
        "confusion_matrix_row_normalized_percent": [[0.0] * 10 for _ in range(10)],
    }
    return run


def _wait_for_job(client: TestClient, headers: dict[str, str], job_id: str) -> dict[str, Any]:
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        result = client.get(f"/api/apg/jobs/{job_id}", headers=headers).json()
        if result["status"] in {"finished", "failed", "stopped"}:
            return result
        time.sleep(0.01)
    raise AssertionError("APG job did not reach a terminal state")


def test_apg_result_routes_require_auth_and_return_saved_file_data(
    client: TestClient,
    auth_headers: dict[str, str],
    tmp_path: Path,
) -> None:
    manager = client.app.state.apg_jobs
    manager.results_root = tmp_path
    (tmp_path / "final_metrics.json").write_text(json.dumps({"runs": [_run("apg", 17, 0.8)]}))

    assert client.get("/api/apg/runs").status_code == 401
    response = client.get("/api/apg/runs", headers=auth_headers)
    assert response.status_code == 200
    assert response.json()["runs"][0]["source_label"] == "SIMULATED"
    assert response.json()["runs"][0]["rounds"][0]["split"] == "VALIDATION"

    detail = client.get("/api/apg/runs/apg-17", headers=auth_headers)
    assert detail.status_code == 200
    assert detail.json()["run"]["rounds"][0]["inputs"]["inputs_available"] is True


def test_empty_apg_results_do_not_report_learning_verdicts(
    client: TestClient,
    auth_headers: dict[str, str],
    tmp_path: Path,
) -> None:
    client.app.state.apg_jobs.results_root = tmp_path
    (tmp_path / "final_metrics.json").write_text(json.dumps({"runs": []}), encoding="utf-8")

    response = client.get("/api/apg/runs", headers=auth_headers)

    assert response.status_code == 200
    assert response.json()["runs"] == []
    assert response.json()["verdicts"] == []


def test_start_job_uses_unique_file_and_does_not_overwrite_matrix(
    client: TestClient,
    auth_headers: dict[str, str],
    tmp_path: Path,
) -> None:
    manager = client.app.state.apg_jobs
    manager.results_root = tmp_path
    matrix_file = tmp_path / "final_metrics.json"
    original = '{"runs":[],"sentinel":"keep"}'
    matrix_file.write_text(original, encoding="utf-8")
    manager.runner = _successful_runner

    response = client.post(
        "/api/apg/jobs",
        headers=auth_headers,
        json={"topology": "apg", "data_setting": "iid_stratified", "seed": 17},
    )
    assert response.status_code == 202
    job = _wait_for_job(client, auth_headers, response.json()["job_id"])
    assert job["status"] == "finished"
    assert job["rounds_completed"] == 30
    assert (tmp_path / "ui_runs" / f"{job['job_id']}.json").exists()
    assert matrix_file.read_text(encoding="utf-8") == original
    with client.app.state.session_factory() as session:
        from app.db.models import AutonomyExecutionEvent
        from sqlalchemy import select

        audit = list(session.scalars(select(AutonomyExecutionEvent)).all())
    assert any(event.action == "run_apg_simulation" and event.mode == "FULL_ACCESS" for event in audit)


def test_global_stop_blocks_start_and_stops_active_training(
    client: TestClient,
    auth_headers: dict[str, str],
    tmp_path: Path,
) -> None:
    manager = client.app.state.apg_jobs
    manager.results_root = tmp_path
    manager.runner = _successful_runner

    client.post("/api/autonomy/stop", headers=auth_headers)
    blocked = client.post(
        "/api/apg/jobs",
        headers=auth_headers,
        json={"topology": "apg", "data_setting": "iid_stratified", "seed": 17},
    )
    assert blocked.status_code == 423
    assert manager.list_jobs() == []

    client.post("/api/autonomy/reset-stop", headers=auth_headers)
    entered = threading.Event()

    def stoppable_runner(**kwargs: Any) -> dict[str, Any]:
        entered.set()
        while not kwargs["should_stop"]():
            time.sleep(0.005)
        raise InterruptedError("Global STOP requested")

    manager.runner = stoppable_runner
    started = client.post(
        "/api/apg/jobs",
        headers=auth_headers,
        json={"topology": "apg", "data_setting": "iid_stratified", "seed": 42},
    )
    assert started.status_code == 202
    assert entered.wait(1)
    client.post("/api/autonomy/stop", headers=auth_headers)
    job = _wait_for_job(client, auth_headers, started.json()["job_id"])
    assert job["status"] == "stopped"
    assert not (tmp_path / "ui_runs" / f"{job['job_id']}.json").exists()


def test_verdict_rules_require_adaptation_learning_and_three_test_seeds() -> None:
    matched = [
        _run(topology, seed, accuracy)
        for seed in (17, 42, 2026)
        for topology, accuracy in (("apg", 0.9), ("mesh", 0.7), ("ring", 0.8))
    ]
    verdict = build_verdict(matched, "iid_stratified", "all_simulated")
    assert verdict["verdict"] == "Topology is adapting and improves over ring/mesh (3 seeds)"
    assert verdict["rules"]["topology_adapts"] is True
    assert verdict["rules"]["topology_follows_inputs"] is True
    assert verdict["rules"]["model_learns"] is True
    assert verdict["rules"]["benefit_over_best_fixed_topology"] is True
    assert verdict["rules"]["test_split"] == "TEST"
    assert verdict["rules"]["validation_split"] == "VALIDATION"

    one_seed = [run for run in matched if run["seed"] == 17]
    insufficient = build_verdict(one_seed, "iid_stratified", "all_simulated")
    assert insufficient["verdict"] == "Not enough data (run more seeds)"
    assert insufficient["rules"]["benefit_over_best_fixed_topology"] is None

    non_adapting = [
        _run("mesh", seed, 0.8)
        for seed in (17, 42, 2026)
    ]
    no_adaptation = build_verdict(non_adapting, "iid_stratified", "all_simulated")
    assert no_adaptation["verdict"] == "Topology is not adapting (same graph every round)"
