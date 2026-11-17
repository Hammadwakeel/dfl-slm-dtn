from datetime import datetime, timezone

from app.db.models import (
    AgentRun,
    AgentRunEvent,
    AutonomyExecutionEvent,
    Device,
)


def _supervisor_run(run_id: str, status: str, timestamp: datetime, **state: object) -> AgentRun:
    return AgentRun(
        id=run_id,
        task="Reach 95.00% mean worker validation accuracy on mnist",
        status=status,
        stage="monitoring" if status == "running" else "finished",
        progress=40,
        selected_task="run_nas" if status == "running" else None,
        selected_reason="NAS should find a stronger model." if status == "running" else None,
        goal={"dataset": "mnist", "target_accuracy": 0.95},
        state={"completed_steps": [], **state},
        created_at=timestamp,
        updated_at=timestamp,
    )


def test_database_assistant_reports_the_active_supervisor_run(client, auth_headers) -> None:
    timestamp = datetime.now(timezone.utc)
    with client.app.state.session_factory.begin() as session:
        session.add(
            _supervisor_run(
                "run-current",
                "running",
                timestamp,
                best={"validation_accuracy": 0.9124, "step": "baseline_training", "group_id": "g-1"},
                active={"step": "run_nas", "group_id": "nas-1", "progress": 0.5},
            )
        )
        session.add(_supervisor_run("run-old", "stopped", timestamp))
        session.flush()
        session.add(
            AgentRunEvent(
                run_id="run-current",
                event_type="decision",
                message="run_nas chosen by slm: NAS should find a stronger model.",
                details={},
                created_at=timestamp,
            )
        )
        session.add(
            AutonomyExecutionEvent(
                action="enable_nas",
                mode="AGENT",
                timestamp=timestamp,
                result_status="executed",
                dataset_name="mnist",
                run_id="nas-1",
                details={"parameters": {"dataset_name": "mnist"}, "evidence": {"source": "supervisor", "run_id": "run-current"}},
            )
        )

    response = client.post(
        "/api/database-assistant/ask",
        headers=auth_headers,
        json={"question": "What is happening right now, and what is the progress?"},
    )

    assert response.status_code == 200
    result = response.json()
    answer = result["answer"]
    assert "Supervisor run run-current: running at stage monitoring (40%)" in answer
    assert "Best mean worker validation accuracy: 91.24%" in answer
    assert "Active experiment: run_nas (group nas-1), 50% of rounds done." in answer
    assert "NAS should find a stronger model." in answer
    assert "Recorded action enable_nas (executed)" in answer
    assert "run-old" not in answer
    assert any(source["record_type"] == "supervisor_run" for source in result["sources"])


def test_database_assistant_history_includes_finished_runs(client, auth_headers) -> None:
    timestamp = datetime.now(timezone.utc)
    with client.app.state.session_factory.begin() as session:
        run = _supervisor_run("run-finished", "completed", timestamp)
        run.result = {"outcome": "objective_achieved", "summary": "Accuracy 0.96 meets the target."}
        session.add(run)

    response = client.post(
        "/api/database-assistant/ask",
        headers=auth_headers,
        json={"question": "What happened in the last agent run?"},
    )

    assert response.status_code == 200
    assert "Outcome: objective_achieved - Accuracy 0.96 meets the target." in response.json()["answer"]


def test_database_assistant_reports_worker_status_without_exposing_credentials(
    client,
    auth_headers,
) -> None:
    with client.app.state.session_factory.begin() as session:
        worker = session.get(Device, "worker1")
        assert worker is not None
        worker.online = True
        worker.api_token = "do-not-return-this-worker-token"
        worker.latest_status = {"cpu_utilization_percent": 37.5}

    response = client.post(
        "/api/database-assistant/ask",
        headers=auth_headers,
        json={"question": "Is worker1 online?"},
    )

    assert response.status_code == 200
    result = response.json()
    assert "worker1 is recorded as online" in result["answer"]
    assert "37.5%" in result["answer"]
    assert "do-not-return-this-worker-token" not in response.text
    assert result["sources"][0]["record_type"] == "device"


def test_database_assistant_declines_questions_without_supported_database_records(
    client,
    auth_headers,
) -> None:
    response = client.post(
        "/api/database-assistant/ask",
        headers=auth_headers,
        json={"question": "Write a poem about distributed learning"},
    )

    assert response.status_code == 200
    assert "only answer from records stored in the coordinator database" in response.json()["answer"]
    assert response.json()["sources"] == []
