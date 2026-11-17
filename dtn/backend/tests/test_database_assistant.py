from datetime import datetime, timezone

from app.db.models import (
    AgentRun,
    AgentRunEvent,
    AutonomyExecutionEvent,
    AutonomyLoop,
    Device,
)


def test_database_assistant_answers_from_agent_run_records(client, auth_headers) -> None:
    timestamp = datetime.now(timezone.utc)
    with client.app.state.session_factory.begin() as session:
        run = AgentRun(
            id="run-database-fact",
            task="Inspect worker health",
            status="running",
            stage="executing_tool",
            progress=40,
            selected_task="get_status",
            selected_reason="The task requests current worker health.",
            created_at=timestamp,
            updated_at=timestamp,
        )
        session.add(run)
        session.flush()
        session.add(
            AgentRunEvent(
                run_id=run.id,
                event_type="tool_started",
                message="Agent started get_status.",
                details={},
                created_at=timestamp,
            )
        )

    response = client.post(
        "/api/database-assistant/ask",
        headers=auth_headers,
        json={"question": "What is happening with the agent right now?"},
    )

    assert response.status_code == 200
    result = response.json()
    assert "Agent run run-database-fact: running at stage executing_tool (40%)" in result["answer"]
    assert "The task requests current worker health." in result["answer"]
    assert "Agent started get_status." in result["answer"]
    assert any(source["record_id"] == "run-database-fact" for source in result["sources"])


def test_database_assistant_reports_only_active_workflow_progress(
    client,
    auth_headers,
) -> None:
    timestamp = datetime.now(timezone.utc)
    workflow_id = "workflow-current-progress"
    with client.app.state.session_factory.begin() as session:
        session.add(
            AutonomyLoop(
                id="loop-current-progress",
                status="RUNNING",
                max_iterations=100,
                iteration=2,
                time_limit_seconds=86400,
                stop_rule="target_metric",
                target_metric=0.8,
                best_metric=0.6942,
                no_improvement_count=0,
                workflow_id=workflow_id,
                created_at=timestamp,
                deadline_at=timestamp,
                updated_at=timestamp,
            )
        )
        session.add(
            AutonomyExecutionEvent(
                action="start_training",
                mode="AGENT",
                timestamp=timestamp,
                result_status="executed",
                dataset_name="cifar10",
                run_id="cifar-group-current",
                details={
                    "parameters": {
                        "dataset_name": "cifar10",
                        "rounds": 5,
                        "epochs": 10,
                    },
                    "reason": "Retrain the selected NAS architecture.",
                    "evidence": {"workflow_id": workflow_id},
                },
            )
        )
        session.add(
            AutonomyLoop(
                id="loop-old-stopped",
                status="STOPPED",
                max_iterations=100,
                iteration=34,
                time_limit_seconds=86400,
                stop_rule="target_metric",
                target_metric=0.8,
                best_metric=0.236,
                no_improvement_count=0,
                workflow_id="workflow-old",
                created_at=timestamp,
                deadline_at=timestamp,
                updated_at=timestamp,
                failure_reason="Stopped by user.",
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
    assert "loop-current-progress is RUNNING" in answer
    assert "2 experiment(s) recorded" in answer
    assert "Best recorded accuracy: 69.42%" in answer
    assert "Latest recorded workflow action: start_training (executed) for cifar10" in answer
    assert "experiment/group ID cifar-group-current" in answer
    assert "rounds=5, epochs=10" in answer
    assert "does not confirm that its experiment is still running or has completed" in answer
    assert "loop-old-stopped" not in answer
    assert any(source["record_type"] == "autonomy_execution" for source in result["sources"])


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
