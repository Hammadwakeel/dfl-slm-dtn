from datetime import datetime, timezone
from threading import Event
from uuid import UUID, uuid4

from app.db.models import AgentRun
from app.services.agent_runs import open_checkpointer
from app.services.langgraph_agent import run_agent


def test_agent_run_resumes_saved_checkpoint_without_replaying_unconfirmed_tool(
    tmp_path,
) -> None:
    database_url = f"sqlite:///{tmp_path.as_posix()}/checkpoints.sqlite3"
    model_messages = []
    tool_calls = []
    messages = [
        {"role": "system", "content": "Check current state before changing it."},
        {"role": "user", "content": "Start one training run."},
    ]

    with open_checkpointer(database_url) as checkpointer:
        try:
            run_agent(
                messages=messages,
                model_call=lambda _: {
                    "role": "assistant",
                    "content": "I selected training because you requested one run.",
                    "tool_calls": [
                        {
                            "function": {
                                "name": "start_training",
                                "arguments": {"dataset_name": "mnist"},
                            }
                        }
                    ],
                },
                tool_call=lambda name, arguments: (_ for _ in ()).throw(
                    RuntimeError("Simulated coordinator crash during remote action")
                ),
                max_tool_calls=4,
                checkpointer=checkpointer,
                thread_id="run-crash",
                on_update=lambda update, state: None,
            )
        except RuntimeError as error:
            assert "Simulated coordinator crash" in str(error)
        else:
            raise AssertionError("The simulated tool failure should escape the graph")

    with open_checkpointer(database_url) as checkpointer:
        def resume_model(current_messages):
            model_messages.extend(current_messages)
            return {"role": "assistant", "content": "The current state was verified."}

        result = run_agent(
            messages=[{"role": "system", "content": "Use a fresh system snapshot."}],
            model_call=resume_model,
            tool_call=lambda name, arguments: tool_calls.append(name) or {"status": "executed"},
            max_tool_calls=4,
            checkpointer=checkpointer,
            thread_id="run-crash",
            resume=True,
            on_update=lambda update, state: None,
        )

    assert result["reply"] == "The current state was verified."
    assert result["trace"] == []
    assert tool_calls == []
    assert any(
        "outcome is unknown" in message.get("content", "")
        for message in model_messages
        if message.get("role") == "tool"
    )
    assert model_messages[0]["content"] == "Use a fresh system snapshot."


def test_agent_run_stop_persists_checkpoint_and_can_continue(tmp_path) -> None:
    database_url = f"sqlite:///{tmp_path.as_posix()}/paused.sqlite3"
    stop_checks = iter([False, True])
    executed = []
    messages = [{"role": "user", "content": "Check worker status."}]

    with open_checkpointer(database_url) as checkpointer:
        paused = run_agent(
            messages=messages,
            model_call=lambda _: {
                "role": "assistant",
                "content": "I selected status because the task asks for current health.",
                "tool_calls": [{"function": {"name": "get_status", "arguments": {}}}],
            },
            tool_call=lambda name, arguments: executed.append(name) or {"status": "executed"},
            max_tool_calls=4,
            checkpointer=checkpointer,
            thread_id="run-paused",
            stop_requested=lambda: next(stop_checks),
            on_update=lambda update, state: None,
        )

    assert paused["paused"] is True
    assert executed == []

    with open_checkpointer(database_url) as checkpointer:
        resumed = run_agent(
            messages=[{"role": "system", "content": "Fresh snapshot"}],
            model_call=lambda _: {"role": "assistant", "content": "Worker status is healthy."},
            tool_call=lambda name, arguments: executed.append(name) or {"status": "executed"},
            max_tool_calls=4,
            checkpointer=checkpointer,
            thread_id="run-paused",
            resume=True,
            on_update=lambda update, state: None,
        )

    assert resumed["paused"] is False
    assert resumed["reply"] == "Worker status is healthy."
    assert executed == []


def test_agent_run_api_persists_start_stop_and_events(
    client,
    auth_headers: dict[str, str],
    monkeypatch,
) -> None:
    from app.routers import agent_runs

    started = Event()
    release = Event()

    def wait_for_stop(request, run_id, task, *, resume):
        started.set()
        assert release.wait(5)
        with request.app.state.session_factory.begin() as session:
            run = session.get(AgentRun, run_id)
            run.status = "paused"
            run.stage = "paused"
            run.completed_at = datetime.now(timezone.utc)

    monkeypatch.setattr(agent_runs, "_execute_run", wait_for_stop)
    response = client.post(
        "/api/slm/agent/runs",
        json={"task": "Check worker status"},
        headers=auth_headers,
    )
    assert response.status_code == 202
    run_id = response.json()["id"]
    assert UUID(run_id)
    assert started.wait(5)

    stop_response = client.post(
        f"/api/slm/agent/runs/{run_id}/stop",
        headers=auth_headers,
    )
    assert stop_response.status_code == 200
    assert stop_response.json()["status"] == "stop_requested"
    assert any(event["event_type"] == "stop_requested" for event in stop_response.json()["events"])

    release.set()
    detail = client.get(f"/api/slm/agent/runs/{run_id}", headers=auth_headers)
    assert detail.status_code == 200
    assert detail.json()["task"] == "Check worker status"


def test_failed_run_without_checkpoint_can_be_restarted_from_original_task(
    client,
    auth_headers: dict[str, str],
    monkeypatch,
) -> None:
    from app.routers import agent_runs

    run_id = str(uuid4())
    timestamp = datetime.now(timezone.utc)
    with client.app.state.session_factory.begin() as session:
        session.add(
            AgentRun(
                id=run_id,
                task="Check coordinator status",
                status="failed",
                stage="failed",
                progress=0,
                created_at=timestamp,
                updated_at=timestamp,
            )
        )

    started = Event()
    resume_values = []

    def restart_without_checkpoint(request, run_id, task, *, resume):
        resume_values.append(resume)
        started.set()
        with request.app.state.session_factory.begin() as session:
            run = session.get(AgentRun, run_id)
            run.status = "completed"
            run.stage = "completed"

    monkeypatch.setattr(agent_runs, "_execute_run", restart_without_checkpoint)
    detail = client.get(f"/api/slm/agent/runs/{run_id}", headers=auth_headers).json()
    assert detail["can_resume"] is True
    assert detail["checkpoint_available"] is False

    response = client.post(
        f"/api/slm/agent/runs/{run_id}/resume",
        headers=auth_headers,
    )
    assert response.status_code == 202
    assert started.wait(5)
    assert resume_values == [False]


def test_agent_run_records_snapshot_decision_tool_result_and_final_reply(
    client,
    monkeypatch,
) -> None:
    from starlette.requests import Request

    from app.db.models import AgentRunEvent
    from app.routers import agent_runs

    run_id = str(uuid4())
    timestamp = datetime.now(timezone.utc)
    with client.app.state.session_factory.begin() as session:
        session.add(
            AgentRun(
                id=run_id,
                task="Check worker health",
                status="queued",
                stage="queued",
                progress=0,
                created_at=timestamp,
                updated_at=timestamp,
            )
        )

    monkeypatch.setattr(
        agent_runs,
        "_agent_context",
        lambda request, settings, controller: {
            "captured_at": timestamp.isoformat(),
            "nodes": [{"node_id": "worker1", "online": True}],
        },
    )
    model_responses = iter(
        [
            {
                "message": {
                    "role": "assistant",
                    "content": "I selected status to inspect current worker health.",
                    "tool_calls": [
                        {"function": {"name": "get_status", "arguments": {}}}
                    ],
                }
            },
            {"message": {"role": "assistant", "content": "Worker health is online."}},
        ]
    )
    monkeypatch.setattr(
        agent_runs,
        "_forward",
        lambda settings, path, *, payload=None, query="", method=None: next(model_responses),
    )
    client.app.state.autonomy.dispatch = lambda action, **kwargs: {
        "status": "executed",
        "result": {"online": True},
    }

    request = Request({"type": "http", "app": client.app})
    agent_runs._execute_run(request, run_id, "Check worker health", resume=False)

    with client.app.state.session_factory() as session:
        run = session.get(AgentRun, run_id)
        event_types = session.query(AgentRunEvent.event_type).filter_by(run_id=run_id).all()
    assert run.status == "completed"
    assert run.result["reply"] == "Worker health is online."
    assert run.result["trace"][0]["tool"] == "get_status"
    assert {event_type for (event_type,) in event_types} >= {
        "context_snapshot",
        "task_selected",
        "tool_started",
        "tool_result",
        "completed",
    }
