from datetime import datetime, timedelta, timezone
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from jsonschema import ValidationError

from app.db.models import (
    AutonomyDecision,
    AutonomyExecutionEvent,
    AutonomyLoop,
    Base,
    SlmConversation,
    SlmMessage,
)
from app.db.session import create_db_engine, create_session_factory
from app.services.autonomy import ACTION_VALIDATOR, AutonomyController, AutonomyError
from common.agent_tools import agent_tool_action
from common.dataset_nas import build_dataset_architecture


def _set_rounds(rounds: int = 5) -> dict[str, object]:
    return {"action": "set_rounds", "parameters": {"rounds": rounds}}


def test_model_actions_accept_creation_design_and_installed_manifest() -> None:
    design = {
        "model_type": "custom",
        "layers": [
            {"type": "patch_embedding", "patch_size": 1, "embed_dim": 32},
            {
                "type": "transformer_encoder",
                "embed_dim": 32,
                "num_heads": 4,
                "ff_dim": 64,
                "dropout": 0.1,
                "activation": "gelu",
            },
            {"type": "global_average_pooling1d"},
            {"type": "dense", "units": 10, "activation": "linear"},
        ],
        "training": {"optimizer": "adamw", "learning_rate": 0.0005, "batch_size": 64},
    }
    for dataset_name in ("mnist", "cifar10"):
        ACTION_VALIDATOR.validate(
            {
                "action": "create_initial_model",
                "parameters": {"model_name": dataset_name},
            }
        )
        ACTION_VALIDATOR.validate(
            {
                "action": "create_initial_model",
                "parameters": {"model_name": dataset_name, "architecture": {}},
            }
        )
        ACTION_VALIDATOR.validate(
            {
                "action": "create_initial_model",
                "parameters": {
                    "model_name": dataset_name,
                    "architecture": design,
                    "seed": 42,
                },
            }
        )
        ACTION_VALIDATOR.validate(
            {
                "action": "start_training",
                "parameters": {
                    "dataset_name": dataset_name,
                    "rounds": 10,
                    "epochs": 5,
                    "architecture": {
                        **design,
                        "dataset_name": dataset_name,
                        "input_shape": [28, 28, 1],
                        "classes": [str(index) for index in range(10)],
                        "num_classes": 10,
                        "parameter_count": 10_986,
                        "initialization_seed": 42,
                        "normalization": {
                            "method": "standard",
                            "mean": [0.13],
                            "standard_deviation": [0.3],
                        },
                    },
                },
            }
        )
    nas_architecture = build_dataset_architecture(
        input_shape=(32, 32, 3),
        num_classes=10,
        channels=(16, 16),
        kernel_size=3,
        block_type="separable_conv2d",
        head_units=0,
        dropout=0.0,
        attention=False,
    )
    ACTION_VALIDATOR.validate(
        {
            "action": "start_training",
            "parameters": {
                "dataset_name": "cifar10",
                "rounds": 5,
                "epochs": 10,
                "architecture": nas_architecture,
            },
        }
    )


def test_agent_tool_arguments_coerce_schema_declared_scalar_strings() -> None:
    action = agent_tool_action(
        "start_training",
        {
            "dataset_name": "cifar10",
            "rounds": "1",
            "epochs": "1",
        },
    )

    assert action["parameters"] == {
        "dataset_name": "cifar10",
        "rounds": 1,
        "epochs": 1,
    }
    ACTION_VALIDATOR.validate(action)


def test_preprocess_action_accepts_official_cifar10() -> None:
    for dataset_split in ("stratified", "dirichlet_a01", "dirichlet_a07"):
        ACTION_VALIDATOR.validate(
            {
                "action": "preprocess_dataset",
                "parameters": {
                    "dataset_name": "cifar10",
                    "normalization": "standard",
                    "dataset_split": dataset_split,
                    "target_size": {"height": 32, "width": 32},
                    "seed": 42,
                    "replace_existing": True,
                },
            }
        )
    ACTION_VALIDATOR.validate(
        {
            "action": "create_initial_model",
            "parameters": {
                "model_name": "cifar10",
                "architecture": {
                    "id": "b2_c8_k3",
                    "blocks": 2,
                    "channels": 8,
                    "kernel_size": 3,
                },
            },
        }
    )
    ACTION_VALIDATOR.validate(
        {
            "action": "start_training",
            "parameters": {
                "dataset_name": "cifar10",
                "rounds": 1,
                "epochs": 1,
            },
        }
    )


def test_training_action_accepts_added_dirichlet_splits_for_mnist() -> None:
    for dataset_split in ("dirichlet_a01", "dirichlet_a07"):
        ACTION_VALIDATOR.validate(
            {
                "action": "start_training",
                "parameters": {
                    "dataset_name": "mnist",
                    "rounds": 1,
                    "epochs": 1,
                    "dataset_split": dataset_split,
                },
            }
        )


def test_model_action_schema_rejects_unknown_layers() -> None:
    with pytest.raises(ValidationError):
        ACTION_VALIDATOR.validate(
            {
                "action": "create_initial_model",
                "parameters": {
                    "model_name": "mnist",
                    "architecture": {
                        "model_type": "custom",
                        "layers": [
                            {"type": "random_layer"},
                            {"type": "dense", "units": 10, "activation": "linear"},
                        ],
                    },
                },
            }
        )


def test_invalid_action_error_includes_the_field_path() -> None:
    from app.services.autonomy import AutonomyError, validate_action

    with pytest.raises(AutonomyError, match="parameters.architecture.model_id"):
        validate_action(
            {
                "action": "create_initial_model",
                "parameters": {
                    "model_name": "mnist",
                    "architecture": {
                        "model_type": "pretrained_deit",
                        "model_id": "unsupported",
                    },
                },
            }
        )


def test_autonomy_routes_require_authentication(client: TestClient) -> None:
    for path, method in (
        ("/api/autonomy/state", "get"),
        ("/api/autonomy/stop", "post"),
        ("/api/autonomy/reset-stop", "post"),
        ("/api/autonomy/loops", "post"),
    ):
        response = client.request(method.upper(), path, json={})
        assert response.status_code == 401


def test_autonomy_history_limit_bounds_the_combined_event_list(
    client: TestClient,
    auth_headers: dict[str, str],
) -> None:
    controller = client.app.state.autonomy
    controller._record_execution_event(
        _set_rounds(4),
        "executed",
        mode="agent",
        reason="Test the recorded action limit.",
    )
    controller._record_execution_event(
        _set_rounds(5),
        "executed",
        mode="agent",
        reason="Test the newest event.",
    )

    response = client.get(
        "/api/autonomy/history?limit=1",
        headers=auth_headers,
    )

    assert response.status_code == 200
    assert len(response.json()) == 1
    assert response.json()[0]["details"]["parameters"] == {"rounds": 5}


def test_autonomy_loop_preserves_workflow_run_identifier(
    client: TestClient,
    auth_headers: dict[str, str],
) -> None:
    workflow_id = str(uuid4())

    response = client.post(
        "/api/autonomy/loops",
        json={
            "max_iterations": 2,
            "time_limit_seconds": 60,
            "stop_rule": "max_iterations",
            "workflow_id": workflow_id,
        },
        headers=auth_headers,
    )

    assert response.status_code == 200
    assert response.json()["workflow_id"] == workflow_id
    assert client.get("/api/autonomy/state", headers=auth_headers).json()["loops"][0][
        "workflow_id"
    ] == workflow_id


def test_delete_workflow_removes_its_saved_chat_actions_and_loop(
    client: TestClient,
    auth_headers: dict[str, str],
) -> None:
    now = datetime.now(timezone.utc)
    workflow_id = str(uuid4())
    conversation_id = str(uuid4())
    with client.app.state.session_factory.begin() as session:
        session.add(
            AutonomyLoop(
                id=str(uuid4()),
                workflow_id=workflow_id,
                status="STOPPED",
                max_iterations=2,
                iteration=1,
                time_limit_seconds=60,
                stop_rule="max_iterations",
                target_metric=None,
                best_metric=0.5,
                no_improvement_count=0,
                failure_reason="finished",
                created_at=now,
                deadline_at=now + timedelta(seconds=60),
                updated_at=now,
            )
        )
        session.add(
            SlmConversation(
                id=conversation_id,
                workflow_id=workflow_id,
                mode="agent",
                title="Workflow chat",
                created_at=now,
                updated_at=now,
                messages=[
                    SlmMessage(
                        role="assistant",
                        content="Completed a training action.",
                        created_at=now,
                    )
                ],
            )
        )
    client.app.state.autonomy._record_execution_event(
        _set_rounds(),
        "executed",
        mode="AGENT",
        reason="Workflow action",
        evidence={"workflow_id": workflow_id, "conversation_id": conversation_id},
    )
    client.app.state.autonomy._record_execution_event(
        _set_rounds(),
        "executed",
        mode="FULL_ACCESS",
        reason="Workflow setup action",
        evidence={"source": "agent_workflow", "workflow_id": workflow_id},
    )

    response = client.request(
        "DELETE",
        "/api/autonomy/workflows",
        json={"workflow_ids": [workflow_id], "conversation_ids": [conversation_id]},
        headers=auth_headers,
    )

    assert response.status_code == 200
    assert response.json() == {
        "deleted_workflows": 1,
        "deleted_conversations": 1,
        "deleted_actions": 2,
    }
    with client.app.state.session_factory() as session:
        assert session.query(AutonomyLoop).filter_by(workflow_id=workflow_id).count() == 0
        assert session.get(SlmConversation, conversation_id) is None
        assert session.query(AutonomyExecutionEvent).count() == 0


def test_delete_legacy_unlinked_workflow_removes_only_unlinked_setup_actions(
    client: TestClient,
    auth_headers: dict[str, str],
) -> None:
    controller = client.app.state.autonomy
    controller._record_execution_event(
        _set_rounds(),
        "executed",
        mode="FULL_ACCESS",
        reason="Unlinked setup action",
        evidence={"source": "agent_workflow"},
    )
    controller._record_execution_event(
        _set_rounds(),
        "executed",
        mode="FULL_ACCESS",
        reason="Unrelated action",
        evidence={"source": "other"},
    )
    workflow_id = str(uuid4())
    controller._record_execution_event(
        _set_rounds(),
        "executed",
        mode="FULL_ACCESS",
        reason="Linked setup action",
        evidence={"source": "agent_workflow", "workflow_id": workflow_id},
    )

    response = client.request(
        "DELETE",
        "/api/autonomy/workflows",
        json={"workflow_ids": ["legacy-unlinked"]},
        headers=auth_headers,
    )

    assert response.status_code == 200
    assert response.json() == {
        "deleted_workflows": 0,
        "deleted_conversations": 0,
        "deleted_actions": 1,
    }
    with client.app.state.session_factory() as session:
        remaining = session.query(AutonomyExecutionEvent).all()
        assert len(remaining) == 2
        remaining_evidence = [event.details["evidence"] for event in remaining]
        assert {"source": "other"} in remaining_evidence
        assert {"source": "agent_workflow", "workflow_id": workflow_id} in remaining_evidence


def test_delete_workflow_rejects_a_running_loop(
    client: TestClient,
    auth_headers: dict[str, str],
) -> None:
    now = datetime.now(timezone.utc)
    workflow_id = str(uuid4())
    loop_id = str(uuid4())
    with client.app.state.session_factory.begin() as session:
        session.add(
            AutonomyLoop(
                id=loop_id,
                workflow_id=workflow_id,
                status="RUNNING",
                max_iterations=2,
                iteration=0,
                time_limit_seconds=60,
                stop_rule="max_iterations",
                target_metric=None,
                best_metric=None,
                no_improvement_count=0,
                failure_reason=None,
                created_at=now,
                deadline_at=now + timedelta(seconds=60),
                updated_at=now,
            )
        )

    response = client.request(
        "DELETE",
        "/api/autonomy/workflows",
        json={"workflow_ids": [workflow_id]},
        headers=auth_headers,
    )

    assert response.status_code == 409
    assert "running workflow cannot be deleted" in response.json()["detail"]
    assert client.app.state.autonomy.state()["loops"][0]["id"] == loop_id


def test_dispatch_executes_immediately_and_has_no_approval_controls(
    client: TestClient,
    auth_headers: dict[str, str],
    monkeypatch,
) -> None:
    from app.routers import slm

    forwarded = []

    def fake_forward(settings, path, *, payload=None, query="", method=None):
        forwarded.append((path, payload))
        return {"recorded": True}

    monkeypatch.setattr(slm, "_forward", fake_forward)
    controller = client.app.state.autonomy
    executed: list[dict[str, object]] = []
    controller._execute = lambda action: executed.append(action) or {"available": True}

    state = client.get("/api/autonomy/state", headers=auth_headers)
    result = client.post(
        "/api/autonomy/dispatch",
        json={"action": _set_rounds(), "reason": "Set rounds"},
        headers=auth_headers,
    )

    assert state.status_code == 200
    assert "mode" not in state.json()
    assert result.json() == {"status": "executed", "result": {"available": True}}
    assert forwarded == [
        (
            "/action-history",
            {
                "action": _set_rounds(),
                "result": {"available": True},
            },
        )
    ]
    assert len(executed) == 1
    assert client.get("/api/autonomy/decisions", headers=auth_headers).status_code == 404
    assert client.put(
        "/api/autonomy/mode",
        json={"mode": "ASK_HUMAN"},
        headers=auth_headers,
    ).status_code == 404
    history = client.get("/api/autonomy/history", headers=auth_headers).json()
    automatic = next(event for event in history if event["event"] == "executed automatically")
    assert automatic["action"] == "set_rounds"
    assert automatic["result_status"] == "executed"
    assert automatic["details"]["parameters"] == {"rounds": 5}


def test_dispatch_validates_action_and_preserves_audit_on_execution_failure(
    client: TestClient,
    auth_headers: dict[str, str],
) -> None:
    controller = client.app.state.autonomy
    controller._execute = lambda action: {"ok": True}
    invalid = client.post(
        "/api/autonomy/dispatch",
        json={"action": {"action": "set_rounds", "parameters": {"rounds": 1000}}},
        headers=auth_headers,
    )
    assert invalid.status_code == 422

    def fail(_action):
        raise RuntimeError("sensitive exception detail")

    controller._execute = fail
    with pytest.raises(RuntimeError, match="sensitive exception detail"):
        controller.dispatch(_set_rounds(), reason="execution should be audited")
    history = client.get("/api/autonomy/history", headers=auth_headers).json()
    failed = next(event for event in history if event["event"] == "automatic execution failed")
    assert failed["action"] == "set_rounds"
    assert failed["result_status"] == "failed"
    assert "sensitive exception detail" not in str(failed)


def test_agent_dispatch_records_agent_execution_mode(
    client: TestClient,
) -> None:
    controller = client.app.state.autonomy
    controller._execute = lambda action: {"available": True, "data": {"rounds": 8}}

    result = controller.dispatch(
        _set_rounds(8),
        reason="Agent requested a measured training configuration.",
        evidence={"source": "langgraph_agent"},
        mode="AGENT",
    )

    assert result["status"] == "executed"
    with client.app.state.session_factory() as session:
        event = session.query(AutonomyExecutionEvent).one()
        assert event.mode == "AGENT"
        assert event.details["evidence"]["source"] == "langgraph_agent"


def test_global_stop_blocks_writes_but_allows_reads_and_reset(
    client: TestClient,
    auth_headers: dict[str, str],
) -> None:
    controller = client.app.state.autonomy
    executed: list[dict[str, object]] = []
    controller._execute = lambda action: executed.append(action) or {"available": True}

    assert client.post("/api/autonomy/stop", headers=auth_headers).json()["stop_requested"]
    blocked = client.post(
        "/api/autonomy/dispatch",
        json={"action": _set_rounds()},
        headers=auth_headers,
    )
    read_only = client.post(
        "/api/autonomy/dispatch",
        json={"action": {"action": "get_status", "parameters": {}}},
        headers=auth_headers,
    )
    assert blocked.status_code == 423
    assert read_only.status_code == 200
    assert len(executed) == 1
    assert client.post("/api/autonomy/reset-stop", headers=auth_headers).json()["stop_requested"] is False


def test_loops_stop_on_maximum_target_and_no_improvement(
    client: TestClient,
    auth_headers: dict[str, str],
) -> None:
    controller = client.app.state.autonomy
    maximum = controller.start_loop(max_iterations=1, time_limit_seconds=60)
    result = controller.record_iteration(maximum["id"], metric=0.5)
    assert result["status"] == "STOPPED"
    assert result["failure_reason"] == "maximum iterations reached"

    targeted = controller.start_loop(
        max_iterations=5,
        time_limit_seconds=60,
        stop_rule="target_metric",
        target_metric=0.8,
    )
    result = controller.record_iteration(targeted["id"], metric=0.8)
    assert result["failure_reason"] == "target metric reached"

    plateau = controller.start_loop(
        max_iterations=5,
        time_limit_seconds=60,
        stop_rule="no_improvement_2",
    )
    controller.record_iteration(plateau["id"], metric=0.5)
    controller.record_iteration(plateau["id"], metric=0.4)
    result = controller.record_iteration(plateau["id"], metric=0.3)
    assert result["failure_reason"] == "no improvement for two iterations"
    assert client.get("/api/autonomy/state", headers=auth_headers).json()["loops"][0]["id"] == plateau["id"]


def test_loop_worker_failure_and_global_stop_are_recorded(
    client: TestClient,
    auth_headers: dict[str, str],
) -> None:
    controller = client.app.state.autonomy
    loop = controller.start_loop(max_iterations=5, time_limit_seconds=60)
    paused = controller.record_iteration(loop["id"], worker_failure="worker2 timed out")
    assert paused["status"] == "PAUSED"
    assert paused["failure_reason"] == "worker2 timed out"

    another = controller.start_loop(max_iterations=5, time_limit_seconds=60)
    stopped = client.post("/api/autonomy/stop", headers=auth_headers)
    assert stopped.json()["loops"][0]["status"] == "STOPPED"
    with pytest.raises(AutonomyError, match="Global STOP"):
        controller.start_loop(max_iterations=1, time_limit_seconds=30)
    client.post("/api/autonomy/reset-stop", headers=auth_headers)
    reopened = AutonomyController(controller._sessions, lambda action: {"ok": True})
    assert any(loop["id"] == another["id"] for loop in reopened.state()["loops"])


def test_loop_time_limit_stops_iteration(
    client: TestClient,
) -> None:
    clock = [datetime(2026, 1, 1, tzinfo=timezone.utc)]
    controller = AutonomyController(
        client.app.state.session_factory,
        lambda action: {"ok": True},
        clock=lambda: clock[0],
    )
    loop = controller.start_loop(max_iterations=5, time_limit_seconds=1)
    clock[0] += timedelta(seconds=2)
    stopped = controller.record_iteration(loop["id"], metric=0.4)
    assert stopped["status"] == "STOPPED"
    assert stopped["failure_reason"] == "time limit reached"


def test_starting_and_stopping_loops_persist_with_controller_restart(
    client: TestClient,
) -> None:
    controller = client.app.state.autonomy
    loop = controller.start_loop(max_iterations=3, time_limit_seconds=60)
    reopened = AutonomyController(controller._sessions, lambda action: {"ok": True})
    state = reopened.state()
    assert state["loops"][0]["id"] == loop["id"]
    assert state["loops"][0]["status"] == "RUNNING"
    assert reopened.stop_loop(loop["id"], "finished by operator")["status"] == "STOPPED"
