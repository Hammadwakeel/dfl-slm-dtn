import pytest
from fastapi.testclient import TestClient
from jsonschema import ValidationError

from app.db.models import AutonomyExecutionEvent
from app.services.autonomy import ACTION_VALIDATOR
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
        ("/api/autonomy/dispatch", "post"),
        ("/api/autonomy/history", "get"),
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

