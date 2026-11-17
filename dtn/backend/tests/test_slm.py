from uuid import uuid4

import pytest
from fastapi.testclient import TestClient


def test_slm_proxy_requires_coordinator_authentication(client: TestClient) -> None:
    for path, method, payload in (
        ("/api/slm/status", "get", None),
        ("/api/slm/conversations", "get", None),
        ("/api/slm/logs", "get", None),
        ("/api/slm/results", "get", None),
        ("/api/slm/agent/chat", "post", {"message": "start training"}),
        ("/api/slm/command", "post", {"action": "get_status", "parameters": {}}),
    ):
        response = getattr(client, method)(path, json=payload) if payload else getattr(
            client, method
        )(path)
        assert response.status_code == 401


def test_slm_proxy_includes_safe_master_error_detail(
    client: TestClient,
    monkeypatch,
) -> None:
    from io import BytesIO
    from urllib.error import HTTPError

    from fastapi import HTTPException

    from app.routers import slm

    class RejectingOpener:
        def open(self, request, timeout):
            raise HTTPError(
                request.full_url,
                400,
                "Bad Request",
                {},
                BytesIO(b'{"error":"Message content exceeds the 32000 character limit"}'),
            )

    monkeypatch.setattr(slm, "build_opener", lambda *args, **kwargs: RejectingOpener())

    with pytest.raises(HTTPException) as error:
        slm._forward(
            client.app.state.settings,
            "/agent/model",
            payload={"messages": []},
        )

    assert error.value.status_code == 400
    assert error.value.detail == (
        "The master request failed (HTTP 400): "
        "Message content exceeds the 32000 character limit"
    )


def test_langgraph_agent_receives_cluster_snapshot_executes_tools_and_persists_mode(
    client: TestClient,
    auth_headers: dict[str, str],
    monkeypatch,
) -> None:
    from app.db.models import AutonomyExecutionEvent, Device
    from app.routers import slm

    model_requests = []
    workflow_id = str(uuid4())
    model_responses = iter(
        [
            {
                "message": {
                    "role": "assistant",
                    "content": "The user asked to change a setting, so I will apply the selected round count.",
                    "tool_calls": [
                        {
                            "function": {
                                "name": "set_rounds",
                                "arguments": {"rounds": 8},
                            }
                        }
                    ],
                }
            },
            {
                "message": {
                    "role": "assistant",
                    "content": "I changed the round count to eight.",
                }
            },
            {
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "function": {
                                "name": "get_status",
                                "arguments": {},
                            }
                        }
                    ],
                }
            },
            {
                "message": {
                    "role": "assistant",
                    "content": "The last action set the training rounds to eight.",
                }
            },
        ]
    )

    def fake_forward(settings, path, *, payload=None, query="", method=None):
        if path == "/results":
            return {"configuration": {"rounds": 5}, "active_group_id": None, "groups": []}
        if path == "/action-history":
            return {"recorded": True}
        if path == "/agent/model":
            model_requests.append(payload)
            return next(model_responses)
        raise AssertionError(f"Unexpected upstream path: {path}")

    monkeypatch.setattr(slm, "_forward", fake_forward)
    with client.app.state.session_factory() as session:
        device = session.get(Device, "worker1")
        assert device is not None
        device.latest_status = {"cpu_utilization_percent": 23.5}
        session.commit()
    client.app.state.datasets.list_metadata = lambda: []
    client.app.state.autonomy._execute = lambda action: {
        "available": True,
        "data": {
            "rounds": action.get("parameters", {}).get("rounds", 8),
        },
    }

    response = client.post(
        "/api/slm/agent/chat",
        json={
            "message": "Set the training round count to eight.",
            "workflow_id": workflow_id,
        },
        headers=auth_headers,
    )

    assert response.status_code == 200
    body = response.json()
    assert body["mode"] == "agent"
    assert body["reply"] == "I changed the round count to eight."
    assert body["trace"][0]["tool"] == "set_rounds"
    assert body["trace"][0]["decision"] == (
        "The user asked to change a setting, so I will apply the selected round count."
    )
    assert body["workflow_id"] == workflow_id
    prompt = model_requests[0]["messages"][0]["content"]
    assert "a concise user-facing decision summary" in prompt
    assert "do not repeat the same call unchanged" in prompt
    assert '"node_id":"worker1"' in prompt
    assert '"cpu_utilization_percent":23.5' in prompt
    assert '"previous_actions":[]' in prompt

    followup = client.post(
        "/api/slm/agent/chat",
        json={
            "message": "What did you last change?",
            "conversation_id": body["conversation_id"],
            "workflow_id": workflow_id,
        },
        headers=auth_headers,
    )
    assert followup.status_code == 200
    followup_prompt = model_requests[2]["messages"][0]["content"]
    assert '"action":"set_rounds"' in followup_prompt
    assert '"mode":"AGENT"' in followup_prompt
    assert '"result":{"available":true,"data":{"rounds":8}}' in followup_prompt
    with client.app.state.session_factory() as session:
        event = session.query(AutonomyExecutionEvent).one()
        conversation = session.query(slm.SlmConversation).one()
        assert event.mode == "AGENT"
        assert event.details["evidence"]["workflow_id"] == workflow_id
        assert conversation.mode == "agent"
        assert conversation.workflow_id == workflow_id


def test_agent_chat_persists_slm_selected_workflow_termination(
    client: TestClient,
    auth_headers: dict[str, str],
    monkeypatch,
) -> None:
    from app.routers import slm

    calls = []

    def fake_forward(settings, path, *, payload=None, query="", method=None):
        if path == "/results":
            return {"configuration": {}, "active_group_id": None, "groups": []}
        if path == "/agent/model":
            calls.append(payload)
            return {
                "message": {
                    "role": "assistant",
                    "content": "The stated validation target is met.",
                    "tool_calls": [{
                        "function": {
                            "name": "finish_agent_workflow",
                            "arguments": {
                                "outcome": "objective_achieved",
                                "summary": "Validation target reached.",
                                "evidence": "The completed result snapshot shows 0.91 validation accuracy.",
                            },
                        },
                    }],
                }
            }
        raise AssertionError(f"Unexpected upstream path: {path}")

    monkeypatch.setattr(slm, "_forward", fake_forward)
    client.app.state.datasets.list_metadata = lambda: []
    workflow_id = str(uuid4())

    response = client.post(
        "/api/slm/agent/chat",
        json={
            "message": "Stop when validation accuracy reaches 90%.",
            "workflow_dataset": "mnist",
            "workflow_id": workflow_id,
        },
        headers=auth_headers,
    )

    assert response.status_code == 200
    body = response.json()
    assert body["terminated"] is True
    assert body["workflow_id"] == workflow_id
    assert body["trace"][0]["tool"] == "finish_agent_workflow"
    assert body["trace"][0]["result"]["status"] == "workflow_complete"
    assert "Validation target reached." in body["reply"]
    assert len(calls) == 1
    assert "System snapshot JSON:" in calls[0]["messages"][0]["content"]
    with client.app.state.session_factory() as session:
        assistant = session.query(slm.SlmMessage).filter_by(role="assistant").one()
        assert assistant.response_data["terminated"] is True
    assert client.get(
        "/api/slm/conversations",
        headers=auth_headers,
    ).json()[0]["workflow_id"] == workflow_id


def test_dataset_bound_agent_workflow_rejects_mnist_only_topology_search(
    client: TestClient,
    auth_headers: dict[str, str],
    monkeypatch,
) -> None:
    from app.routers import slm

    responses = iter(
        [
            {
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "function": {
                                "name": "optimize_topology",
                                "arguments": {"rounds": 1, "epochs": 1, "seeds": [17], "alphas": [0.3]},
                            }
                        }
                    ],
                }
            },
            {
                "message": {
                    "role": "assistant",
                    "content": "I will use dataset-specific CIFAR-10 NAS instead.",
                }
            },
        ]
    )
    monkeypatch.setattr(
        slm,
        "_forward",
        lambda settings, path, *, payload=None, query="", method=None: (
            {"configuration": {}, "active_group_id": None, "groups": []}
            if path == "/results"
            else next(responses)
            if path == "/agent/model"
            else {"recorded": True}
        ),
    )
    client.app.state.datasets.list_metadata = lambda: []
    client.app.state.autonomy._execute = lambda action: pytest.fail(
        f"Dataset-bound workflow unexpectedly executed {action['action']}"
    )

    response = client.post(
        "/api/slm/agent/chat",
        json={
            "message": "Start topology search for my selected dataset.",
            "workflow_dataset": "cifar10",
        },
        headers=auth_headers,
    )

    assert response.status_code == 200
    body = response.json()
    assert body["trace"][0]["result"]["status"] == "invalid_tool_call"
    assert "currently runs MNIST only" in body["trace"][0]["result"]["error"]


def test_agent_workflow_allows_the_slm_to_choose_nas_before_training(
    client: TestClient,
    auth_headers: dict[str, str],
    monkeypatch,
) -> None:
    from app.routers import slm

    responses = iter(
        [
            {
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "function": {
                                "name": "enable_nas",
                                "arguments": {
                                    "enabled": True,
                                    "dataset_name": "cifar10",
                                    "search_time_limit_seconds": 90,
                                },
                            }
                        }
                    ],
                }
            },
            {
                "message": {
                    "role": "assistant",
                    "content": "I will start the required baseline first.",
                }
            },
        ]
    )
    monkeypatch.setattr(
        slm,
        "_forward",
        lambda settings, path, *, payload=None, query="", method=None: (
            {"configuration": {}, "active_group_id": None, "groups": []}
            if path == "/results"
            else next(responses)
            if path == "/agent/model"
            else {"recorded": True}
        ),
    )
    client.app.state.datasets.list_metadata = lambda: []
    executed: list[dict[str, object]] = []
    client.app.state.autonomy._execute = lambda action: (
        executed.append(action) or {"available": True, "data": {"group_id": "cifar-nas"}}
    )

    response = client.post(
        "/api/slm/agent/chat",
        json={
            "message": "Choose the best next CIFAR-10 action based on the current state.",
            "workflow_dataset": "cifar10",
        },
        headers=auth_headers,
    )

    assert response.status_code == 200
    result = response.json()["trace"][0]["result"]
    assert result["status"] == "executed"
    assert executed == [{
        "action": "enable_nas",
        "parameters": {
            "enabled": True,
            "dataset_name": "cifar10",
            "search_time_limit_seconds": 90,
        },
    }]


def test_agent_workflow_accepts_stringified_empty_training_architecture(
    client: TestClient,
    auth_headers: dict[str, str],
    monkeypatch,
) -> None:
    from app.routers import slm

    responses = iter(
        [
            {
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "function": {
                                "name": "start_training",
                                "arguments": {
                                    "rounds": "5",
                                    "epochs": "10",
                                    "dataset_name": "cifar10",
                                    "architecture": "{}",
                                },
                            }
                        }
                    ],
                }
            },
            {"message": {"role": "assistant", "content": "The baseline started."}},
        ]
    )
    executed: list[dict[str, object]] = []

    def fake_forward(settings, path, *, payload=None, query="", method=None):
        if path == "/results":
            return {"configuration": {}, "active_group_id": None, "groups": []}
        if path == "/action-history":
            return {"recorded": True}
        if path == "/agent/model":
            return next(responses)
        raise AssertionError(f"Unexpected upstream path: {path}")
    monkeypatch.setattr(slm, "_forward", fake_forward)
    client.app.state.datasets.list_metadata = lambda: []

    def execute(action):
        executed.append(action)
        return {"available": True, "data": {"group_id": "baseline"}}

    client.app.state.autonomy._execute = execute

    response = client.post(
        "/api/slm/agent/chat",
        json={
            "message": "Start the required CIFAR-10 baseline.",
            "workflow_dataset": "cifar10",
        },
        headers=auth_headers,
    )

    assert response.status_code == 200
    assert response.json()["reply"] == "The baseline started."
    assert executed == [
        {
            "action": "start_training",
            "parameters": {
                "rounds": 5,
                "epochs": 10,
                "dataset_name": "cifar10",
            },
        }
    ]


def test_workflow_rejects_repeated_training_without_a_second_model_call(
    client: TestClient,
    auth_headers: dict[str, str],
    monkeypatch,
) -> None:
    from app.routers import slm

    workflow_id = str(uuid4())
    architecture = {
        "id": "b1_c8_k3",
        "blocks": 1,
        "channels": 8,
        "kernel_size": 3,
    }
    baseline = {
        "group_id": "mnist-baseline",
        "kind": "decentralized",
        "dataset_name": "mnist",
        "status": "completed",
        "rounds": 5,
        "epochs": 10,
        "topology": "directed_ring",
        "dataset_split": "stratified",
        "architecture": architecture,
        "runs": [],
    }
    stopped_duplicate = {
        **baseline,
        "group_id": "mnist-stopped-duplicate",
        "status": "stopped",
        "error": "Training was cancelled",
    }
    model_calls = 0
    executed: list[dict[str, object]] = []

    def fake_forward(settings, path, *, payload=None, query="", method=None):
        nonlocal model_calls
        if path == "/results":
            return {
                "configuration": {
                    "rounds": 5,
                    "epochs": 10,
                    "topology": "directed_ring",
                    "dataset_split": "stratified",
                    "architecture": architecture,
                },
                "active_group_id": None,
                "groups": [baseline, stopped_duplicate],
            }
        if path == "/agent/model":
            model_calls += 1
            return {
                "message": {
                    "role": "assistant",
                    "content": "Repeat the baseline.",
                    "tool_calls": [
                        {
                            "function": {
                                "name": "start_training",
                                "arguments": {
                                    "dataset_name": "mnist",
                                    "rounds": 5,
                                    "epochs": 10,
                                },
                            }
                        }
                    ],
                }
            }
        raise AssertionError(f"Unexpected upstream path: {path}")

    monkeypatch.setattr(slm, "_forward", fake_forward)
    client.app.state.datasets.list_metadata = lambda: []
    client.app.state.autonomy._execute = lambda action: (
        executed.append(action) or {"available": True, "data": {}}
    )

    response = client.post(
        "/api/slm/agent/chat",
        json={
            "message": "Do not repeat the fixed baseline.",
            "workflow_dataset": "mnist",
            "workflow_id": workflow_id,
            "workflow_previous_group_id": "mnist-stopped-duplicate",
        },
        headers=auth_headers,
    )

    assert response.status_code == 200
    result = response.json()["trace"][0]["result"]
    assert result["status"] == "invalid_tool_call"
    assert "already completed" in result["error"]
    assert model_calls == 1
    assert executed == []


def test_agent_retries_action_rejected_by_live_state(
    client: TestClient,
    auth_headers: dict[str, str],
    monkeypatch,
) -> None:
    from app.routers import slm

    workflow_id = str(uuid4())
    model_requests = []
    responses = iter(
        [
            {
                "message": {
                    "role": "assistant",
                    "content": "Stop the current training process.",
                    "tool_calls": [{
                        "function": {"name": "stop_training", "arguments": {}},
                    }],
                }
            },
            {
                "message": {
                    "role": "assistant",
                    "content": "Start a changed CIFAR training experiment.",
                    "tool_calls": [{
                        "function": {
                            "name": "start_training",
                            "arguments": {
                                "dataset_name": "cifar10",
                                "rounds": 6,
                                "epochs": 20,
                            },
                        }
                    }],
                }
            },
        ]
    )

    def fake_forward(settings, path, *, payload=None, query="", method=None):
        if path == "/results":
            return {"configuration": {}, "active_group_id": None, "groups": []}
        if path == "/agent/model":
            model_requests.append(payload["messages"])
            if len(model_requests) == 2:
                assert payload["messages"][-1]["role"] == "tool"
                assert "No training group is active" in payload["messages"][-1]["content"]
            return next(responses)
        if path == "/action-history":
            return {"recorded": True}
        raise AssertionError(f"Unexpected upstream path: {path}")

    monkeypatch.setattr(slm, "_forward", fake_forward)
    client.app.state.datasets.list_metadata = lambda: []
    executed: list[dict[str, object]] = []

    def execute(action):
        executed.append(action)
        if action["action"] == "stop_training":
            return {
                "available": True,
                "data": {"accepted": False, "message": "No training group is active."},
            }
        return {
            "available": True,
            "data": {"accepted": True, "group_id": "cifar-tuned"},
        }

    client.app.state.autonomy._execute = execute
    response = client.post(
        "/api/slm/agent/chat",
        json={
            "message": "Continue CIFAR-10 training.",
            "workflow_dataset": "cifar10",
            "workflow_id": workflow_id,
        },
        headers=auth_headers,
    )

    assert response.status_code == 200
    body = response.json()
    assert len(model_requests) == 2
    assert body["trace"][0]["result"] == {
        "status": "invalid_tool_call",
        "retryable": True,
        "error": "No training group is active.",
    }
    assert body["trace"][1]["result"]["status"] == "executed"
    assert [action["action"] for action in executed] == [
        "stop_training",
        "start_training",
    ]


def test_dataset_workflow_retries_read_only_status_tool(
    client: TestClient,
    auth_headers: dict[str, str],
    monkeypatch,
) -> None:
    from app.routers import slm

    model_requests = []
    responses = iter(
        [
            {
                "message": {
                    "role": "assistant",
                    "content": "Inspect historical logs.",
                    "tool_calls": [{
                        "function": {"name": "get_logs", "arguments": {"limit": 10}},
                    }],
                }
            },
            {
                "message": {
                    "role": "assistant",
                    "content": "Start the changed CIFAR experiment.",
                    "tool_calls": [{
                        "function": {
                            "name": "start_training",
                            "arguments": {
                                "dataset_name": "cifar10",
                                "rounds": 6,
                                "epochs": 50,
                            },
                        }
                    }],
                }
            },
        ]
    )

    def fake_forward(settings, path, *, payload=None, query="", method=None):
        if path == "/results":
            return {"configuration": {}, "active_group_id": None, "groups": []}
        if path == "/agent/model":
            model_requests.append(payload["messages"])
            if len(model_requests) == 2:
                assert payload["messages"][-1]["role"] == "tool"
                assert "already supplies current status and results" in payload["messages"][-1]["content"]
            return next(responses)
        if path == "/action-history":
            return {"recorded": True}
        raise AssertionError(f"Unexpected upstream path: {path}")

    monkeypatch.setattr(slm, "_forward", fake_forward)
    client.app.state.datasets.list_metadata = lambda: []
    executed: list[dict[str, object]] = []
    client.app.state.autonomy._execute = lambda action: (
        executed.append(action)
        or {"available": True, "data": {"accepted": True, "group_id": "cifar-tuned"}}
    )

    response = client.post(
        "/api/slm/agent/chat",
        json={
            "message": "Continue the CIFAR-10 workflow.",
            "workflow_dataset": "cifar10",
            "workflow_id": str(uuid4()),
        },
        headers=auth_headers,
    )

    assert response.status_code == 200
    body = response.json()
    assert len(model_requests) == 2
    assert body["trace"][0]["result"]["retryable"] is True
    assert "status and results" in body["trace"][0]["result"]["error"]
    assert [action["action"] for action in executed] == ["start_training"]


def test_agent_retries_reused_initial_model_as_no_progress(
    client: TestClient,
    auth_headers: dict[str, str],
    monkeypatch,
) -> None:
    from app.routers import slm

    responses = iter(
        [
            {
                "message": {
                    "role": "assistant",
                    "content": "Create the initial CIFAR model.",
                    "tool_calls": [{
                        "function": {
                            "name": "create_initial_model",
                            "arguments": {"model_name": "cifar10"},
                        }
                    }],
                }
            },
            {
                "message": {
                    "role": "assistant",
                    "content": "Start CIFAR training with the existing model.",
                    "tool_calls": [{
                        "function": {
                            "name": "start_training",
                            "arguments": {
                                "dataset_name": "cifar10",
                                "rounds": 6,
                                "epochs": 50,
                            },
                        }
                    }],
                }
            },
        ]
    )

    def fake_forward(settings, path, *, payload=None, query="", method=None):
        if path == "/results":
            return {"configuration": {}, "active_group_id": None, "groups": []}
        if path == "/agent/model":
            return next(responses)
        if path == "/action-history":
            return {"recorded": True}
        raise AssertionError(f"Unexpected upstream path: {path}")

    monkeypatch.setattr(slm, "_forward", fake_forward)
    client.app.state.datasets.list_metadata = lambda: []
    executed: list[dict[str, object]] = []

    def execute(action):
        executed.append(action)
        if action["action"] == "create_initial_model":
            return {
                "available": True,
                "data": {"dataset_name": "cifar10", "reused_existing_model": True},
            }
        return {
            "available": True,
            "data": {"accepted": True, "group_id": "cifar-tuned"},
        }

    client.app.state.autonomy._execute = execute
    response = client.post(
        "/api/slm/agent/chat",
        json={
            "message": "Continue the CIFAR-10 workflow.",
            "workflow_dataset": "cifar10",
            "workflow_id": str(uuid4()),
        },
        headers=auth_headers,
    )

    assert response.status_code == 200
    body = response.json()
    assert body["trace"][0]["result"] == {
        "status": "invalid_tool_call",
        "retryable": True,
        "error": (
            "The initial model already exists and was reused. Choose the next "
            "experiment/configuration action instead of recreating it."
        ),
    }
    assert body["trace"][1]["result"]["status"] == "executed"
    assert [action["action"] for action in executed] == [
        "create_initial_model",
        "start_training",
    ]


def test_agent_workflow_preserves_slm_selected_training_parameters(
    client: TestClient,
    auth_headers: dict[str, str],
    monkeypatch,
) -> None:
    from app.routers import slm

    responses = iter(
        [
            {
                "message": {
                    "role": "assistant",
                    "content": "A short training run is appropriate for this diagnostic comparison.",
                    "tool_calls": [
                        {
                            "function": {
                                "name": "start_training",
                                "arguments": {
                                    "rounds": 1,
                                    "epochs": 1,
                                    "dataset_name": "cifar10",
                                },
                            }
                        }
                    ],
                }
            },
            {"message": {"role": "assistant", "content": "The requested diagnostic run was started."}},
        ]
    )
    monkeypatch.setattr(
        slm,
        "_forward",
        lambda settings, path, *, payload=None, query="", method=None: (
            {"configuration": {}, "active_group_id": None, "groups": []}
            if path == "/results"
            else next(responses)
            if path == "/agent/model"
            else {"recorded": True}
        ),
    )
    client.app.state.datasets.list_metadata = lambda: []
    executed: list[dict[str, object]] = []

    def execute(action):
        executed.append(action)
        return {"available": True, "data": {"group_id": "normalized-training"}}

    client.app.state.autonomy._execute = execute

    response = client.post(
        "/api/slm/agent/chat",
        json={"message": "Improve the last experiment.", "workflow_dataset": "cifar10"},
        headers=auth_headers,
    )

    assert response.status_code == 200
    result = response.json()["trace"][0]["result"]
    assert result["status"] == "executed"
    assert "workflow_adjustments" not in result
    assert executed == [{
        "action": "start_training",
        "parameters": {
            "rounds": 1,
            "epochs": 1,
            "dataset_name": "cifar10",
        },
    }]


def test_agent_workflow_requires_the_slm_to_select_topology_split_explicitly(
    client: TestClient,
    auth_headers: dict[str, str],
    monkeypatch,
) -> None:
    from app.routers import slm

    responses = iter(
        [
            {
                "message": {
                    "role": "assistant",
                    "content": "Validate the selected topology.",
                    "tool_calls": [
                        {
                            "function": {
                                "name": "start_training",
                                "arguments": {
                                    "rounds": 1,
                                    "epochs": 1,
                                    "dataset_name": "mnist",
                                    "dataset_split": "dirichlet_a03",
                                },
                            }
                        }
                    ],
                }
            },
            {"message": {"role": "assistant", "content": "The selected topology is training."}},
        ]
    )
    executed: list[dict[str, object]] = []

    def fake_forward(settings, path, *, payload=None, query="", method=None):
        if path == "/results":
            return {
                "configuration": {},
                "active_group_id": None,
                "groups": [{
                    "group_id": "mnist-topology",
                    "kind": "topology_optimization",
                    "dataset_name": "mnist",
                    "status": "completed",
                    "selected_configuration": {
                        "topology": "chain",
                        "alpha": 0.3,
                        "dataset_split": "dirichlet_a03",
                    },
                }],
            }
        if path == "/action-history":
            return {"recorded": True}
        if path == "/agent/model":
            return next(responses)
        raise AssertionError(f"Unexpected upstream path: {path}")

    monkeypatch.setattr(slm, "_forward", fake_forward)
    client.app.state.datasets.list_metadata = lambda: []

    def execute(action):
        executed.append(action)
        return {"available": True, "data": {"group_id": "mnist-selected-topology"}}

    client.app.state.autonomy._execute = execute
    response = client.post(
        "/api/slm/agent/chat",
        json={
            "message": "Retrain using the selected topology.",
            "workflow_dataset": "mnist",
            "workflow_previous_group_id": "mnist-topology",
        },
        headers=auth_headers,
    )

    assert response.status_code == 200
    result = response.json()["trace"][0]["result"]
    assert result["status"] == "executed"
    assert "workflow_adjustments" not in result
    assert executed == [{
        "action": "start_training",
        "parameters": {
            "rounds": 1,
            "epochs": 1,
            "dataset_name": "mnist",
            "dataset_split": "dirichlet_a03",
        },
    }]


def test_agent_topology_search_preserves_slm_selected_budget(
    client: TestClient,
    auth_headers: dict[str, str],
    monkeypatch,
) -> None:
    from app.routers import slm

    responses = iter(
        [
            {
                "message": {
                    "role": "assistant",
                    "content": "The baseline suggests comparing topologies.",
                    "tool_calls": [
                        {
                            "function": {
                                "name": "optimize_topology",
                                "arguments": {"seeds": [42], "alphas": [0.3]},
                            }
                        }
                    ],
                }
            },
            {"message": {"role": "assistant", "content": "The topology search was started."}},
        ]
    )
    monkeypatch.setattr(
        slm,
        "_forward",
        lambda settings, path, *, payload=None, query="", method=None: (
            {"configuration": {}, "active_group_id": None, "groups": []}
            if path == "/results"
            else next(responses)
            if path == "/agent/model"
            else {"recorded": True}
        ),
    )
    client.app.state.datasets.list_metadata = lambda: []
    executed: list[dict[str, object]] = []

    def execute(action):
        executed.append(action)
        return {"available": True, "data": {"group_id": "topology-search-1"}}

    client.app.state.autonomy._execute = execute
    response = client.post(
        "/api/slm/agent/chat",
        json={"message": "Compare the available topologies.", "workflow_dataset": "mnist"},
        headers=auth_headers,
    )

    assert response.status_code == 200
    assert executed == [{
        "action": "optimize_topology",
        "parameters": {
            "seeds": [42],
            "alphas": [0.3],
        },
    }]


def test_agent_retrains_with_the_selected_completed_nas_architecture(
    client: TestClient,
    auth_headers: dict[str, str],
    monkeypatch,
) -> None:
    from app.routers import slm

    selected_architecture = {
        "id": "b1_c8_k3",
        "blocks": 1,
        "channels": 8,
        "kernel_size": 3,
    }
    responses = iter(
        [
            {
                "message": {
                    "role": "assistant",
                    "content": "Retrain the selected NAS model to verify distributed accuracy.",
                    "tool_calls": [
                        {
                            "function": {
                                "name": "start_training",
                                "arguments": {
                                    "rounds": 5,
                                    "epochs": 10,
                                    "dataset_name": "cifar10",
                                    "architecture": selected_architecture,
                                },
                            }
                        }
                    ],
                }
            },
            {"message": {"role": "assistant", "content": "The selected NAS model is training."}},
        ]
    )
    executed: list[dict[str, object]] = []

    def fake_forward(settings, path, *, payload=None, query="", method=None):
        if path == "/results":
            return {
                "configuration": {},
                "active_group_id": None,
                "groups": [{
                    "group_id": "dataset-nas-1",
                    "kind": "dataset_architecture_search",
                    "dataset_name": "cifar10",
                    "status": "completed",
                    "selected_architecture": selected_architecture,
                }],
            }
        if path == "/agent/model":
            return next(responses)
        return {"recorded": True}

    monkeypatch.setattr(slm, "_forward", fake_forward)
    client.app.state.datasets.list_metadata = lambda: []

    def execute(action):
        executed.append(action)
        return {"available": True, "data": {"group_id": "retraining-1"}}

    client.app.state.autonomy._execute = execute
    response = client.post(
        "/api/slm/agent/chat",
        json={
            "message": "Validate the selected NAS model.",
            "workflow_dataset": "cifar10",
            "workflow_id": str(uuid4()),
            "workflow_previous_group_id": "dataset-nas-1",
        },
        headers=auth_headers,
    )

    assert response.status_code == 200
    assert executed == [{
        "action": "start_training",
        "parameters": {
            "rounds": 5,
            "epochs": 10,
            "dataset_name": "cifar10",
            "architecture": selected_architecture,
        },
    }]


def test_agent_snapshot_compacts_large_action_history() -> None:
    import json

    from fastapi import HTTPException

    from app.routers.slm import (
        AGENT_CONTEXT_CHAR_LIMIT,
        AGENT_HISTORY_CHAR_LIMIT,
        _bound_agent_context,
        _bound_agent_chat_history,
        _compact_agent_action_history,
        _compact_agent_tool_value,
    )

    actions = [
        {
            "action": "preprocess_dataset",
            "mode": "AGENT",
            "result_status": "executed",
            "dataset_name": "cifar10",
            "details": {
                "action": "preprocess_dataset",
                "mode": "AGENT",
                "result_status": "executed",
                "dataset_name": "cifar10",
                "reason": "Ship dataset " + ("x" * 1000),
                "parameters": {
                    "dataset_name": "cifar10",
                    "dataset_split": "dirichlet_a01",
                },
                "result": {"worker_payload": "y" * 5000},
                "evidence": {"snapshot": "z" * 5000},
            },
        }
        for _ in range(20)
    ]

    compacted = _compact_agent_action_history(actions)
    serialized = json.dumps(compacted, separators=(",", ":"))

    assert len(compacted) == 20
    assert len(serialized) < 12_000
    assert '"action":"preprocess_dataset"' in serialized
    assert '"mode":"AGENT"' in serialized
    assert '"dataset_split":"dirichlet_a01"' in serialized
    assert "worker_payload" not in serialized
    assert "snapshot" not in serialized

    bounded_context = _bound_agent_context(
        {
            "previous_actions": [
                {"action": f"recent-{index}", "details": "x" * 1800}
                for index in range(3)
            ]
        }
    )
    serialized_context = json.dumps(
        bounded_context,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    assert len(serialized_context) <= AGENT_CONTEXT_CHAR_LIMIT
    assert [action["action"] for action in bounded_context["previous_actions"]] == [
        "recent-0",
        "recent-1",
    ]

    with pytest.raises(HTTPException, match="safe size limit"):
        _bound_agent_context({"oversized": "x" * (AGENT_CONTEXT_CHAR_LIMIT + 1)})

    bounded_history = _bound_agent_chat_history(
        [
            {"role": "user", "content": "old " * 2000},
            {"role": "assistant", "content": "recent " * 200},
        ]
    )
    assert sum(len(message["content"]) for message in bounded_history) == AGENT_HISTORY_CHAR_LIMIT
    assert bounded_history[-1]["content"] == "recent " * 200

    compacted_tool_result = _compact_agent_tool_value(
        {
            "workers": {
                "worker1": {"weights_sha256": "a" * 64},
                "worker2": {"weights_sha256": "b" * 64},
            },
            "architecture": {
                "model_type": "small_cnn",
                "id": "b1_c8_k3",
                "layers": [{"type": "conv"}] * 100,
            },
            "confusion_matrix": [[0] * 10 for _ in range(10)],
        }
    )
    assert compacted_tool_result["workers"] == ["worker1", "worker2"]
    assert compacted_tool_result["architecture"]["id"] == "b1_c8_k3"
    assert "layers" not in compacted_tool_result["architecture"]
    assert "confusion_matrix" not in compacted_tool_result

    compacted_results = _compact_agent_tool_value(
        {
            "result": {
                "available": True,
                "data": {
                    "active_group_id": None,
                    "groups": [
                        {
                            "group_id": f"run-{index}",
                            "dataset_name": "mnist",
                            "status": "completed",
                            "runs": [{"worker_metrics": "x" * 5000}],
                        }
                        for index in range(10)
                    ],
                    "topologies": {"mesh": [["worker1", "worker2"]] * 100},
                },
            }
        }
    )
    serialized_results = json.dumps(compacted_results, separators=(",", ":"))
    assert len(serialized_results) < 2_000
    assert '"active_group_id":null' in serialized_results
    assert '"group_count":10' in serialized_results
    assert "topologies" not in serialized_results
    assert "worker_metrics" not in serialized_results


def test_agent_snapshot_compacts_topology_optimization_evidence() -> None:
    import json

    from app.routers.slm import _compact_agent_experiment_group

    candidates = [
        {
            "topology": topology,
            "alpha": 0.3,
            "completed_runs": 1,
            "mean_final_accuracy": accuracy,
            "mean_reward": 0.8,
            "mean_components": {"accuracy_gain": 1.0},
            "per_run_rewards": [{"worker_metrics": "x" * 1000}],
        }
        for topology, accuracy in (
            ("directed_ring", 0.962409),
            ("bidirectional_ring", 0.981484),
            ("mesh", 0.987686),
            ("chain", 0.9614),
            ("star", 0.9819),
        )
    ]
    group = {
        "group_id": "topology-search",
        "kind": "topology_optimization",
        "dataset_name": "mnist",
        "status": "completed",
        "selected_configuration": {
            "topology": "directed_ring",
            "alpha": 0.3,
            "dataset_split": "dirichlet_a03",
        },
        "runs": [{"worker_metrics": "y" * 5000}],
        "candidates": candidates,
    }

    compacted = _compact_agent_experiment_group(group)
    serialized = json.dumps(compacted, separators=(",", ":"))

    assert compacted["selected_validation_accuracy"] == 0.962409
    assert len(compacted["candidates"]) == 5
    assert "per_run_rewards" not in serialized
    assert "worker_metrics" not in serialized
    assert len(serialized) < 1_500


def test_agent_snapshot_uses_only_actions_from_the_bound_workflow() -> None:
    from app.routers.slm import _filter_agent_action_history

    actions = [
        {
            "action": "start_training",
            "details": {
                "dataset_name": "mnist",
                "evidence": {"workflow_id": "current-workflow"},
            },
        },
        {
            "action": "enable_nas",
            "details": {
                "dataset_name": "mnist",
                "evidence": {"workflow_id": "previous-workflow"},
            },
        },
        {
            "action": "start_training",
            "details": {
                "dataset_name": "cifar10",
                "evidence": {"workflow_id": "current-workflow"},
            },
        },
    ]

    selected = _filter_agent_action_history(
        actions,
        workflow_dataset="mnist",
        workflow_id="current-workflow",
    )

    assert selected == actions[:1]


def test_agent_workflow_stops_when_the_model_returns_no_native_tool_call(
    client: TestClient,
    auth_headers: dict[str, str],
    monkeypatch,
) -> None:
    from app.routers import slm

    def fake_forward(settings, path, *, payload=None, query="", method=None):
        if path == "/results":
            return {"configuration": {}, "active_group_id": None, "groups": []}
        if path == "/agent/model":
            return {
                "message": {
                    "role": "assistant",
                    "content": "I will call the unavailable inspect_results tool.",
                }
            }
        raise AssertionError(f"Unexpected upstream path: {path}")

    monkeypatch.setattr(slm, "_forward", fake_forward)
    client.app.state.datasets.list_metadata = lambda: []

    response = client.post(
        "/api/slm/agent/chat",
        json={
            "message": "Reach 95% validation accuracy on MNIST.",
            "workflow_dataset": "mnist",
        },
        headers=auth_headers,
    )

    assert response.status_code == 502
    assert response.json()["detail"] == (
        "The SLM returned no native tool call for this Agent workflow. "
        "The workflow was halted instead of repeating a no-progress decision."
    )


def test_slm_conversation_delete_removes_messages_and_requires_auth(
    client: TestClient,
    auth_headers: dict[str, str],
    monkeypatch,
) -> None:
    from app.routers import slm
    from app.db.models import SlmConversation, SlmMessage
    from datetime import datetime, timezone
    from uuid import uuid4

    conversation_id = str(uuid4())
    with client.app.state.session_factory.begin() as session:
        session.add(
            SlmConversation(
                id=conversation_id,
                title="Delete me",
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc),
            )
        )
        session.add(
            SlmMessage(
                conversation_id=conversation_id,
                role="user",
                content="This chat should be deleted",
                created_at=datetime.now(timezone.utc),
            )
        )

    monkeypatch.setattr(
        slm,
        "_forward",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("deleting chat history must not call the master")
        ),
    )
    path = f"/api/slm/conversations/{conversation_id}"
    assert client.delete(path).status_code == 401
    response = client.delete(path, headers=auth_headers)
    assert response.status_code == 204
    assert client.get(path, headers=auth_headers).status_code == 404
    with client.app.state.session_factory() as session:
        assert session.query(SlmMessage).filter_by(
            conversation_id=conversation_id
        ).count() == 0


def test_state_changing_command_executes_automatically(
    client: TestClient,
    auth_headers: dict[str, str],
    monkeypatch,
) -> None:
    from app.routers import slm

    monkeypatch.setattr(
        slm,
        "_forward",
        lambda settings, path, *, payload=None, query="", method=None: {"recorded": True},
    )
    client.app.state.autonomy._execute = lambda action: {"available": True}
    response = client.post(
        "/api/slm/command",
        json={"action": {"action": "set_rounds", "parameters": {"rounds": 8}}},
        headers=auth_headers,
    )
    assert response.status_code == 200
    assert response.json() == {"status": "executed", "result": {"available": True}}


def test_command_requires_wrapped_action_and_dispatches_once(
    client: TestClient,
    auth_headers: dict[str, str],
    monkeypatch,
) -> None:
    from app.routers import slm

    monkeypatch.setattr(
        slm,
        "_forward",
        lambda settings, path, *, payload=None, query="", method=None: {"recorded": True},
    )
    class FakeDispatcher:
        def __init__(self) -> None:
            self.calls = []

        def dispatch(self, action, *, reason, evidence):
            self.calls.append((action, reason, evidence))
            return {"status": "executed", "result": {"available": True}}

    fake = FakeDispatcher()
    client.app.state.autonomy = fake
    action = {
        "action": "optimize_topology",
        "parameters": {
            "rounds": 10,
            "epochs": 5,
            "seeds": [17, 42, 2026],
            "alphas": [0.3],
        },
    }

    flat = client.post("/api/slm/command", json=action, headers=auth_headers)
    assert flat.status_code == 422
    assert flat.json()["detail"] == "Unexpected command fields"
    assert fake.calls == []

    wrapped = client.post(
        "/api/slm/command",
        json={"action": action},
        headers=auth_headers,
    )
    assert wrapped.status_code == 200
    assert wrapped.json()["status"] == "executed"
    assert fake.calls == [(action, "Authenticated coordinator command", {})]


def test_experiment_results_are_forwarded(
    client: TestClient,
    auth_headers: dict[str, str],
    monkeypatch,
) -> None:
    from app.routers import slm

    forwarded = []

    def fake_forward(settings, path, *, payload=None, query=""):
        forwarded.append((settings.master_slm_url, settings.master_slm_token, path, payload, query))
        return {"groups": []}

    monkeypatch.setattr(slm, "_forward", fake_forward)
    results = client.get("/api/slm/results", headers=auth_headers)

    assert results.status_code == 200
    assert forwarded == [
        (
            "http://127.0.0.1:5003",
            "test-master-slm-token-that-is-long-and-random",
            "/results",
            None,
            "",
        ),
    ]

def test_delete_experiment_result_uses_authenticated_master_proxy(
    client: TestClient,
    auth_headers: dict[str, str],
    monkeypatch,
) -> None:
    from app.routers import slm

    forwarded = {}
    monkeypatch.setattr(
        slm,
        "_forward",
        lambda settings, path, **kwargs: forwarded.update(path=path, **kwargs)
        or {"deleted": True, "group_id": "nas-result"},
    )

    response = client.delete(
        "/api/slm/results/nas-result",
        headers=auth_headers,
    )

    assert response.status_code == 200
    assert forwarded == {"path": "/results/nas-result", "method": "DELETE"}


def test_slm_proxy_validates_chat_message_size(
    client: TestClient,
    auth_headers: dict[str, str],
) -> None:
    response = client.post(
        "/api/slm/agent/chat",
        json={"message": "x" * 4001},
        headers=auth_headers,
    )
    assert response.status_code == 422
