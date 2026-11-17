import json
import sqlite3
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator, ValidationError

from master.app import (
    ACTION_NAMES,
    ACTION_SCHEMA,
    ServiceConfig,
    create_app,
)


def config(tmp_path: Path) -> ServiceConfig:
    return ServiceConfig(
        api_token="slm-test-token-that-is-at-least-32-characters",
        status_api_token="status-test-token-that-is-at-least-32-characters",
        chat_api_token="browser-chat-token-that-is-at-least-32-characters",
        log_path=str(tmp_path / "events.jsonl"),
        action_history_path=str(tmp_path / "action_history.sqlite3"),
    )


def memory() -> dict[str, int]:
    return {
        "total_bytes": 8_000_000_000,
        "available_bytes": 6_000_000_000,
        "used_bytes": 2_000_000_000,
    }


def test_schema_allows_exactly_the_declared_actions() -> None:
    Draft202012Validator.check_schema(ACTION_SCHEMA)
    names = {
        branch["properties"]["action"]["const"]
        for branch in ACTION_SCHEMA["oneOf"]
    }
    assert names == set(ACTION_NAMES)
    assert len(names) == 17


def test_agent_model_forwards_schema_tools_to_ollama(
    tmp_path: Path,
    monkeypatch,
) -> None:
    import master.app as service_module

    app = create_app(config(tmp_path))
    forwarded = {}

    def fake_ollama(settings, path, payload):
        forwarded.update({"path": path, "payload": payload})
        return {
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "function": {
                            "name": "optimize_topology",
                            "arguments": {"rounds": 10, "epochs": 5},
                        }
                    }
                ],
            }
        }

    monkeypatch.setattr(service_module, "_ollama_request", fake_ollama)
    response = app.test_client().post(
        "/agent/model",
        headers={"Authorization": f"Bearer {config(tmp_path).api_token}"},
        json={"messages": [{"role": "user", "content": "Optimize the topology."}]},
    )

    assert response.status_code == 200
    assert response.json["message"]["tool_calls"][0]["function"]["name"] == "optimize_topology"
    assert forwarded["path"] == "/api/chat"
    assert forwarded["payload"]["model"] == "qwen3.5:4b"
    assert forwarded["payload"]["think"] is False
    assert forwarded["payload"]["options"]["num_ctx"] == 4096
    assert forwarded["payload"]["options"]["temperature"] == 0.1
    tools = forwarded["payload"]["tools"]
    assert {
        tool["function"]["name"] for tool in tools
    } == {
        "preprocess_dataset",
        "create_initial_model",
        "start_training",
        "stop_training",
        "set_topology",
        "optimize_topology",
        "set_rounds",
        "set_epochs",
        "enable_nas",
        "get_status",
        "get_logs",
        "get_results",
        "compare_runs",
        "finish_agent_workflow",
    }
    initial_model_tool = next(
        tool["function"]
        for tool in tools
        if tool["function"]["name"] == "create_initial_model"
    )
    assert initial_model_tool["parameters"]["required"] == ["model_name"]
    assert list(initial_model_tool["parameters"]["properties"]) == ["model_name"]


def test_agent_model_rejects_malformed_messages(tmp_path: Path) -> None:
    app = create_app(config(tmp_path))
    response = app.test_client().post(
        "/agent/model",
        headers={"Authorization": f"Bearer {config(tmp_path).api_token}"},
        json={"messages": [{"role": "system", "content": []}]},
    )
    assert response.status_code == 400


def test_action_schema_accepts_valid_values_and_rejects_ranges_and_injection() -> None:
    validator = Draft202012Validator(ACTION_SCHEMA)
    validator.validate({"action": "set_rounds", "parameters": {"rounds": 10}})
    validator.validate(
        {
            "action": "start_training",
            "parameters": {"rounds": 10, "epochs": 5, "dataset_name": "mnist"},
        }
    )
    validator.validate(
        {
            "action": "optimize_topology",
            "parameters": {"alphas": [0.3]},
        }
    )
    for invalid in (
        {"action": "set_rounds", "parameters": {"rounds": 0}},
        {"action": "set_epochs", "parameters": {"epochs": 51}},
        {"action": "run_shell", "parameters": {"command": "whoami"}},
        {
            "action": "get_status",
            "parameters": {},
            "shell": "touch /tmp/pwned",
        },
        {"action": "set_topology", "parameters": {"topology": "execute"}},
        {
            "action": "optimize_topology",
            "parameters": {"rounds": 0, "epochs": 5},
        },
        {
            "action": "optimize_topology",
            "parameters": {"alphas": [0.5]},
        },
        {
            "action": "optimize_topology",
            "parameters": {"alphas": [0.3, 0.5]},
        },
    ):
        try:
            validator.validate(invalid)
        except ValidationError:
            continue
        raise AssertionError(f"Invalid action was accepted: {invalid}")


def test_conversational_outcomes_are_schema_valid_and_have_no_side_effects(
    tmp_path: Path,
) -> None:
    app = create_app(config(tmp_path), memory_reader=memory)
    client = app.test_client()
    for outcome in ("clarify", "refuse", "redirect"):
        response = client.post(
            "/command",
            json={"action": outcome, "parameters": {}},
            headers={"Authorization": "Bearer slm-test-token-that-is-at-least-32-characters"},
        )
        assert response.status_code == 200
        assert response.get_json()["result"] == {
            "available": True,
            "data": {"outcome": outcome, "action_required": False},
        }


def test_all_slm_endpoints_reject_missing_and_wrong_tokens(tmp_path: Path) -> None:
    app = create_app(config(tmp_path), memory_reader=memory)
    client = app.test_client()
    for method, path, payload in (
        ("get", "/status", None),
        ("get", "/logs", None),
        ("post", "/chat", {"message": "show status"}),
        ("post", "/reply", {"message": "what next", "context": {}}),
        ("post", "/command", {"action": "get_status", "parameters": {}}),
    ):
        response = getattr(client, method)(path, json=payload) if payload else getattr(
            client, method
        )(path)
        assert response.status_code == 401
        assert response.headers["WWW-Authenticate"] == "Bearer"
    unauthorized = client.get(
        "/status",
        headers={"Authorization": "Bearer incorrect-token"},
    )
    assert unauthorized.status_code == 401


def test_command_persists_results_for_every_successful_action(
    tmp_path: Path,
) -> None:
    class TrainingCoordinator:
        def configure(self, **kwargs):
            if "topology" in kwargs:
                raise ValueError("Topology update was rejected")
            return {"configuration": kwargs}

    app = create_app(
        config(tmp_path),
        status_reader=lambda: {"node_id": "master"},
        training_coordinator=TrainingCoordinator(),
    )
    client = app.test_client()
    headers = {"Authorization": "Bearer " + config(tmp_path).api_token}
    changed = client.post(
        "/command",
        json={"action": "set_epochs", "parameters": {"epochs": 3}},
        headers=headers,
    )
    read_only = client.post(
        "/command",
        json={"action": "get_status", "parameters": {}},
        headers=headers,
    )
    unavailable = client.post(
        "/command",
        json={"action": "set_topology", "parameters": {"topology": "mesh"}},
        headers=headers,
    )

    assert changed.status_code == 200
    assert read_only.status_code == 200
    assert unavailable.status_code == 200
    assert unavailable.json["result"]["available"] is False
    from master.app import _recent_action_history

    history = _recent_action_history(config(tmp_path).action_history_path)
    assert len(history) == 2
    assert history[0]["action"] == "get_status"
    assert history[0]["result"]["data"] == {"node_id": "master"}
    assert history[1]["action"] == "set_epochs"
    assert history[1]["parameters"] == {"epochs": 3}
    assert history[1]["result"]["data"] == {"configuration": {"epochs": 3}}


def test_rag_chat_retrieves_action_results_and_persists_conversation_history(
    tmp_path: Path,
) -> None:
    contexts: list[dict] = []

    def generate_rag(message: str, context: dict) -> tuple[str, float]:
        contexts.append(context)
        assert message
        return "The dataset was prepared on five workers. Create the initial model next.", 12.0

    settings = config(tmp_path)
    app = create_app(settings, generate_rag=generate_rag)
    client = app.test_client()
    headers = {"Authorization": f"Bearer {settings.chat_api_token}"}
    action = {
        "action": "preprocess_dataset",
        "parameters": {"dataset_name": "mnist"},
    }
    recorded = client.post(
        "/action-history",
        json={
            "action": action,
            "result": {
                "available": True,
                "data": {"worker_count": 5, "shards": ["worker1", "worker2", "worker3", "worker4", "worker5"]},
            },
        },
        headers={"Authorization": f"Bearer {settings.api_token}"},
    )
    first = client.post(
        "/rag/chat",
        json={"message": "Dataset was preprocessed and shipped to five workers. What next?"},
        headers=headers,
    )

    assert recorded.status_code == 200
    assert first.status_code == 200
    first_body = first.get_json()
    conversation_id = first_body["conversation_id"]
    assert set(first_body) == {"conversation_id", "reply"}
    assert first_body["reply"].startswith("The dataset was prepared")
    assert contexts[0]["recent_chat_history"] == []
    assert contexts[0]["relevant_performed_actions"][0]["result"]["data"]["worker_count"] == 5

    followup = client.post(
        "/rag/chat",
        json={
            "message": "Which workers received the shards?",
            "conversation_id": conversation_id,
        },
        headers=headers,
    )
    conversation = client.get(
        f"/rag/conversations/{conversation_id}",
        headers=headers,
    )
    summaries = client.get("/rag/conversations", headers=headers)
    deleted = client.delete(
        f"/rag/conversations/{conversation_id}",
        headers=headers,
    )
    after_delete = client.get(
        f"/rag/conversations/{conversation_id}",
        headers=headers,
    )

    assert followup.status_code == 200
    assert contexts[1]["recent_chat_history"] == [
        {"role": "user", "content": "Dataset was preprocessed and shipped to five workers. What next?"},
        {"role": "assistant", "content": "The dataset was prepared on five workers. Create the initial model next."},
    ]
    assert conversation.status_code == 200
    assert [message["role"] for message in conversation.get_json()["messages"]] == [
        "user",
        "assistant",
        "user",
        "assistant",
    ]
    assert summaries.status_code == 200
    assert summaries.get_json()[0]["id"] == conversation_id
    assert summaries.get_json()[0]["message_count"] == 4
    assert deleted.status_code == 204
    assert after_delete.status_code == 404


def test_rag_chat_includes_live_completed_training_results(
    tmp_path: Path,
) -> None:
    contexts: list[dict] = []

    class TrainingCoordinator:
        def results(self):
            return {
                "active_group_id": None,
                "groups": [{
                    "group_id": "mnist-ring-complete",
                    "kind": "decentralized",
                    "dataset_name": "mnist",
                    "status": "completed",
                    "rounds": 1,
                    "epochs": 5,
                    "runs": [{
                        "run_id": "mnist-seed-17",
                        "status": "completed",
                        "seed": 17,
                        "rounds": [{
                            "round": 1,
                            "workers": {
                                "worker1": {
                                    "validation_accuracy": 0.94,
                                    "validation_macro_precision": 0.93,
                                    "validation_macro_recall": 0.92,
                                    "validation_macro_f1": 0.925,
                                    "validation_sample_count": 1200,
                                    "training_loss": 0.12,
                                    "validation_confusion_matrix_counts": [[0] * 10 for _ in range(10)],
                                },
                            },
                        }],
                    }],
                }],
            }

    def generate_rag(message: str, context: dict) -> tuple[str, float]:
        contexts.append(context)
        assert message == "training is completed now after training what should i do?"
        return "Training is complete. Review the final metrics and compare them with saved runs.", 12.0

    settings = config(tmp_path)
    app = create_app(
        settings,
        generate_rag=generate_rag,
        training_coordinator=TrainingCoordinator(),
    )
    response = app.test_client().post(
        "/rag/chat",
        json={"message": "training is completed now after training what should i do?"},
        headers={"Authorization": f"Bearer {settings.chat_api_token}"},
    )

    assert response.status_code == 200
    training_context = contexts[0]["live_training_results"]
    assert training_context["active_group_id"] is None
    group = training_context["recent_groups"][0]
    assert group["group_id"] == "mnist-ring-complete"
    assert group["status"] == "completed"
    assert group["runs"][0]["completed_rounds"] == 1
    assert group["runs"][0]["final_round"] == 1
    assert group["runs"][0]["final_worker_metrics"]["worker1"]["validation_accuracy"] == 0.94


def test_rag_chat_token_is_scoped_and_cors_is_restricted(tmp_path: Path) -> None:
    settings = config(tmp_path)
    app = create_app(
        settings,
        generate_rag=lambda message, context: ("A factual answer.", 1.0),
    )
    client = app.test_client()
    origin = "http://192.168.11.12:5173"
    chat_headers = {
        "Authorization": f"Bearer {settings.chat_api_token}",
        "Origin": origin,
    }
    preflight = client.options("/rag/chat", headers={"Origin": origin})
    chat = client.post("/rag/chat", json={"message": "What happened?"}, headers=chat_headers)
    denied_master_route = client.get("/status", headers=chat_headers)
    denied_origin = client.options(
        "/rag/chat",
        headers={"Origin": "https://untrusted.example"},
    )

    assert preflight.status_code == 204
    assert preflight.headers["Access-Control-Allow-Origin"] == origin
    assert chat.status_code == 200
    assert chat.headers["Access-Control-Allow-Origin"] == origin
    assert denied_master_route.status_code == 401
    assert denied_origin.status_code == 403


def test_rag_retrieval_bounds_large_action_results(tmp_path: Path) -> None:
    from master.app import (
        RAG_CONTEXT_LIMIT,
        _initialize_action_history,
        _record_action_history,
        _retrieve_rag_context,
    )

    path = config(tmp_path).action_history_path
    _initialize_action_history(path)
    for index in range(8):
        _record_action_history(
            path,
            {
                "action": "preprocess_dataset",
                "parameters": {"dataset_name": f"mnist-{index}"},
            },
            {"available": True, "data": {"payload": "x" * 4000}},
        )

    context = _retrieve_rag_context(path, None, "dataset preprocessing results")

    assert context is not None
    assert len(json.dumps(context, ensure_ascii=False, separators=(",", ":"))) <= RAG_CONTEXT_LIMIT
    assert len(context["relevant_performed_actions"]) < 5
    assert context["relevant_performed_actions"][0]["result"]["truncated"] is True


def test_rag_retrieval_preserves_recent_chat_before_large_action_history(
    tmp_path: Path,
) -> None:
    from master.app import (
        RAG_CONTEXT_LIMIT,
        _initialize_action_history,
        _record_action_history,
        _retrieve_rag_context,
    )

    path = config(tmp_path).action_history_path
    conversation_id = "nas-completion-context"
    _initialize_action_history(path)
    with sqlite3.connect(path) as connection:
        connection.execute(
            """
            INSERT INTO slm_conversations (id, title, created_at, updated_at)
            VALUES (?, ?, ?, ?)
            """,
            (conversation_id, "NAS completed", "now", "now"),
        )
        connection.execute(
            """
            INSERT INTO slm_messages (conversation_id, role, content, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (conversation_id, "user", "I have already performed NAS.", "now"),
        )
    for index in range(8):
        _record_action_history(
            path,
            {
                "action": "preprocess_dataset",
                "parameters": {"dataset_name": f"mnist-{index}"},
            },
            {"available": True, "data": {"payload": "x" * 4000}},
        )

    context = _retrieve_rag_context(
        path,
        conversation_id,
        "What should I do next?",
    )

    assert context is not None
    assert len(json.dumps(context, ensure_ascii=False, separators=(",", ":"))) <= RAG_CONTEXT_LIMIT
    assert context["recent_chat_history"] == [
        {"role": "user", "content": "I have already performed NAS."}
    ]


def test_rag_retrieval_preserves_preprocessing_and_worker_upload_summary(
    tmp_path: Path,
) -> None:
    from master.app import (
        _initialize_action_history,
        _record_action_history,
        _retrieve_rag_context,
    )

    path = config(tmp_path).action_history_path
    _initialize_action_history(path)
    worker_ids = [f"worker{index}" for index in range(1, 6)]
    _record_action_history(
        path,
        {
            "action": "preprocess_dataset",
            "parameters": {"dataset_name": "mnist"},
        },
        {
            "available": True,
            "data": {
                "dataset_name": "mnist",
                "image_count": 70000,
                "target_size": {"height": 28, "width": 28},
                "channels": 1,
                "image_shape": [28, 28, 1],
                "normalization": {"method": "standard"},
                "split_strategy": "stratified",
                "split_method": "official_mnist_train_validation_test",
                "official_test_set_preserved": True,
                "split_sizes": {"train": 54001, "validation": 5999, "test": 10000},
                "split_indices_disjoint": True,
                "worker_shards_disjoint": True,
                "uploaded_workers": {
                    worker_id: {"rows": 10800, "shard_sha256": "a" * 64}
                    for worker_id in worker_ids
                },
                "large_detail": "x" * 4000,
            },
        },
    )

    context = _retrieve_rag_context(path, None, "What should I do next for MNIST?")

    assert context is not None
    action_result = context["relevant_performed_actions"][0]["result"]
    assert action_result["truncated"] is True
    summary = action_result["summary"]["data"]
    assert summary["dataset_name"] == "mnist"
    assert summary["split_sizes"] == {
        "train": 54001,
        "validation": 5999,
        "test": 10000,
    }
    assert list(summary["uploaded_workers"]) == worker_ids
    assert summary["uploaded_workers"]["worker5"] == 10800
    assert "large_detail" not in summary


def test_action_history_initialization_migrates_existing_database(tmp_path: Path) -> None:
    from master.app import _initialize_action_history, _recent_action_history

    path = tmp_path / "existing-action-history.sqlite3"
    with sqlite3.connect(path) as connection:
        connection.execute(
            """
            CREATE TABLE slm_action_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                action TEXT NOT NULL,
                parameters_json TEXT NOT NULL,
                action_timestamp TEXT NOT NULL,
                dataset_name TEXT,
                run_id TEXT
            )
            """
        )
        connection.execute(
            """
            INSERT INTO slm_action_history
                (action, parameters_json, action_timestamp, dataset_name, run_id)
            VALUES (?, ?, ?, ?, ?)
            """,
            ("set_epochs", '{"epochs":3}', "2026-11-06T00:00:00+00:00", None, None),
        )

    _initialize_action_history(str(path))
    history = _recent_action_history(str(path))

    assert history[0]["action"] == "set_epochs"
    assert history[0]["result"] == {}


def test_training_results_context_tracks_most_recent_completed_workflow() -> None:
    from master.app import _training_results_context

    class TrainingCoordinator:
        def results(self):
            return {
                "active_group_id": None,
                "groups": [
                    {"kind": "decentralized", "status": "completed"},
                    {"kind": "dataset_architecture_search", "status": "completed"},
                    {"kind": "topology_optimization", "status": "completed"},
                ],
            }

    context = _training_results_context(TrainingCoordinator())

    assert context["latest_completed_workflow"] == "perform topology learning"


def test_training_results_context_tracks_completed_workflows_beyond_recent_groups() -> None:
    from master.app import _training_results_context

    class TrainingCoordinator:
        def results(self):
            return {
                "active_group_id": None,
                "groups": [
                    {"kind": "dataset_architecture_search", "status": "completed"},
                    {"kind": "topology_optimization", "status": "failed"},
                    {"kind": "decentralized", "status": "completed"},
                    {"kind": "centralized_baseline", "status": "completed"},
                ],
            }

    context = _training_results_context(TrainingCoordinator())

    assert context["completed_workflows"] == ["perform NAS", "perform training"]
    assert len(context["recent_groups"]) == 3


def test_status_endpoint_reads_hardware_and_ollama_metadata(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from master import app as service_module

    status_payload = {
        "node_id": "master",
        "hostname": "dfl-master",
        "cpu_utilization_percent": 3.0,
    }
    monkeypatch.setattr(
        service_module,
        "_ollama_request",
        lambda *_args: {
            "models": [{"name": "qwen3.5:4b", "size": 4_000_000_000}],
        },
    )
    app = service_module.create_app(
        config(tmp_path),
        status_reader=lambda: status_payload,
        memory_reader=memory,
    )
    response = app.test_client().get(
        "/status",
        headers={"Authorization": "Bearer slm-test-token-that-is-at-least-32-characters"},
    )
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["device_status"] == status_payload
    assert payload["ollama"]["model_available"] is True
    assert payload["ollama"]["model_size_bytes"] == 4_000_000_000
    assert payload["ollama"]["context_tokens"] == 4096
    assert payload["memory"] == memory()


def test_unimplemented_command_returns_not_available_without_success(tmp_path: Path) -> None:
    app = create_app(config(tmp_path), memory_reader=memory)
    response = app.test_client().post(
        "/command",
        json={
            "action": "start_training",
            "parameters": {"rounds": 3, "epochs": 2, "dataset_name": "mnist"},
        },
        headers={"Authorization": "Bearer slm-test-token-that-is-at-least-32-characters"},
    )
    assert response.status_code == 200
    result = response.get_json()["result"]
    assert result["available"] is False
    assert "not available yet" in result["message"]


def test_invalid_direct_command_is_rejected(tmp_path: Path) -> None:
    app = create_app(config(tmp_path), memory_reader=memory)
    response = app.test_client().post(
        "/command",
        json={"action": "enable_nas", "parameters": {"enabled": "yes"}},
        headers={"Authorization": "Bearer slm-test-token-that-is-at-least-32-characters"},
    )
    assert response.status_code == 422


def test_get_logs_returns_real_service_events(tmp_path: Path) -> None:
    app = create_app(
        config(tmp_path),
        status_reader=lambda: {"node_id": "master", "hostname": "dfl-master"},
        memory_reader=memory,
    )
    client = app.test_client()
    response = client.post(
        "/command",
        json={"action": "get_status", "parameters": {}},
        headers={"Authorization": "Bearer slm-test-token-that-is-at-least-32-characters"},
    )
    assert response.status_code == 200
    logs = client.get(
        "/logs?limit=5",
        headers={"Authorization": "Bearer slm-test-token-that-is-at-least-32-characters"},
    )
    assert logs.status_code == 200
    assert any(row["event"] == "command_completed" for row in logs.get_json()["logs"])


def test_training_actions_dispatch_to_configured_worker_coordinator(tmp_path: Path) -> None:
    class FakeCoordinator:
        def __init__(self):
            self.configuration = {"rounds": 10, "epochs": 5, "topology": "ring"}

        def configure(self, **values):
            self.configuration.update(values)
            return dict(self.configuration)

        def start_training(
            self,
            *,
            rounds,
            epochs,
            dataset_split=None,
            dataset_name=None,
            architecture=None,
        ):
            return {
                "accepted": True,
                "rounds": rounds,
                "epochs": epochs,
                "dataset_split": dataset_split,
                "dataset_name": dataset_name,
                "architecture": architecture,
            }

        def start_optimization(self, **values):
            return {"accepted": True, **values}

        def stop_training(self):
            return {"accepted": True}

        def results(self):
            return {"groups": [{"status": "completed"}]}

    app = create_app(
        config(tmp_path),
        memory_reader=memory,
        training_coordinator=FakeCoordinator(),
    )
    client = app.test_client()
    headers = {"Authorization": "Bearer slm-test-token-that-is-at-least-32-characters"}
    requests = (
        ({"action": "set_rounds", "parameters": {"rounds": 4}}, {"rounds": 4}),
        ({"action": "set_epochs", "parameters": {"epochs": 2}}, {"epochs": 2}),
        ({"action": "set_topology", "parameters": {"topology": "mesh"}}, {"topology": "mesh"}),
    )
    for payload, expected in requests:
        response = client.post("/command", json=payload, headers=headers)
        assert response.status_code == 200
        assert response.get_json()["result"]["data"] | expected == response.get_json()["result"]["data"]

    started = client.post(
        "/command",
        json={
            "action": "start_training",
            "parameters": {
                "rounds": 3,
                "epochs": 2,
                "dataset_name": "mnist",
                "architecture": {"id": "b1_c16_k5", "blocks": 1, "channels": 16, "kernel_size": 5},
            },
        },
        headers=headers,
    )
    assert started.status_code == 200
    assert started.get_json()["result"]["data"] == {
        "accepted": True,
        "rounds": 3,
        "epochs": 2,
        "dataset_split": None,
        "dataset_name": "mnist",
        "architecture": {"id": "b1_c16_k5", "blocks": 1, "channels": 16, "kernel_size": 5},
    }
    stopped = client.post(
        "/command",
        json={"action": "stop_training", "parameters": {}},
        headers=headers,
    )
    assert stopped.status_code == 200
    results = client.post(
        "/command",
        json={"action": "get_results", "parameters": {}},
        headers=headers,
    )
    assert results.status_code == 200
    assert results.get_json()["result"]["data"]["groups"][0]["status"] == "completed"
    assert client.get("/results", headers=headers).status_code == 200


def test_optimizer_action_defaults_to_only_dirichlet_alpha_03(tmp_path: Path) -> None:
    class FakeCoordinator:
        def start_optimization(self, **values):
            return {"accepted": True, **values}

    app = create_app(
        config(tmp_path),
        memory_reader=memory,
        training_coordinator=FakeCoordinator(),
    )
    response = app.test_client().post(
        "/command",
        json={
            "action": "optimize_topology",
            "parameters": {"rounds": 1, "epochs": 1, "seeds": [17]},
        },
        headers={"Authorization": f"Bearer {config(tmp_path).api_token}"},
    )

    assert response.status_code == 200
    result = response.get_json()["result"]["data"]
    assert result["rounds"] == 1
    assert result["epochs"] == 1
    assert result["seeds"] == [17]
    assert result["alphas"] == [0.3]

    assert result["alphas"] == [0.3]

def test_enable_nas_starts_search_on_mnist_by_default(tmp_path: Path) -> None:
    class FakeCoordinator:
        def start_dataset_nas(self, **values):
            return {"accepted": True, **values}

    app = create_app(
        config(tmp_path),
        memory_reader=memory,
        training_coordinator=FakeCoordinator(),
    )
    response = app.test_client().post(
        "/command",
        json={
            "action": "enable_nas",
            "parameters": {
                "enabled": True,
                "search_time_limit_seconds": 90,
                "smoke_test": True,
            },
        },
        headers={"Authorization": f"Bearer {config(tmp_path).api_token}"},
    )
    assert response.status_code == 200
    data = response.get_json()["result"]["data"]
    assert data["dataset_name"] == "mnist"
    assert data["search_time_limit_seconds"] == 90
    assert data["smoke_test"] is True


def test_delete_experiment_result_forwards_to_coordinator(tmp_path: Path) -> None:
    class FakeCoordinator:
        def delete_group(self, group_id):
            return {"deleted": True, "group_id": group_id}

    app = create_app(
        config(tmp_path),
        memory_reader=memory,
        training_coordinator=FakeCoordinator(),
    )
    response = app.test_client().delete(
        "/results/dataset-nas-result-1",
        headers={"Authorization": f"Bearer {config(tmp_path).api_token}"},
    )

    assert response.status_code == 200
    assert response.get_json() == {"deleted": True, "group_id": "dataset-nas-result-1"}


def test_enable_nas_uses_the_only_enabled_dataset_when_name_is_omitted(tmp_path: Path) -> None:
    class FakeCoordinator:
        def start_dataset_nas(self, **_values):
            return {"dataset_name": _values["dataset_name"]}

    app = create_app(
        config(tmp_path),
        memory_reader=memory,
        training_coordinator=FakeCoordinator(),
    )
    response = app.test_client().post(
        "/command",
        json={"action": "enable_nas", "parameters": {"enabled": True}},
        headers={"Authorization": "Bearer " + config(tmp_path).api_token},
    )
    assert response.status_code == 200
    data = response.get_json()["result"]["data"]
    assert data["dataset_name"] == "mnist"


def test_enable_nas_rejects_out_of_range_search_budget(tmp_path: Path) -> None:
    app = create_app(config(tmp_path), memory_reader=memory)
    response = app.test_client().post(
        "/command",
        json={
            "action": "enable_nas",
            "parameters": {"enabled": True, "search_time_limit_seconds": 601},
        },
        headers={"Authorization": f"Bearer {config(tmp_path).api_token}"},
    )
    assert response.status_code == 422


def test_supported_topology_is_reported_as_available(tmp_path: Path) -> None:
    class FakeCoordinator:
        def configure(self, **values):
            return values

    app = create_app(
        config(tmp_path),
        memory_reader=memory,
        training_coordinator=FakeCoordinator(),
    )
    response = app.test_client().post(
        "/command",
        json={"action": "set_topology", "parameters": {"topology": "star"}},
        headers={"Authorization": "Bearer slm-test-token-that-is-at-least-32-characters"},
    )

    assert response.status_code == 200
    assert response.get_json()["result"]["available"] is True
    assert response.get_json()["result"]["data"] == {"topology": "star"}


def test_create_app_initializes_coordinator_from_base64_worker_config(
    tmp_path: Path,
    monkeypatch,
) -> None:
    import base64
    import json

    from master import training

    coordinator = object()
    monkeypatch.delenv("DFL_WORKERS_JSON", raising=False)
    workers = [
        {"node_id": f"worker{index}", "base_url": f"http://127.0.0.1:{5003 + index}", "token": "x" * 32}
        for index in range(1, 5)
    ]
    worker_json = json.dumps(workers).encode("utf-8")
    monkeypatch.setenv("DFL_WORKERS_JSON_B64", base64.b64encode(worker_json).decode("ascii"))
    monkeypatch.setattr(training, "TrainingCoordinator", lambda config: coordinator)

    app = create_app(config(tmp_path), memory_reader=memory)

    assert app.config["TRAINING_COORDINATOR"] is coordinator


def test_cluster_worker_inventory_endpoint_authenticates_and_updates_coordinator(
    tmp_path: Path,
) -> None:
    class FakeCoordinator:
        def __init__(self) -> None:
            self.workers = None

        def configure_workers(self, workers):
            self.workers = workers
            return {"configured_workers": [worker.node_id for worker in workers]}

    coordinator = FakeCoordinator()
    app = create_app(config(tmp_path), memory_reader=memory, training_coordinator=coordinator)
    client = app.test_client()
    workers = [
        {
            "node_id": f"worker{index}",
            "base_url": f"http://127.0.0.1:{5003 + index}",
            "token": f"worker{index}-token-" + "x" * 32,
        }
        for index in range(1, 5)
    ]

    unauthenticated = client.put("/cluster/workers", json={"workers": workers})
    response = client.put(
        "/cluster/workers",
        json={"workers": workers},
        headers={"Authorization": "Bearer slm-test-token-that-is-at-least-32-characters"},
    )

    assert unauthenticated.status_code == 401
    assert response.status_code == 200
    assert response.get_json()["configured_workers"] == [f"worker{i}" for i in range(1, 5)]
    assert [worker.node_id for worker in coordinator.workers] == [f"worker{i}" for i in range(1, 5)]


def test_rag_reply_uses_the_local_slm_with_context_and_configured_limits(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from master import app as service_module

    calls = []

    def fake_ollama(_config, path, payload):
        calls.append((path, payload))
        return {"message": {"role": "assistant", "content": " MNIST is ready. Recommended next action: perform training. "}}

    monkeypatch.setattr(service_module, "_ollama_request", fake_ollama)
    reply, latency = service_module._generate_rag_reply(
        config(tmp_path),
        "What should I do after preprocessing?",
        {"relevant_performed_actions": [{"action": "preprocess_dataset", "result": {"data": {"worker_count": 5}}}]},
    )

    assert reply == "MNIST is ready. Recommended next action: perform training."
    assert latency >= 0
    path, payload = calls[0]
    assert path == "/api/chat"
    assert payload["model"] == "qwen3.5:4b"
    assert payload["think"] is False
    assert payload["stream"] is False
    assert payload["options"]["num_ctx"] == 4096
    assert payload["keep_alive"] == "15m"
    assert "tools" not in payload
    assert payload["messages"][0]["role"] == "system"
    assert "only advise" in payload["messages"][0]["content"]
    user_content = payload["messages"][1]["content"]
    assert '"worker_count":5' in user_content
    assert user_content.endswith("What should I do after preprocessing?")


def test_rag_reply_falls_back_to_rule_based_suggestion_when_slm_is_unavailable(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from master import app as service_module

    def unavailable(*_args):
        raise service_module.ServiceError("Could not reach the local Ollama service", 503)

    monkeypatch.setattr(service_module, "_ollama_request", unavailable)
    reply, _latency = service_module._generate_rag_reply(
        config(tmp_path),
        "What next?",
        {
            "live_training_results": {
                "active_group_id": None,
                "recent_groups": [{"status": "completed", "kind": "decentralized"}],
            }
        },
    )

    assert "rule-based suggestion" in reply
    assert reply.endswith("Recommended next action: perform NAS.")


@pytest.mark.parametrize(
    ("name", "value"),
    [("DFL_OLLAMA_NUM_CTX", "1024"), ("DFL_OLLAMA_KEEP_ALIVE", "forever")],
)
def test_service_config_rejects_invalid_ollama_limits(monkeypatch, name: str, value: str) -> None:
    monkeypatch.setenv("DFL_API_TOKEN", "slm-test-token-that-is-at-least-32-characters")
    monkeypatch.setenv("DFL_STATUS_API_TOKEN", "status-test-token-that-is-at-least-32-characters")
    monkeypatch.setenv(name, value)

    with pytest.raises(ValueError, match=name):
        ServiceConfig.from_env()
