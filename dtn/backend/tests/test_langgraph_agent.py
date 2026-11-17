from app.services.langgraph_agent import run_agent


def test_graph_executes_model_selected_tool_then_returns_model_summary() -> None:
    model_messages = []
    tool_calls = []

    def model_call(messages):
        model_messages.append(messages)
        if len(model_messages) == 1:
            return {
                "role": "assistant",
                "content": "The requested check requires reading the current worker status.",
                "tool_calls": [
                    {
                        "function": {
                            "name": "get_status",
                            "arguments": {},
                        }
                    }
                ],
            }
        return {
            "role": "assistant",
            "content": "All workers are online.",
        }

    def tool_call(name, arguments):
        tool_calls.append((name, arguments))
        return {"status": "executed", "result": {"online_workers": 3}}

    response = run_agent(
        messages=[{"role": "user", "content": "Check the worker state."}],
        model_call=model_call,
        tool_call=tool_call,
        max_tool_calls=4,
    )

    assert response["reply"] == "All workers are online."
    assert response["tool_calls"] == 1
    assert response["trace"][0]["tool"] == "get_status"
    assert response["trace"][0]["decision"] == (
        "The requested check requires reading the current worker status."
    )
    assert tool_calls == [("get_status", {})]
    assert model_messages[1][-1]["role"] == "tool"


def test_graph_returns_failed_tool_result_before_the_next_model_decision() -> None:
    model_messages = []
    model_calls = 0

    def model_call(messages):
        nonlocal model_calls
        model_calls += 1
        model_messages.append(messages)
        if model_calls == 1:
            return {
                "role": "assistant",
                "content": "Try the selected experiment.",
                "tool_calls": [{
                    "function": {
                        "name": "start_training",
                        "arguments": {"dataset_name": "cifar10", "rounds": -1},
                    },
                }],
            }
        if model_calls == 2:
            assert messages[-1]["role"] == "tool"
            assert "rounds must be between 1 and 100" in messages[-1]["content"]
            return {
                "role": "assistant",
                "content": "The experiment was rejected; I will inspect the current results instead.",
                "tool_calls": [{
                    "function": {
                        "name": "get_results",
                        "arguments": {},
                    },
                }],
            }
        return {"role": "assistant", "content": "The current results are recorded."}

    response = run_agent(
        messages=[{"role": "user", "content": "Improve the experiment."}],
        model_call=model_call,
        tool_call=lambda name, arguments: {
            "error": "rounds must be between 1 and 100"
        } if name == "start_training" else {"status": "executed"},
        max_tool_calls=3,
    )

    assert model_calls == 3
    assert response["trace"][0]["result"]["error"] == "rounds must be between 1 and 100"
    assert response["trace"][1]["tool"] == "get_results"
    assert model_messages[1][-1]["content"] == (
        '{"error": "rounds must be between 1 and 100"}'
    )


def test_graph_retries_invalid_tool_call_and_stops_after_valid_action() -> None:
    model_messages = []
    model_calls = 0
    tool_calls = []

    def model_call(messages):
        nonlocal model_calls
        model_calls += 1
        model_messages.append(messages)
        if model_calls == 1:
            return {
                "role": "assistant",
                "content": "Start the architecture search.",
                "tool_calls": [{
                    "function": {
                        "name": "enable_nas",
                        "arguments": {"search_time_limit_seconds": 3600},
                    },
                }],
            }
        assert messages[-1]["role"] == "tool"
        assert "search_time_limit_seconds must be between 90 and 600" in messages[-1]["content"]
        return {
            "role": "assistant",
            "content": "Retry with a valid search limit.",
            "tool_calls": [{
                "function": {
                    "name": "enable_nas",
                    "arguments": {"search_time_limit_seconds": 600},
                },
            }],
        }

    def tool_call(name, arguments):
        tool_calls.append((name, arguments))
        if arguments["search_time_limit_seconds"] > 600:
            return {
                "status": "invalid_tool_call",
                "retryable": True,
                "error": "search_time_limit_seconds must be between 90 and 600",
            }
        return {"status": "executed"}

    response = run_agent(
        messages=[{"role": "user", "content": "Improve CIFAR accuracy."}],
        model_call=model_call,
        tool_call=tool_call,
        max_tool_calls=1,
    )

    assert model_calls == 2
    assert len(model_messages[1]) > 1
    assert tool_calls == [
        ("enable_nas", {"search_time_limit_seconds": 3600}),
        ("enable_nas", {"search_time_limit_seconds": 600}),
    ]
    assert response["tool_calls"] == 2
    assert [entry["result"]["status"] for entry in response["trace"]] == [
        "invalid_tool_call",
        "executed",
    ]


def test_graph_bounds_repeated_invalid_tool_calls() -> None:
    model_calls = 0
    tool_calls = 0

    def model_call(messages):
        nonlocal model_calls
        model_calls += 1
        return {
            "role": "assistant",
            "content": "Try the invalid action again.",
            "tool_calls": [{
                "function": {
                    "name": "enable_nas",
                    "arguments": {"search_time_limit_seconds": 3600},
                },
            }],
        }

    def tool_call(name, arguments):
        nonlocal tool_calls
        tool_calls += 1
        return {
            "status": "invalid_tool_call",
            "retryable": True,
            "error": "search_time_limit_seconds must be between 90 and 600",
        }

    response = run_agent(
        messages=[{"role": "user", "content": "Enable NAS."}],
        model_call=model_call,
        tool_call=tool_call,
        max_tool_calls=1,
    )

    assert model_calls == 2
    assert tool_calls == 2
    assert response["tool_calls"] == 2
    assert len(response["trace"]) == 2
    assert "repeated tool-validation failures" in response["reply"]


def test_graph_stops_at_configured_tool_limit() -> None:
    executed = []
    model_calls = 0

    def model_call(messages):
        nonlocal model_calls
        model_calls += 1
        return {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "function": {
                        "name": "start_training",
                        "arguments": {
                            "dataset_name": "mnist",
                            "rounds": 10,
                            "epochs": 5,
                        },
                    }
                }
            ],
        }

    response = run_agent(
        messages=[{"role": "user", "content": "Run training"}],
        model_call=model_call,
        tool_call=lambda name, arguments: executed.append(name) or {"status": "executed"},
        max_tool_calls=1,
    )

    assert executed == ["start_training"]
    assert response["tool_calls"] == 1
    assert "configured agent tool-call limit" in response["reply"]
    assert model_calls == 1


def test_graph_serializes_multi_tool_output_but_executes_only_first_call() -> None:
    executed = []
    model_calls = 0

    def model_call(messages):
        nonlocal model_calls
        model_calls += 1
        if model_calls == 1:
            return {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"function": {"name": "get_status", "arguments": {}}},
                    {"function": {"name": "start_training", "arguments": {}}},
                ],
            }
        assert messages[-2]["role"] == "tool"
        assert messages[-1]["role"] == "tool"
        return {"role": "assistant", "content": "I checked the status first."}

    response = run_agent(
        messages=[{"role": "user", "content": "Check then train"}],
        model_call=model_call,
        tool_call=lambda name, arguments: executed.append(name) or {"status": "executed"},
        max_tool_calls=3,
    )

    assert executed == ["get_status"]
    assert response["trace"][1]["result"]["error"] == "Only the first call in this batch was executed"


def test_graph_stops_when_the_model_selects_workflow_termination() -> None:
    model_calls = 0

    def model_call(messages):
        nonlocal model_calls
        model_calls += 1
        return {
            "role": "assistant",
            "content": "The target is met by the recorded validation result.",
            "tool_calls": [{
                "function": {
                    "name": "finish_agent_workflow",
                    "arguments": {
                        "outcome": "objective_achieved",
                        "summary": "Validation target reached.",
                        "evidence": "The latest completed experiment achieved 0.91 validation accuracy.",
                    },
                },
            }],
        }

    result = run_agent(
        messages=[{"role": "user", "content": "Reach 90% accuracy."}],
        model_call=model_call,
        tool_call=lambda name, arguments: {
            "status": "workflow_complete",
            **arguments,
        },
        max_tool_calls=4,
    )

    assert model_calls == 1
    assert result["terminated"] is True
    assert result["trace"][0]["tool"] == "finish_agent_workflow"
    assert result["trace"][0]["result"]["evidence"].endswith("0.91 validation accuracy.")
    assert "objective_achieved" in result["reply"]
    assert "Validation target reached." in result["reply"]
