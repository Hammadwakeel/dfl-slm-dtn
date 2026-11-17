from jsonschema import Draft202012Validator

from common.agent_tools import ACTION_SCHEMA_DOCUMENT, agent_tool_action, agent_tool_definitions


def test_agent_tools_are_generated_from_executable_schema_actions() -> None:
    expected = {
        branch["properties"]["action"]["const"]
        for branch in ACTION_SCHEMA_DOCUMENT["$defs"]["action"]["oneOf"]
        if branch["properties"]["action"].get("x-agent-tool") is True
    }

    tools = agent_tool_definitions()

    # finish_agent_workflow is a control tool handled by the coordinator, not a schema action.
    assert {tool["function"]["name"] for tool in tools} == expected | {"finish_agent_workflow"}
    for tool in tools:
        Draft202012Validator.check_schema(tool["function"]["parameters"])
        assert "$ref" not in str(tool["function"]["parameters"])


def test_agent_tool_action_normalizes_stringified_empty_architecture() -> None:
    action = agent_tool_action(
        "start_training",
        {
            "rounds": "1",
            "epochs": "1",
            "dataset_name": "cifar10",
            "architecture": "{}",
        },
    )

    assert action == {
        "action": "start_training",
        "parameters": {
            "rounds": 1,
            "epochs": 1,
            "dataset_name": "cifar10",
        },
    }
    Draft202012Validator(
        {**ACTION_SCHEMA_DOCUMENT, **ACTION_SCHEMA_DOCUMENT["$defs"]["action"]}
    ).validate(action)


def test_agent_tool_action_decodes_stringified_architecture() -> None:
    action = agent_tool_action(
        "start_training",
        {
            "rounds": 1,
            "epochs": 1,
            "dataset_name": "cifar10",
            "architecture": '{"id":"b1_c8_k3","blocks":1,"channels":8,"kernel_size":3}',
        },
    )

    assert action["parameters"]["architecture"] == {
        "id": "b1_c8_k3",
        "blocks": 1,
        "channels": 8,
        "kernel_size": 3,
    }
    Draft202012Validator(
        {**ACTION_SCHEMA_DOCUMENT, **ACTION_SCHEMA_DOCUMENT["$defs"]["action"]}
    ).validate(action)
