import pytest

from scripts.m4_hardware_check import (
    HardwareCheckError,
    _approved_action_data,
    _await_nas_terminal,
)


def test_approved_action_data_unwraps_real_nested_nas_response():
    response = {
        "status": "approved",
        "result": {
            "action": {
                "action": "enable_nas",
                "parameters": {
                    "enabled": True,
                    "dataset_name": "sample-dataset",
                    "search_time_limit_seconds": 30,
                },
            },
            "result": {
                "available": True,
                "data": {
                    "accepted": True,
                    "group_id": "dataset-nas-fixture",
                    "search_time_limit_seconds": 30,
                },
            },
        },
    }

    assert _approved_action_data(response) == {
        "accepted": True,
        "group_id": "dataset-nas-fixture",
        "search_time_limit_seconds": 30,
    }


def test_approved_action_data_unwraps_local_model_response():
    response = {
        "status": "approved",
        "result": {
            "available": True,
            "data": {
                "dataset_name": "sample-dataset",
                "parameter_count": 898,
            },
        },
    }

    assert _approved_action_data(response) == {
        "dataset_name": "sample-dataset",
        "parameter_count": 898,
    }


@pytest.mark.parametrize(
    "response",
    [
        {"status": "failed", "detail": "action execution failed"},
        {"status": "approved", "result": {"action": {}, "result": {"error": "failed"}}},
    ],
)
def test_approved_action_data_rejects_failure_envelopes(response):
    with pytest.raises(HardwareCheckError):
        _approved_action_data(response)


def test_nas_monitor_waits_until_the_group_is_terminal():
    class Client:
        def __init__(self):
            self.responses = [
                {
                    "active_group_id": "dataset-nas-fixture",
                    "groups": [{"group_id": "dataset-nas-fixture", "status": "running"}],
                },
                {
                    "active_group_id": None,
                    "groups": [{"group_id": "dataset-nas-fixture", "status": "completed"}],
                },
            ]

        def request(self, method, path, *, timeout):
            assert method == "GET"
            assert path == "/api/slm/results"
            return self.responses.pop(0)

    client = Client()
    terminal = _await_nas_terminal(
        client,
        "dataset-nas-fixture",
        poll_interval_seconds=0,
    )

    assert terminal["status"] == "completed"
    assert client.responses == []
