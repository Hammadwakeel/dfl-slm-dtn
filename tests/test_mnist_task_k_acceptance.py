from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import mnist_task_k_acceptance as acceptance


@pytest.fixture
def baseline_group_ids() -> set[str]:
    return {"prior-group"}


@pytest.fixture
def completed_dataset_group() -> dict[str, object]:
    return {
        "group_id": "new-group",
        "kind": "dataset_architecture_search",
        "dataset_name": "mnist_demo",
        "status": "completed",
        "error": None,
    }


class FakeResultsClient:
    def __init__(self, groups: list[dict[str, object]]) -> None:
        self.groups = groups

    def request(self, method: str, path: str) -> dict[str, object]:
        assert method == "GET"
        assert path == "/api/slm/results"
        return {"groups": self.groups}


def _disable_preflight_and_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(acceptance, "_preflight", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(acceptance.time, "sleep", lambda _seconds: None)


def test_approved_action_with_result_data_polls_returned_group_id(
    monkeypatch: pytest.MonkeyPatch,
    baseline_group_ids: set[str],
    completed_dataset_group: dict[str, object],
) -> None:
    _disable_preflight_and_sleep(monkeypatch)
    client = FakeResultsClient([completed_dataset_group])
    approval = {
        "status": "approved",
        "result": {"available": True, "data": {"group_id": "new-group"}},
    }

    group = acceptance._wait_for_approved_async_group(
        client,
        None,
        [],
        approval,
        baseline_group_ids=baseline_group_ids,
        dataset_name="mnist_demo",
        timeout_seconds=5,
    )

    assert group["group_id"] == "new-group"


def test_approved_async_action_without_result_data_finds_new_group(
    monkeypatch: pytest.MonkeyPatch,
    baseline_group_ids: set[str],
    completed_dataset_group: dict[str, object],
) -> None:
    _disable_preflight_and_sleep(monkeypatch)
    client = FakeResultsClient(
        [
            {"group_id": "prior-group", "dataset_name": "mnist_demo", "status": "completed"},
            completed_dataset_group,
        ]
    )
    approval = {"status": "approved", "result": {"available": True}}

    group = acceptance._wait_for_approved_async_group(
        client,
        None,
        [],
        approval,
        baseline_group_ids=baseline_group_ids,
        dataset_name="mnist_demo",
        timeout_seconds=5,
    )

    assert group["group_id"] == "new-group"


def test_failed_async_group_raises_after_reaching_terminal_state(
    monkeypatch: pytest.MonkeyPatch,
    baseline_group_ids: set[str],
) -> None:
    _disable_preflight_and_sleep(monkeypatch)
    client = FakeResultsClient(
        [
            {
                "group_id": "new-group",
                "kind": "dataset_architecture_search",
                "dataset_name": "mnist_demo",
                "status": "failed",
                "error": "fixture failure",
            }
        ]
    )
    approval = {"status": "approved", "result": {"available": True}}

    with pytest.raises(acceptance.TaskKError, match="ended failed: fixture failure"):
        acceptance._wait_for_approved_async_group(
            client,
            None,
            [],
            approval,
            baseline_group_ids=baseline_group_ids,
            dataset_name="mnist_demo",
            timeout_seconds=5,
        )