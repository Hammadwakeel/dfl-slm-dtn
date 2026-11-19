import pytest
from fastapi.testclient import TestClient


def test_slm_proxy_requires_coordinator_authentication(client: TestClient) -> None:
    for path, method, payload in (
        ("/api/slm/status", "get", None),
        ("/api/slm/logs", "get", None),
        ("/api/slm/results", "get", None),
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
    one = client.get("/api/slm/results?group_id=mesh-1795-abc", headers=auth_headers)
    few = client.get("/api/slm/results?limit=3", headers=auth_headers)
    bad = client.get("/api/slm/results?group_id=../x", headers=auth_headers)

    assert results.status_code == one.status_code == few.status_code == 200
    assert bad.status_code == 422
    master = ("http://127.0.0.1:5003", "test-master-slm-token-that-is-long-and-random", "/results", None)
    full = client.get("/api/slm/results?limit=1&compact=false", headers=auth_headers)
    assert full.status_code == 200
    assert forwarded == [
        (*master, "?limit=20&compact=1"),
        (*master, "?group_id=mesh-1795-abc&compact=1"),
        (*master, "?limit=3&compact=1"),
        (*master, "?limit=1&compact=0"),
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
