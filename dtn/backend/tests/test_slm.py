from fastapi.testclient import TestClient


def test_slm_proxy_requires_coordinator_authentication(client: TestClient) -> None:
    for path, method, payload in (
        ("/api/slm/status", "get", None),
        ("/api/slm/logs", "get", None),
        ("/api/slm/results", "get", None),
        ("/api/slm/chat", "post", {"message": "status"}),
        ("/api/slm/command", "post", {"action": "get_status", "parameters": {}}),
        ("/api/slm/baseline", "post", {"rounds": 1, "epochs": 1}),
    ):
        response = getattr(client, method)(path, json=payload) if payload else getattr(
            client, method
        )(path)
        assert response.status_code == 401


def test_slm_chat_is_forwarded_with_the_master_token(
    client: TestClient,
    auth_headers: dict[str, str],
    monkeypatch,
) -> None:
    from app.routers import slm

    forwarded = {}

    def fake_forward(settings, path, *, payload=None, query=""):
        forwarded.update(
            {
                "base_url": settings.master_slm_url,
                "token": settings.master_slm_token,
                "path": path,
                "payload": payload,
            }
        )
        return {"reply": "real upstream response", "action": {"action": "get_status"}}

    monkeypatch.setattr(slm, "_forward", fake_forward)
    response = client.post(
        "/api/slm/chat",
        json={"message": "Check status"},
        headers=auth_headers,
    )
    assert response.status_code == 200
    assert response.json()["reply"] == "real upstream response"
    assert forwarded == {
        "base_url": "http://127.0.0.1:5003",
        "token": "test-master-slm-token-that-is-long-and-random",
        "path": "/chat",
        "payload": {"message": "Check status"},
    }


def test_experiment_results_and_baseline_are_forwarded(
    client: TestClient,
    auth_headers: dict[str, str],
    monkeypatch,
) -> None:
    from app.routers import slm

    forwarded = []

    def fake_forward(settings, path, *, payload=None, query=""):
        forwarded.append((settings.master_slm_url, settings.master_slm_token, path, payload, query))
        return {"groups": [], "accepted": True}

    monkeypatch.setattr(slm, "_forward", fake_forward)
    results = client.get("/api/slm/results", headers=auth_headers)
    baseline = client.post(
        "/api/slm/baseline",
        json={"rounds": 2, "epochs": 1},
        headers=auth_headers,
    )

    assert results.status_code == 200
    assert baseline.status_code == 200
    assert forwarded == [
        (
            "http://127.0.0.1:5003",
            "test-master-slm-token-that-is-long-and-random",
            "/results",
            None,
            "",
        ),
        (
            "http://127.0.0.1:5003",
            "test-master-slm-token-that-is-long-and-random",
            "/baseline",
            {"rounds": 2, "epochs": 1},
            "",
        ),
    ]


def test_baseline_rejects_out_of_range_parameters(
    client: TestClient,
    auth_headers: dict[str, str],
) -> None:
    response = client.post(
        "/api/slm/baseline",
        json={"rounds": 0, "epochs": 1},
        headers=auth_headers,
    )
    assert response.status_code == 422



def test_slm_proxy_validates_chat_message_size(
    client: TestClient,
    auth_headers: dict[str, str],
) -> None:
    response = client.post(
        "/api/slm/chat",
        json={"message": "x" * 4001},
        headers=auth_headers,
    )
    assert response.status_code == 422
