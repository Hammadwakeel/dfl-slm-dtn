from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect


def test_websocket_requires_token(client: TestClient) -> None:
    with client.websocket_connect("/ws/devices") as websocket:
        websocket.send_json({"type": "auth", "token": "invalid"})
        try:
            websocket.receive_json()
        except WebSocketDisconnect as error:
            assert error.code == 1008
        else:
            raise AssertionError("unauthorized WebSocket was not closed")


def test_websocket_sends_registry_snapshot_after_authentication(
    client: TestClient,
    settings,
) -> None:
    with client.websocket_connect("/ws/devices") as websocket:
        websocket.send_json({"type": "auth", "token": settings.api_token})
        snapshot = websocket.receive_json()
    assert snapshot["type"] == "device_snapshot"
    assert [device["node_id"] for device in snapshot["devices"]] == [
        "master",
        "worker1",
        "worker2",
        "worker3",
    ]
