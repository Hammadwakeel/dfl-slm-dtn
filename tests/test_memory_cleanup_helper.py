import json
import threading
from http.server import ThreadingHTTPServer
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from worker import memory_cleanup_helper


def test_privileged_helper_requires_token_and_only_drops_caches(monkeypatch):
    token = "memory-cleanup-token-that-is-long-enough"
    writes = []
    sync_calls = []

    class CacheFile:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def write(self, value):
            writes.append(value)

    monkeypatch.setattr(memory_cleanup_helper, "TOKEN", token)
    monkeypatch.setattr(
        memory_cleanup_helper.os,
        "sync",
        lambda: sync_calls.append(True),
        raising=False,
    )
    monkeypatch.setattr(
        memory_cleanup_helper,
        "open",
        lambda path, mode, encoding: CacheFile(),
        raising=False,
    )
    server = ThreadingHTTPServer(
        ("127.0.0.1", 0),
        memory_cleanup_helper.MemoryCleanupHandler,
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        base_url = f"http://127.0.0.1:{server.server_port}"
        unauthorized_health = Request(f"{base_url}/healthz")
        with pytest.raises(HTTPError) as error:
            urlopen(unauthorized_health, timeout=3)
        assert error.value.code == 401

        authorized_health = Request(
            f"{base_url}/healthz",
            headers={"Authorization": "Be" + "arer " + token},
        )
        with urlopen(authorized_health, timeout=3) as response:
            assert response.status == 200
            assert json.loads(response.read().decode("utf-8")) == {"status": "ok"}

        unauthorized = Request(
            f"{base_url}/drop-caches",
            data=b"",
            method="POST",
        )
        with pytest.raises(HTTPError) as error:
            urlopen(unauthorized, timeout=3)
        assert error.value.code == 401
        assert sync_calls == []

        authorized = Request(
            f"{base_url}/drop-caches",
            data=b"",
            headers={"Authorization": f"Bearer {token}"},
            method="POST",
        )
        with urlopen(authorized, timeout=3) as response:
            assert response.status == 200
            assert json.loads(response.read().decode("utf-8")) == {"status": "ok"}
        assert sync_calls == [True]
        assert writes == ["3\n"]
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_reboot_endpoint_requires_token_and_schedules_only_the_fixed_action(monkeypatch):
    token = "worker-token-that-is-long-enough-for-tests"
    monkeypatch.setattr(memory_cleanup_helper, "TOKEN", token)
    reboot_started = threading.Event()
    commands = []
    monkeypatch.setattr(
        memory_cleanup_helper.subprocess,
        "Popen",
        lambda command, **_kwargs: (commands.append(command), reboot_started.set()),
    )
    server = ThreadingHTTPServer(
        ("127.0.0.1", 0),
        memory_cleanup_helper.MemoryCleanupHandler,
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        base_url = f"http://127.0.0.1:{server.server_port}"
        unauthorized = Request(
            f"{base_url}/reboot",
            data=b"",
            method="POST",
        )
        with pytest.raises(HTTPError) as error:
            urlopen(unauthorized, timeout=3)
        assert error.value.code == 401
        assert not reboot_started.is_set()

        request = Request(
            f"{base_url}/reboot",
            data=b"",
            headers={"Authorization": f"Bearer {token}"},
            method="POST",
        )
        with urlopen(request, timeout=3) as response:
            assert response.status == 202
            assert json.loads(response.read().decode("utf-8")) == {
                "status": "accepted",
                "action": "reboot",
            }
        assert reboot_started.wait(timeout=2)
        assert commands == [["/usr/bin/systemctl", "reboot"]]
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
