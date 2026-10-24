"""Tests for status authentication, metrics parsing, and HTTP behavior."""

import json
import threading
import unittest
from http.server import ThreadingHTTPServer
from http.client import HTTPConnection
from typing import Any

from common.api_client import ApiClient, NodeEndpoint
from common.auth import is_authorized
from common.status_agent import make_handler
from common.status_metrics import (
    calculate_cpu_percent,
    parse_tegrastats,
    read_temperatures,
)

TOKEN = "test-token-with-more-than-thirty-two-characters"


class AuthTests(unittest.TestCase):
    def test_authorizes_exact_bearer_token(self) -> None:
        self.assertTrue(is_authorized(f"Bearer {TOKEN}", TOKEN))

    def test_rejects_missing_or_invalid_credentials(self) -> None:
        for value in (None, "", TOKEN, "Bearer ", f"Bearer {TOKEN}x"):
            with self.subTest(value=value):
                self.assertFalse(is_authorized(value, TOKEN))


class MetricsTests(unittest.TestCase):
    def test_calculates_cpu_busy_percentage_from_procfs_counters(self) -> None:
        self.assertEqual(calculate_cpu_percent((1000, 600), (1100, 650)), 50.0)
        self.assertIsNone(calculate_cpu_percent(None, (1100, 650)))
        self.assertIsNone(calculate_cpu_percent((1000, 600), (1000, 600)))

    def test_parses_tegrastats_cpu_and_gpu_utilization(self) -> None:
        readings = parse_tegrastats(
            "RAM 2000/7620MB CPU [10%@729,20%@729,0%@729] GR3D_FREQ 37%"
        )
        self.assertEqual(readings["cpu_utilization_percent"], 10.0)
        self.assertEqual(readings["gpu_utilization_percent"], 37.0)

    def test_missing_tegrastats_fields_are_not_fabricated(self) -> None:
        self.assertEqual(
            parse_tegrastats("RAM 2000/7620MB"),
            {"cpu_utilization_percent": None, "gpu_utilization_percent": None},
        )

    def test_reads_temporary_thermal_zones(self) -> None:
        from pathlib import Path
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            zone = root / "thermal_zone0"
            zone.mkdir()
            (zone / "type").write_text("cpu-thermal", encoding="ascii")
            (zone / "temp").write_text("42500", encoding="ascii")
            self.assertEqual(read_temperatures(root), {"cpu-thermal": 42.5})


class HttpAuthTests(unittest.TestCase):
    def setUp(self) -> None:
        payload: dict[str, Any] = {
            "node_id": "worker-test",
            "role": "worker",
            "cpu_utilization_percent": 12.5,
        }
        self.server = ThreadingHTTPServer(
            ("127.0.0.1", 0),
            make_handler(TOKEN, lambda: payload),
        )
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.port = self.server.server_port

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    def request(self, path: str, token: str | None = None) -> tuple[int, dict[str, Any]]:
        connection = HTTPConnection("127.0.0.1", self.port, timeout=2)
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        connection.request("GET", path, headers=headers)
        response = connection.getresponse()
        result = json.loads(response.read().decode("utf-8"))
        connection.close()
        return response.status, result

    def test_status_rejects_missing_and_wrong_tokens(self) -> None:
        self.assertEqual(self.request("/v1/status")[0], 401)
        self.assertEqual(self.request("/v1/status", "wrong-token")[0], 401)

    def test_status_returns_real_provider_payload_for_valid_token(self) -> None:
        status, payload = self.request("/v1/status", TOKEN)
        self.assertEqual(status, 200)
        self.assertEqual(payload["node_id"], "worker-test")
        self.assertEqual(payload["cpu_utilization_percent"], 12.5)

    def test_shared_api_client_uses_bearer_authentication(self) -> None:
        client = ApiClient(
            NodeEndpoint(
                node_id="worker-test",
                base_url=f"http://127.0.0.1:{self.port}",
                token=TOKEN,
            )
        )
        self.assertEqual(client.get_status()["node_id"], "worker-test")

    def test_health_endpoint_does_not_expose_device_status(self) -> None:
        status, payload = self.request("/healthz")
        self.assertEqual(status, 200)
        self.assertEqual(payload, {"status": "ok"})


if __name__ == "__main__":
    unittest.main()
