"""Authenticated HTTP status service for a physical Jetson node."""

import json
import logging
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable

from common.auth import is_authorized
from common.status_metrics import collect_status

LOGGER = logging.getLogger("dfl_slm.status_agent")


class StatusSampler:
    """Refresh real device metrics in the background and expose the latest reading."""

    def __init__(self, node_id: str, role: str, interval_seconds: float = 2.0) -> None:
        if interval_seconds <= 0:
            raise ValueError("Sampling interval must be positive")
        self._node_id = node_id
        self._role = role
        self._interval_seconds = interval_seconds
        self._previous_cpu: tuple[int, int] | None = None
        self._snapshot: dict[str, Any] | None = None
        self._last_error: str | None = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="status-sampler", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=self._interval_seconds + 2)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            if self._snapshot is None:
                detail = self._last_error or "first metrics sample is not ready"
                raise RuntimeError(f"Device metrics are unavailable: {detail}")
            return dict(self._snapshot)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                snapshot, cpu_counters = collect_status(
                    self._node_id,
                    self._role,
                    self._previous_cpu,
                )
            except (OSError, ValueError, RuntimeError) as error:
                LOGGER.exception("Could not collect device status")
                with self._lock:
                    self._last_error = str(error)
            else:
                self._previous_cpu = cpu_counters
                with self._lock:
                    self._snapshot = snapshot
                    self._last_error = None
            self._stop.wait(self._interval_seconds)


def make_handler(
    token: str,
    status_provider: Callable[[], dict[str, Any]],
) -> type[BaseHTTPRequestHandler]:
    """Build a request handler bound to the configured token and snapshot source."""
    if len(token) < 32:
        raise ValueError("DFL_API_TOKEN must contain at least 32 characters")

    class StatusHandler(BaseHTTPRequestHandler):
        server_version = "DFLStatus/1.0"
        sys_version = ""

        def do_GET(self) -> None:
            if self.path == "/healthz":
                self._send_json(200, {"status": "ok"})
                return
            if self.path != "/v1/status":
                self._send_json(404, {"error": "not_found"})
                return
            if not is_authorized(self.headers.get("Authorization"), token):
                self._send_json(401, {"error": "unauthorized"})
                return
            try:
                payload = status_provider()
            except RuntimeError as error:
                LOGGER.error("Status request failed: %s", error)
                self._send_json(503, {"error": "status_unavailable"})
                return
            self._send_json(200, payload)

        def _send_json(self, status: int, payload: dict[str, Any]) -> None:
            body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format_string: str, *args: Any) -> None:
            LOGGER.info("%s - %s", self.client_address[0], format_string % args)

    return StatusHandler


def main() -> None:
    logging.basicConfig(
        level=os.environ.get("DFL_LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    node_id = os.environ.get("DFL_NODE_ID", "").strip()
    role = os.environ.get("DFL_NODE_ROLE", "").strip()
    token = os.environ.get("DFL_API_TOKEN", "")
    host = os.environ.get("DFL_BIND_HOST", "0.0.0.0")
    try:
        port = int(os.environ.get("DFL_PORT", "5001"))
        interval = float(os.environ.get("DFL_SAMPLE_INTERVAL_SECONDS", "2"))
    except ValueError as error:
        raise SystemExit(f"Invalid numeric DFL_PORT or sampling interval: {error}") from error
    if not node_id or not role:
        raise SystemExit("DFL_NODE_ID and DFL_NODE_ROLE must be configured")
    if not 1 <= port <= 65535:
        raise SystemExit("DFL_PORT must be between 1 and 65535")
    if not 0.2 <= interval <= 60:
        raise SystemExit("DFL_SAMPLE_INTERVAL_SECONDS must be between 0.2 and 60")

    sampler = StatusSampler(node_id, role, interval)
    sampler.start()
    server = ThreadingHTTPServer((host, port), make_handler(token, sampler.snapshot))
    server.daemon_threads = True
    LOGGER.info("Starting authenticated status API for %s (%s) on %s:%s", node_id, role, host, port)
    try:
        server.serve_forever(poll_interval=0.5)
    except KeyboardInterrupt:
        LOGGER.info("Stopping status API")
    finally:
        server.shutdown()
        server.server_close()
        sampler.stop()


if __name__ == "__main__":
    main()
