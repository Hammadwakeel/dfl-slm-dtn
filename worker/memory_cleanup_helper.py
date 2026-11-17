"""Loopback-only privileged helper for dropping Linux filesystem caches."""

import hmac
import json
import logging
import os
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


LOGGER = logging.getLogger("dfl_memory_cleanup")
TOKEN = os.environ.get("DFL_MEMORY_CLEANUP_TOKEN", "")


class MemoryCleanupHandler(BaseHTTPRequestHandler):
    server_version = "DFLMemoryCleanup/1.0"
    sys_version = ""

    def do_GET(self) -> None:
        if self.path != "/healthz":
            self._send_json(404, {"error": "not_found"})
            return
        if not self._authorized():
            self._send_json(401, {"error": "unauthorized"})
            return
        self._send_json(200, {"status": "ok"})

    def do_POST(self) -> None:
        if self.path not in {"/drop-caches", "/reboot"}:
            self._send_json(404, {"error": "not_found"})
            return
        if not self._authorized():
            self._send_json(401, {"error": "unauthorized"})
            return
        if self.headers.get("Content-Length", "0") != "0":
            self._send_json(400, {"error": "request_body_not_allowed"})
            return

        if self.path == "/reboot":
            self._send_json(202, {"status": "accepted", "action": "reboot"})
            threading.Timer(0.5, self._reboot).start()
            return

        try:
            os.sync()
            with open("/proc/sys/vm/drop_caches", "w", encoding="ascii") as stream:
                stream.write("3\n")
        except OSError:
            LOGGER.exception("Could not drop Linux filesystem caches")
            self._send_json(500, {"error": "cache_drop_failed"})
            return
        self._send_json(200, {"status": "ok"})

    @staticmethod
    def _reboot() -> None:
        try:
            subprocess.Popen(
                ["/usr/bin/systemctl", "reboot"],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
        except OSError:
            LOGGER.exception("Could not start the requested Jetson reboot")

    def _authorized(self) -> bool:
        authorization = self.headers.get("Authorization", "")
        supplied = (
            authorization.removeprefix("Bearer ")
            if authorization.startswith("Bearer ")
            else ""
        )
        return bool(TOKEN) and hmac.compare_digest(
            supplied.encode(),
            TOKEN.encode(),
        )

    def _send_json(self, status_code: int, payload: dict[str, str]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, _format: str, *_args: object) -> None:
        return


def main() -> None:
    if len(TOKEN) < 32:
        raise RuntimeError("DFL_MEMORY_CLEANUP_TOKEN must contain at least 32 characters")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    server = ThreadingHTTPServer(("127.0.0.1", 5005), MemoryCleanupHandler)
    LOGGER.info("Privileged memory cleanup helper listening on 127.0.0.1:5005")
    server.serve_forever()


if __name__ == "__main__":
    main()
