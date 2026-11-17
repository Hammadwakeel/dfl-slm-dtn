"""Verify health, required OpenAPI routes, and safe autonomy state after restart."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "dtn" / "backend"))

from app.core.config import Settings  # noqa: E402


class BackendVerificationError(RuntimeError):
    pass


def _get(url: str, token: str | None = None) -> tuple[int, object]:
    headers = {"Accept": "application/json"}
    if token is not None:
        headers["Authorization"] = f"Bearer {token}"
    request = Request(url, headers=headers)
    try:
        with urlopen(request, timeout=15) as response:
            return response.status, json.loads(response.read().decode("utf-8"))
    except HTTPError as error:
        raise BackendVerificationError(f"GET {url} returned HTTP {error.code}") from error
    except (URLError, TimeoutError, OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BackendVerificationError(
            f"GET {url} failed: {type(error).__name__}"
        ) from error


def main() -> None:
    settings = Settings.from_env()
    base_url = f"http://127.0.0.1:{settings.port}"
    health_status, health = _get(f"{base_url}/healthz")
    if health_status != 200 or health != {"status": "ok"}:
        raise BackendVerificationError("Backend health check did not return 200/ok")
    print("healthz: HTTP 200")

    openapi_status, openapi = _get(f"{base_url}/openapi.json")
    if openapi_status != 200 or not isinstance(openapi, dict):
        raise BackendVerificationError("OpenAPI request did not return an object")
    paths = openapi.get("paths")
    if not isinstance(paths, dict) or "/api/slm/suggest-next-action" not in paths:
        raise BackendVerificationError(
            "Required route /api/slm/suggest-next-action is missing from OpenAPI"
        )
    print("openapi: HTTP 200; /api/slm/suggest-next-action present")

    state_status, state = _get(
        f"{base_url}/api/autonomy/state",
        settings.api_token,
    )
    if state_status != 200 or not isinstance(state, dict):
        raise BackendVerificationError("Authenticated autonomy state check failed")
    if state.get("mode") != "ASK_HUMAN" or state.get("stop_requested") is not False:
        raise BackendVerificationError("Autonomy must be ASK_HUMAN with global STOP inactive")
    if state.get("pending_decisions") != 0 or any(
        loop.get("status") == "RUNNING" for loop in state.get("loops", [])
    ):
        raise BackendVerificationError("Autonomy has a pending decision or running loop")
    print(
        "autonomy: HTTP 200; "
        f"mode={state['mode']}; global_STOP={state['stop_requested']}; "
        f"pending_decisions={state['pending_decisions']}; running_loops=0"
    )


if __name__ == "__main__":
    main()
