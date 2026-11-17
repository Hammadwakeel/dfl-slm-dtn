"""Poll backend health for ten minutes and verify the persistent task remains active."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "dtn" / "backend"))

from app.core.config import Settings  # noqa: E402

DURATION_SECONDS = 600
INTERVAL_SECONDS = 30
TASK_NAME = "DFL Backend Coordinator"


def _get_json(url: str, token: str | None = None) -> tuple[int, object]:
    headers = {"Accept": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    try:
        with urlopen(Request(url, headers=headers), timeout=10) as response:
            return response.status, json.loads(response.read().decode("utf-8"))
    except HTTPError as error:
        return error.code, None
    except (URLError, TimeoutError, OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Backend poll failed: {type(error).__name__}") from error


def _task_state() -> str:
    command = (
        "$t=Get-ScheduledTask -TaskName 'DFL Backend Coordinator' -ErrorAction Stop; "
        "$i=Get-ScheduledTaskInfo -TaskName $t.TaskName; "
        "[pscustomobject]@{State=$t.State;LastTaskResult=$i.LastTaskResult}|ConvertTo-Json -Compress"
    )
    result = subprocess.run(
        ["powershell.exe", "-NoLogo", "-NoProfile", "-NonInteractive", "-Command", command],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=15,
        check=False,
    )
    if result.returncode:
        raise RuntimeError("Could not query Scheduled Task state")
    try:
        task = json.loads(result.stdout)
    except json.JSONDecodeError as error:
        raise RuntimeError("Scheduled Task query returned invalid JSON") from error
    if task.get("State") != 4:
        raise RuntimeError(f"Scheduled Task is not running: state={task.get('State')}")
    return str(task.get("LastTaskResult"))


def main() -> None:
    settings = Settings.from_env()
    base = f"http://127.0.0.1:{settings.port}"
    started = time.monotonic()
    deadline = started + DURATION_SECONDS
    samples = 0
    while True:
        status, body = _get_json(f"{base}/healthz")
        if status != 200 or body != {"status": "ok"}:
            raise RuntimeError(f"/healthz returned HTTP {status} with unexpected body")
        task_result = _task_state()
        samples += 1
        elapsed = round(time.monotonic() - started, 1)
        print(
            f"sample={samples}; elapsed_seconds={elapsed}; healthz=200; "
            f"task=running; task_last_result={task_result}; "
            f"checked_at={datetime.now().astimezone().isoformat(timespec='seconds')}"
        )
        if time.monotonic() >= deadline:
            break
        time.sleep(min(INTERVAL_SECONDS, max(0, deadline - time.monotonic())))

    openapi_status, spec = _get_json(f"{base}/openapi.json")
    if openapi_status != 200 or not isinstance(spec, dict):
        raise RuntimeError("OpenAPI check failed after stability interval")
    paths = spec.get("paths")
    if not isinstance(paths, dict) or "/api/datasets/{dataset_name}/preprocessing-report" not in paths:
        raise RuntimeError("Preprocessing-report route is absent after stability interval")
    state_status, state = _get_json(f"{base}/api/autonomy/state", settings.api_token)
    if (
        state_status != 200
        or not isinstance(state, dict)
        or state.get("mode") != "ASK_HUMAN"
        or state.get("stop_requested") is not False
        or state.get("pending_decisions") != 0
        or any(loop.get("status") == "RUNNING" for loop in state.get("loops", []))
    ):
        raise RuntimeError("Autonomy state was not safe after stability interval")
    elapsed = round(time.monotonic() - started, 1)
    print(
        f"STABILITY PASS; elapsed_seconds={elapsed}; samples={samples}; "
        "report_route=present; mode=ASK_HUMAN; STOP=False"
    )


if __name__ == "__main__":
    main()
