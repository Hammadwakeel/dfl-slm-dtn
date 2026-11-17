"""Read-only verification of the local backend listener and autonomy state."""

from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from sqlalchemy.engine import make_url


ROOT = Path(__file__).resolve().parents[1]
BACKEND_URL = "http://127.0.0.1:8000"
EXPECTED_REVISION = "0003_full_access_execution_audit"
REQUIRED_ROUTES = (
    "/api/autonomy/state",
    "/api/autonomy/history",
    "/api/slm/status",
    "/api/slm/results",
)
POWERSHELL_PROCESS_QUERY = r"""
$ErrorActionPreference = 'Stop'
$listener = Get-NetTCPConnection -LocalPort 8000 -State Listen | Select-Object -First 1
if (-not $listener) { throw 'No listener on port 8000' }
$processes = Get-CimInstance Win32_Process
$process = $processes | Where-Object { $_.ProcessId -eq $listener.OwningProcess }
if (-not $process) { throw 'Listener process not found' }
$started = (Get-Process -Id $process.ProcessId -ErrorAction Stop).StartTime
$source = Get-Item 'dtn\backend\app\services\autonomy.py'
$chain = @()
$runnerFound = $false
$current = $process
for ($i = 0; $i -lt 12 -and $current; $i++) {
  $runnerFound = $runnerFound -or ($current.CommandLine -match 'backend_task_runner\.ps1')
  $chain += [pscustomobject]@{ name = $current.Name; pid = $current.ProcessId }
  $parentId = [int]$current.ParentProcessId
  if ($parentId -eq 0) { break }
  $current = $processes | Where-Object { $_.ProcessId -eq $parentId }
}
[pscustomobject]@{
  pid = $process.ProcessId
  name = $process.Name
  commandIsBackend = ($process.Name -match '^python(?:\.exe)?$' -and $process.CommandLine -match '(?i)-m\s+app\.main')
  startTime = $started.ToString('yyyy-MM-ddTHH:mm:ss.fffzzz')
  sourceMtime = $source.LastWriteTime.ToString('yyyy-MM-ddTHH:mm:ss.fffzzz')
  runnerFound = $runnerFound
  parentChain = $chain
} | ConvertTo-Json -Compress -Depth 5
"""


def _process_info() -> dict[str, Any]:
    result = subprocess.run(
        [
            "powershell.exe",
            "-NoLogo",
            "-NoProfile",
            "-NonInteractive",
            "-Command",
            POWERSHELL_PROCESS_QUERY,
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=20,
        check=False,
    )
    if result.returncode:
        raise RuntimeError(f"PowerShell process inspection failed (exit {result.returncode})")
    try:
        info = json.loads(result.stdout)
    except json.JSONDecodeError as error:
        raise RuntimeError("PowerShell process inspection returned invalid JSON") from error
    if not isinstance(info, dict):
        raise RuntimeError("PowerShell process inspection returned an unexpected value")
    return info


def _get_json(path: str, *, token: str | None = None) -> tuple[int, Any]:
    headers = {"Accept": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = Request(BACKEND_URL + path, headers=headers, method="GET")
    try:
        with urlopen(request, timeout=10) as response:
            return response.status, json.loads(response.read().decode("utf-8"))
    except HTTPError as error:
        raise RuntimeError(f"GET {path} returned HTTP {error.code}") from error
    except (URLError, TimeoutError, OSError) as error:
        raise RuntimeError(f"GET {path} failed ({type(error).__name__})") from error
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"GET {path} returned invalid JSON") from error


def _database_path() -> Path:
    database_url = os.environ.get("DTN_DATABASE_URL")
    if not database_url:
        return ROOT / "dtn" / "backend" / "data" / "dtn.sqlite3"
    try:
        parsed = make_url(database_url)
    except Exception as error:
        raise RuntimeError("DTN_DATABASE_URL is not a valid SQLite URL") from error
    if not parsed.drivername.startswith("sqlite") or not parsed.database:
        raise RuntimeError("Configured backend database is not a file-backed SQLite database")
    database_path = Path(parsed.database).expanduser()
    if not database_path.is_absolute():
        database_path = ROOT / database_path
    return database_path.resolve()


def _database_revision() -> str:
    database_path = _database_path()
    connection_uri = database_path.as_uri() + "?mode=ro"
    try:
        connection = sqlite3.connect(connection_uri, uri=True, timeout=5)
        try:
            row = connection.execute(
                "SELECT version_num FROM alembic_version"
            ).fetchone()
        finally:
            connection.close()
    except sqlite3.Error as error:
        raise RuntimeError(f"Could not read Alembic revision ({type(error).__name__})") from error
    if row is None or not isinstance(row[0], str):
        raise RuntimeError("Alembic revision table has no revision id")
    return row[0]


def _running_loop_count(state: dict[str, Any]) -> int:
    loops = state.get("loops")
    if not isinstance(loops, list):
        raise RuntimeError("Autonomy state omitted its loops list")
    return sum(isinstance(loop, dict) and loop.get("status") == "RUNNING" for loop in loops)


def main() -> int:
    checks: dict[str, bool] = {}
    try:
        process = _process_info()
        listener_start = datetime.fromisoformat(process["startTime"])
        source_mtime = datetime.fromisoformat(process["sourceMtime"])
        listener_newer = listener_start > source_mtime
        checks["listener_is_backend"] = bool(process.get("commandIsBackend"))
        checks["listener_newer_than_autonomy_source"] = listener_newer
        checks["backend_task_runner_in_parent_chain"] = bool(process.get("runnerFound"))
        print(f"listener_pid={process['pid']}")
        print(f"listener_start_local={listener_start.isoformat()}")
        print(f"autonomy_py_mtime_local={source_mtime.isoformat()}")
        print(f"listener_newer_than_source={listener_newer}")
        print(f"backend_task_runner_in_parent_chain={process.get('runnerFound')}")
        print(
            "listener_parent_chain="
            + " -> ".join(
                f"{item['name']}:{item['pid']}" for item in process.get("parentChain", [])
            )
        )
    except (RuntimeError, KeyError, TypeError, ValueError, subprocess.SubprocessError) as error:
        print(f"listener_inspection=FAIL ({error})")
        return 1

    try:
        health_status, health = _get_json("/healthz")
        health_ok = health_status == 200 and isinstance(health, dict) and health.get("status") == "ok"
        checks["healthz"] = health_ok
        print(f"healthz_http={health_status}; status={health.get('status') if isinstance(health, dict) else 'invalid'}")
    except RuntimeError as error:
        checks["healthz"] = False
        print(f"healthz=FAIL ({error})")

    try:
        openapi_status, openapi = _get_json("/openapi.json")
        paths = openapi.get("paths", {}) if isinstance(openapi, dict) else {}
        missing_routes = [route for route in REQUIRED_ROUTES if route not in paths]
        routes_ok = openapi_status == 200 and not missing_routes
        checks["required_routes"] = routes_ok
        print(f"openapi_http={openapi_status}; missing_routes={json.dumps(missing_routes)}")
    except RuntimeError as error:
        checks["required_routes"] = False
        print(f"openapi=FAIL ({error})")

    try:
        revision = _database_revision()
        checks["expected_database_revision"] = revision == EXPECTED_REVISION
        print(f"alembic_revision={revision}")
    except RuntimeError as error:
        checks["expected_database_revision"] = False
        print(f"alembic_revision=FAIL ({error})")

    token = os.environ.get("DTN_API_TOKEN", "")
    if not token:
        checks["autonomy_state"] = False
        print("autonomy_state=FAIL (DTN_API_TOKEN is not set in the process environment)")
    else:
        try:
            state_status, state = _get_json("/api/autonomy/state", token=token)
            if state_status != 200 or not isinstance(state, dict):
                raise RuntimeError("Autonomy state response was not a JSON object")
            running_loops = _running_loop_count(state)
            checks["autonomy_state"] = (
                state.get("mode") == "ASK_HUMAN"
                and state.get("stop_requested") is False
                and state.get("pending_decisions") == 0
                and running_loops == 0
            )
            print(
                f"autonomy_mode={state.get('mode')}; STOP={state.get('stop_requested')}; "
                f"pending={state.get('pending_decisions')}; running_loops={running_loops}"
            )
        except RuntimeError as error:
            checks["autonomy_state"] = False
            print(f"autonomy_state=FAIL ({error})")

    print(f"overall={'PASS' if all(checks.values()) else 'FAIL'}")
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    sys.exit(main())