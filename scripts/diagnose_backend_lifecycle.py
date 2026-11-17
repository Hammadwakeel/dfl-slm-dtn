"""Print redacted backend logs, process state, and recent Windows lifecycle events."""

from __future__ import annotations

import json
import os
import re
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "dtn" / "backend" / "data"
SENSITIVE_KEY = re.compile(r"(?i)(token|secret|password|api[_-]?key|credential)")
REDACTIONS = (
    re.compile(r"(?i)(authorization\s*[:=]\s*bearer\s+)[^\s,;]+"),
    re.compile(r"(?i)(\bbearer\s+)[A-Za-z0-9._~+/-]+=*"),
    re.compile(
        r"(?i)(\b(?:api[_-]?token|token|password|secret|api[_-]?key|credential)\b"
        r"\s*[:=]\s*[\"']?)[^\s,;\"'}]+"
    ),
)
POWERSHELL = r'''
$ErrorActionPreference = 'Continue'
$since = (Get-Date).AddHours(-3)
$processes = @(Get-CimInstance Win32_Process -ErrorAction SilentlyContinue |
  Where-Object { $_.Name -match '^python(w)?\.exe$' -and $_.CommandLine -match 'app\.main' } |
  ForEach-Object { [pscustomobject]@{ ProcessId = $_.ProcessId; Name = $_.Name; StartTime = $_.CreationDate } })
$tasks = @(Get-ScheduledTask -ErrorAction SilentlyContinue |
  Where-Object { $_.TaskName -match '(?i)dfl|backend' -or $_.TaskPath -match '(?i)dfl|backend' } |
  ForEach-Object {
    $info = Get-ScheduledTaskInfo -TaskName $_.TaskName -TaskPath $_.TaskPath -ErrorAction SilentlyContinue
    [pscustomobject]@{
      TaskName = $_.TaskName; TaskPath = $_.TaskPath; State = $_.State
      LastRunTime = $info.LastRunTime; LastTaskResult = $info.LastTaskResult
      Actions = @($_.Actions | ForEach-Object {
        [pscustomobject]@{ Execute = $_.Execute; Arguments = $_.Arguments; WorkingDirectory = $_.WorkingDirectory }
      })
    }
  })
$events = @()
foreach ($logName in @('Application', 'System', 'Microsoft-Windows-TaskScheduler/Operational')) {
  try {
    $found = @(Get-WinEvent -FilterHashtable @{ LogName = $logName; StartTime = $since } -ErrorAction Stop |
      Where-Object {
        $_.ProviderName -match '(?i)application error|windows error reporting|user32|winlogon|kernel-general|task.?scheduler' -or
        $_.Message -match '(?i)python(\.exe)?|app\.main|logoff|log off|terminal|console|session|dfl.*backend|backend.*dfl'
      } |
      Select-Object -Last 100 |
      ForEach-Object {
        [pscustomobject]@{
          TimeCreated = $_.TimeCreated; LogName = $_.LogName; ProviderName = $_.ProviderName
          Id = $_.Id; Level = $_.LevelDisplayName; Message = $_.Message
        }
      })
    $events += $found
  } catch {
    $events += [pscustomobject]@{ LogName = $logName; QueryError = $_.Exception.Message }
  }
}
[pscustomobject]@{ Processes = $processes; Tasks = $tasks; Events = $events } | ConvertTo-Json -Depth 6 -Compress
'''


def _secret_values() -> list[str]:
    values = [
        value
        for key, value in os.environ.items()
        if SENSITIVE_KEY.search(key) and value
    ]
    env_path = ROOT / ".env"
    if env_path.is_file():
        for line in env_path.read_text(encoding="utf-8", errors="replace").splitlines():
            name, separator, value = line.partition("=")
            if separator and SENSITIVE_KEY.search(name.strip()) and value.strip():
                values.append(value.strip().strip("\"'"))
    return sorted(set(values), key=len, reverse=True)


def _redact(value: str, secrets: list[str]) -> str:
    redacted = value
    for secret in secrets:
        redacted = redacted.replace(secret, "[REDACTED]")
    for pattern in REDACTIONS:
        redacted = pattern.sub(r"\1[REDACTED]", redacted)
    return redacted


def _powershell_data(secrets: list[str]) -> dict[str, Any]:
    process = subprocess.run(
        ["powershell.exe", "-NoLogo", "-NoProfile", "-NonInteractive", "-Command", POWERSHELL],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if process.returncode:
        detail = _redact(process.stderr.strip(), secrets)
        raise RuntimeError(f"PowerShell lifecycle query failed: {detail}")
    try:
        result = json.loads(process.stdout)
    except json.JSONDecodeError as error:
        raise RuntimeError("PowerShell lifecycle query returned invalid JSON") from error
    if not isinstance(result, dict):
        raise RuntimeError("PowerShell lifecycle query returned an unexpected value")
    return result


def main() -> None:
    secrets = _secret_values()
    for filename in ("backend.stderr.log", "backend.stdout.log"):
        path = DATA_DIR / filename
        print(f"--- {filename} (last 60 lines; redacted) ---")
        if not path.is_file():
            print("[missing]")
            continue
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        for line in lines[-60:]:
            print(_redact(line, secrets))

    state = _powershell_data(secrets)
    processes = state.get("Processes") or []
    if isinstance(processes, dict):
        processes = [processes]
    print("--- python app.main processes ---")
    if processes:
        for process in processes:
            print(
                f"pid={process.get('ProcessId')}; name={process.get('Name')}; "
                f"start_time={process.get('StartTime')}"
            )
    else:
        print("none")

    tasks = state.get("Tasks") or []
    if isinstance(tasks, dict):
        tasks = [tasks]
    print("--- matching scheduled tasks ---")
    if tasks:
        for task in tasks:
            print(
                f"task={task.get('TaskPath')}{task.get('TaskName')}; state={task.get('State')}; "
                f"last_run={task.get('LastRunTime')}; last_result={task.get('LastTaskResult')}"
            )
            for action in task.get("Actions") or []:
                print(
                    "action_path=" + _redact(str(action.get("Execute", "")), secrets)
                    + "; arguments=" + _redact(str(action.get("Arguments", "")), secrets)
                    + "; working_directory="
                    + _redact(str(action.get("WorkingDirectory", "")), secrets)
                )
    else:
        print("none")

    events = state.get("Events") or []
    if isinstance(events, dict):
        events = [events]
    print("--- relevant Windows events (last 3 hours) ---")
    if events:
        for event in events:
            if event.get("QueryError"):
                print(f"log={event.get('LogName')}; query_error={_redact(str(event['QueryError']), secrets)}")
                continue
            message = " ".join(str(event.get("Message", "")).split())
            print(
                f"time={event.get('TimeCreated')}; log={event.get('LogName')}; "
                f"provider={event.get('ProviderName')}; id={event.get('Id')}; "
                f"level={event.get('Level')}; message={_redact(message, secrets)}"
            )
    else:
        print("none found")
    print(f"diagnostic_time={datetime.now().astimezone().isoformat(timespec='seconds')}")


if __name__ == "__main__":
    main()
