"""Register and start the persistent local backend Scheduled Task."""

from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TASK_NAME = "DFL Backend Coordinator"
RUNNER = ROOT / "scripts" / "backend_task_runner.ps1"


def _ps_quote(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def main() -> None:
    if not RUNNER.is_file():
        raise FileNotFoundError(f"Task runner is missing: {RUNNER}")

    runner = _ps_quote(str(RUNNER))
    root = _ps_quote(str(ROOT))
    task_name = _ps_quote(TASK_NAME)
    command = f"""
$ErrorActionPreference = 'Stop'
$taskName = {task_name}
$repoRoot = {root}
$runnerPath = {runner}
$identity = "$env:USERDOMAIN\\$env:USERNAME"
$powerShellPath = Join-Path $PSHOME 'powershell.exe'
$arguments = '-NoLogo -NoProfile -NonInteractive -WindowStyle Hidden -ExecutionPolicy Bypass -File "' + $runnerPath + '"'
$action = New-ScheduledTaskAction -Execute $powerShellPath -Argument $arguments -WorkingDirectory $repoRoot
$trigger = New-ScheduledTaskTrigger -AtLogOn -User $identity
$settings = New-ScheduledTaskSettingsSet -RestartCount 5 -RestartInterval (New-TimeSpan -Minutes 1) -StartWhenAvailable -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -ExecutionTimeLimit ([TimeSpan]::Zero) -MultipleInstances IgnoreNew
$principal = New-ScheduledTaskPrincipal -UserId $identity -LogonType Interactive -RunLevel Limited
Register-ScheduledTask -TaskName $taskName -Action $action -Trigger $trigger -Settings $settings -Principal $principal -Description 'Starts the local DFL coordinator at user logon and restarts it after task failure.' -Force | Out-Null
Start-ScheduledTask -TaskName $taskName
$task = Get-ScheduledTask -TaskName $taskName
$info = Get-ScheduledTaskInfo -TaskName $taskName
[pscustomobject]@{{
  TaskName = $task.TaskName
  State = $task.State
  LastTaskResult = $info.LastTaskResult
  ActionPath = $task.Actions[0].Execute
  ActionArguments = $task.Actions[0].Arguments
  WorkingDirectory = $task.Actions[0].WorkingDirectory
  Trigger = 'AtLogOn'
  RestartIntervalMinutes = 1
  RestartCount = 5
}} | ConvertTo-Json -Compress
"""
    result = subprocess.run(
        [
            "powershell.exe",
            "-NoLogo",
            "-NoProfile",
            "-NonInteractive",
            "-Command",
            command,
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=45,
        check=False,
    )
    if result.returncode:
        raise RuntimeError(
            f"Scheduled Task registration failed (exit {result.returncode}): "
            f"{result.stderr.strip()[-1000:]}"
        )
    print(result.stdout.strip())


if __name__ == "__main__":
    main()
