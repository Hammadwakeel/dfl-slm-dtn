$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$dataRoot = Join-Path $repoRoot "dtn\backend\data"
$stdoutPath = Join-Path $dataRoot "backend.stdout.log"
$stderrPath = Join-Path $dataRoot "backend.stderr.log"

$listener = Get-NetTCPConnection -LocalPort 8000 -State Listen -ErrorAction SilentlyContinue
if ($listener) {
    throw "Port 8000 already has a listener; refusing to start a second backend."
}

$python = (Get-Command python -ErrorAction Stop).Source
$env:PYTHONPATH = "dtn\backend"
$process = Start-Process `
    -FilePath $python `
    -ArgumentList "-m app.main" `
    -WorkingDirectory $repoRoot `
    -RedirectStandardOutput $stdoutPath `
    -RedirectStandardError $stderrPath `
    -WindowStyle Hidden `
    -PassThru

$deadline = (Get-Date).AddSeconds(45)
do {
    Start-Sleep -Seconds 1
    if ($process.HasExited) {
        throw "Backend exited during startup (exit code $($process.ExitCode))."
    }
    $health = $null
    try {
        $health = Invoke-RestMethod -Uri "http://127.0.0.1:8000/healthz" -TimeoutSec 2
    }
    catch {
        $health = $null
    }
    if ($health.status -eq "ok") {
        Write-Output "Hidden backend process started (PID $($process.Id)); /healthz returned ok."
        exit 0
    }
} while ((Get-Date) -lt $deadline)

throw "Backend did not become healthy within 45 seconds (PID $($process.Id))."
