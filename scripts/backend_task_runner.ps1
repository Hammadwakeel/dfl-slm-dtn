$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$dataRoot = Join-Path $repoRoot "dtn\backend\data"
$stdoutPath = Join-Path $dataRoot "backend.stdout.log"
$stderrPath = Join-Path $dataRoot "backend.stderr.log"

Set-Location $repoRoot
$env:PYTHONPATH = "dtn\backend"
try {
	$python = (Get-Command python -ErrorAction Stop).Source
	$process = Start-Process `
		-FilePath $python `
		-ArgumentList @("-m", "app.main") `
		-WorkingDirectory $repoRoot `
		-RedirectStandardOutput $stdoutPath `
		-RedirectStandardError $stderrPath `
		-WindowStyle Hidden `
		-PassThru `
		-Wait `
		-ErrorAction Stop
	exit $process.ExitCode
}
catch {
	$message = "$(Get-Date -Format o) ERROR backend task runner: $($_.Exception.Message)`r`n"
	[System.IO.File]::AppendAllText($stderrPath, $message)
	exit 1
}
