# Creates dtn/backend/.venv-eval, the TensorFlow CPU environment the backend uses
# to evaluate every worker's model on the shared validation/test splits.
$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$venv = Join-Path $repoRoot "dtn\backend\.venv-eval"
$python = (Get-Command python -ErrorAction Stop).Source
if (-not (Test-Path (Join-Path $venv "Scripts\python.exe"))) {
    & $python -m venv $venv
    if ($LASTEXITCODE -ne 0) { throw "Could not create $venv" }
}
$venvPython = Join-Path $venv "Scripts\python.exe"
& $venvPython -m pip install --quiet -r (Join-Path $repoRoot "dtn\backend\requirements-eval.txt")
if ($LASTEXITCODE -ne 0) { throw "Installing evaluation requirements failed" }
& $venvPython -c "import tensorflow as tf; print('evaluation TensorFlow', tf.__version__)"
