param(
    [string]$SshKey = (Join-Path $env:USERPROFILE '.ssh\id_ed25519')
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$repoRoot = Split-Path -Parent $PSScriptRoot
$envPath = Join-Path $repoRoot '.env'
$masterIp = '192.168.11.15'
$servicePort = 5003
$workers = @(
    @{ Id = 'worker1'; Ip = '192.168.11.123'; TokenName = 'WORKER1_API_TOKEN' },
    @{ Id = 'worker2'; Ip = '192.168.11.14'; TokenName = 'WORKER2_API_TOKEN' },
    @{ Id = 'worker3'; Ip = '192.168.11.13'; TokenName = 'WORKER3_API_TOKEN' },
    @{ Id = 'worker4'; Ip = '192.168.11.75'; TokenName = 'WORKER4_API_TOKEN' },
    @{ Id = 'worker5'; Ip = '192.168.11.76'; TokenName = 'WORKER5_API_TOKEN' }
)
$values = @{}
$lines = [System.Collections.Generic.List[string]]::new()
if (Test-Path -LiteralPath $envPath) {
    foreach ($line in Get-Content -LiteralPath $envPath) {
        $lines.Add($line)
        if ($line -match '^([A-Z0-9_]+)=(.*)$') {
            $values[$Matches[1]] = $Matches[2]
        }
    }
}

function Set-EnvValue([string]$Name, [string]$Value) {
    $script:values[$Name] = $Value
    for ($index = 0; $index -lt $script:lines.Count; $index++) {
        if ($script:lines[$index] -match "^$Name=") {
            $script:lines[$index] = "$Name=$Value"
            return
        }
    }
    $script:lines.Add("$Name=$Value")
}

function New-ApiToken {
    $bytes = New-Object byte[] 48
    $rng = [System.Security.Cryptography.RandomNumberGenerator]::Create()
    try {
        $rng.GetBytes($bytes)
    }
    finally {
        $rng.Dispose()
    }
    return [Convert]::ToBase64String($bytes).TrimEnd('=').Replace('+', '-').Replace('/', '_')
}

if (-not (Test-Path -LiteralPath $SshKey -PathType Leaf)) {
    throw "SSH private key not found: $SshKey"
}
if (-not $values.ContainsKey('MASTER_API_TOKEN') -or $values['MASTER_API_TOKEN'].Length -lt 32) {
    throw 'MASTER_API_TOKEN must be present in the ignored root .env file'
}
if (-not $values.ContainsKey('DTN_API_TOKEN') -or $values['DTN_API_TOKEN'].Length -lt 32) {
    throw 'DTN_API_TOKEN must be present in the ignored root .env file for read-only browser chat'
}
if (-not $values.ContainsKey('MASTER_SLM_API_TOKEN') -or
    $values['MASTER_SLM_API_TOKEN'].Length -lt 32 -or
    $values['MASTER_SLM_API_TOKEN'] -like 'replace-*') {
    Set-EnvValue 'MASTER_SLM_API_TOKEN' (New-ApiToken)
}
if (-not $values.ContainsKey('MASTER_OLLAMA_MODEL') -or -not $values['MASTER_OLLAMA_MODEL']) {
    Set-EnvValue 'MASTER_OLLAMA_MODEL' 'qwen3.5:4b'
}
foreach ($worker in $workers) {
    if (-not $values.ContainsKey($worker.TokenName) -or $values[$worker.TokenName].Length -lt 32) {
        throw "$($worker.TokenName) must be set in the ignored .env file."
    }
}
Set-EnvValue 'MASTER_SLM_PORT' "$servicePort"

$utf8NoBom = New-Object System.Text.UTF8Encoding($false)
[System.IO.File]::WriteAllText($envPath, [string]::Join("`n", $lines) + "`n", $utf8NoBom)
$acl = Get-Acl -LiteralPath $envPath
if (-not $acl.AreAccessRulesProtected) {
    throw 'The root .env file must have a restricted ACL before adding the SLM token'
}

$headers = @(
    'Host *',
    "  IdentityFile `"$SshKey`"",
    '  IdentitiesOnly yes',
    '  BatchMode yes',
    '  ConnectTimeout 10'
)
$sshConfigPath = Join-Path ([System.IO.Path]::GetTempPath()) ("dfl-ssh-" + [guid]::NewGuid().ToString('N') + '.conf')
$stageName = '.dfl-slm-master-stage'
$stageRoot = Join-Path ([System.IO.Path]::GetTempPath()) ("dfl-master-" + [guid]::NewGuid().ToString('N'))
$localEnv = Join-Path $stageRoot 'master.env'
$remote = "hammad@$masterIp"
$tempFiles = [System.Collections.Generic.List[string]]::new()
New-Item -ItemType Directory -Path (Join-Path $stageRoot 'master') -Force | Out-Null
New-Item -ItemType Directory -Path (Join-Path $stageRoot 'common') -Force | Out-Null
New-Item -ItemType Directory -Path (Join-Path $stageRoot 'data') -Force | Out-Null
New-Item -ItemType Directory -Path (Join-Path $stageRoot 'topology_opt') -Force | Out-Null
$tempFiles.Add($stageRoot)

$workerEntries = @(
    foreach ($worker in $workers) {
        [pscustomobject]@{
            node_id = $worker.Id
            base_url = "http://$($worker.Ip):5004"
            token = $values[$worker.TokenName]
        }
    }
)
$workersJson = ConvertTo-Json -InputObject $workerEntries -Compress
$workersJsonBase64 = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($workersJson))
$nodeEnv = @(
    "DFL_API_TOKEN=$($values['MASTER_SLM_API_TOKEN'])",
    "DFL_STATUS_API_TOKEN=$($values['MASTER_API_TOKEN'])",
    "DFL_CHAT_API_TOKEN=$($values['DTN_API_TOKEN'])",
    'DFL_CHAT_ALLOWED_ORIGINS=http://localhost:5173,http://192.168.11.12:5173',
    "DFL_PORT=$servicePort",
    'DFL_BIND_HOST=0.0.0.0',
    "SLM_MODEL_NAME=$($values['MASTER_OLLAMA_MODEL'])",
    'DFL_OLLAMA_URL=http://127.0.0.1:11434',
    'DFL_STATUS_URL=http://127.0.0.1:5002',
    'DFL_LOG_FILE=/var/log/dfl-slm-master/events.jsonl',
    'DFL_ACTION_HISTORY_PATH=/var/lib/dfl-slm-master/action_history.sqlite3',
    'DFL_OLLAMA_TIMEOUT_SECONDS=120',
    'DFL_OLLAMA_NUM_CTX=4096',
    'DFL_OLLAMA_KEEP_ALIVE=15m',
    "DFL_WORKERS_JSON_B64=$workersJsonBase64",
    'DFL_TRAINING_ROUNDS=10',
    'DFL_LOCAL_EPOCHS=5',
    'DFL_TOPOLOGY=directed_ring',
    'DFL_TRAINING_PEER_TIMEOUT_SECONDS=60',
    'DFL_TRAINING_ROUND_TIMEOUT_SECONDS=7200',
    'DFL_EVALUATOR_PYTHON=/usr/bin/python3',
    'DFL_EVALUATOR_SCRIPT=/opt/dfl-slm-dtn/master/cifar_evaluator.py',
    'DFL_TEST_DATA_PATH=/opt/dfl-slm-dtn/data/cifar10_test.npz',
    'DFL_BASELINE_TRAIN_DATA_PATH=/opt/dfl-slm-dtn/data/cifar10_train.npz',
    'DFL_EVALUATOR_LOG_PATH=/var/log/dfl-slm-master/cifar-evaluator.log',
    'DFL_TRAINING_RESULTS_PATH=/var/lib/dfl-slm-master/experiments.sqlite3',
    'DFL_TOPOLOGY_REWARD_CONFIG=/opt/dfl-slm-dtn/topology_opt/reward_config.json',
    'PYTHONDONTWRITEBYTECODE=1'
)
[System.IO.File]::WriteAllText($localEnv, [string]::Join("`n", $nodeEnv) + "`n", $utf8NoBom)
$tempFiles.Add($localEnv)
[System.IO.File]::WriteAllText($sshConfigPath, [string]::Join("`n", $headers), $utf8NoBom)

try {
    $requiredData = @(
        (Join-Path $repoRoot 'data\raw\cifar10_test.npz'),
        (Join-Path $repoRoot 'data\raw\cifar10_train.npz')
    )
    foreach ($path in $requiredData) {
        if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
            throw "CIFAR-10 evaluation/baseline data is missing: $path"
        }
    }
    & ssh -F $sshConfigPath $remote "mkdir -p '$stageName/master/tests' '$stageName/common' '$stageName/data' '$stageName/topology_opt' && chmod 700 '$stageName' '$stageName/master' '$stageName/master/tests' '$stageName/common' '$stageName/data' '$stageName/topology_opt'"
    if ($LASTEXITCODE -ne 0) {
        throw 'Could not create protected SLM staging directories on the master'
    }

    $sources = @(
        (Join-Path $repoRoot 'master\app.py'),
        (Join-Path $repoRoot 'master\__init__.py'),
        (Join-Path $repoRoot 'master\requirements.txt'),
        (Join-Path $repoRoot 'master\wsgi.py'),
        (Join-Path $repoRoot 'master\training.py'),
        (Join-Path $repoRoot 'master\cifar_evaluator.py')
    )
    & scp -F $sshConfigPath @sources "${remote}:$stageName/master/"
    if ($LASTEXITCODE -ne 0) {
        throw 'Could not stage SLM application files on the master'
    }
    & scp -F $sshConfigPath `
        (Join-Path $repoRoot 'master\tests\test_app.py') `
        "${remote}:$stageName/master/tests/"
    if ($LASTEXITCODE -ne 0) {
        throw 'Could not stage SLM application tests on the master'
    }
    & scp -F $sshConfigPath (Join-Path $repoRoot 'common\slm_action_schema.json') "${remote}:$stageName/common/"
    if ($LASTEXITCODE -ne 0) {
        throw 'Could not stage the action schema on the master'
    }
    & scp -F $sshConfigPath `
        (Join-Path $repoRoot 'common\__init__.py') `
        (Join-Path $repoRoot 'common\cifar_model.py') `
        (Join-Path $repoRoot 'common\dataset_nas.py') `
        (Join-Path $repoRoot 'common\nas.py') `
        (Join-Path $repoRoot 'common\model_design.py') `
        "${remote}:$stageName/common/"
    if ($LASTEXITCODE -ne 0) {
        throw 'Could not stage the shared CIFAR-10 model on the master'
    }
    & scp -F $sshConfigPath `
        (Join-Path $repoRoot 'topology_opt\__init__.py') `
        (Join-Path $repoRoot 'topology_opt\reward.py') `
        (Join-Path $repoRoot 'topology_opt\selector.py') `
        (Join-Path $repoRoot 'topology_opt\reward_config.json') `
        "${remote}:$stageName/topology_opt/"
    if ($LASTEXITCODE -ne 0) {
        throw 'Could not stage the topology optimizer on the master'
    }
    & scp -F $sshConfigPath @requiredData "${remote}:$stageName/data/"
    if ($LASTEXITCODE -ne 0) {
        throw 'Could not stage the held-out and centralized-baseline data on the master'
    }
    & scp -F $sshConfigPath `
        (Join-Path $PSScriptRoot 'install_master_slm.sh') `
        (Join-Path $PSScriptRoot 'dfl-slm-master.service') `
        "${remote}:$stageName/"
    if ($LASTEXITCODE -ne 0) {
        throw 'Could not stage the SLM service installer on the master'
    }
    & scp -F $sshConfigPath $localEnv "${remote}:$stageName/master.env"
    if ($LASTEXITCODE -ne 0) {
        throw 'Could not stage protected SLM environment values on the master'
    }

    & ssh -F $sshConfigPath $remote "sed -i 's/\r$//' '$stageName/install_master_slm.sh' '$stageName/dfl-slm-master.service' && sudo -n bash '$stageName/install_master_slm.sh' 2>&1"
    if ($LASTEXITCODE -ne 0) {
        throw "Master SLM install failed; protected staging files remain at ~/$stageName for diagnosis"
    }
    & ssh -F $sshConfigPath $remote "rm -rf -- '$stageName'"
    if ($LASTEXITCODE -ne 0) {
        throw 'SLM is installed, but the temporary staging directory could not be cleaned up'
    }
}
finally {
    Remove-Item -LiteralPath $sshConfigPath -Force -ErrorAction SilentlyContinue
    foreach ($path in $tempFiles) {
        Remove-Item -LiteralPath $path -Recurse -Force -ErrorAction SilentlyContinue
    }
}

Write-Output "Master SLM service is installed on $masterIp`:$servicePort."
Write-Output 'Advisory chat and the agent share the single Ollama model; the old LoRA reply service is retired.'
Write-Output 'The existing dfl-slm-status.service continues on port 5002.'
Write-Output "The coordinator token is stored locally in the ignored file: $envPath"
