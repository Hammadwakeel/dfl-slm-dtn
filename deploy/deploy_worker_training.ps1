param(
    [string]$SshKey = (Join-Path $env:USERPROFILE '.ssh\id_ed25519'),
    [string[]]$WorkerIds = @('worker1', 'worker2', 'worker3')
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$repoRoot = Split-Path -Parent $PSScriptRoot
$envPath = Join-Path $repoRoot '.env'
$nodes = @(
    @{ Id = 'worker1'; Ip = '192.168.11.123'; TokenName = 'WORKER1_API_TOKEN' },
    @{ Id = 'worker2'; Ip = '192.168.11.14'; TokenName = 'WORKER2_API_TOKEN' },
    @{ Id = 'worker3'; Ip = '192.168.11.13'; TokenName = 'WORKER3_API_TOKEN' }
)

if (-not (Test-Path -LiteralPath $SshKey -PathType Leaf)) {
    throw "SSH private key not found: $SshKey"
}
if (-not (Test-Path -LiteralPath $envPath -PathType Leaf)) {
    throw 'The ignored root .env file is required for worker service credentials.'
}
$acl = Get-Acl -LiteralPath $envPath
if (-not $acl.AreAccessRulesProtected) {
    throw 'The root .env file must have a restricted ACL before deployment.'
}

$values = @{}
foreach ($line in Get-Content -LiteralPath $envPath) {
    if ($line -match '^([A-Z0-9_]+)=(.*)$') {
        $values[$Matches[1]] = $Matches[2]
    }
}
foreach ($node in $nodes) {
    if (-not $values.ContainsKey($node.TokenName) -or $values[$node.TokenName].Length -lt 32) {
        throw "$($node.TokenName) is missing or too short in the ignored .env file."
    }
}
$selected = @($nodes | Where-Object { $WorkerIds -contains $_.Id })
if ($selected.Count -ne $WorkerIds.Count) {
    throw 'WorkerIds may contain only worker1, worker2, and worker3.'
}

$utf8NoBom = New-Object System.Text.UTF8Encoding($false)
$tempRoot = Join-Path ([System.IO.Path]::GetTempPath()) ("dfl-worker-" + [guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $tempRoot -Force | Out-Null
$tempEnvs = [System.Collections.Generic.List[string]]::new()
try {
    foreach ($node in $selected) {
        $shardPath = Join-Path $repoRoot "data\worker_shards\$($node.Id).npz"
        if (-not (Test-Path -LiteralPath $shardPath -PathType Leaf)) {
            throw "Training shard is missing: $shardPath"
        }
        $peerEntries = @(
            foreach ($peer in $nodes) {
                if ($peer.Id -ne $node.Id) {
                    [pscustomobject]@{
                        node_id = $peer.Id
                        base_url = "http://$($peer.Ip):5004"
                        token = $values[$peer.TokenName]
                    }
                }
            }
        )
        $peerJson = ConvertTo-Json -InputObject $peerEntries -Compress
        $peerJsonBase64 = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($peerJson))
        $workerEnv = Join-Path $tempRoot "$($node.Id).env"
        $tempEnvs.Add($workerEnv)
        $lines = @(
            "DFL_NODE_ID=$($node.Id)",
            "DFL_API_TOKEN=$($values[$node.TokenName])",
            'DFL_TRAINING_PORT=5004',
            'DFL_BIND_HOST=0.0.0.0',
            'DFL_TRAINING_DATA_PATH=/opt/dfl-slm-dtn/worker/data/train_shard.npz',
            "DFL_TRAINING_LOG_PATH=/var/log/dfl-slm-worker-training/$($node.Id).jsonl",
            "DFL_PEERS_JSON_B64=$peerJsonBase64",
            'DFL_PEER_TIMEOUT_SECONDS=60'
        )
        [System.IO.File]::WriteAllText($workerEnv, [string]::Join("`n", $lines) + "`n", $utf8NoBom)

        $stageName = ".dfl-worker-training-stage-$($node.Id)"
        $remote = "hammad@$($node.Ip)"
        & ssh -i $SshKey -o IdentitiesOnly=yes -o BatchMode=yes $remote `
            "mkdir -p '$stageName/worker' '$stageName/common' && chmod 700 '$stageName' '$stageName/worker' '$stageName/common'"
        if ($LASTEXITCODE -ne 0) {
            throw "Could not create protected staging directory on $($node.Ip)."
        }

        $sources = @(
            (Join-Path $PSScriptRoot 'dfl-slm-worker-training.service'),
            (Join-Path $PSScriptRoot 'install_worker_training.sh')
        )
        & scp -i $SshKey -o IdentitiesOnly=yes -o BatchMode=yes @sources `
            "$($remote):$stageName/"
        if ($LASTEXITCODE -ne 0) {
            throw "Could not stage the training service on $($node.Ip)."
        }
        & scp -i $SshKey -o IdentitiesOnly=yes -o BatchMode=yes `
            (Join-Path $repoRoot 'common\__init__.py') `
            (Join-Path $repoRoot 'common\auth.py') `
            (Join-Path $repoRoot 'common\cifar_model.py') `
            "$($remote):$stageName/common/"
        if ($LASTEXITCODE -ne 0) {
            throw "Could not stage shared training helpers on $($node.Ip)."
        }
        & scp -i $SshKey -o IdentitiesOnly=yes -o BatchMode=yes `
            (Join-Path $repoRoot 'worker\training_service.py') `
            (Join-Path $repoRoot 'worker\__init__.py') `
            "$($remote):$stageName/worker/"
        if ($LASTEXITCODE -ne 0) {
            throw "Could not stage the worker service package on $($node.Ip)."
        }
        & scp -i $SshKey -o IdentitiesOnly=yes -o BatchMode=yes $shardPath `
            "$($remote):$stageName/train_shard.npz"
        if ($LASTEXITCODE -ne 0) {
            throw "Could not copy the $($node.Id) training shard."
        }
        & scp -i $SshKey -o IdentitiesOnly=yes -o BatchMode=yes $workerEnv `
            "$($remote):$stageName/worker-training.env"
        if ($LASTEXITCODE -ne 0) {
            throw "Could not stage protected service configuration on $($node.Ip)."
        }

        & ssh -i $SshKey -o IdentitiesOnly=yes -o BatchMode=yes $remote `
            "sed -i 's/\r$//' '$stageName/install_worker_training.sh' '$stageName/dfl-slm-worker-training.service' && sudo -n bash '$stageName/install_worker_training.sh'"
        if ($LASTEXITCODE -ne 0) {
            throw "Training service installation failed on $($node.Ip); staging files remain at ~/$stageName."
        }
        & ssh -i $SshKey -o IdentitiesOnly=yes -o BatchMode=yes $remote `
            "rm -rf -- '$stageName'"
        if ($LASTEXITCODE -ne 0) {
            throw "The service is installed, but staging cleanup failed on $($node.Ip)."
        }
        Write-Output "Installed authenticated training service and one local shard on $($node.Id) ($($node.Ip))."
    }
}
finally {
    foreach ($path in $tempEnvs) {
        Remove-Item -LiteralPath $path -Force -ErrorAction SilentlyContinue
    }
    Remove-Item -LiteralPath $tempRoot -Recurse -Force -ErrorAction SilentlyContinue
}
