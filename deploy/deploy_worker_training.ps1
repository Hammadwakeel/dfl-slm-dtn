param(
    [string]$SshKey = (Join-Path $env:USERPROFILE '.ssh\id_ed25519'),
    [string[]]$WorkerIds = @('worker1', 'worker2', 'worker3', 'worker4', 'worker5')
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$repoRoot = Split-Path -Parent $PSScriptRoot
$envPath = Join-Path $repoRoot '.env'
$nodes = @(
    @{ Id = 'worker1'; Ip = '192.168.11.123'; User = 'hammad'; TokenName = 'WORKER1_API_TOKEN' },
    @{ Id = 'worker2'; Ip = '192.168.11.14'; User = 'hammad'; TokenName = 'WORKER2_API_TOKEN' },
    @{ Id = 'worker3'; Ip = '192.168.11.13'; User = 'hammad'; TokenName = 'WORKER3_API_TOKEN' },
    @{ Id = 'worker4'; Ip = '192.168.11.75'; User = 'jetson5'; TokenName = 'WORKER4_API_TOKEN' },
    @{ Id = 'worker5'; Ip = '192.168.11.76'; User = 'hammad'; TokenName = 'WORKER5_API_TOKEN' }
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
    throw 'WorkerIds may contain only worker1, worker2, worker3, worker4, and worker5.'
}

$utf8NoBom = New-Object System.Text.UTF8Encoding($false)
$tempRoot = Join-Path ([System.IO.Path]::GetTempPath()) ("dfl-worker-" + [guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $tempRoot -Force | Out-Null
$tempEnvs = [System.Collections.Generic.List[string]]::new()
try {
    $pythonPath = $null
    foreach ($candidate in @('.venv-tf\Scripts\python.exe', '.venv\Scripts\python.exe')) {
        $candidatePath = Join-Path $repoRoot $candidate
        if (Test-Path -LiteralPath $candidatePath -PathType Leaf) {
            & $candidatePath -c 'import numpy' 2>$null
            if ($LASTEXITCODE -eq 0) {
                $pythonPath = $candidatePath
                break
            }
        }
    }
    if ($null -eq $pythonPath) {
        $pythonPath = 'python'
    }
    & $pythonPath (Join-Path $repoRoot 'scripts\reshard_cifar10.py') --output-root (Join-Path $tempRoot 'shards') --worker-count 5
    if ($LASTEXITCODE -ne 0) {
        throw 'Could not produce disjoint CIFAR-10 shards for all five workers.'
    }
    foreach ($node in $selected) {
        $shardPath = Join-Path $tempRoot "shards\stratified\$($node.Id).npz"
        $alpha03Path = Join-Path $tempRoot "shards\dirichlet_a03\$($node.Id).npz"
        $alpha05Path = Join-Path $tempRoot "shards\dirichlet_a05\$($node.Id).npz"
        foreach ($requiredShard in @($shardPath, $alpha03Path, $alpha05Path)) {
            if (-not (Test-Path -LiteralPath $requiredShard -PathType Leaf)) {
                throw "Training shard is missing: $requiredShard"
            }
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
        $shardPaths = @{
            dirichlet_a03 = '/opt/dfl-slm-dtn/worker/data/shards/dirichlet_a03.npz'
            dirichlet_a05 = '/opt/dfl-slm-dtn/worker/data/shards/dirichlet_a05.npz'
        }
        $shardPathsBase64 = [Convert]::ToBase64String(
            [Text.Encoding]::UTF8.GetBytes((ConvertTo-Json -InputObject $shardPaths -Compress))
        )
        $workerEnv = Join-Path $tempRoot "$($node.Id).env"
        $tempEnvs.Add($workerEnv)
        $lines = @(
            "DFL_NODE_ID=$($node.Id)",
            "DFL_API_TOKEN=$($values[$node.TokenName])",
            'DFL_TRAINING_PORT=5004',
            'DFL_BIND_HOST=0.0.0.0',
            'DFL_TRAINING_DATA_PATH=/opt/dfl-slm-dtn/worker/data/train_shard.npz',
            'DFL_TRAINING_DATASET_ROOT=/var/lib/dfl-slm-dtn/datasets',
            'HF_HOME=/var/lib/dfl-slm-dtn/huggingface',
            "DFL_TRAINING_SHARDS_JSON_B64=$shardPathsBase64",
            "DFL_TRAINING_LOG_PATH=/var/log/dfl-slm-worker-training/$($node.Id).jsonl",
            "DFL_PEERS_JSON_B64=$peerJsonBase64",
            'DFL_PEER_TIMEOUT_SECONDS=600'
        )
        [System.IO.File]::WriteAllText($workerEnv, [string]::Join("`n", $lines) + "`n", $utf8NoBom)

        $stageName = ".dfl-worker-training-stage-$($node.Id)"
        $remote = "$($node.User)@$($node.Ip)"
        & ssh -i $SshKey -o IdentitiesOnly=yes -o BatchMode=yes $remote `
            "mkdir -p '$stageName/worker' '$stageName/common' '$stageName/shards' && chmod 700 '$stageName' '$stageName/worker' '$stageName/common' '$stageName/shards'"
        if ($LASTEXITCODE -ne 0) {
            throw "Could not create protected staging directory on $($node.Ip)."
        }

        $sources = @(
            (Join-Path $PSScriptRoot 'dfl-slm-worker-training.service'),
            (Join-Path $PSScriptRoot 'dfl-slm-memory-cleanup.service'),
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
            (Join-Path $repoRoot 'common\dataset_nas.py') `
            (Join-Path $repoRoot 'common\nas.py') `
            (Join-Path $repoRoot 'common\model_design.py') `
            "$($remote):$stageName/common/"
        if ($LASTEXITCODE -ne 0) {
            throw "Could not stage shared training helpers on $($node.Ip)."
        }
        & scp -i $SshKey -o IdentitiesOnly=yes -o BatchMode=yes `
            (Join-Path $repoRoot 'worker\training_service.py') `
            (Join-Path $repoRoot 'worker\memory_cleanup_helper.py') `
            (Join-Path $repoRoot 'worker\model_factory.py') `
            (Join-Path $repoRoot 'worker\nas_search.py') `
            (Join-Path $repoRoot 'worker\__init__.py') `
            "$($remote):$stageName/worker/"
        if ($LASTEXITCODE -ne 0) {
            throw "Could not stage the worker service package on $($node.Ip)."
        }
        & scp -i $SshKey -o IdentitiesOnly=yes -o BatchMode=yes `
            (Join-Path $repoRoot 'worker\transformer-requirements.txt') `
            (Join-Path $repoRoot 'worker\legacy-keras-requirements.txt') `
            "$($remote):$stageName/worker/"
        if ($LASTEXITCODE -ne 0) {
            throw "Could not stage worker transformer dependencies on $($node.Ip)."
        }
        & scp -i $SshKey -o IdentitiesOnly=yes -o BatchMode=yes $shardPath `
            "$($remote):$stageName/train_shard.npz"
        if ($LASTEXITCODE -ne 0) {
            throw "Could not copy the $($node.Id) training shard."
        }
        & scp -i $SshKey -o IdentitiesOnly=yes -o BatchMode=yes $alpha03Path `
            "$($remote):$stageName/shards/dirichlet_a03.npz"
        if ($LASTEXITCODE -ne 0) {
            throw "Could not copy the alpha 0.3 shard for $($node.Id)."
        }
        & scp -i $SshKey -o IdentitiesOnly=yes -o BatchMode=yes $alpha05Path `
            "$($remote):$stageName/shards/dirichlet_a05.npz"
        if ($LASTEXITCODE -ne 0) {
            throw "Could not copy the alpha 0.5 shard for $($node.Id)."
        }
        & scp -i $SshKey -o IdentitiesOnly=yes -o BatchMode=yes $workerEnv `
            "$($remote):$stageName/worker-training.env"
        if ($LASTEXITCODE -ne 0) {
            throw "Could not stage protected service configuration on $($node.Ip)."
        }

        & ssh -tt -i $SshKey -o IdentitiesOnly=yes -o BatchMode=yes $remote `
            "sed -i 's/\r$//' '$stageName/install_worker_training.sh' '$stageName/dfl-slm-worker-training.service' && sudo env DFL_SERVICE_USER='$($node.User)' bash '$stageName/install_worker_training.sh'"
        if ($LASTEXITCODE -ne 0) {
            $installExitCode = $LASTEXITCODE
            & ssh -i $SshKey -o IdentitiesOnly=yes -o BatchMode=yes $remote `
                "rm -rf -- '$stageName'"
            if ($LASTEXITCODE -ne 0) {
                throw "Training service installation failed on $($node.Ip) (exit $installExitCode), and protected staging cleanup also failed."
            }
            throw "Training service installation failed on $($node.Ip) (exit $installExitCode); removed protected staging files."
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
