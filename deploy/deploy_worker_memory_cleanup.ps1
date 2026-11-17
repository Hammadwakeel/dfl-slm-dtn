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
if (-not (Get-Acl -LiteralPath $envPath).AreAccessRulesProtected) {
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
$tempRoot = Join-Path ([System.IO.Path]::GetTempPath()) (
    'dfl-worker-memory-cleanup-' + [guid]::NewGuid().ToString('N')
)
New-Item -ItemType Directory -Path $tempRoot -Force | Out-Null
$identity = [System.Security.Principal.WindowsIdentity]::GetCurrent().Name
& icacls $tempRoot /inheritance:r /grant:r `
    "${identity}:(OI)(CI)F" 'SYSTEM:(OI)(CI)F' | Out-Null
if ($LASTEXITCODE -ne 0) {
    Remove-Item -LiteralPath $tempRoot -Recurse -Force
    throw 'Could not restrict permissions on temporary deployment files.'
}

try {
    foreach ($node in $selected) {
        $stageName = ".dfl-worker-memory-cleanup-stage-$($node.Id)"
        $remote = "$($node.User)@$($node.Ip)"
        $workerStage = Join-Path $tempRoot $node.Id
        New-Item -ItemType Directory -Path $workerStage -Force | Out-Null
        $environmentPath = Join-Path $workerStage 'memory-cleanup.env'
        [System.IO.File]::WriteAllText(
            $environmentPath,
            "DFL_MEMORY_CLEANUP_TOKEN=$($values[$node.TokenName])`n",
            $utf8NoBom
        )

        & ssh -i $SshKey -o IdentitiesOnly=yes -o BatchMode=yes $remote `
            "mkdir -p '$stageName' && chmod 700 '$stageName'"
        if ($LASTEXITCODE -ne 0) {
            throw "Could not create protected staging directory on $($node.Ip)."
        }

        & scp -i $SshKey -o IdentitiesOnly=yes -o BatchMode=yes `
            (Join-Path $repoRoot 'worker\memory_cleanup_helper.py') `
            (Join-Path $PSScriptRoot 'dfl-slm-memory-cleanup.service') `
            (Join-Path $PSScriptRoot 'install_memory_cleanup_helper.sh') `
            $environmentPath "$($remote):$stageName/"
        if ($LASTEXITCODE -ne 0) {
            throw "Could not stage memory cleanup helper files on $($node.Ip)."
        }

        & ssh -tt -i $SshKey -o IdentitiesOnly=yes -o BatchMode=yes $remote `
            "sed -i 's/\r$//' '$stageName/install_memory_cleanup_helper.sh' '$stageName/dfl-slm-memory-cleanup.service' && sudo bash '$stageName/install_memory_cleanup_helper.sh'"
        if ($LASTEXITCODE -ne 0) {
            $installExitCode = $LASTEXITCODE
            & ssh -i $SshKey -o IdentitiesOnly=yes -o BatchMode=yes $remote `
                "rm -rf -- '$stageName'"
            if ($LASTEXITCODE -ne 0) {
                throw "Memory cleanup installation failed on $($node.Ip) (exit $installExitCode), and protected staging cleanup also failed."
            }
            throw "Memory cleanup installation failed on $($node.Ip) (exit $installExitCode); removed protected staging files."
        }

        & ssh -i $SshKey -o IdentitiesOnly=yes -o BatchMode=yes $remote `
            "rm -rf -- '$stageName'"
        if ($LASTEXITCODE -ne 0) {
            throw "The helper is installed on $($node.Ip), but staging cleanup failed."
        }
        Write-Output "Installed authenticated memory cleanup helper on $($node.Id) ($($node.Ip))."
    }
}
finally {
    Remove-Item -LiteralPath $tempRoot -Recurse -Force -ErrorAction SilentlyContinue
}
