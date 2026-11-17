param(
    [string]$SshKey = (Join-Path $env:USERPROFILE '.ssh\id_ed25519')
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$repoRoot = Split-Path -Parent $PSScriptRoot
$envPath = Join-Path $repoRoot '.env'
$requiredTokens = @('MASTER_API_TOKEN', 'WORKER1_API_TOKEN', 'WORKER2_API_TOKEN', 'WORKER3_API_TOKEN', 'WORKER4_API_TOKEN', 'WORKER5_API_TOKEN')
$nodes = @(
    @{ Id = 'master'; Role = 'master'; Ip = '192.168.11.15'; User = 'hammad'; Port = 5002; TokenName = 'MASTER_API_TOKEN' },
    @{ Id = 'worker1'; Role = 'worker'; Ip = '192.168.11.123'; User = 'hammad'; Port = 5001; TokenName = 'WORKER1_API_TOKEN' },
    @{ Id = 'worker2'; Role = 'worker'; Ip = '192.168.11.14'; User = 'hammad'; Port = 5001; TokenName = 'WORKER2_API_TOKEN' },
    @{ Id = 'worker3'; Role = 'worker'; Ip = '192.168.11.13'; User = 'hammad'; Port = 5001; TokenName = 'WORKER3_API_TOKEN' },
    @{ Id = 'worker4'; Role = 'worker'; Ip = '192.168.11.75'; User = 'jetson5'; Port = 5001; TokenName = 'WORKER4_API_TOKEN' },
    @{ Id = 'worker5'; Role = 'worker'; Ip = '192.168.11.76'; User = 'hammad'; Port = 5001; TokenName = 'WORKER5_API_TOKEN' }
)

if (-not (Test-Path -LiteralPath $SshKey -PathType Leaf)) {
    throw "SSH private key not found: $SshKey"
}

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

$rng = [System.Security.Cryptography.RandomNumberGenerator]::Create()
$newTokenNames = [System.Collections.Generic.List[string]]::new()
foreach ($name in $requiredTokens) {
    if (-not $values.ContainsKey($name) -or $values[$name].Length -lt 32 -or $values[$name] -like 'replace-*') {
        $bytes = New-Object byte[] 48
        $rng.GetBytes($bytes)
        $token = [Convert]::ToBase64String($bytes).TrimEnd('=').Replace('+', '-').Replace('/', '_')
        $values[$name] = $token
        $existingLine = -1
        for ($index = 0; $index -lt $lines.Count; $index++) {
            if ($lines[$index] -match "^$name=") {
                $existingLine = $index
                break
            }
        }
        if ($existingLine -ge 0) {
            $lines[$existingLine] = "$name=$token"
        }
        else {
            $lines.Add("$name=$token")
        }
        $newTokenNames.Add($name)
    }
}
$rng.Dispose()
if ($newTokenNames.Count -gt 0 -or -not (Test-Path -LiteralPath $envPath)) {
    $utf8NoBom = New-Object System.Text.UTF8Encoding($false)
    [System.IO.File]::WriteAllText($envPath, [string]::Join("`n", $lines) + "`n", $utf8NoBom)
    $acl = Get-Acl -LiteralPath $envPath
    $acl.SetAccessRuleProtection($true, $false)
    foreach ($rule in @($acl.Access)) {
        [void]$acl.RemoveAccessRuleAll($rule)
    }
    $currentUser = [System.Security.Principal.WindowsIdentity]::GetCurrent().Name
    foreach ($identity in @($currentUser, 'NT AUTHORITY\SYSTEM', 'BUILTIN\Administrators')) {
        $rule = [System.Security.AccessControl.FileSystemAccessRule]::new(
            $identity,
            [System.Security.AccessControl.FileSystemRights]::FullControl,
            [System.Security.AccessControl.AccessControlType]::Allow
        )
        $acl.AddAccessRule($rule)
    }
    Set-Acl -LiteralPath $envPath -AclObject $acl
}

$headers = @(
    'Host *',
    "  IdentityFile `"$SshKey`"",
    '  IdentitiesOnly yes',
    '  BatchMode yes',
    '  ConnectTimeout 10'
)
$sshConfigPath = Join-Path ([System.IO.Path]::GetTempPath()) ("dfl-ssh-" + [guid]::NewGuid().ToString('N') + '.conf')
[System.IO.File]::WriteAllText($sshConfigPath, [string]::Join("`n", $headers), (New-Object System.Text.UTF8Encoding($false)))
$tempFiles = [System.Collections.Generic.List[string]]::new()

try {
    foreach ($node in $nodes) {
        $token = $values[$node.TokenName]
        if ($token.Length -lt 32) {
            throw "Token $($node.TokenName) must be at least 32 characters"
        }

        $localEnv = Join-Path ([System.IO.Path]::GetTempPath()) ("dfl-node-" + [guid]::NewGuid().ToString('N') + '.env')
        $tempFiles.Add($localEnv)
        $nodeEnv = @(
            "DFL_NODE_ID=$($node.Id)",
            "DFL_NODE_ROLE=$($node.Role)",
            "DFL_API_TOKEN=$token",
            "DFL_PORT=$($node.Port)",
            'DFL_BIND_HOST=0.0.0.0',
            'DFL_LOG_LEVEL=INFO',
            'DFL_SAMPLE_INTERVAL_SECONDS=2'
        )
        [System.IO.File]::WriteAllText(
            $localEnv,
            [string]::Join("`n", $nodeEnv) + "`n",
            (New-Object System.Text.UTF8Encoding($false))
        )

        $stageName = "dfl-slm-stage-$($node.Id)"
        $destination = "$($node.User)@$($node.Ip):$stageName"
        $sshArgs = @('-F', $sshConfigPath, "$($node.User)@$($node.Ip)")
        $stageCommand = "mkdir -p '$stageName' && chmod 700 '$stageName'"
        & ssh @sshArgs $stageCommand
        if ($LASTEXITCODE -ne 0) {
            throw "Could not create protected staging directory on $($node.Ip)"
        }

        & scp -F $sshConfigPath -r `
            (Join-Path $repoRoot 'common') `
            (Join-Path $PSScriptRoot 'install_jetson.sh') `
            (Join-Path $PSScriptRoot 'dfl-slm-status.service') `
            "$destination/"
        if ($LASTEXITCODE -ne 0) {
            throw "Could not stage service files on $($node.Ip)"
        }
        & scp -F $sshConfigPath $localEnv "$destination/node.env"
        if ($LASTEXITCODE -ne 0) {
            throw "Could not stage protected node configuration on $($node.Ip)"
        }

        $installCommand = "sudo -n env DFL_SERVICE_USER='$($node.User)' bash '$stageName/install_jetson.sh' '$($node.Id)' '$($node.Role)' '$($node.Port)' '$($node.User)'"
        & ssh @sshArgs $installCommand
        if ($LASTEXITCODE -ne 0) {
            throw "Service installation failed on $($node.Ip); staging files remain at ~/$stageName for diagnosis"
        }

        & ssh @sshArgs "rm -rf -- '$stageName'"
        if ($LASTEXITCODE -ne 0) {
            throw "Service is installed on $($node.Ip), but temporary staging cleanup failed"
        }
    }
}
finally {
    Remove-Item -LiteralPath $sshConfigPath -Force -ErrorAction SilentlyContinue
    foreach ($tempFile in $tempFiles) {
        Remove-Item -LiteralPath $tempFile -Force -ErrorAction SilentlyContinue
    }
}

Write-Output 'Status services were installed on the master and all five workers.'
Write-Output "Local API tokens are stored in the ignored file: $envPath"
