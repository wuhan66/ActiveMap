param(
    [ValidateSet("all", "nts", "hdpi")]
    [string]$Target = "all",
    [switch]$KeepArchive
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$ReleaseId = "local-" + (Get-Date).ToUniversalTime().ToString("yyyyMMddTHHmmssZ")
$ReleaseRoot = Join-Path $ProjectRoot ".local\releases"
$Archive = Join-Path $ReleaseRoot "$ReleaseId.tar.gz"
New-Item -ItemType Directory -Force -Path $ReleaseRoot | Out-Null

tar -czf $Archive `
    --exclude="./.local" `
    --exclude="./.tmp" `
    --exclude="./.git" `
    --exclude="./.venv" `
    --exclude="./.pytest_cache" `
    --exclude="./.ruff_cache" `
    --exclude="./.mypy_cache" `
    --exclude="*/__pycache__" `
    --exclude="./data/raw" `
    --exclude="./data/interim" `
    --exclude="./data/processed" `
    --exclude="./outputs" `
    --exclude="./reports/generated" `
    --exclude="./reports/server_runs" `
    -C $ProjectRoot .
if ($LASTEXITCODE -ne 0) { throw "Failed to create canonical archive" }

$Sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $Archive).Hash.ToLowerInvariant()
$Servers = @{
    nts = @{
        Host = "nts-server.harmo.icu"
        Port = 50053
        Repository = "/home/wh/projects/activemap-v1-joint-debug"
        Storage = "/mnt/mydisk/wh/ActiveMap"
        Identity = $null
    }
    hdpi = @{
        Host = "hdpi-sys1"
        Port = 50050
        Repository = "/home/wh/projects/activemap-v1"
        Storage = "/home/wh/ActiveMap"
        Identity = "C:\Users\Administrator\.ssh\cprea_hdpi"
    }
}
$Names = if ($Target -eq "all") { @("nts", "hdpi") } else { @($Target) }

foreach ($Name in $Names) {
    $Server = $Servers[$Name]
    $Destination = "wh@$($Server.Host)"
    $SshArgs = @("-p", [string]$Server.Port)
    $ScpArgs = @("-P", [string]$Server.Port)
    if ($Server.Identity) {
        $SshArgs += @("-i", $Server.Identity)
        $ScpArgs += @("-i", $Server.Identity)
    }
    $Incoming = "$($Server.Storage)/code-sync/incoming/$ReleaseId.tar.gz"
    $Staging = "$($Server.Storage)/code-sync/staging/$ReleaseId"
    $Backup = "$($Server.Storage)/code-sync/backups/$ReleaseId"
    $Receipt = "$($Server.Storage)/code-sync/receipts/$ReleaseId.txt"

    & ssh @SshArgs $Destination "mkdir -p '$($Server.Storage)/code-sync/incoming' '$($Server.Storage)/code-sync/staging' '$($Server.Storage)/code-sync/backups' '$($Server.Storage)/code-sync/receipts'"
    if ($LASTEXITCODE -ne 0) { throw "Failed to initialize $Name" }
    & scp @ScpArgs $Archive "${Destination}:${Incoming}"
    if ($LASTEXITCODE -ne 0) { throw "Failed to upload canonical archive to $Name" }

    $Remote = @"
set -euo pipefail
echo '$Sha256  $Incoming' | sha256sum -c -
mkdir -p '$Staging' '$Backup' '$($Server.Repository)'
tar -xzf '$Incoming' -C '$Staging'
rsync -a --backup --backup-dir='$Backup' --exclude='.local/' '$Staging/' '$($Server.Repository)/'
printf 'release_id=%s\nsha256=%s\nsource=%s\nrepository=%s\nsynced_at=%s\n' \
  '$ReleaseId' '$Sha256' 'D:\desktop\ActiveMap\activemap-v1' \
  '$($Server.Repository)' "`$(date --iso-8601=seconds)" > '$Receipt'
"@
    & ssh @SshArgs $Destination $Remote
    if ($LASTEXITCODE -ne 0) { throw "Failed to install canonical archive on $Name" }
    Write-Host "[$Name] $ReleaseId $Sha256 -> $($Server.Repository)"
}

if (-not $KeepArchive) {
    Remove-Item -LiteralPath $Archive
}
