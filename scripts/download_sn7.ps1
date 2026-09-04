param(
    [string]$Destination = "data/raw/sn7",
    [switch]$Extract
)

$ErrorActionPreference = "Stop"

$Bucket = "spacenet-dataset"
$Key = "spacenet/SN7_buildings/tarballs/SN7_buildings_train.tar.gz"
$ArchiveName = "SN7_buildings_train.tar.gz"
$MinimumFreeBytes = 30GB

if (-not (Get-Command aws -ErrorAction SilentlyContinue)) {
    throw "AWS CLI is required. Install AWS CLI v2 and retry."
}

$HeadJson = aws s3api head-object `
    --bucket $Bucket `
    --key $Key `
    --no-sign-request `
    --output json

if ($LASTEXITCODE -ne 0) {
    throw "Unable to read SpaceNet 7 archive metadata from S3."
}

$Head = $HeadJson | ConvertFrom-Json
$ExpectedBytes = [int64]$Head.ContentLength
$DestinationPath = [System.IO.Path]::GetFullPath($Destination)
$DriveRoot = [System.IO.Path]::GetPathRoot($DestinationPath)
$Drive = [System.IO.DriveInfo]::new($DriveRoot)

if ($Drive.AvailableFreeSpace -lt $MinimumFreeBytes) {
    $FreeGB = [math]::Round($Drive.AvailableFreeSpace / 1GB, 1)
    throw "At least 30 GB free space is required; only $FreeGB GB is available."
}

New-Item -ItemType Directory -Force -Path $DestinationPath | Out-Null
$ArchivePath = Join-Path $DestinationPath $ArchiveName

if ((Test-Path -LiteralPath $ArchivePath) -and
    ((Get-Item -LiteralPath $ArchivePath).Length -eq $ExpectedBytes)) {
    Write-Host "Archive already exists with the expected size: $ArchivePath"
} else {
    Write-Host "Downloading $ExpectedBytes bytes to $ArchivePath"
    aws s3 cp "s3://$Bucket/$Key" $ArchivePath --no-sign-request --only-show-errors
    if ($LASTEXITCODE -ne 0) {
        throw "SpaceNet 7 download failed. Re-run this script to retry."
    }
}

$DownloadedBytes = (Get-Item -LiteralPath $ArchivePath).Length
if ($DownloadedBytes -ne $ExpectedBytes) {
    throw "Archive size mismatch: expected $ExpectedBytes, found $DownloadedBytes."
}

$Hash = Get-FileHash -LiteralPath $ArchivePath -Algorithm SHA256
Write-Host "SHA256: $($Hash.Hash)"

if ($Extract) {
    $ExtractPath = Join-Path $DestinationPath "train"
    New-Item -ItemType Directory -Force -Path $ExtractPath | Out-Null
    tar -xzf $ArchivePath -C $ExtractPath
    if ($LASTEXITCODE -ne 0) {
        throw "Archive extraction failed."
    }
    Write-Host "Extracted SpaceNet 7 training data to $ExtractPath"
}
