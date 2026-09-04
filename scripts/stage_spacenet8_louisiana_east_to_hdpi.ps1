param(
    [Parameter(Mandatory = $true)]
    [int]$DownloadProcessId
)

$ErrorActionPreference = 'Stop'

$root = 'D:\桌面\ActiveMap\datasets\spacenet8\raw'
$partial = Join-Path $root 'Louisiana-East_Training_Public.tar.gz.partial'
$archive = Join-Path $root 'Louisiana-East_Training_Public.tar.gz'
$expectedBytes = 3336023039
$logRoot = 'D:\桌面\ActiveMap\logs'
$log = Join-Path $logRoot 'spacenet8_louisiana_east_stage_20260904.log'

New-Item -ItemType Directory -Force -Path $logRoot | Out-Null
Start-Transcript -Path $log -Append | Out-Null
try {
    Wait-Process -Id $DownloadProcessId
    $item = Get-Item -LiteralPath $partial
    if ($item.Length -ne $expectedBytes) {
        throw "Incomplete Louisiana-East archive: $($item.Length) / $expectedBytes bytes"
    }
    Move-Item -LiteralPath $partial -Destination $archive

    & scp -P 50050 $archive 'wh@100.100.1.101:/home/wh/ActiveMap/datasets/spacenet8/raw/Louisiana-East_Training_Public.tar.gz'
    if ($LASTEXITCODE -ne 0) {
        throw "scp failed with exit code $LASTEXITCODE"
    }
    & ssh -o BatchMode=yes -p 50050 'wh@100.100.1.101' 'nohup bash /home/wh/projects/activemap-v1/scripts/run_spacenet8_louisiana_external_validation_hdpi.sh >/home/wh/ActiveMap/logs/spacenet8_louisiana_east_external_20260904.log 2>&1 & echo $!'
    if ($LASTEXITCODE -ne 0) {
        throw "HDPI queue launch failed with exit code $LASTEXITCODE"
    }
}
finally {
    Stop-Transcript | Out-Null
}
