param(
    [string]$RemoteHost = "wh@hdpi-sys1",
    [int]$RemotePort = 50050,
    [string]$IdentityFile = "C:\Users\86159\.ssh\activemap_ed25519",
    [string]$RemoteExport = "/home/wh/ActiveMap/artifacts/paper_results/sn7_step0_frozen_test_v3_cap20_physical_gpu_20260808",
    [string]$RemoteSlices = "/home/wh/ActiveMap/artifacts/paper_results/sn7_step0_frozen_test_operation_slices_v3_cap20_20260808",
    [string]$PaperDir = "",
    [string]$Python = "D:\Anaconda3\python.exe"
)

$ErrorActionPreference = "Stop"
$Repo = Split-Path -Parent $PSScriptRoot
if ([string]::IsNullOrWhiteSpace($PaperDir)) {
    $PaperFolder = -join ([char]0x8BBA, [char]0x6587)
    $PaperDir = Join-Path (Split-Path -Parent $Repo) "$PaperFolder\iclr-2027-style-files\iclr2027"
}
$Cache = Join-Path $Repo "artifacts\sn7_frozen_test_closeout_20260808"
$Stage = "$Cache.partial"
$ExportName = Split-Path -Leaf $RemoteExport
$SlicesName = Split-Path -Leaf $RemoteSlices
$ExportDir = Join-Path $Cache $ExportName
$SlicesDir = Join-Path $Cache $SlicesName
$OperationSlicesJson = Join-Path $SlicesDir "benefit_vs_notool\operation_slices.json"
$QualityScripts = "C:\Users\86159\.codex\skills\paper-writing-suite\scripts"

if (Test-Path $Stage) {
    throw "Refusing stale partial closeout directory: $Stage"
}
if (-not (Test-Path (Join-Path $ExportDir "manifest.json"))) {
    New-Item -ItemType Directory -Path $Stage | Out-Null
    & scp -r -i $IdentityFile -P $RemotePort "${RemoteHost}:$RemoteExport" $Stage
    if ($LASTEXITCODE -ne 0) { throw "Failed to download frozen export" }
    & scp -r -i $IdentityFile -P $RemotePort "${RemoteHost}:$RemoteSlices" $Stage
    if ($LASTEXITCODE -ne 0) { throw "Failed to download operation slices" }
    if (-not (Test-Path (Join-Path $Stage "$ExportName\manifest.json"))) {
        throw "Downloaded frozen export lacks manifest.json"
    }
    if (-not (Test-Path (Join-Path $Stage "$SlicesName\manifest.json"))) {
        throw "Downloaded operation slices lack manifest.json"
    }
    if (-not (Test-Path (Join-Path $Stage "$SlicesName\benefit_vs_notool\operation_slices.json"))) {
        throw "Downloaded operation slices lack benefit-vs-no-tool JSON"
    }
    Move-Item -LiteralPath $Stage -Destination $Cache
}

Push-Location $Repo
try {
    & $Python scripts\import_sn7_frozen_test_into_iclr.py $ExportDir $PaperDir --operation-slices $OperationSlicesJson
    if ($LASTEXITCODE -ne 0) { throw "Frozen-result importer failed" }
} finally {
    Pop-Location
}

Push-Location $PaperDir
try {
    & latexmk -pdf -interaction=nonstopmode iclr2027_conference.tex
    if ($LASTEXITCODE -ne 0) { throw "LaTeX build failed" }
    & $Python "$QualityScripts\check_numeric_evidence.py" .
    if ($LASTEXITCODE -ne 0) { throw "Numeric evidence gate failed" }
    & $Python "$QualityScripts\research_quality_gate.py" --mode full-paper .
    if ($LASTEXITCODE -ne 0) { throw "Research quality gate failed" }
    & $Python "$QualityScripts\record_build.py" .
    if ($LASTEXITCODE -ne 0) { throw "Build attestation failed" }
} finally {
    Pop-Location
}

Write-Host "SN7 frozen result imported, compiled, and verified."
