# Prepare Inflect-Micro-v2 (9.3M) for the s2s Lab host agent.
#
# - Creates tools/inflect/.venv (Python 3.11+ recommended)
# - Downloads pinned inference artifacts into models/inflect-micro-v2
# - Installs CPU PyTorch + model requirements
#
# Usage:
#   .\scripts\setup_inflect.ps1
#   .\scripts\setup_inflect.ps1 -Cuda

[CmdletBinding()]
param(
    [switch]$Cuda,
    [string]$ModelsDir = "",
    [string]$Python = "python"
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
if (-not $ModelsDir) {
    $ModelsDir = Join-Path $Root "models\inflect-micro-v2"
}
$VenvDir = Join-Path $Root "tools\inflect\.venv"
$Revision = "1e0f60061e50c6849ce4c79c9aff0887fa631d81"
$Repo = "owensong/Inflect-Micro-v2"
$ExpectedSize = 37529995
$ExpectedSha = "3eede065c9ccfa88ade0a5a9a5c23de34afcbbb32213e59aad44d5cf100fdee8"

New-Item -ItemType Directory -Force -Path $ModelsDir | Out-Null
New-Item -ItemType Directory -Force -Path (Split-Path -Parent $VenvDir) | Out-Null

function Get-FileSha256([string]$Path) {
    $hash = Get-FileHash -Algorithm SHA256 -LiteralPath $Path
    return $hash.Hash.ToLowerInvariant()
}

Write-Host "Downloading Inflect-Micro-v2 ($Revision) -> $ModelsDir"
$downloadPy = @"
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='$Repo',
    revision='$Revision',
    local_dir=r'$ModelsDir',
    allow_patterns=[
        'model.pth',
        'config.json',
        'inference.py',
        'inflect_vits_frontend.py',
        'inflect_nano_v2_frontend.py',
        'requirements.txt',
        'requirements-tested.txt',
        'runtime/**',
    ],
)
print('download complete')
"@
& $Python -c $downloadPy
if ($LASTEXITCODE -ne 0) {
    throw "huggingface download failed (pip install huggingface_hub if needed)"
}

$weights = Join-Path $ModelsDir "model.pth"
if (-not (Test-Path -LiteralPath $weights)) {
    throw "model.pth missing after download"
}
$size = (Get-Item -LiteralPath $weights).Length
if ($size -ne $ExpectedSize) {
    throw "model.pth size $size != expected $ExpectedSize"
}
$sha = Get-FileSha256 $weights
if ($sha -ne $ExpectedSha) {
    throw "model.pth sha256 $sha != expected $ExpectedSha"
}
Write-Host "model.pth verified ($ExpectedSize bytes, sha256 ok)"

if (-not (Test-Path -LiteralPath (Join-Path $VenvDir "Scripts\python.exe"))) {
    Write-Host "Creating venv at $VenvDir"
    & $Python -m venv $VenvDir
}
$VenvPython = Join-Path $VenvDir "Scripts\python.exe"
& $VenvPython -m pip install --upgrade pip
if ($Cuda) {
    & $VenvPython -m pip install torch --index-url https://download.pytorch.org/whl/cu124
} else {
    & $VenvPython -m pip install torch --index-url https://download.pytorch.org/whl/cpu
}
$req = Join-Path $ModelsDir "requirements.txt"
if (Test-Path -LiteralPath $req) {
    & $VenvPython -m pip install -r $req
} else {
    & $VenvPython -m pip install numpy soundfile
}
& $VenvPython -m pip install huggingface_hub

Write-Host ""
Write-Host "Inflect-Micro-v2 ready."
Write-Host "  Model:  $ModelsDir"
Write-Host "  Python: $VenvPython"
Write-Host "  Smoke:  `$env:S2S_INFLECT_MODEL_DIR='$ModelsDir'; & '$VenvPython' '$Root\scripts\tts_inflect_server.py' --port 8095"
Write-Host "Note: phonemizer may require eSpeak-ng on PATH for best English quality."
