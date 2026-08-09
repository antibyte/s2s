# Prepare Audio8 TTS Preview 0.6B for the s2s Lab host agent.
#
# - Creates tools/audio8/.venv (Python 3.11+ recommended)
# - Downloads pinned HF snapshot into models/audio8-tts-preview-0.6b
# - Installs torch + transformers + soundfile
#
# Usage:
#   .\scripts\setup_audio8.ps1
#   .\scripts\setup_audio8.ps1 -Cuda

[CmdletBinding()]
param(
    [switch]$Cuda,
    [string]$ModelsDir = "",
    [string]$Python = "python"
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
if (-not $ModelsDir) {
    $ModelsDir = Join-Path $Root "models\audio8-tts-preview-0.6b"
}
$VenvDir = Join-Path $Root "tools\audio8\.venv"
$Revision = "f9612f13a0ab40facf3d050fc908b9e6db05c2be"
$Repo = "Audio8/Audio8-TTS-Preview-0.6b"
$ExpectedWeightSize = 1202342528

New-Item -ItemType Directory -Force -Path $ModelsDir | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $ModelsDir "voices") | Out-Null
New-Item -ItemType Directory -Force -Path (Split-Path -Parent $VenvDir) | Out-Null

Write-Host "Downloading Audio8 TTS Preview ($Revision) -> $ModelsDir"
$downloadPy = @"
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='$Repo',
    revision='$Revision',
    local_dir=r'$ModelsDir',
    allow_patterns=[
        'model.safetensors',
        'codec.pth',
        'config.json',
        'configuration_arktts.py',
        'modeling_arktts.py',
        'modeling_arktts_codec.py',
        'processing_arktts.py',
        'preprocessor_config.json',
        'processor_config.json',
        'tokenizer.json',
        'tokenizer_config.json',
        'special_tokens_map.json',
        'generation_config.json',
    ],
)
print('download complete')
"@
& $Python -c $downloadPy
if ($LASTEXITCODE -ne 0) {
    throw "huggingface download failed (pip install huggingface_hub if needed)"
}

$weights = Join-Path $ModelsDir "model.safetensors"
if (-not (Test-Path -LiteralPath $weights)) {
    throw "model.safetensors missing after download"
}
$size = (Get-Item -LiteralPath $weights).Length
if ($size -ne $ExpectedWeightSize) {
    throw "model.safetensors size $size != expected $ExpectedWeightSize"
}
foreach ($name in @(
    "codec.pth",
    "config.json",
    "configuration_arktts.py",
    "modeling_arktts.py",
    "modeling_arktts_codec.py",
    "processing_arktts.py",
    "tokenizer.json"
)) {
    $path = Join-Path $ModelsDir $name
    if (-not (Test-Path -LiteralPath $path)) {
        throw "missing required artifact: $name"
    }
}
Write-Host "weights verified ($ExpectedWeightSize bytes)"

if (-not (Test-Path -LiteralPath (Join-Path $VenvDir "Scripts\python.exe"))) {
    Write-Host "Creating venv at $VenvDir"
    & $Python -m venv $VenvDir
}
$VenvPython = Join-Path $VenvDir "Scripts\python.exe"
& $VenvPython -m pip install --upgrade pip
if ($Cuda) {
    & $VenvPython -m pip install "torch>=2.5.0" "torchaudio>=2.5.0" --index-url https://download.pytorch.org/whl/cu124
} else {
    & $VenvPython -m pip install "torch>=2.5.0" "torchaudio>=2.5.0" --index-url https://download.pytorch.org/whl/cpu
}
& $VenvPython -m pip install "transformers>=4.57.0,<5" "soundfile>=0.12" "safetensors>=0.4" "numpy" "huggingface_hub"

Write-Host ""
Write-Host "Audio8 TTS Preview ready."
Write-Host "  Model:  $ModelsDir"
Write-Host "  Python: $VenvPython"
Write-Host "  Smoke:  & '$VenvPython' '$Root\scripts\tts_audio8_server.py' --model-dir '$ModelsDir' --port 8096"
Write-Host "  Optional clone voices: place <name>.wav + matching <name>.txt under $ModelsDir\voices"
Write-Host "  Then enable experimental catalog variants and select audio8-tts-preview-0.6b in the lab UI."
