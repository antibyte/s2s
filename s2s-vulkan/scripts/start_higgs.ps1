# Start Higgs TTS 3 4B (bosonai/higgs-tts-3-4b) as an OpenAI-compatible TTS server.
#
# Requires NVIDIA GPU + Docker with nvidia-container-toolkit (recommended),
# or a host install of SGLang-Omni / vLLM-Omni.
#
# Lab backend id: higgs-tts-3-4b
# Default endpoint: http://127.0.0.1:8086/v1/audio/speech
#
# Usage:
#   .\scripts\start_higgs.ps1
#   .\scripts\start_higgs.ps1 -Port 8086 -ModelsDir D:\models

param(
    [int]$Port = 8086,
    [string]$ModelsDir = "",
    [string]$Image = "s2s-tts-higgs:local",
    [switch]$Build,
    [switch]$Native
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
if (-not $ModelsDir) {
    $ModelsDir = Join-Path $Root "models"
}
$ModelPath = Join-Path $ModelsDir "higgs-tts-3-4b"
$Config = Join-Path $ModelPath "config.json"

if (-not (Test-Path $Config)) {
    Write-Host "Model not found at $ModelPath"
    Write-Host "Download first (HF CLI):"
    Write-Host "  hf download bosonai/higgs-tts-3-4b --local-dir `"$ModelPath`""
    Write-Host "Or use the Lab UI model download for backend id higgs-tts-3-4b."
    exit 1
}

if ($Native) {
    $serve = Get-Command sgl-omni -ErrorAction SilentlyContinue
    if (-not $serve) {
        Write-Error "sgl-omni not on PATH. Install SGLang-Omni or omit -Native to use Docker."
    }
    Write-Host "Starting native sgl-omni on port $Port ..."
    & sgl-omni serve --model-path $ModelPath --host 0.0.0.0 --port $Port
    exit $LASTEXITCODE
}

if ($Build) {
    Push-Location $Root
    try {
        docker build -f docker/Dockerfile.higgs -t $Image .
    } finally {
        Pop-Location
    }
}

$Name = "s2s-tts-higgs"
docker rm -f $Name 2>$null | Out-Null
Write-Host "Starting $Name ($Image) → http://127.0.0.1:$Port/v1/audio/speech"
docker run --rm -d `
    --name $Name `
    --gpus all `
    --shm-size 32g `
    -p "${Port}:8086" `
    -v "${ModelsDir}:/models" `
    -e HIGGS_MODEL_PATH=/models/higgs-tts-3-4b `
    -e HIGGS_PORT=8086 `
    --label s2s.lab.managed=true `
    --label stage=tts `
    --label backend-id=higgs-tts-3-4b `
    $Image

Write-Host "Health: curl http://127.0.0.1:$Port/v1/models"
Write-Host "s2s-vulkan: --tts http --tts-url http://127.0.0.1:$Port/v1/audio/speech --tts-model bosonai/higgs-tts-3-4b"
