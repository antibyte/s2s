# Start VibeVoice Realtime 0.5B (cstr/vibevoice-realtime-0.5b-GGUF) via CrispASR.
#
# Lab backend id: vibevoice-realtime-0.5b
# Default endpoint: http://127.0.0.1:8089/v1/audio/speech
#
# Usage:
#   .\scripts\start_vibevoice.ps1
#   .\scripts\start_vibevoice.ps1 -Build -Cuda

param(
    [int]$Port = 8089,
    [string]$ModelsDir = "",
    [string]$Image = "s2s-tts-vibevoice:local",
    [switch]$Build,
    [switch]$Cuda
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
if (-not $ModelsDir) {
    $ModelsDir = Join-Path $Root "models"
}
$ModelDir = Join-Path $ModelsDir "vibevoice"
$Model = Join-Path $ModelDir "vibevoice-realtime-0.5b-q4_k.gguf"

if (-not (Test-Path $Model)) {
    Write-Host "Model not found at $Model"
    Write-Host "Download first:"
    Write-Host "  hf download cstr/vibevoice-realtime-0.5b-GGUF --local-dir `"$ModelDir`""
    Write-Host "Or use Lab UI download for backend id vibevoice-realtime-0.5b."
    exit 1
}

if ($Cuda) {
    $Image = "s2s-tts-vibevoice:cuda"
}

if ($Build) {
    Push-Location $Root
    try {
        $args = @(
            "build", "-f", "docker/Dockerfile.vibevoice",
            "-t", $Image
        )
        if ($Cuda) {
            $args += @("--build-arg", "CRISPASR_IMAGE=ghcr.io/crispstrobe/crispasr:main-cuda")
        }
        $args += "."
        docker @args
    } finally {
        Pop-Location
    }
}

$Name = if ($Cuda) { "s2s-tts-vibevoice-cuda" } else { "s2s-tts-vibevoice" }
docker rm -f $Name 2>$null | Out-Null

$run = @(
    "run", "--rm", "-d",
    "--name", $Name,
    "--shm-size", "2g",
    "-p", "${Port}:8089",
    "-v", "${ModelsDir}:/models",
    "-e", "VIBEVOICE_MODEL=/models/vibevoice/vibevoice-realtime-0.5b-q4_k.gguf",
    "-e", "VIBEVOICE_VOICE=/models/vibevoice/vibevoice-voice-emma.gguf",
    "-e", "VIBEVOICE_VOICE_DIR=/models/vibevoice",
    "-e", "VIBEVOICE_PORT=8089",
    "--label", "s2s.lab.managed=true",
    "--label", "stage=tts",
    "--label", "backend-id=vibevoice-realtime-0.5b"
)
if ($Cuda) {
    $run += @("--gpus", "all")
}
$run += $Image

Write-Host "Starting $Name ($Image) → http://127.0.0.1:$Port/v1/audio/speech"
docker @run
Write-Host "Health: curl http://127.0.0.1:$Port/health"
Write-Host "s2s-vulkan: --tts http --tts-url http://127.0.0.1:$Port/v1/audio/speech --tts-model vibevoice-realtime-0.5b"
