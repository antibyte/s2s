# Start Kokoro TTS OpenAI-compatible HTTP server on :8084
#
#   .\scripts\start_kokoro.ps1
#   .\scripts\start_kokoro.ps1 -Backend onnx -Download
#
# Then in lab: TTS → Kokoro, or:
#   s2s-vulkan --tts http --tts-url http://127.0.0.1:8084/v1/audio/speech

param(
    [string]$HostAddr = "127.0.0.1",
    [int]$Port = 8084,
    [ValidateSet("auto", "kokoro", "onnx")]
    [string]$Backend = "auto",
    [string]$Speaker = "af_bella",
    [switch]$Download
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$Script = Join-Path $Root "scripts\tts_kokoro_server.py"

if (-not (Test-Path $Script)) {
    throw "Missing $Script"
}

$py = Get-Command python -ErrorAction SilentlyContinue
if (-not $py) { throw "python not found on PATH" }

# Best-effort deps (quiet if already present)
Write-Host "Ensuring FastAPI stack…" -ForegroundColor Cyan
& python -m pip install -q fastapi uvicorn soundfile numpy 2>$null

if ($Backend -eq "kokoro" -or $Backend -eq "auto") {
    Write-Host "Trying hexgrad/kokoro (optional)…" -ForegroundColor Cyan
    & python -m pip install -q "kokoro" "misaki[en]" 2>$null
}
if ($Backend -eq "onnx" -or $Backend -eq "auto") {
    Write-Host "Ensuring kokoro-onnx fallback…" -ForegroundColor Cyan
    & python -m pip install -q kokoro-onnx 2>$null
}

$args = @(
    $Script,
    "--host", $HostAddr,
    "--port", "$Port",
    "--backend", $Backend,
    "--speaker", $Speaker
)
if ($Download) { $args += "--download" }

Write-Host "Starting Kokoro TTS on http://${HostAddr}:$Port/v1/audio/speech" -ForegroundColor Green
& python @args
