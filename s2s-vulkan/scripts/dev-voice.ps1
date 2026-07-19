# Start local voice stack for s2s-vulkan web lab (network-ready HTTPS UI).
# Usage (from repo s2s-vulkan/):
#   powershell -ExecutionPolicy Bypass -File scripts\dev-voice.ps1

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

$env:RUST_LOG = "info"

function Ensure-PortFree([int]$Port) {
  $conns = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
  foreach ($c in $conns) {
    Write-Host "Stopping PID $($c.OwningProcess) on port $Port"
    Stop-Process -Id $c.OwningProcess -Force -ErrorAction SilentlyContinue
  }
}

Write-Host "==> Whisper STT :8082"
Ensure-PortFree 8082
Start-Process -FilePath "python" -ArgumentList @(
  "$Root\scripts\whisper_stt_server.py",
  "--host", "127.0.0.1", "--port", "8082",
  "--model", "base", "--device", "cpu", "--compute-type", "int8"
) -WorkingDirectory $Root -WindowStyle Minimized

Write-Host "==> s2s-vulkan :8765 (0.0.0.0)"
Ensure-PortFree 8765
$bin = Join-Path $Root "target\release\s2s-vulkan.exe"
if (-not (Test-Path $bin)) { cargo build --release }
Start-Process -FilePath $bin -ArgumentList @(
  "--mode", "websocket",
  "--host", "0.0.0.0", "--port", "8765",
  "--whisper-url", "http://127.0.0.1:8082",
  "--llm-base-url", "http://127.0.0.1:11434/v1",
  "--model-name", "llama3.2:1b",
  "--tts", "auto",
  "--supertonic-model-dir", "$Root\models\supertonic\onnx",
  "--supertonic-voice", "M1",
  "--tts-sample-rate", "16000",
  "--min-speech-ms", "250",
  "--min-silence-ms", "350",
  "--thresh", "0.45",
  "--skip-health"
) -WorkingDirectory $Root -WindowStyle Minimized

Write-Host "==> HTTPS web lab :9999"
Ensure-PortFree 9999
Start-Process -FilePath "python" -ArgumentList @(
  "$Root\web\serve.py",
  "--host", "0.0.0.0", "--port", "9999",
  "--backend", "127.0.0.1:8765"
) -WorkingDirectory (Join-Path $Root "web") -WindowStyle Minimized

Write-Host ""
Write-Host "Open (accept cert warning once):"
Write-Host "  https://127.0.0.1:9999"
Write-Host "  https://<LAN-IP>:9999"
Write-Host ""
Write-Host "Optional LLM: ollama pull llama3.2:1b"
Write-Host "Without Ollama model, STT still works and replies via echo+Windows TTS."
