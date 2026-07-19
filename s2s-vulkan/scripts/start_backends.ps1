# Start Vulkan-capable GGML backends for s2s-vulkan (Windows).
# Adjust paths to your builds/models.

param(
    [string]$WhisperBin = "whisper-server",
    [string]$WhisperModel = "models\ggml-small.bin",
    [int]$WhisperPort = 8082,

    [string]$LlamaBin = "llama-server",
    [string]$LlamaModel = "models\model.gguf",
    [int]$LlamaPort = 8081,

    [string]$TtsBin = "",
    [int]$TtsPort = 8083,

    [string]$GgmlBackend = "Vulkan0"
)

$ErrorActionPreference = "Stop"
$env:GGML_BACKEND = $GgmlBackend

Write-Host "GGML_BACKEND=$env:GGML_BACKEND" -ForegroundColor Cyan

# --- whisper.cpp (STT) ---
if (Get-Command $WhisperBin -ErrorAction SilentlyContinue) {
    Write-Host "Starting whisper-server on :$WhisperPort ..." -ForegroundColor Green
    Start-Process -FilePath $WhisperBin -ArgumentList @(
        "-m", $WhisperModel,
        "--host", "127.0.0.1",
        "--port", "$WhisperPort",
        "--language", "auto",
        "--no-timestamps"
    ) -WindowStyle Minimized
} else {
    Write-Warning "whisper-server not found ($WhisperBin). Build whisper.cpp with -DGGML_VULKAN=1"
}

# --- llama.cpp (LLM) ---
if (Get-Command $LlamaBin -ErrorAction SilentlyContinue) {
    Write-Host "Starting llama-server on :$LlamaPort ..." -ForegroundColor Green
    Start-Process -FilePath $LlamaBin -ArgumentList @(
        "-m", $LlamaModel,
        "-ngl", "999",
        "-c", "8192",
        "--host", "127.0.0.1",
        "--port", "$LlamaPort"
    ) -WindowStyle Minimized
} else {
    Write-Warning "llama-server not found ($LlamaBin). Build llama.cpp with -DGGML_VULKAN=ON"
}

# --- optional TTS HTTP wrapper ---
if ($TtsBin -and (Test-Path $TtsBin)) {
    Write-Host "Starting TTS server on :$TtsPort ..." -ForegroundColor Green
    Start-Process -FilePath $TtsBin -ArgumentList @("--port", "$TtsPort") -WindowStyle Minimized
} else {
    Write-Host "No TTS binary configured — use: s2s-vulkan --tts system  (or provide HTTP TTS)" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "Then run:" -ForegroundColor Cyan
Write-Host "  cargo run --release -- --mode local --tts system"
Write-Host "  cargo run --release -- --mode local --tts http --tts_url http://127.0.0.1:$TtsPort/v1/audio/speech"
