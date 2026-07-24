# Qwen3-TTS on Intel Arc via Vulkan (experimental)
#
# Status 2026-07: Arc B580 runs fast (RTF ~0.3 with graph-opt off) but produces
# WRONG logits from sample step 0 vs CPU (even BF16). Audio is unintelligible.
# Upstream qwentts Vulkan CI is NVIDIA-only. Prefer Supertonic for voice chat
# until ggml-vulkan/Intel numerics are fixed or a SYCL build is available.
#
# Usage (when experimenting):
#   .\scripts\start_qwen_vulkan.ps1
#   # then s2s: --tts http --tts-url http://127.0.0.1:8083/v1/audio/speech --supertonic-voice aiden

$ErrorActionPreference = "Stop"
$Build = "C:\b\qwentts-build"
$Models = "C:\b\qwentts-models"
$VulkanSdk = "C:\VulkanSDK\1.4.350.0"

if (-not (Test-Path "$Build\tts-server.exe")) {
    throw "Missing $Build\tts-server.exe — build qwentts with -DGGML_VULKAN=ON first"
}

$env:Path = "$Build;$VulkanSdk\Bin;$env:Path"
$env:GGML_BACKEND = "Vulkan0"
# Best-known Arc flags (speed). Quality still wrong on B580 as of testing.
$env:GGML_VK_DISABLE_GRAPH_OPTIMIZE = "1"
$env:GGML_VK_DISABLE_COOPMAT = "1"
$env:GGML_VK_DISABLE_COOPMAT2 = "1"

$model = Join-Path $Models "qwen-talker-0.6b-customvoice-Q4_K_M.gguf"
$codec = Join-Path $Models "qwen-tokenizer-12hz-Q8_0.gguf"

Write-Host "Starting Qwen tts-server on Vulkan0 (experimental quality)..."
& "$Build\tts-server.exe" `
    --model $model `
    --codec $codec `
    --alias qwen3-tts-vulkan `
    --host 127.0.0.1 `
    --port 8083 `
    --lang auto `
    --clamp-fp16
