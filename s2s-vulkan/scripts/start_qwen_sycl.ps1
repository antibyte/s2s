# Start Qwen3-TTS on Intel Arc via oneAPI SYCL (correct numerics on B580).
# Prereq: oneAPI, build at D:\b\qwentts-sycl, models under C:\b\qwentts-models
#
#   .\scripts\start_qwen_sycl.ps1

$ErrorActionPreference = "Stop"

$OneApi = if ($env:S2S_ONEAPI_ROOT) { $env:S2S_ONEAPI_ROOT } elseif ($env:ONEAPI_ROOT) { $env:ONEAPI_ROOT } else { "D:\Intel\oneAPI" }
$Build = if ($env:S2S_QWENTTS_SYCL) { $env:S2S_QWENTTS_SYCL } else { "D:\b\qwentts-sycl" }
$Models = if ($env:S2S_QWENTTS_MODELS) { $env:S2S_QWENTTS_MODELS } else { "C:\b\qwentts-models" }
$Port = if ($env:S2S_TTS_PORT) { [int]$env:S2S_TTS_PORT } else { 8083 }

if (-not (Test-Path "$OneApi\setvars.bat")) { throw "oneAPI missing: $OneApi" }
if (-not (Test-Path "$Build\tts-server.exe")) {
    throw "tts-server missing under $Build - rebuild SYCL first"
}

$model = Join-Path $Models "qwen-talker-0.6b-customvoice-Q4_K_M.gguf"
$codec = Join-Path $Models "qwen-tokenizer-12hz-Q8_0.gguf"
if (-not (Test-Path $model) -or -not (Test-Path $codec)) {
    throw "Models missing in $Models"
}

# Launch via cmd so setvars applies to the server process tree.
$inner = @"
set VS2022INSTALLDIR=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools
call `"$OneApi\setvars.bat`" intel64 vs2022
set PATH=$Build;%PATH%
set GGML_BACKEND=SYCL0
set ONEAPI_DEVICE_SELECTOR=level_zero:0
set GGML_SYCL_ENABLE_FLASH_ATTN=1
set GGML_SYCL_DISABLE_GRAPH=1
set GGML_SYCL_PRIORITIZE_DMMV=1
set GGML_SYCL_DEV2DEV_MEMCPY=0
set GGML_SYCL_DISABLE_OPT=0
set GGML_SYCL_DISABLE_DNN=0
set GGML_SYCL_USE_LEVEL_ZERO_API=1
set QWEN_CODE_SAMPLER=host
set ZES_ENABLE_SYSMAN=1
cd /d $Build
REM Default german for DE voice stacks; per-request "language" JSON overrides when set.
tts-server.exe --model $model --codec $codec --alias qwen3-tts-sycl --host 127.0.0.1 --port $Port --lang german --clamp-fp16
"@
$bat = Join-Path $env:TEMP "run_qwen_sycl_server.cmd"
Set-Content -Path $bat -Value $inner -Encoding ASCII
Write-Host "Starting SYCL tts-server on :$Port (GGML_BACKEND=SYCL0, Arc L0)..."
cmd /c $bat
