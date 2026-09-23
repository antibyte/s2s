param(
    [ValidateSet("base", "linux-gpu", "nvidia", "intel-sycl", "windows")]
    [string]$Platform = "base",
    [switch]$Build,
    [switch]$NoWeb
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

$composeFiles = @("-f", "docker-compose.yml")
switch ($Platform) {
    "linux-gpu" {
        $composeFiles += @("-f", "docker/docker-compose.linux-gpu.yml")
    }
    "intel-sycl" {
        $composeFiles += @("-f", "docker/docker-compose.intel-sycl.yml")
    }
    "nvidia" {
        $composeFiles += @("-f", "docker/docker-compose.nvidia.yml")
    }
    "windows" {
        $composeFiles += @("-f", "docker/docker-compose.windows.yml")
    }
}

$profiles = @(
    "--profile", "backends",
    "--profile", "managed",
    "--profile", "fallback",
    "--profile", "web"
)
if ($Platform -eq "intel-sycl") {
    $profiles += @("--profile", "tts")
}
$compose = @("compose") + $composeFiles + $profiles

if ($Build) {
    & docker @compose build
    if ($LASTEXITCODE -ne 0) {
        throw "Docker image build failed."
    }
}

# The controller is intentionally limited to starting and stopping pre-created,
# registry-labelled containers. Browser input can therefore never create an
# arbitrary image, command, mount or device mapping.
& docker @compose create
if ($LASTEXITCODE -ne 0) {
    throw "Could not create the managed lab containers. Build or pull the images first."
}

$optionalContainers = @(
    "s2s-whisper-tiny",
    "s2s-whisper-base",
    "s2s-whisper-small",
    "s2s-confucius-cpu",
    "s2s-confucius-cuda",
    "s2s-confucius-vulkan",
    "s2s-parakeet-cpu",
    "s2s-parakeet-cuda",
    "s2s-parakeet-xpu",
    "s2s-tts-qwen-vulkan",
    "s2s-tts-qwen-sycl-aot",
    "s2s-tts-qwen-sycl-jit",
    "s2s-tts-kokoro"
)
foreach ($container in $optionalContainers) {
    & docker container inspect $container 2>$null | Out-Null
    if ($LASTEXITCODE -eq 0) {
        & docker stop --time 10 $container 2>$null | Out-Null
    }
}

$initialServices = @("docker-proxy", "model-init", "supertonic", "confucius", "llama", "s2s")
if (-not $NoWeb) {
    $initialServices += "web"
}
& docker @compose up -d @initialServices
if ($LASTEXITCODE -ne 0) {
    throw "Could not start the initial Speech Lab stack."
}

Write-Host "Speech Lab ready: ASR=confucius4-r2t2, LLM=Granite 3.3, TTS=supertonic"
if (-not $NoWeb) {
    $webPort = if ($env:WEB_PORT) { $env:WEB_PORT } else { "8088" }
    Write-Host "Open http://127.0.0.1:$webPort"
}
