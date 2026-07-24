$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
$source = Join-Path $repoRoot "target/qwentts-pinned"
$build = Join-Path $repoRoot "target/qwentts-sycl-pinned-native"

if (-not (Test-Path -LiteralPath $source)) {
    & (Join-Path $PSScriptRoot "prepare_qwentts_source.ps1") -Destination $source
}

$env:S2S_QWENTTS_SOURCE = $source
$env:S2S_QWENTTS_BUILD_DIR = $build
& cmd.exe /d /c (Join-Path $PSScriptRoot "build_qwentts_native.cmd")
if ($LASTEXITCODE -ne 0) {
    throw "Pinned qwentts SYCL build failed."
}

& cmd.exe /d /c "call D:\Intel\oneAPI\setvars.bat --force >nul && `"$build\test-abi-c.exe`""
if ($LASTEXITCODE -ne 0) {
    throw "Pinned qwentts ABI test failed."
}

Write-Host "Pinned native qwentts build ready: $build"
