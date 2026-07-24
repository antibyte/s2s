param(
    [string]$Destination = "",
    [string]$QwenRevision = "82cd05b9f3a175612dc89fd6943e610fab096ef5",
    [string]$GgmlRevision = "c044c6f03892f9d5e98213b05f8afea1f8b0d3c9"
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
if (-not $Destination) {
    $Destination = Join-Path $repoRoot "target/qwentts-pinned"
}
$Destination = [System.IO.Path]::GetFullPath($Destination)
if (Test-Path -LiteralPath $Destination) {
    throw "Destination already exists; refusing to overwrite: $Destination"
}

$qwenPatch = Join-Path $repoRoot "patches/qwentts/0101-qwentts-rebased.patch"
$ggmlPatch = Join-Path $repoRoot "patches/qwentts/0102-ggml-rebased.patch"

& git clone --filter=blob:none https://github.com/ServeurpersoCom/qwentts.cpp.git $Destination
if ($LASTEXITCODE -ne 0) { throw "qwentts clone failed" }
& git -C $Destination checkout --detach $QwenRevision
if ($LASTEXITCODE -ne 0) { throw "qwentts revision checkout failed" }
& git -C $Destination submodule update --init ggml
if ($LASTEXITCODE -ne 0) { throw "ggml submodule initialization failed" }
& git -C (Join-Path $Destination "ggml") checkout --detach $GgmlRevision
if ($LASTEXITCODE -ne 0) { throw "ggml revision checkout failed" }

& git -C $Destination apply --check $qwenPatch
if ($LASTEXITCODE -ne 0) { throw "qwentts patch check failed" }
& git -C $Destination apply $qwenPatch
if ($LASTEXITCODE -ne 0) { throw "qwentts patch failed" }
& git -C (Join-Path $Destination "ggml") apply --check $ggmlPatch
if ($LASTEXITCODE -ne 0) { throw "ggml patch check failed" }
& git -C (Join-Path $Destination "ggml") apply $ggmlPatch
if ($LASTEXITCODE -ne 0) { throw "ggml patch failed" }

Write-Host "Prepared qwentts source: $Destination"
Write-Host "Build with:"
Write-Host "`$env:S2S_QWENTTS_SOURCE='$Destination'; scripts\build_qwentts_native.cmd"
