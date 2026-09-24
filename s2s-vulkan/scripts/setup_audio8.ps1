# Prepare Audio8 TTS Preview weights for the s2s Lab host agent.
#
# - Creates tools/audio8/.venv (Python 3.11+ recommended)
# - Downloads a pinned HF snapshot into models/audio8-tts-preview-<size>
# - Installs torch + transformers + soundfile
#
# Usage:
#   .\scripts\setup_audio8.ps1
#   .\scripts\setup_audio8.ps1 -Cuda
#   .\scripts\setup_audio8.ps1 -Model 0.1b
#   .\scripts\setup_audio8.ps1 -Model both -Cuda

[CmdletBinding()]
param(
    [switch]$Cuda,
    [ValidateSet("0.6b", "0.1b", "both")]
    [string]$Model = "0.6b",
    [string]$ModelsDir = "",
    [string]$Python = "python"
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$VenvDir = Join-Path $Root "tools\audio8\.venv"

$Catalog = @{
    "0.6b" = @{
        Repo = "Audio8/Audio8-TTS-Preview-0.6b"
        Revision = "f9612f13a0ab40facf3d050fc908b9e6db05c2be"
        RelDir = "audio8-tts-preview-0.6b"
        ExpectedWeightSize = 1202342528
        Alias = "audio8-tts-preview-0.6b"
    }
    "0.1b" = @{
        Repo = "Audio8/Audio8-TTS-Preview-0.1b"
        Revision = "7a644014c398a0495d5efd1da7461bfeb4dbddcd"
        RelDir = "audio8-tts-preview-0.1b"
        ExpectedWeightSize = 339605208
        Alias = "audio8-tts-preview-0.1b"
    }
}

function Install-Audio8Snapshot {
    param(
        [hashtable]$Spec,
        [string]$TargetDir
    )
    New-Item -ItemType Directory -Force -Path $TargetDir | Out-Null
    New-Item -ItemType Directory -Force -Path (Join-Path $TargetDir "voices") | Out-Null

    $repo = $Spec.Repo
    $revision = $Spec.Revision
    Write-Host "Downloading $($Spec.Alias) ($revision) -> $TargetDir"
    $downloadPy = @"
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='$repo',
    revision='$revision',
    local_dir=r'$TargetDir',
    allow_patterns=[
        'model.safetensors',
        'codec.pth',
        'config.json',
        'configuration_arktts.py',
        'modeling_arktts.py',
        'modeling_arktts_codec.py',
        'processing_arktts.py',
        'preprocessor_config.json',
        'processor_config.json',
        'tokenizer.json',
        'tokenizer_config.json',
        'special_tokens_map.json',
        'generation_config.json',
    ],
)
print('download complete')
"@
    & $Python -c $downloadPy
    if ($LASTEXITCODE -ne 0) {
        throw "huggingface download failed (pip install huggingface_hub if needed)"
    }

    $weights = Join-Path $TargetDir "model.safetensors"
    if (-not (Test-Path -LiteralPath $weights)) {
        throw "model.safetensors missing after download"
    }
    $size = (Get-Item -LiteralPath $weights).Length
    if ($size -ne $Spec.ExpectedWeightSize) {
        throw "model.safetensors size $size != expected $($Spec.ExpectedWeightSize)"
    }
    foreach ($name in @(
        "codec.pth",
        "config.json",
        "configuration_arktts.py",
        "modeling_arktts.py",
        "modeling_arktts_codec.py",
        "processing_arktts.py",
        "tokenizer.json"
    )) {
        $path = Join-Path $TargetDir $name
        if (-not (Test-Path -LiteralPath $path)) {
            throw "missing required artifact: $name"
        }
    }
    Write-Host "weights verified ($($Spec.ExpectedWeightSize) bytes)"
}

$selected = if ($Model -eq "both") { @("0.6b", "0.1b") } else { @($Model) }
if ($ModelsDir -and $selected.Count -gt 1) {
    throw "-ModelsDir cannot be combined with -Model both"
}

New-Item -ItemType Directory -Force -Path (Split-Path -Parent $VenvDir) | Out-Null
$installed = @()
foreach ($key in $selected) {
    $spec = $Catalog[$key]
    $target = if ($ModelsDir) {
        $ModelsDir
    } else {
        Join-Path $Root "models\$($spec.RelDir)"
    }
    Install-Audio8Snapshot -Spec $spec -TargetDir $target
    $installed += [pscustomobject]@{ Spec = $spec; Dir = $target }
}

if (-not (Test-Path -LiteralPath (Join-Path $VenvDir "Scripts\python.exe"))) {
    Write-Host "Creating venv at $VenvDir"
    & $Python -m venv $VenvDir
}
$VenvPython = Join-Path $VenvDir "Scripts\python.exe"
& $VenvPython -m pip install --upgrade pip
if ($Cuda) {
    & $VenvPython -m pip install "torch>=2.5.0" "torchaudio>=2.5.0" --index-url https://download.pytorch.org/whl/cu124
} else {
    & $VenvPython -m pip install "torch>=2.5.0" "torchaudio>=2.5.0" --index-url https://download.pytorch.org/whl/cpu
}
& $VenvPython -m pip install "transformers>=4.57.0,<5" "soundfile>=0.12" "safetensors>=0.4" "numpy" "huggingface_hub"

Write-Host ""
Write-Host "Audio8 TTS Preview ready."
Write-Host "  Python: $VenvPython"
foreach ($item in $installed) {
    $spec = $item.Spec
    $dir = $item.Dir
    Write-Host "  Model:  $($spec.Alias) -> $dir"
    Write-Host "  Smoke:  & '$VenvPython' '$Root\scripts\tts_audio8_server.py' --model-dir '$dir' --model-id '$($spec.Repo)' --model-alias '$($spec.Alias)' --revision '$($spec.Revision)' --port 8096"
    Write-Host "  Optional clone voices: place <name>.wav + matching <name>.txt under $dir\voices"
    Write-Host "  Then select $($spec.Alias) in the lab UI."
}
