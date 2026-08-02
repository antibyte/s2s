# Creates the isolated CPython 3.11 runtime used by the allowlisted XTTS-v2
# profiles. Large ONNX artifacts remain managed by the Lab catalog.

[CmdletBinding()]
param(
    [string]$Python = "py",
    [string[]]$PythonArgs = @("-3.11"),
    [string]$VenvDir = "",
    [string]$ModelsDir = "",
    [switch]$ForceRecreate
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$toolsRoot = [IO.Path]::GetFullPath((Join-Path $root "tools\xtts-v2"))
if (-not $VenvDir) {
    $VenvDir = Join-Path $toolsRoot ".venv"
}
if (-not $ModelsDir) {
    $ModelsDir = if ($env:S2S_MODELS_HOST_DIR) {
        $env:S2S_MODELS_HOST_DIR
    } else {
        Join-Path $root "models"
    }
}
$venv = [IO.Path]::GetFullPath($VenvDir)
$models = [IO.Path]::GetFullPath($ModelsDir)
$venvPrefix = $toolsRoot.TrimEnd("\", "/") + [IO.Path]::DirectorySeparatorChar
if (
    $venv.Equals($toolsRoot, [StringComparison]::OrdinalIgnoreCase) -or
    -not $venv.StartsWith($venvPrefix, [StringComparison]::OrdinalIgnoreCase)
) {
    throw "VenvDir must be a child directory of $toolsRoot"
}

$pythonCommand = Get-Command $Python -ErrorAction Stop
$version = & $pythonCommand.Source @PythonArgs -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
if ($LASTEXITCODE -ne 0) {
    throw "unable to inspect Python at $($pythonCommand.Source)"
}
if ($version.Trim() -ne "3.11") {
    throw "XTTS-v2 WebGPU requires CPython 3.11 for the hash-locked Windows wheels; found $version"
}

$venvPython = Join-Path $venv "Scripts\python.exe"
if ($ForceRecreate -and (Test-Path -LiteralPath $venv)) {
    $resolved = [IO.Path]::GetFullPath($venv)
    if (-not $resolved.StartsWith($venvPrefix, [StringComparison]::OrdinalIgnoreCase)) {
        throw "refusing to remove VenvDir outside $toolsRoot"
    }
    Remove-Item -LiteralPath $resolved -Recurse -Force
}
if (-not (Test-Path -LiteralPath $venvPython -PathType Leaf)) {
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $venv) | Out-Null
    & $pythonCommand.Source @PythonArgs -m venv $venv
    if ($LASTEXITCODE -ne 0) {
        throw "failed to create XTTS-v2 virtual environment"
    }
}

$requirements = Join-Path $PSScriptRoot "requirements-xtts-v2-win-cp311.txt"
if (-not (Test-Path -LiteralPath $requirements -PathType Leaf)) {
    throw "hash-locked XTTS-v2 requirements are missing: $requirements"
}
$env:PIP_DISABLE_PIP_VERSION_CHECK = "1"
& $venvPython -m pip install --require-hashes --no-build-isolation -r $requirements
if ($LASTEXITCODE -ne 0) {
    throw "failed to install the hash-locked XTTS-v2 runtime"
}

$token = if ($env:S2S_HF_TOKEN) { $env:S2S_HF_TOKEN } else { $env:HF_TOKEN }
if ([string]::IsNullOrWhiteSpace($token)) {
    throw "XTTS-v2 runtime sources require HF_TOKEN. Accept access at https://huggingface.co/pltobing/XTTSv2-Streaming-ONNX and retry."
}

$upstream = Join-Path $toolsRoot "upstream"
$onnxDir = Join-Path $models "xtts-v2\onnx"
New-Item -ItemType Directory -Force -Path $upstream, $onnxDir | Out-Null
$env:S2S_XTTS_UPSTREAM_DIR = $upstream
$env:S2S_XTTS_ONNX_DIR = $onnxDir
$env:S2S_XTTS_REVISION = "975b202585dea4ae6ca7f6118121cdf1011d7d28"

& $venvPython -c @'
from __future__ import annotations

import hashlib
import os
import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download

repo = "pltobing/XTTSv2-Streaming-ONNX"
revision = os.environ["S2S_XTTS_REVISION"]
token = os.environ.get("S2S_HF_TOKEN") or os.environ.get("HF_TOKEN")
upstream = Path(os.environ["S2S_XTTS_UPSTREAM_DIR"])
onnx_dir = Path(os.environ["S2S_XTTS_ONNX_DIR"])
manifest = {
    "xtts_onnx_orchestrator.py": (41695, "19e27d236a8b9fbff0d0ba65e675e24a146843b0"),
    "xtts_streaming_pipeline.py": (29385, "64b2b66fdbb2ee0d2f1b733d5a4ae27745126052"),
    "xtts_tokenizer.py": (30800, "c6b1bddda2c49a2f8bb362e1d340a40b34e3fad5"),
    "zh_num2words.py": (59359, "035043b73e5f5911716a2fec431e8be51fda30e6"),
    "xtts_onnx/metadata.json": (645, "d8fd28fb6ad8c6748993288839bc4a3d33d53ef7"),
}

for name, (expected_size, expected_blob) in manifest.items():
    downloaded = Path(
        hf_hub_download(
            repo_id=repo,
            filename=name,
            revision=revision,
            token=token,
            local_dir=upstream,
        )
    )
    data = downloaded.read_bytes()
    blob = hashlib.sha1(
        f"blob {len(data)}\0".encode("ascii") + data,
        usedforsecurity=False,
    ).hexdigest()
    if len(data) != expected_size or blob != expected_blob:
        raise SystemExit(
            f"pinned source verification failed for {name}: "
            f"size={len(data)} blob={blob}"
        )

metadata = upstream / "xtts_onnx" / "metadata.json"
shutil.copy2(metadata, onnx_dir / "metadata.json")
'@
if ($LASTEXITCODE -ne 0) {
    throw "failed to fetch or verify the pinned gated XTTS-v2 runtime sources"
}

& $venvPython -c @'
import onnxruntime as ort
import onnxruntime_ep_webgpu as webgpu_ep
import numpy
import soundfile
from importlib.metadata import version

assert ort.__version__ == "1.28.0"
assert version("onnxruntime-ep-webgpu") == "0.1.0"
print(
    f"XTTS imports ok; onnxruntime={ort.__version__}; "
    f"webgpu_ep={version('onnxruntime-ep-webgpu')}; numpy={numpy.__version__}"
)
'@
if ($LASTEXITCODE -ne 0) {
    throw "XTTS-v2 import smoke test failed"
}

Write-Host ""
Write-Host "XTTS-v2 runtime ready: $venvPython"
Write-Host "Runtime sources: $upstream"
Write-Host "Next: download XTTS-v2 in the Lab UI, then restart host_idle_agent.ps1."
