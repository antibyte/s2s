# Creates the isolated Python runtime used by the allowlisted Chatterbox profile.
# Model weights are intentionally not downloaded here; use the Lab catalog so
# revision, size, and SHA-256 verification remain centralized.

[CmdletBinding()]
param(
    [string]$Python = "python",
    [string]$VenvDir = "",
    [switch]$ForceRecreate
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
if (-not $VenvDir) {
    $VenvDir = Join-Path $root "tools\chatterbox\.venv"
}
$venv = [IO.Path]::GetFullPath($VenvDir)
$venvRoot = [IO.Path]::GetFullPath((Join-Path $root "tools\chatterbox"))
if (
    $venv.Equals($venvRoot, [StringComparison]::OrdinalIgnoreCase) -or
    -not $venv.StartsWith(
        $venvRoot.TrimEnd("\", "/") + [IO.Path]::DirectorySeparatorChar,
        [StringComparison]::OrdinalIgnoreCase
    )
) {
    throw "VenvDir must be a child directory of $venvRoot"
}

$pythonCommand = Get-Command $Python -ErrorAction Stop
$version = & $pythonCommand.Source -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
if ($LASTEXITCODE -ne 0) {
    throw "unable to inspect Python at $($pythonCommand.Source)"
}
$parts = @($version.Trim().Split(".") | ForEach-Object { [int]$_ })
if ($parts.Count -ne 2 -or $parts[0] -ne 3 -or $parts[1] -lt 10 -or $parts[1] -gt 13) {
    throw "Chatterbox requires Python 3.10-3.13; found $version"
}
if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    throw "Git is required because chatterbox-tts installs ResembleAI Perth from GitHub"
}

$venvPython = Join-Path $venv "Scripts\python.exe"
if ($ForceRecreate -and (Test-Path -LiteralPath $venv)) {
    $resolved = [IO.Path]::GetFullPath($venv)
    if (
        $resolved.StartsWith(
            $venvRoot.TrimEnd("\", "/") + [IO.Path]::DirectorySeparatorChar,
            [StringComparison]::OrdinalIgnoreCase
        )
    ) {
        Remove-Item -LiteralPath $resolved -Recurse -Force
    } else {
        throw "refusing to remove VenvDir outside $venvRoot"
    }
}
if (-not (Test-Path -LiteralPath $venvPython -PathType Leaf)) {
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $venv) | Out-Null
    & $pythonCommand.Source -m venv $venv
    if ($LASTEXITCODE -ne 0) {
        throw "failed to create Chatterbox virtual environment"
    }
}

$env:PIP_DISABLE_PIP_VERSION_CHECK = "1"
& $venvPython -m pip install --upgrade "pip>=25.1,<26"
if ($LASTEXITCODE -ne 0) {
    throw "failed to update pip"
}
& $venvPython -m pip install "setuptools==80.9.0"
if ($LASTEXITCODE -ne 0) {
    throw "failed to install the Perth-compatible setuptools runtime"
}
$chatterboxCommit = "5de7a54aa4e5e2baadb0182dde554908b48b85c2"
$chatterboxSource = "chatterbox-tts @ git+https://github.com/resemble-ai/chatterbox.git@$chatterboxCommit"
& $venvPython -m pip install --upgrade --force-reinstall --no-deps $chatterboxSource
if ($LASTEXITCODE -ne 0) {
    throw "failed to install Chatterbox from commit $chatterboxCommit"
}
& $venvPython -c @"
import chatterbox
import torch
from chatterbox.mtl_tts import ChatterboxMultilingualTTS
print(f"chatterbox import ok; torch={torch.__version__}; cuda={torch.cuda.is_available()}")
"@
if ($LASTEXITCODE -ne 0) {
    throw "Chatterbox import smoke test failed"
}

Write-Host ""
Write-Host "Chatterbox runtime ready: $venvPython"
Write-Host "Next: download 'Chatterbox Multilingual V3' in the Lab UI, then restart host_idle_agent.ps1."
