# Install Intel oneAPI Base Toolkit to a custom drive (default: D:\Intel\oneAPI).
# Requires elevated PowerShell (Admin / UAC).
#
#   .\scripts\install_oneapi.ps1
#   .\scripts\install_oneapi.ps1 -InstallDir "D:\Intel\oneAPI"

[CmdletBinding()]
param(
    [string]$InstallDir = "D:\Intel\oneAPI",
    [string]$TempDir = "D:\Intel\oneAPI-tmp",
    [string]$LogDir = "D:\Intel\oneAPI-logs"
)

$ErrorActionPreference = "Continue"
$transcript = Join-Path $env:USERPROFILE "oneapi-install-transcript.txt"

function Test-Admin {
    $id = [Security.Principal.WindowsIdentity]::GetCurrent()
    $p = New-Object Security.Principal.WindowsPrincipal($id)
    return $p.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

# Always log to user profile first (works before D: setup / elevation)
function Log([string]$msg) {
    $line = "$(Get-Date -Format o)  $msg"
    Write-Host $line
    Add-Content -Path $transcript -Value $line -ErrorAction SilentlyContinue
}

Log "=== install_oneapi.ps1 start ==="
Log "InstallDir=$InstallDir TempDir=$TempDir LogDir=$LogDir Admin=$(Test-Admin)"

if (-not (Test-Admin)) {
    Log "Re-launching elevated..."
    $script = $MyInvocation.MyCommand.Path
    $arg = "-NoProfile -ExecutionPolicy Bypass -File `"$script`" -InstallDir `"$InstallDir`" -TempDir `"$TempDir`" -LogDir `"$LogDir`""
    $p = Start-Process powershell -Verb RunAs -ArgumentList $arg -Wait -PassThru
    Log "Elevated child exit=$($p.ExitCode)"
    exit $p.ExitCode
}

try {
    New-Item -ItemType Directory -Force -Path $InstallDir, $TempDir, $LogDir | Out-Null
    Log "Created dirs on target drive"

    $root = [System.IO.Path]::GetPathRoot($InstallDir)  # e.g. D:\
    $driveName = $root.TrimEnd('\').TrimEnd(':')       # D
    $drv = Get-PSDrive -Name $driveName -ErrorAction Stop
    $freeGB = [math]::Round($drv.Free / 1GB, 1)
    Log "Drive ${driveName}: free=${freeGB} GB"
    if ($drv.Free -lt 40GB) {
        throw "Need ~40+ GB free on ${driveName}: (have ${freeGB} GB)"
    }

    # Keep extract off C:
    $env:TEMP = $TempDir
    $env:TMP = $TempDir
    Log "TEMP/TMP=$TempDir"

    $exe = Get-ChildItem "$env:LOCALAPPDATA\Temp\WinGet" -Recurse -Filter "*offline.exe" -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -match "oneapi|base-toolkit" } |
        Sort-Object LastWriteTime -Descending |
        Select-Object -First 1

    if (-not $exe) {
        $exe = Get-ChildItem "$env:LOCALAPPDATA\Temp\WinGet" -Recurse -Filter "*offline.exe" -ErrorAction SilentlyContinue |
            Sort-Object LastWriteTime -Descending |
            Select-Object -First 1
    }

    if (-not $exe) {
        Log "No cached offline kit — winget download (needs ~3GB free on C: for download only)..."
        $cFree = [math]::Round((Get-PSDrive C).Free / 1GB, 1)
        Log "C: free=${cFree} GB"
        $wgLog = Join-Path $LogDir "winget.txt"
        winget install --id Intel.OneAPI.BaseToolkit -e `
            --accept-package-agreements --accept-source-agreements --disable-interactivity `
            *>&1 | Tee-Object -FilePath $wgLog
        Log "winget exit=$LASTEXITCODE"
        $exe = Get-ChildItem "$env:LOCALAPPDATA\Temp\WinGet" -Recurse -Filter "*offline.exe" -ErrorAction SilentlyContinue |
            Sort-Object LastWriteTime -Descending |
            Select-Object -First 1
    }

    if (-not $exe) {
        throw "Offline installer EXE not found after winget. Download Base Toolkit offline to D:\ and re-run."
    }

    # Copy installer to D: so C: is not required during long install
    $localExe = Join-Path $TempDir $exe.Name
    if (-not (Test-Path $localExe) -or (Get-Item $localExe).Length -ne $exe.Length) {
        Log "Copying installer to $localExe ($([math]::Round($exe.Length/1GB,2)) GB)..."
        Copy-Item -Force $exe.FullName $localExe
    }
    Log "Installer: $localExe"

    # Intel silent custom directory
    $argList = @(
        "-s", "--eula", "accept",
        "-a", "--silent", "--eula", "accept",
        "--install-dir=$InstallDir",
        "-p=NEED_VS2019_INTEGRATION=0",
        "-p=NEED_VS2022_INTEGRATION=0",
        "-p=NEED_VS2026_INTEGRATION=0",
        "--log-dir=$LogDir"
    )
    Log "Args: $($argList -join ' ')"
    Log "Starting installer (20–60+ min)..."

    $proc = Start-Process -FilePath $localExe -ArgumentList $argList -Wait -PassThru -NoNewWindow
    Log "Installer ExitCode=$($proc.ExitCode)"

    $setvars = Join-Path $InstallDir "setvars.bat"
    if ($proc.ExitCode -ne 0 -or -not (Test-Path $setvars)) {
        Get-ChildItem $LogDir -Filter "*.log" -ErrorAction SilentlyContinue |
            Sort-Object LastWriteTime -Descending |
            Select-Object -First 8 |
            ForEach-Object { Log "log: $($_.Name) ($($_.Length) bytes)" }
        # dump last errors
        $latest = Get-ChildItem $LogDir -Filter "*.log" -ErrorAction SilentlyContinue |
            Sort-Object LastWriteTime -Descending | Select-Object -First 1
        if ($latest) {
            Select-String -Path $latest.FullName -Pattern "Not enough|Failed|ERROR|error|abort" |
                Select-Object -Last 15 |
                ForEach-Object { Log "  $($_.Line.Substring(0,[Math]::Min(200,$_.Line.Length)))" }
        }
        throw "Install failed or setvars.bat missing (exit=$($proc.ExitCode))"
    }

    try {
        [Environment]::SetEnvironmentVariable("ONEAPI_ROOT", $InstallDir, "Machine")
    } catch {
        Log "Machine ONEAPI_ROOT not set: $_"
    }
    [Environment]::SetEnvironmentVariable("S2S_ONEAPI_ROOT", $InstallDir, "User")
    [Environment]::SetEnvironmentVariable("ONEAPI_ROOT", $InstallDir, "User")

    Log "SUCCESS: $setvars"
    Log "Next: cmd /c `"`"$InstallDir\setvars.bat`" intel64 vs2022 && ... build_qwentts_sycl.ps1`""
    exit 0
}
catch {
    Log "FATAL: $_"
    exit 1
}
