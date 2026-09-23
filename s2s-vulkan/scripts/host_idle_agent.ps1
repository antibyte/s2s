# Windows host supervisor for native Vulkan and SYCL inference backends.
#
# The Linux Docker Desktop VM cannot use an Intel Arc Vulkan device directly.
# This agent watches the shared data directory, accepts only schema-validated
# allowlisted profiles, and owns every native process it starts.
#
# Usage:
#   .\scripts\host_idle_agent.ps1
#   .\scripts\host_idle_agent.ps1 -DataDir D:\s2s\data -ModelsDir D:\s2s\models
#
# Executable overrides:
#   S2S_HOST_CRISPASR_EXE
#   S2S_HOST_LLAMA_EXE
#   S2S_HOST_QWEN_SYCL_EXE
#   S2S_HOST_SUPERTONIC_EXE
#   S2S_HOST_CHATTERBOX_PYTHON
#   S2S_HOST_XTTS_PYTHON
#   S2S_HOST_INFLECT_PYTHON
#   S2S_HOST_AUDIO8_PYTHON
#   S2S_ONEAPI_ROOT
#
# Compatibility: idle-unload.request and idle-reload.request are still handled.

[CmdletBinding()]
param(
    [string]$DataDir = "",
    [string]$ModelsDir = "",
    [string]$Volume = "",
    [ValidateRange(1, 60)]
    [int]$PollSeconds = 2,
    [ValidateRange(-1, 31)]
    [int]$VulkanDevice = -1,
    [switch]$Once
)

$ErrorActionPreference = "Stop"
$Script:SchemaVersion = 2
$Script:AgentVersion = "2.1.0"
$Script:Root = Split-Path -Parent $PSScriptRoot
$Script:ResolvedVulkanDevice = if ($VulkanDevice -ge 0) { $VulkanDevice } else { 0 }
$Script:AllowedStages = @("asr", "tts", "llm")
$Script:AllowedDockerContainers = @(
    "s2s-tts-vibevoice",
    "s2s-tts-vibevoice-cuda",
    "s2s-parakeet-cpu",
    "s2s-llama-fallback",
    "s2s-whisper-tiny",
    "s2s-whisper-base",
    "s2s-whisper-small"
)

if (-not $DataDir) {
    $DataDir = if ($env:S2S_DATA_HOST_DIR) {
        $env:S2S_DATA_HOST_DIR
    } else {
        Join-Path $Script:Root "data"
    }
}
if (-not $ModelsDir) {
    $ModelsDir = if ($env:S2S_MODELS_HOST_DIR) {
        $env:S2S_MODELS_HOST_DIR
    } else {
        Join-Path $Script:Root "models"
    }
}

$Script:DataDir = [IO.Path]::GetFullPath($DataDir)
$Script:ModelsDir = [IO.Path]::GetFullPath($ModelsDir)
$Script:AgentDir = Join-Path $Script:DataDir "host-agent"
$Script:CommandDir = Join-Path $Script:AgentDir "commands"
$Script:ResultDir = Join-Path $Script:AgentDir "results"
$Script:LogDir = Join-Path $Script:AgentDir "logs"
$Script:StatusPath = Join-Path $Script:AgentDir "status.json"
$Script:StatePath = Join-Path $Script:AgentDir "owned-processes.json"
$Script:Owned = @{}
$Script:VulkanStatusCache = $null

foreach ($path in @(
    $Script:DataDir,
    $Script:ModelsDir,
    $Script:AgentDir,
    $Script:CommandDir,
    $Script:ResultDir,
    $Script:LogDir
)) {
    New-Item -ItemType Directory -Force -Path $path | Out-Null
}

function Get-ConfiguredExecutable {
    param(
        [Parameter(Mandatory = $true)][string]$EnvironmentName,
        [Parameter(Mandatory = $true)][string]$DefaultPath
    )
    $configured = [Environment]::GetEnvironmentVariable($EnvironmentName)
    if ([string]::IsNullOrWhiteSpace($configured)) {
        $configured = $DefaultPath
    }
    return [IO.Path]::GetFullPath($configured)
}

function Get-OneApiEnvironment {
    $oneApiRoot = if ($env:S2S_ONEAPI_ROOT) {
        $env:S2S_ONEAPI_ROOT
    } elseif ($env:ONEAPI_ROOT) {
        $env:ONEAPI_ROOT
    } else {
        @(
            "D:\Intel\oneAPI",
            "E:\Intel\oneAPI",
            "C:\Program Files (x86)\Intel\oneAPI",
            "C:\Program Files\Intel\oneAPI"
        ) | Where-Object { Test-Path -LiteralPath (Join-Path $_ "setvars.bat") } |
            Select-Object -First 1
    }
    if ([string]::IsNullOrWhiteSpace($oneApiRoot)) {
        throw "oneAPI runtime not found; set S2S_ONEAPI_ROOT"
    }
    $setvars = [IO.Path]::GetFullPath((Join-Path $oneApiRoot "setvars.bat"))
    if (-not (Test-Path -LiteralPath $setvars -PathType Leaf)) {
        throw "oneAPI setvars.bat missing: $setvars"
    }

    $command = "call `"$setvars`" --force >nul && set"
    $lines = & $env:ComSpec /d /s /c $command
    if ($LASTEXITCODE -ne 0) {
        throw "oneAPI setvars.bat failed with exit code $LASTEXITCODE"
    }
    $environment = @{}
    foreach ($line in $lines) {
        $separator = $line.IndexOf("=")
        if ($separator -le 0) {
            continue
        }
        $name = $line.Substring(0, $separator)
        if ($name -notmatch '^[A-Za-z_][A-Za-z0-9_]*$') {
            continue
        }
        $environment[$name] = $line.Substring($separator + 1)
    }
    if (-not $environment.ContainsKey("Path")) {
        throw "oneAPI setvars.bat did not return PATH"
    }
    return $environment
}

$Script:CrispAsrExe = Get-ConfiguredExecutable `
    -EnvironmentName "S2S_HOST_CRISPASR_EXE" `
    -DefaultPath (Join-Path $Script:Root "tools\CrispASR\build-vulkan\bin\crispasr.exe")
$Script:LlamaExe = Get-ConfiguredExecutable `
    -EnvironmentName "S2S_HOST_LLAMA_EXE" `
    -DefaultPath (Join-Path $Script:Root "tools\llama.cpp\build\bin\Release\llama-server.exe")
$Script:QwenSyclExe = Get-ConfiguredExecutable `
    -EnvironmentName "S2S_HOST_QWEN_SYCL_EXE" `
    -DefaultPath (Join-Path $Script:Root "tools\qwentts\build-sycl\bin\Release\tts-server.exe")
$Script:SupertonicExe = Get-ConfiguredExecutable `
    -EnvironmentName "S2S_HOST_SUPERTONIC_EXE" `
    -DefaultPath (Join-Path $Script:Root "target\release\s2s-vulkan.exe")
$Script:ChatterboxPython = Get-ConfiguredExecutable `
    -EnvironmentName "S2S_HOST_CHATTERBOX_PYTHON" `
    -DefaultPath (Join-Path $Script:Root "tools\chatterbox\.venv\Scripts\python.exe")
$Script:XTTSPython = Get-ConfiguredExecutable `
    -EnvironmentName "S2S_HOST_XTTS_PYTHON" `
    -DefaultPath (Join-Path $Script:Root "tools\xtts-v2\.venv\Scripts\python.exe")
$Script:InflectPython = Get-ConfiguredExecutable `
    -EnvironmentName "S2S_HOST_INFLECT_PYTHON" `
    -DefaultPath (Join-Path $Script:Root "tools\inflect\.venv\Scripts\python.exe")
$Script:Audio8Python = Get-ConfiguredExecutable `
    -EnvironmentName "S2S_HOST_AUDIO8_PYTHON" `
    -DefaultPath (Join-Path $Script:Root "tools\audio8\.venv\Scripts\python.exe")

function New-Profile {
    param(
        [string]$Name,
        [string]$Stage,
        [int]$Port,
        [string]$Executable,
        [string[]]$BackendIds,
        [string[]]$VariantIds
    )
    return [pscustomobject]@{
        name = $Name
        stage = $Stage
        port = $Port
        executable = $Executable
        backend_ids = @($BackendIds)
        variant_ids = @($VariantIds)
    }
}

$Script:Profiles = @{
    "llama-confucius" = New-Profile `
        -Name "llama-confucius" -Stage "asr" -Port 8082 `
        -Executable $Script:LlamaExe `
        -BackendIds @("confucius4-r2t2") `
        -VariantIds @("confucius4-r2t2-vulkan-windows")
    "crispasr-whisper" = New-Profile `
        -Name "crispasr-whisper" -Stage "asr" -Port 8082 `
        -Executable $Script:CrispAsrExe `
        -BackendIds @("fw-tiny", "fw-base", "fw-small", "wcpp-base") `
        -VariantIds @(
            "fw-tiny-vulkan-windows-b580",
            "fw-base-vulkan-windows-b580",
            "fw-small-vulkan-windows-b580",
            "wcpp-base-vulkan-windows-b580"
        )
    "crispasr-parakeet" = New-Profile `
        -Name "crispasr-parakeet" -Stage "asr" -Port 8082 `
        -Executable $Script:CrispAsrExe `
        -BackendIds @("parakeet-tdt-0.6b-v3") `
        -VariantIds @("parakeet-tdt-0.6b-v3-vulkan-windows-b580")
    "crispasr-voxtral" = New-Profile `
        -Name "crispasr-voxtral" -Stage "asr" -Port 8087 `
        -Executable $Script:CrispAsrExe `
        -BackendIds @("voxtral-mini-4b-realtime") `
        -VariantIds @("voxtral-mini-4b-realtime-vulkan-windows-b580")
    "crispasr-qwen3-asr" = New-Profile `
        -Name "crispasr-qwen3-asr" -Stage "asr" -Port 8082 `
        -Executable $Script:CrispAsrExe `
        -BackendIds @("qwen3-asr-0.6b") `
        -VariantIds @(
            "qwen3-asr-0.6b-host-cpu",
            "qwen3-asr-0.6b-vulkan-windows-b580"
        )
    "crispasr-canary" = New-Profile `
        -Name "crispasr-canary" -Stage "asr" -Port 8082 `
        -Executable $Script:CrispAsrExe `
        -BackendIds @("canary-1b-v2") `
        -VariantIds @(
            "canary-1b-v2-host-cpu",
            "canary-1b-v2-vulkan-windows-b580"
        )
    "crispasr-funasr-mlt" = New-Profile `
        -Name "crispasr-funasr-mlt" -Stage "asr" -Port 8082 `
        -Executable $Script:CrispAsrExe `
        -BackendIds @("fun-asr-mlt-nano") `
        -VariantIds @(
            "fun-asr-mlt-nano-host-cpu",
            "fun-asr-mlt-nano-vulkan-windows-b580"
        )
    "crispasr-kokoro" = New-Profile `
        -Name "crispasr-kokoro" -Stage "tts" -Port 8084 `
        -Executable $Script:CrispAsrExe `
        -BackendIds @("kokoro") `
        -VariantIds @("kokoro-vulkan-windows-b580")
    "crispasr-vibevoice" = New-Profile `
        -Name "crispasr-vibevoice" -Stage "tts" -Port 8089 `
        -Executable $Script:CrispAsrExe `
        -BackendIds @("vibevoice-realtime-0.5b") `
        -VariantIds @("vibevoice-realtime-0.5b-vulkan-windows-b580")
    "crispasr-chatterbox" = New-Profile `
        -Name "crispasr-chatterbox" -Stage "tts" -Port 8090 `
        -Executable $Script:CrispAsrExe `
        -BackendIds @("chatterbox-multilingual-v3") `
        -VariantIds @("chatterbox-multilingual-v3-vulkan-windows-b580")
    "crispasr-piper" = New-Profile `
        -Name "crispasr-piper" -Stage "tts" -Port 8092 `
        -Executable $Script:CrispAsrExe `
        -BackendIds @("piper") `
        -VariantIds @(
            "piper-host-cpu",
            "piper-vulkan-windows-b580"
        )
    "crispasr-cosyvoice3" = New-Profile `
        -Name "crispasr-cosyvoice3" -Stage "tts" -Port 8093 `
        -Executable $Script:CrispAsrExe `
        -BackendIds @("cosyvoice3-0.5b") `
        -VariantIds @(
            "cosyvoice3-0.5b-host-cpu",
            "cosyvoice3-0.5b-vulkan-windows-b580"
        )
    "crispasr-omnivoice" = New-Profile `
        -Name "crispasr-omnivoice" -Stage "tts" -Port 8094 `
        -Executable $Script:CrispAsrExe `
        -BackendIds @("omnivoice") `
        -VariantIds @(
            "omnivoice-host-cpu",
            "omnivoice-vulkan-windows-b580"
        )
    "chatterbox-python" = New-Profile `
        -Name "chatterbox-python" -Stage "tts" -Port 8090 `
        -Executable $Script:ChatterboxPython `
        -BackendIds @("chatterbox-multilingual-v3") `
        -VariantIds @(
            "chatterbox-multilingual-v3-cpu-windows",
            "chatterbox-multilingual-v3-cuda-windows"
        )
    "xtts-webgpu" = New-Profile `
        -Name "xtts-webgpu" -Stage "tts" -Port 8091 `
        -Executable $Script:XTTSPython `
        -BackendIds @("xtts-v2") `
        -VariantIds @("xtts-v2-webgpu-vulkan-windows-b580")
    "xtts-cpu" = New-Profile `
        -Name "xtts-cpu" -Stage "tts" -Port 8091 `
        -Executable $Script:XTTSPython `
        -BackendIds @("xtts-v2") `
        -VariantIds @("xtts-v2-cpu-windows")
    "inflect-python" = New-Profile `
        -Name "inflect-python" -Stage "tts" -Port 8095 `
        -Executable $Script:InflectPython `
        -BackendIds @("inflect-micro-v2") `
        -VariantIds @(
            "inflect-micro-v2-host-cpu",
            "inflect-micro-v2-host-cuda"
        )
    "audio8-python" = New-Profile `
        -Name "audio8-python" -Stage "tts" -Port 8096 `
        -Executable $Script:Audio8Python `
        -BackendIds @("audio8-tts-preview-0.6b") `
        -VariantIds @(
            "audio8-tts-preview-0.6b-host-cpu",
            "audio8-tts-preview-0.6b-host-cuda"
        )
    "llama-granite" = New-Profile `
        -Name "llama-granite" -Stage "llm" -Port 8081 `
        -Executable $Script:LlamaExe `
        -BackendIds @("local-fallback") `
        -VariantIds @("local-fallback-vulkan-windows-b580")
    "qwen-sycl" = New-Profile `
        -Name "qwen-sycl" -Stage "tts" -Port 8083 `
        -Executable $Script:QwenSyclExe `
        -BackendIds @("qwen3-tts-0.6b") `
        -VariantIds @("qwen3-tts-sycl-windows-experimental")
    "supertonic-webgpu" = New-Profile `
        -Name "supertonic-webgpu" -Stage "tts" -Port 8085 `
        -Executable $Script:SupertonicExe `
        -BackendIds @("supertonic") `
        -VariantIds @("supertonic-webgpu-vulkan-windows-b580")
}

function Write-JsonAtomic {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)]$Value
    )
    $json = $Value | ConvertTo-Json -Depth 12 -Compress
    $temp = "$Path.$([guid]::NewGuid().ToString('N')).tmp"
    try {
        [IO.File]::WriteAllText($temp, $json, [Text.UTF8Encoding]::new($false))
        # Prefer atomic replace when available (PowerShell 7 / modern .NET).
        # Windows PowerShell 5 only has the 2-argument Move overload.
        $moved = $false
        try {
            [IO.File]::Move($temp, $Path, $true)
            $moved = $true
        } catch [System.MissingMethodException] {
            $moved = $false
        } catch {
            # Some hosts surface the missing overload as MethodInvocationException.
            if ($_.Exception.Message -notmatch 'Move|Überladung|overload|arguments') {
                throw
            }
            $moved = $false
        }
        if (-not $moved) {
            if (Test-Path -LiteralPath $Path) {
                Remove-Item -LiteralPath $Path -Force -ErrorAction Stop
            }
            [IO.File]::Move($temp, $Path)
        }
    }
    finally {
        Remove-Item -LiteralPath $temp -Force -ErrorAction SilentlyContinue
    }
}

function Get-UnixTime {
    return [DateTimeOffset]::UtcNow.ToUnixTimeSeconds()
}

function Resolve-ModelPath {
    param([Parameter(Mandatory = $true)][string]$RelativePath)
    if ([IO.Path]::IsPathRooted($RelativePath)) {
        throw "absolute model paths are forbidden: $RelativePath"
    }
    $rootWithSeparator = $Script:ModelsDir.TrimEnd("\", "/") + [IO.Path]::DirectorySeparatorChar
    $candidate = [IO.Path]::GetFullPath((Join-Path $Script:ModelsDir $RelativePath))
    if (-not $candidate.StartsWith($rootWithSeparator, [StringComparison]::OrdinalIgnoreCase)) {
        throw "model path escapes configured model directory: $RelativePath"
    }
    if (-not (Test-Path -LiteralPath $candidate -PathType Leaf)) {
        throw "required model artifact is missing: $candidate"
    }
    return $candidate
}

function Get-VulkanStatus {
    # vulkaninfo can hang on some Windows GPU stacks; never block the heartbeat
    # loop longer than a few seconds, and cache successful probes.
    $cacheSecs = 30
    $now = [DateTimeOffset]::UtcNow.ToUnixTimeSeconds()
    if (
        $Script:VulkanStatusCache -and
        ($now - [int]$Script:VulkanStatusCache.cached_at) -lt $cacheSecs
    ) {
        return $Script:VulkanStatusCache.status
    }

    $forcedName = $env:S2S_HOST_VULKAN_DEVICE_NAME
    $assumeVulkan = $env:S2S_HOST_ASSUME_VULKAN -eq "1"
    $fallback = [pscustomobject]@{
        available = $assumeVulkan
        device_name = if ($forcedName) { $forcedName } else { "" }
        device_index = $Script:ResolvedVulkanDevice
    }

    $vulkanInfo = Get-Command "vulkaninfo.exe" -ErrorAction SilentlyContinue
    if (-not $vulkanInfo) {
        $vulkanInfo = Get-Command "vulkaninfo" -ErrorAction SilentlyContinue
    }
    if (-not $vulkanInfo) {
        $Script:VulkanStatusCache = @{ cached_at = $now; status = $fallback }
        return $fallback
    }

    $output = ""
    $exitCode = 1
    $probe = $null
    try {
        $probe = Start-Process -FilePath $vulkanInfo.Source -ArgumentList @("--summary") `
            -NoNewWindow -PassThru -RedirectStandardOutput "$env:TEMP\s2s-vulkaninfo.out.txt" `
            -RedirectStandardError "$env:TEMP\s2s-vulkaninfo.err.txt"
        if (-not $probe.WaitForExit(5000)) {
            try { Stop-Process -Id $probe.Id -Force -ErrorAction SilentlyContinue } catch {}
            Write-Warning "vulkaninfo --summary timed out after 5s; reusing cached/fallback device info"
            if ($Script:VulkanStatusCache) {
                return $Script:VulkanStatusCache.status
            }
            $Script:VulkanStatusCache = @{ cached_at = $now; status = $fallback }
            return $fallback
        }
        $exitCode = $probe.ExitCode
        if (Test-Path -LiteralPath "$env:TEMP\s2s-vulkaninfo.out.txt") {
            $output = Get-Content -LiteralPath "$env:TEMP\s2s-vulkaninfo.out.txt" -Raw -ErrorAction SilentlyContinue
        }
    } catch {
        Write-Warning "vulkaninfo probe failed: $($_.Exception.Message)"
        if ($Script:VulkanStatusCache) {
            return $Script:VulkanStatusCache.status
        }
        $Script:VulkanStatusCache = @{ cached_at = $now; status = $fallback }
        return $fallback
    }

    if ($exitCode -ne 0 -or [string]::IsNullOrWhiteSpace($output)) {
        $Script:VulkanStatusCache = @{ cached_at = $now; status = $fallback }
        return $fallback
    }

    $devices = @()
    $currentDevice = $null
    foreach ($line in ($output -split "\r?\n")) {
        if ($line -match "^\s*GPU(?<index>\d+):\s*$") {
            if ($currentDevice -and $currentDevice.name) {
                $devices += [pscustomobject]$currentDevice
            }
            $currentDevice = [ordered]@{
                index = [int]$Matches["index"]
                type = ""
                name = ""
            }
        } elseif ($currentDevice -and $line -match "^\s*deviceType\s*=\s*(.+?)\s*$") {
            $currentDevice.type = $Matches[1].Trim()
        } elseif ($currentDevice -and $line -match "^\s*deviceName\s*=\s*(.+?)\s*$") {
            $currentDevice.name = $Matches[1].Trim()
        }
    }
    if ($currentDevice -and $currentDevice.name) {
        $devices += [pscustomobject]$currentDevice
    }

    $selectedDevice = $null
    if ($VulkanDevice -ge 0) {
        $selectedDevice = $devices |
            Where-Object { $_.index -eq $VulkanDevice } |
            Select-Object -First 1
    } else {
        $selectedDevice = $devices |
            Where-Object { $_.type -eq "PHYSICAL_DEVICE_TYPE_DISCRETE_GPU" } |
            Select-Object -First 1
        if (-not $selectedDevice) {
            $selectedDevice = $devices | Select-Object -First 1
        }
    }

    if ($selectedDevice) {
        $Script:ResolvedVulkanDevice = [int]$selectedDevice.index
    }
    $deviceName = if ($forcedName) {
        $forcedName
    } elseif ($selectedDevice) {
        [string]$selectedDevice.name
    } else {
        "Vulkan device $($Script:ResolvedVulkanDevice)"
    }
    $status = [pscustomobject]@{
        available = $true
        device_name = $deviceName
        device_index = $Script:ResolvedVulkanDevice
    }
    $Script:VulkanStatusCache = @{ cached_at = $now; status = $status }
    return $status
}

function Get-ProcessExecutable {
    param([int]$ProcessId)
    try {
        $process = Get-CimInstance Win32_Process -Filter "ProcessId=$ProcessId" -ErrorAction Stop
        if ($process) {
            return [string]$process.ExecutablePath
        }
    } catch {
        return ""
    }
    return ""
}

function Test-OwnedProcessAlive {
    param($OwnedProcess)
    if (-not $OwnedProcess -or -not $OwnedProcess.pid) {
        return $false
    }
    $actual = Get-ProcessExecutable -ProcessId ([int]$OwnedProcess.pid)
    if (-not $actual) {
        return $false
    }
    return [IO.Path]::GetFullPath($actual).Equals(
        [IO.Path]::GetFullPath([string]$OwnedProcess.executable),
        [StringComparison]::OrdinalIgnoreCase
    )
}

function Save-OwnedState {
    Write-JsonAtomic -Path $Script:StatePath -Value @($Script:Owned.Values)
}

function Load-OwnedState {
    if (-not (Test-Path -LiteralPath $Script:StatePath -PathType Leaf)) {
        return
    }
    try {
        $loaded = Get-Content -Raw -LiteralPath $Script:StatePath | ConvertFrom-Json
        foreach ($entry in @($loaded)) {
            if ($Script:AllowedStages -notcontains [string]$entry.stage) {
                continue
            }
            if (Test-OwnedProcessAlive $entry) {
                $Script:Owned[[string]$entry.stage] = $entry
            }
        }
        Save-OwnedState
    } catch {
        Write-Warning "Ignoring invalid owned process state: $($_.Exception.Message)"
        $Script:Owned = @{}
        Save-OwnedState
    }
}

function Stop-OwnedStage {
    param([Parameter(Mandatory = $true)][string]$Stage)
    if (-not $Script:Owned.ContainsKey($Stage)) {
        return
    }
    $owned = $Script:Owned[$Stage]
    if (Test-OwnedProcessAlive $owned) {
        Write-Host "[host-agent] stopping owned $Stage process pid=$($owned.pid)"
        Stop-Process -Id ([int]$owned.pid) -ErrorAction SilentlyContinue
        try {
            Wait-Process -Id ([int]$owned.pid) -Timeout 10 -ErrorAction Stop
        } catch {
            if (Test-OwnedProcessAlive $owned) {
                Stop-Process -Id ([int]$owned.pid) -Force -ErrorAction SilentlyContinue
            }
        }
    }
    $Script:Owned.Remove($Stage)
    Save-OwnedState
}

function Stop-AllOwned {
    foreach ($stage in @($Script:Owned.Keys)) {
        Stop-OwnedStage -Stage $stage
    }
}

function Stop-AllowlistedContainers {
    param([string[]]$Names)
    foreach ($name in @($Names)) {
        if ($Script:AllowedDockerContainers -notcontains $name) {
            Write-Warning "Ignoring non-allowlisted Docker container '$name'"
            continue
        }
        $running = docker inspect -f "{{.State.Running}}" $name 2>$null
        if ($LASTEXITCODE -eq 0 -and $running -eq "true") {
            Write-Host "[host-agent] docker stop $name"
            docker stop -t 10 $name 2>$null | Out-Null
        }
    }
}

function Assert-Endpoint {
    param(
        [Parameter(Mandatory = $true)][string]$Endpoint,
        [Parameter(Mandatory = $true)][int]$ExpectedPort
    )
    if ([string]::IsNullOrWhiteSpace($Endpoint)) {
        throw "endpoint is required"
    }
    $uri = [Uri]$Endpoint
    if ($uri.Scheme -ne "http") {
        throw "only http endpoints are allowed"
    }
    if ($uri.Port -ne $ExpectedPort) {
        throw "endpoint port $($uri.Port) does not match allowlisted port $ExpectedPort"
    }
    if (@("127.0.0.1", "localhost", "host.docker.internal") -notcontains $uri.Host) {
        throw "endpoint host '$($uri.Host)' is not allowlisted"
    }
}

function Assert-PortAvailable {
    param([int]$Port)
    $listeners = @(Get-NetTCPConnection -State Listen -LocalPort $Port -ErrorAction SilentlyContinue)
    foreach ($listener in $listeners) {
        $owner = [int]$listener.OwningProcess
        $ownedStage = $null
        foreach ($stage in @($Script:Owned.Keys)) {
            if ([int]$Script:Owned[$stage].pid -eq $owner) {
                $ownedStage = $stage
                break
            }
        }
        if ($ownedStage) {
            Stop-OwnedStage -Stage $ownedStage
        } else {
            throw "port $Port is already held by unmanaged pid $owner"
        }
    }
}

function Get-WhisperModel {
    param([string]$BackendId)
    switch ($BackendId) {
        "fw-tiny" { return Resolve-ModelPath "whisper\ggml-tiny.bin" }
        "fw-base" { return Resolve-ModelPath "whisper\ggml-base.bin" }
        "wcpp-base" { return Resolve-ModelPath "whisper\ggml-base.bin" }
        "fw-small" { return Resolve-ModelPath "whisper\ggml-small.bin" }
        default { throw "unsupported Whisper backend '$BackendId'" }
    }
}

function Get-LaunchSpec {
    param(
        [Parameter(Mandatory = $true)]$Profile,
        [Parameter(Mandatory = $true)]$Command
    )
    $vulkanStatus = Get-VulkanStatus
    $effectiveVulkanDevice = [int]$vulkanStatus.device_index
    $commonCrisp = @(
        "--server",
        "--gpu-backend", "vulkan",
        "-dev", "$effectiveVulkanDevice",
        "--host", "127.0.0.1",
        "--port", "$($Profile.port)"
    )
    switch ([string]$Profile.name) {
        "llama-confucius" {
            $model = Resolve-ModelPath "confucius4-r2t2\confucius4-r2t2.Q4_K_M.gguf"
            $projector = Resolve-ModelPath "confucius4-r2t2\confucius4-r2t2.mmproj-Q8_0.gguf"
            return [pscustomobject]@{
                arguments = @(
                    "-m", $model, "--mmproj", $projector, "-dev", "Vulkan$effectiveVulkanDevice",
                    "-ngl", "99", "-c", "4096",
                    "--alias", "confucius4-r2t2", "--host", "127.0.0.1", "--port", "$($Profile.port)"
                )
                environment = @{ GGML_BACKEND = "Vulkan$effectiveVulkanDevice" }
            }
        }
        "crispasr-whisper" {
            $model = Get-WhisperModel -BackendId ([string]$Command.backend_id)
            $args = @("--server", "-m", $model, "--gpu-backend", "vulkan",
                "-dev", "$effectiveVulkanDevice",
                "--host", "127.0.0.1", "--port", "$($Profile.port)", "-l", "de")
            return [pscustomobject]@{ arguments = $args; environment = @{} }
        }
        "crispasr-parakeet" {
            $model = Resolve-ModelPath "parakeet\parakeet-tdt-0.6b-v3-q4_k.gguf"
            return [pscustomobject]@{
                arguments = @("--server", "-m", $model) + $commonCrisp[1..($commonCrisp.Count - 1)]
                environment = @{}
            }
        }
        "crispasr-voxtral" {
            $model = Resolve-ModelPath "voxtral\voxtral-mini-4b-realtime-q4_k.gguf"
            return [pscustomobject]@{
                arguments = @("--server", "--backend", "voxtral4b", "-m", $model) +
                    $commonCrisp[1..($commonCrisp.Count - 1)]
                environment = @{}
            }
        }
        "crispasr-qwen3-asr" {
            $model = Resolve-ModelPath "qwen3-asr\qwen3-asr-0.6b-q4_k.gguf"
            $useVulkan = [string]$Command.variant_id -like "*-vulkan-*"
            # -l auto: multilingual backends; per-request language still wins.
            $args = @("--server", "--backend", "qwen3", "-m", $model, "-l", "auto",
                "--host", "127.0.0.1", "--port", "$($Profile.port)")
            if ($useVulkan) {
                $args += @("--gpu-backend", "vulkan", "-dev", "$effectiveVulkanDevice")
            }
            return [pscustomobject]@{ arguments = $args; environment = @{} }
        }
        "crispasr-canary" {
            $model = Resolve-ModelPath "canary\canary-1b-v2-q4_k.gguf"
            $useVulkan = [string]$Command.variant_id -like "*-vulkan-*"
            $args = @("--server", "--backend", "canary", "-m", $model, "-l", "auto",
                "--host", "127.0.0.1", "--port", "$($Profile.port)")
            if ($useVulkan) {
                $args += @("--gpu-backend", "vulkan", "-dev", "$effectiveVulkanDevice")
            }
            return [pscustomobject]@{ arguments = $args; environment = @{} }
        }
        "crispasr-funasr-mlt" {
            $model = Resolve-ModelPath "funasr\funasr-mlt-nano-2512-q4_k.gguf"
            $useVulkan = [string]$Command.variant_id -like "*-vulkan-*"
            $args = @("--server", "--backend", "fun-asr-mlt-nano", "-m", $model, "-l", "auto",
                "--host", "127.0.0.1", "--port", "$($Profile.port)")
            if ($useVulkan) {
                $args += @("--gpu-backend", "vulkan", "-dev", "$effectiveVulkanDevice")
            }
            return [pscustomobject]@{ arguments = $args; environment = @{} }
        }
        "crispasr-kokoro" {
            $model = Resolve-ModelPath "kokoro-gguf\kokoro-82m-q8_0.gguf"
            $voice = Resolve-ModelPath "kokoro-gguf\kokoro-voice-df_victoria.gguf"
            return [pscustomobject]@{
                arguments = @(
                    "--server", "--backend", "kokoro", "-m", $model,
                    "--voice", $voice, "-l", "de"
                ) + $commonCrisp[1..($commonCrisp.Count - 1)]
                environment = @{ CRISPASR_KOKORO_GEN_GPU = "1" }
            }
        }
        "crispasr-vibevoice" {
            $model = Resolve-ModelPath "vibevoice\vibevoice-realtime-0.5b-q4_k.gguf"
            $voice = Resolve-ModelPath "vibevoice\vibevoice-voice-emma.gguf"
            $voiceDir = Split-Path -Parent $voice
            return [pscustomobject]@{
                arguments = @(
                    "--server", "--backend", "vibevoice-tts", "-m", $model,
                    "--voice", $voice, "--voice-dir", $voiceDir
                ) + $commonCrisp[1..($commonCrisp.Count - 1)]
                environment = @{}
            }
        }
        "crispasr-chatterbox" {
            $model = Resolve-ModelPath "chatterbox\chatterbox-t3-q8_0.gguf"
            $codec = Resolve-ModelPath "chatterbox\chatterbox-s3gen-q8_0.gguf"
            return [pscustomobject]@{
                arguments = @(
                    "--server", "--backend", "chatterbox", "-m", $model,
                    "--codec-model", $codec, "-l", "de", "--tts-steps", "10"
                ) + $commonCrisp[1..($commonCrisp.Count - 1)]
                environment = @{
                    CRISPASR_CHATTERBOX_T3_GPU = "0"
                    CRISPASR_CHATTERBOX_FORCE_GPU = "0"
                    CRISPASR_CHATTERBOX_FULL_CPU = "0"
                }
            }
        }
        "crispasr-piper" {
            # Model GGUF *is* the voice; do not pass a separate --voice name.
            $model = switch ([string]$Command.voice) {
                "thorsten" { Resolve-ModelPath "piper\piper-de_DE-thorsten-medium-f16.gguf" }
                "libritts" { Resolve-ModelPath "piper\piper-en_US-libritts_r-medium-f16.gguf" }
                default { throw "Piper voice '$($Command.voice)' is not allowlisted" }
            }
            $useVulkan = [string]$Command.variant_id -like "*-vulkan-*"
            $args = @(
                "--server", "--backend", "piper", "-m", $model, "-l", "auto",
                "--host", "127.0.0.1", "--port", "$($Profile.port)"
            )
            if ($useVulkan) {
                $args += @("--gpu-backend", "vulkan", "-dev", "$effectiveVulkanDevice")
            }
            return [pscustomobject]@{ arguments = $args; environment = @{} }
        }
        "crispasr-cosyvoice3" {
            $model = Resolve-ModelPath "cosyvoice3\cosyvoice3-llm-q4_k.gguf"
            foreach ($name in @(
                "cosyvoice3-flow-q8_0.gguf",
                "cosyvoice3-hift-f16.gguf",
                "cosyvoice3-s3tok-q4_k.gguf",
                "cosyvoice3-campplus-f16.gguf",
                "cosyvoice3-voices.gguf"
            )) {
                Resolve-ModelPath "cosyvoice3\$name" | Out-Null
            }
            $useVulkan = [string]$Command.variant_id -like "*-vulkan-*"
            $args = @(
                "--server", "--backend", "cosyvoice3-tts", "-m", $model,
                "--voice", "fleurs-de", "-l", "auto",
                "--host", "127.0.0.1", "--port", "$($Profile.port)"
            )
            if ($useVulkan) {
                $args += @("--gpu-backend", "vulkan", "-dev", "$effectiveVulkanDevice")
            }
            return [pscustomobject]@{ arguments = $args; environment = @{} }
        }
        "crispasr-omnivoice" {
            # Zero-shot cloning backend: no baked OpenAI voice name. Speak with
            # the loaded GGUF defaults; attach --voice <wav> later for cloning.
            $model = Resolve-ModelPath "omnivoice\omnivoice-q4_k.gguf"
            $codec = Resolve-ModelPath "omnivoice\omnivoice-tokenizer-q8_0.gguf"
            $useVulkan = [string]$Command.variant_id -like "*-vulkan-*"
            $args = @(
                "--server", "--backend", "omnivoice", "-m", $model,
                "--codec-model", $codec, "-l", "auto",
                "--host", "127.0.0.1", "--port", "$($Profile.port)"
            )
            if ($useVulkan) {
                $args += @("--gpu-backend", "vulkan", "-dev", "$effectiveVulkanDevice")
            }
            return [pscustomobject]@{ arguments = $args; environment = @{} }
        }
        "chatterbox-python" {
            $modelDir = Split-Path -Parent (Resolve-ModelPath "chatterbox\ve.pt")
            foreach ($name in @(
                "t3_mtl23ls_v3.safetensors",
                "s3gen.pt",
                "grapheme_mtl_merged_expanded_v1.json",
                "conds.pt",
                "Cangjie5_TC.json"
            )) {
                Resolve-ModelPath "chatterbox\$name" | Out-Null
            }
            $serverScript = [IO.Path]::GetFullPath(
                (Join-Path $PSScriptRoot "tts_chatterbox_server.py")
            )
            if (-not (Test-Path -LiteralPath $serverScript -PathType Leaf)) {
                throw "Chatterbox server script is missing: $serverScript"
            }
            $device = if (
                [string]$Command.variant_id -eq "chatterbox-multilingual-v3-cuda-windows"
            ) {
                "cuda"
            } else {
                "cpu"
            }
            return [pscustomobject]@{
                arguments = @(
                    $serverScript,
                    "--model-dir", $modelDir,
                    "--device", $device,
                    "--host", "127.0.0.1",
                    "--port", "$($Profile.port)"
                )
                environment = @{
                    HF_HUB_OFFLINE = "1"
                    PYTHONUNBUFFERED = "1"
                    PYTHONUTF8 = "1"
                    TOKENIZERS_PARALLELISM = "false"
                }
            }
        }
        "inflect-python" {
            $modelDir = Split-Path -Parent (Resolve-ModelPath "inflect-micro-v2\model.pth")
            Resolve-ModelPath "inflect-micro-v2\config.json" | Out-Null
            Resolve-ModelPath "inflect-micro-v2\inference.py" | Out-Null
            $serverScript = [IO.Path]::GetFullPath(
                (Join-Path $PSScriptRoot "tts_inflect_server.py")
            )
            if (-not (Test-Path -LiteralPath $serverScript -PathType Leaf)) {
                throw "Inflect server script is missing: $serverScript"
            }
            $device = if ([string]$Command.variant_id -like "*-cuda*") {
                "cuda"
            } else {
                "cpu"
            }
            return [pscustomobject]@{
                arguments = @(
                    $serverScript,
                    "--model-dir", $modelDir,
                    "--device", $device,
                    "--host", "127.0.0.1",
                    "--port", "$($Profile.port)"
                )
                environment = @{
                    S2S_INFLECT_MODEL_DIR = $modelDir
                    S2S_INFLECT_DEVICE = $device
                    PYTHONUNBUFFERED = "1"
                    PYTHONUTF8 = "1"
                }
            }
        }
        "audio8-python" {
            $modelDir = Split-Path -Parent (Resolve-ModelPath "audio8-tts-preview-0.6b\model.safetensors")
            foreach ($name in @(
                "codec.pth",
                "config.json",
                "configuration_arktts.py",
                "modeling_arktts.py",
                "modeling_arktts_codec.py",
                "processing_arktts.py",
                "preprocessor_config.json",
                "processor_config.json",
                "tokenizer.json",
                "tokenizer_config.json",
                "special_tokens_map.json",
                "generation_config.json"
            )) {
                Resolve-ModelPath "audio8-tts-preview-0.6b\$name" | Out-Null
            }
            $serverScript = [IO.Path]::GetFullPath(
                (Join-Path $PSScriptRoot "tts_audio8_server.py")
            )
            if (-not (Test-Path -LiteralPath $serverScript -PathType Leaf)) {
                throw "Audio8 server script is missing: $serverScript"
            }
            $device = if ([string]$Command.variant_id -like "*-cuda*") {
                "cuda"
            } else {
                "cpu"
            }
            return [pscustomobject]@{
                arguments = @(
                    $serverScript,
                    "--model-dir", $modelDir,
                    "--device", $device,
                    "--host", "127.0.0.1",
                    "--port", "$($Profile.port)"
                )
                environment = @{
                    S2S_AUDIO8_MODEL_DIR = $modelDir
                    S2S_AUDIO8_DEVICE = $device
                    HF_HUB_OFFLINE = "1"
                    TRANSFORMERS_OFFLINE = "1"
                    PYTHONUNBUFFERED = "1"
                    PYTHONUTF8 = "1"
                    TOKENIZERS_PARALLELISM = "false"
                }
            }
        }
        { $_ -in @("xtts-webgpu", "xtts-cpu") } {
            $modelDir = Split-Path -Parent (Resolve-ModelPath "xtts-v2\onnx\gpt_model.onnx")
            foreach ($name in @(
                "metadata.json",
                "vocab.json",
                "mel_stats.npy",
                "conditioning_encoder.onnx",
                "speaker_encoder.onnx",
                "hifigan_vocoder.onnx",
                "embeddings\mel_embedding.npy",
                "embeddings\mel_pos_embedding.npy",
                "embeddings\text_embedding.npy",
                "embeddings\text_pos_embedding.npy"
            )) {
                Resolve-ModelPath "xtts-v2\onnx\$name" | Out-Null
            }
            $voicesDir = Split-Path -Parent (Resolve-ModelPath "xtts-v2\voices\de_sample.wav")
            $serverScript = [IO.Path]::GetFullPath(
                (Join-Path $PSScriptRoot "tts_xtts_v2_server.py")
            )
            $upstreamDir = [IO.Path]::GetFullPath(
                (Join-Path $Script:Root "tools\xtts-v2\upstream")
            )
            if (-not (Test-Path -LiteralPath $serverScript -PathType Leaf)) {
                throw "XTTS-v2 server script is missing: $serverScript"
            }
            foreach ($name in @(
                "xtts_streaming_pipeline.py",
                "xtts_onnx_orchestrator.py",
                "xtts_tokenizer.py",
                "zh_num2words.py"
            )) {
                $sourcePath = Join-Path $upstreamDir $name
                if (-not (Test-Path -LiteralPath $sourcePath -PathType Leaf)) {
                    throw "XTTS-v2 pinned runtime source is missing: $sourcePath"
                }
            }
            $mode = if ([string]$Profile.name -eq "xtts-webgpu") {
                "webgpu-vulkan"
            } else {
                "cpu"
            }
            return [pscustomobject]@{
                arguments = @(
                    $serverScript,
                    "--model-dir", $modelDir,
                    "--voices-dir", $voicesDir,
                    "--upstream-dir", $upstreamDir,
                    "--mode", $mode,
                    "--default-language", "de",
                    "--host", "127.0.0.1",
                    "--port", "$($Profile.port)"
                )
                environment = @{
                    HF_HUB_OFFLINE = "1"
                    PYTHONUNBUFFERED = "1"
                    PYTHONUTF8 = "1"
                    TOKENIZERS_PARALLELISM = "false"
                }
            }
        }
        "llama-granite" {
            $model = Resolve-ModelPath "granite\granite-3.3-2b-instruct-q4_k_m.gguf"
            return [pscustomobject]@{
                arguments = @(
                    "-m", $model, "-ngl", "999", "-c", "8192",
                    "--host", "127.0.0.1", "--port", "$($Profile.port)"
                )
                environment = @{ GGML_BACKEND = "Vulkan0" }
            }
        }
        "qwen-sycl" {
            $model = Resolve-ModelPath "qwen\qwen-talker-0.6b-customvoice-Q4_K_M.gguf"
            $codec = Resolve-ModelPath "qwen\qwen-tokenizer-12hz-Q8_0.gguf"
            $environment = Get-OneApiEnvironment
            $environment["GGML_BACKEND"] = "SYCL0"
            $environment["ONEAPI_DEVICE_SELECTOR"] = "level_zero:0"
            $environment["GGML_SYCL_ENABLE_FLASH_ATTN"] = "1"
            $environment["GGML_SYCL_DISABLE_GRAPH"] = "1"
            $environment["GGML_SYCL_PRIORITIZE_DMMV"] = "1"
            $environment["GGML_SYCL_DEV2DEV_MEMCPY"] = "0"
            $environment["GGML_SYCL_DISABLE_OPT"] = "0"
            $environment["GGML_SYCL_DISABLE_DNN"] = "0"
            $environment["GGML_SYCL_USE_LEVEL_ZERO_API"] = "1"
            $environment["QWEN_CODE_SAMPLER"] = "host"
            $environment["ZES_ENABLE_SYSMAN"] = "1"
            return [pscustomobject]@{
                arguments = @(
                    "--model", $model, "--codec", $codec,
                    "--alias", "qwen3-tts-sycl", "--host", "127.0.0.1",
                    "--port", "$($Profile.port)", "--lang", "german", "--clamp-fp16"
                )
                environment = $environment
            }
        }
        "supertonic-webgpu" {
            $onnxDir = Join-Path $Script:ModelsDir "supertonic\onnx"
            $voice = Resolve-ModelPath "supertonic\voice_styles\M1.json"
            foreach ($name in @(
                "duration_predictor.onnx",
                "text_encoder.onnx",
                "vector_estimator.onnx",
                "vocoder.onnx",
                "unicode_indexer.json"
            )) {
                Resolve-ModelPath "supertonic\onnx\$name" | Out-Null
            }
            return [pscustomobject]@{
                arguments = @(
                    "--mode", "tts-server", "--tts", "supertonic",
                    "--supertonic-model-dir", $onnxDir,
                    "--supertonic-voice-path", $voice,
                    "--supertonic-provider", "webgpu-vulkan",
                    "--host", "127.0.0.1", "--port", "$($Profile.port)"
                )
                environment = @{ S2S_GPU = "vulkan" }
            }
        }
        default {
            throw "unknown host profile '$($Profile.name)'"
        }
    }
}

function ConvertTo-CommandLine {
    param([string[]]$Arguments)
    $quoted = foreach ($argument in $Arguments) {
        if ($argument -match '[\s"]') {
            '"' + $argument.Replace('"', '\"') + '"'
        } else {
            $argument
        }
    }
    return [string]::Join(" ", $quoted)
}

function Start-AllowlistedProfile {
    param(
        [Parameter(Mandatory = $true)]$Profile,
        [Parameter(Mandatory = $true)]$Command
    )
    $launch = Get-LaunchSpec -Profile $Profile -Command $Command
    Stop-OwnedStage -Stage ([string]$Profile.stage)
    Assert-PortAvailable -Port ([int]$Profile.port)
    $stamp = [DateTimeOffset]::UtcNow.ToString("yyyyMMdd-HHmmss")
    $baseName = "$($Command.stage)-$($Command.variant_id)-$stamp"
    $stdout = Join-Path $Script:LogDir "$baseName.out.log"
    $stderr = Join-Path $Script:LogDir "$baseName.err.log"
    $oldEnvironment = @{}
    try {
        foreach ($name in $launch.environment.Keys) {
            $oldEnvironment[$name] = [Environment]::GetEnvironmentVariable($name, "Process")
            [Environment]::SetEnvironmentVariable($name, [string]$launch.environment[$name], "Process")
        }
        $process = Start-Process `
            -FilePath ([string]$Profile.executable) `
            -ArgumentList (ConvertTo-CommandLine @($launch.arguments)) `
            -WorkingDirectory (Split-Path -Parent ([string]$Profile.executable)) `
            -RedirectStandardOutput $stdout `
            -RedirectStandardError $stderr `
            -WindowStyle Hidden `
            -PassThru
    } finally {
        foreach ($name in $launch.environment.Keys) {
            [Environment]::SetEnvironmentVariable($name, $oldEnvironment[$name], "Process")
        }
    }
    Start-Sleep -Milliseconds 300
    if ($process.HasExited) {
        $errorTail = ""
        if (Test-Path -LiteralPath $stderr) {
            $errorTail = (Get-Content -LiteralPath $stderr -Tail 10) -join " "
        }
        throw "host process exited during startup (code=$($process.ExitCode)): $errorTail"
    }
    $owned = [pscustomobject]@{
        stage = [string]$Command.stage
        backend_id = [string]$Command.backend_id
        variant_id = [string]$Command.variant_id
        host_profile = [string]$Command.host_profile
        state = "starting"
        pid = [int]$process.Id
        endpoint = [string]$Command.endpoint
        voice = [string]$Command.voice
        executable = [string]$Profile.executable
        stdout = $stdout
        stderr = $stderr
        started_at_unix = Get-UnixTime
    }
    $Script:Owned[[string]$Command.stage] = $owned
    Save-OwnedState
    Write-Host "[host-agent] started $($Command.host_profile) pid=$($process.Id)"
    return $owned
}

function Assert-Command {
    param(
        [Parameter(Mandatory = $true)]$Command,
        [Parameter(Mandatory = $true)][string]$FileStem
    )
    if ([int]$Command.schema_version -ne $Script:SchemaVersion) {
        throw "unsupported schema_version '$($Command.schema_version)'"
    }
    $requestId = [guid]::Empty
    if (-not [guid]::TryParse([string]$Command.request_id, [ref]$requestId)) {
        throw "request_id is not a UUID"
    }
    if ([string]$Command.request_id -ne $FileStem) {
        throw "request_id does not match command filename"
    }
    if (@("start", "stop", "stop_all") -notcontains [string]$Command.action) {
        throw "action '$($Command.action)' is not allowlisted"
    }
    if ([string]$Command.action -eq "stop_all") {
        return $null
    }
    if ($Script:AllowedStages -notcontains [string]$Command.stage) {
        throw "stage '$($Command.stage)' is not allowlisted"
    }
    if (-not $Script:Profiles.ContainsKey([string]$Command.host_profile)) {
        throw "host profile '$($Command.host_profile)' is not allowlisted"
    }
    $profile = $Script:Profiles[[string]$Command.host_profile]
    if ([string]$Command.stage -ne [string]$profile.stage) {
        throw "profile '$($profile.name)' cannot run stage '$($Command.stage)'"
    }
    if ($profile.backend_ids -notcontains [string]$Command.backend_id) {
        throw "backend '$($Command.backend_id)' is not allowlisted for profile '$($profile.name)'"
    }
    if ($profile.variant_ids -notcontains [string]$Command.variant_id) {
        throw "variant '$($Command.variant_id)' is not allowlisted for profile '$($profile.name)'"
    }
    $voice = [string]$Command.voice
    if ($voice.Length -gt 128 -or $voice -match '[\x00-\x1f\x7f]') {
        throw "voice contains invalid characters"
    }
    if ([string]$Command.stage -eq "tts" -and [string]$Command.backend_id -eq "piper" -and
        @("thorsten", "libritts") -notcontains $voice) {
        throw "Piper voice '$voice' is not allowlisted"
    }
    Assert-Endpoint -Endpoint ([string]$Command.endpoint) -ExpectedPort ([int]$profile.port)
    if (-not (Test-Path -LiteralPath $profile.executable -PathType Leaf)) {
        throw "configured executable is missing: $($profile.executable)"
    }
    return $profile
}

function Write-CommandResult {
    param(
        [string]$RequestId,
        [string]$State,
        [int]$ProcessId = 0,
        [string]$ErrorMessage = ""
    )
    $result = [ordered]@{
        schema_version = $Script:SchemaVersion
        request_id = $RequestId
        state = $State
        pid = $ProcessId
        error = $ErrorMessage
        updated_at_unix = Get-UnixTime
    }
    Write-JsonAtomic `
        -Path (Join-Path $Script:ResultDir "$RequestId.json") `
        -Value $result
}

function Handle-CommandFile {
    param([Parameter(Mandatory = $true)][IO.FileInfo]$File)
    $requestId = $File.BaseName
    try {
        $command = Get-Content -Raw -LiteralPath $File.FullName | ConvertFrom-Json
        $profile = Assert-Command -Command $command -FileStem $requestId
        switch ([string]$command.action) {
            "start" {
                $owned = Start-AllowlistedProfile -Profile $profile -Command $command
                Write-CommandResult -RequestId $requestId -State "starting" -ProcessId ([int]$owned.pid)
            }
            "stop" {
                Stop-OwnedStage -Stage ([string]$command.stage)
                Write-CommandResult -RequestId $requestId -State "stopped"
            }
            "stop_all" {
                Stop-AllOwned
                Write-CommandResult -RequestId $requestId -State "stopped"
            }
        }
    } catch {
        Write-Warning "Command $requestId failed: $($_.Exception.Message)"
        Write-CommandResult -RequestId $requestId -State "error" -ErrorMessage $_.Exception.Message
    } finally {
        Remove-Item -LiteralPath $File.FullName -Force -ErrorAction SilentlyContinue
    }
}

function Read-CompatibilityFile {
    param([string]$Name)
    $path = Join-Path $Script:DataDir $Name
    if (Test-Path -LiteralPath $path -PathType Leaf) {
        return Get-Content -Raw -LiteralPath $path
    }
    if (-not $Volume) {
        return $null
    }
    $raw = docker run --rm -v "${Volume}:/data" alpine:3.20 cat "/data/$Name" 2>$null
    if ($LASTEXITCODE -eq 0 -and -not [string]::IsNullOrWhiteSpace($raw)) {
        return $raw
    }
    return $null
}

function Remove-CompatibilityFile {
    param([string]$Name)
    Remove-Item -LiteralPath (Join-Path $Script:DataDir $Name) -Force -ErrorAction SilentlyContinue
    if ($Volume) {
        docker run --rm -v "${Volume}:/data" alpine:3.20 rm -f "/data/$Name" 2>$null | Out-Null
    }
}

function Handle-LegacyIdleRequests {
    $unloadRaw = Read-CompatibilityFile "idle-unload.request"
    if ($unloadRaw) {
        try {
            $request = $unloadRaw | ConvertFrom-Json
            Stop-AllOwned
            if ($request.containers) {
                Stop-AllowlistedContainers @($request.containers)
            }
            Write-Host "[host-agent] compatible idle unload applied"
        } catch {
            Write-Warning "Invalid idle unload request: $($_.Exception.Message)"
        } finally {
            Remove-CompatibilityFile "idle-unload.request"
        }
    }

    $reloadRaw = Read-CompatibilityFile "idle-reload.request"
    if ($reloadRaw) {
        # New host-managed selections are started through UUID commands before the
        # health gate. The old reload marker is acknowledged to avoid duplicate
        # starts; container fallbacks are restored by the Rust controller.
        Remove-CompatibilityFile "idle-reload.request"
        Write-Host "[host-agent] compatible idle reload acknowledged"
    }
}

function Update-OwnedProcesses {
    $changed = $false
    foreach ($stage in @($Script:Owned.Keys)) {
        $entry = $Script:Owned[$stage]
        if (-not (Test-OwnedProcessAlive $entry)) {
            Write-Warning "Owned $stage process pid=$($entry.pid) exited"
            $Script:Owned.Remove($stage)
            $changed = $true
        } elseif ([string]$entry.state -ne "running") {
            $entry.state = "running"
            $changed = $true
        }
    }
    if ($changed) {
        Save-OwnedState
    }
}

function Write-Heartbeat {
    $vulkan = Get-VulkanStatus
    $availableProfiles = @()
    # Advertise any profile whose executable exists. Vulkan-only tools still
    # fail closed at start when the device is missing; CPU Python TTS must
    # remain visible without a working vulkaninfo probe.
    foreach ($profile in $Script:Profiles.Values) {
        if (-not (Test-Path -LiteralPath $profile.executable -PathType Leaf)) {
            continue
        }
        $name = [string]$profile.name
        $needsVulkan = $name -match '^(crispasr-|supertonic-webgpu|qwen-sycl|llama-confucius|llama-granite|xtts-webgpu)'
        if ($needsVulkan -and -not $vulkan.available) {
            continue
        }
        $availableProfiles += $name
    }
    $processes = foreach ($entry in $Script:Owned.Values) {
        [ordered]@{
            stage = [string]$entry.stage
            backend_id = [string]$entry.backend_id
            variant_id = [string]$entry.variant_id
            host_profile = [string]$entry.host_profile
            state = [string]$entry.state
            pid = [int]$entry.pid
            endpoint = [string]$entry.endpoint
            voice = [string]$entry.voice
        }
    }
    $status = [ordered]@{
        schema_version = $Script:SchemaVersion
        agent_version = $Script:AgentVersion
        updated_at_unix = Get-UnixTime
        platform = "windows"
        device_name = [string]$vulkan.device_name
        device_index = [int]$vulkan.device_index
        # The array subexpression prevents PowerShell from unrolling a
        # single accelerator into a JSON string. Rust expects string[].
        accelerators = @($(if ($vulkan.available) { "vulkan" }))
        profiles = @($availableProfiles | Sort-Object -Unique)
        processes = @($processes)
    }
    Write-JsonAtomic -Path $Script:StatusPath -Value $status
}

$initialVulkanStatus = Get-VulkanStatus
Load-OwnedState
Write-Host "[host-agent] data=$($Script:DataDir)"
Write-Host "[host-agent] models=$($Script:ModelsDir)"
Write-Host "[host-agent] Vulkan device index=$($Script:ResolvedVulkanDevice) name=$($initialVulkanStatus.device_name)"
Write-Host "[host-agent] Ctrl+C stops the supervisor; owned inference processes remain tracked"

while ($true) {
    try {
        Update-OwnedProcesses
        foreach ($file in @(Get-ChildItem -LiteralPath $Script:CommandDir -Filter "*.json" -File |
                Sort-Object CreationTimeUtc)) {
            Handle-CommandFile -File $file
        }
        Handle-LegacyIdleRequests
        Write-Heartbeat
    } catch {
        Write-Warning "Supervisor loop failed: $($_.Exception.Message)"
    }
    if ($Once) {
        break
    }
    Start-Sleep -Seconds $PollSeconds
}
