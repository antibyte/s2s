# Windows host supervisor for Vulkan inference backends.
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
#   S2S_HOST_QWEN_EXE
#   S2S_HOST_SUPERTONIC_EXE
#
# Compatibility: idle-unload.request and idle-reload.request are still handled.

[CmdletBinding()]
param(
    [string]$DataDir = "",
    [string]$ModelsDir = "",
    [string]$Volume = "",
    [ValidateRange(1, 60)]
    [int]$PollSeconds = 2,
    [ValidateRange(0, 31)]
    [int]$VulkanDevice = 0,
    [switch]$Once
)

$ErrorActionPreference = "Stop"
$Script:SchemaVersion = 2
$Script:AgentVersion = "2.1.0"
$Script:Root = Split-Path -Parent $PSScriptRoot
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

$Script:CrispAsrExe = Get-ConfiguredExecutable `
    -EnvironmentName "S2S_HOST_CRISPASR_EXE" `
    -DefaultPath (Join-Path $Script:Root "tools\CrispASR\build-vulkan\bin\crispasr.exe")
$Script:LlamaExe = Get-ConfiguredExecutable `
    -EnvironmentName "S2S_HOST_LLAMA_EXE" `
    -DefaultPath (Join-Path $Script:Root "tools\llama.cpp\build\bin\Release\llama-server.exe")
$Script:QwenExe = Get-ConfiguredExecutable `
    -EnvironmentName "S2S_HOST_QWEN_EXE" `
    -DefaultPath (Join-Path $Script:Root "tools\qwentts\build-vulkan\bin\Release\tts-server.exe")
$Script:SupertonicExe = Get-ConfiguredExecutable `
    -EnvironmentName "S2S_HOST_SUPERTONIC_EXE" `
    -DefaultPath (Join-Path $Script:Root "target\release\s2s-vulkan.exe")

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
    "crispasr-piper" = New-Profile `
        -Name "crispasr-piper" -Stage "tts" -Port 8092 `
        -Executable $Script:CrispAsrExe `
        -BackendIds @("piper") `
        -VariantIds @(
            "piper-host-cpu",
            "piper-vulkan-windows-b580"
        )    "llama-granite" = New-Profile `
        -Name "llama-granite" -Stage "llm" -Port 8081 `
        -Executable $Script:LlamaExe `
        -BackendIds @("local-fallback") `
        -VariantIds @("local-fallback-vulkan-windows-b580")
    "qwen-vulkan" = New-Profile `
        -Name "qwen-vulkan" -Stage "tts" -Port 8083 `
        -Executable $Script:QwenExe `
        -BackendIds @("qwen3-tts-0.6b") `
        -VariantIds @("qwen3-tts-vulkan-windows-b580")
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
    [IO.File]::WriteAllText($temp, $json, [Text.UTF8Encoding]::new($false))
    Move-Item -LiteralPath $temp -Destination $Path -Force
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
    $forcedName = $env:S2S_HOST_VULKAN_DEVICE_NAME
    $assumeVulkan = $env:S2S_HOST_ASSUME_VULKAN -eq "1"
    $vulkanInfo = Get-Command "vulkaninfo.exe" -ErrorAction SilentlyContinue
    if (-not $vulkanInfo) {
        $vulkanInfo = Get-Command "vulkaninfo" -ErrorAction SilentlyContinue
    }
    if (-not $vulkanInfo) {
        return [pscustomobject]@{
            available = $assumeVulkan
            device_name = if ($forcedName) { $forcedName } else { "" }
        }
    }

    $output = & $vulkanInfo.Source --summary 2>&1 | Out-String
    if ($LASTEXITCODE -ne 0) {
        return [pscustomobject]@{
            available = $assumeVulkan
            device_name = if ($forcedName) { $forcedName } else { "" }
        }
    }
    $deviceName = $forcedName
    if (-not $deviceName -and $output -match "(?im)^\s*deviceName\s*=\s*(.+?)\s*$") {
        $deviceName = $Matches[1].Trim()
    }
    return [pscustomobject]@{
        available = $true
        device_name = if ($deviceName) { $deviceName } else { "Vulkan device $VulkanDevice" }
    }
}

function Get-ProcessExecutable {
    param([int]$Pid)
    try {
        $process = Get-CimInstance Win32_Process -Filter "ProcessId=$Pid" -ErrorAction Stop
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
    $actual = Get-ProcessExecutable -Pid ([int]$OwnedProcess.pid)
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
    $commonCrisp = @(
        "--server",
        "--gpu-backend", "vulkan",
        "-dev", "$VulkanDevice",
        "--host", "127.0.0.1",
        "--port", "$($Profile.port)"
    )
    switch ([string]$Profile.name) {
        "crispasr-whisper" {
            $model = Get-WhisperModel -BackendId ([string]$Command.backend_id)
            $args = @("--server", "-m", $model, "--gpu-backend", "vulkan", "-dev", "$VulkanDevice",
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
        "crispasr-kokoro" {
            $model = Resolve-ModelPath "kokoro-gguf\kokoro-82m-q8_0.gguf"
            $voice = Resolve-ModelPath "kokoro-gguf\kokoro-voice-df_victoria.gguf"
            return [pscustomobject]@{
                arguments = @(
                    "--server", "--backend", "kokoro", "-m", $model,
                    "--voice", $voice, "-l", "de"
                ) + $commonCrisp[1..($commonCrisp.Count - 1)]
                environment = @{}
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
        "crispasr-piper" {
            # The allowlisted GGUF is the effective voice; arbitrary model paths are forbidden.
            $model = switch ([string]$Command.voice) {
                "thorsten" { Resolve-ModelPath "piper\piper-de_DE-thorsten-medium-f16.gguf" }
                "libritts" { Resolve-ModelPath "piper\piper-en_US-libritts_r-medium-f16.gguf" }
                default { throw "Piper voice '$($Command.voice)' is not allowlisted" }
            }
            $args = @(
                "--server", "--backend", "piper", "-m", $model, "-l", "auto",
                "--host", "127.0.0.1", "--port", "$($Profile.port)"
            )
            if ([string]$Command.variant_id -like "*-vulkan-*") {
                $args += @("--gpu-backend", "vulkan", "-dev", "$VulkanDevice")
            }
            return [pscustomobject]@{ arguments = $args; environment = @{} }
        }        "llama-granite" {
            $model = Resolve-ModelPath "granite\granite-3.3-2b-instruct-q4_k_m.gguf"
            return [pscustomobject]@{
                arguments = @(
                    "-m", $model, "-ngl", "999", "-c", "8192",
                    "--host", "127.0.0.1", "--port", "$($Profile.port)"
                )
                environment = @{ GGML_BACKEND = "Vulkan0" }
            }
        }
        "qwen-vulkan" {
            $model = Resolve-ModelPath "qwen\qwen-talker-0.6b-customvoice-Q4_K_M.gguf"
            $codec = Resolve-ModelPath "qwen\qwen-tokenizer-12hz-Q8_0.gguf"
            return [pscustomobject]@{
                arguments = @(
                    "--model", $model, "--codec", $codec,
                    "--alias", "qwen3-tts-vulkan", "--host", "127.0.0.1",
                    "--port", "$($Profile.port)", "--lang", "auto", "--clamp-fp16"
                )
                environment = @{
                    GGML_BACKEND = "Vulkan0"
                    GGML_VK_DISABLE_GRAPH_OPTIMIZE = "1"
                    GGML_VK_DISABLE_COOPMAT = "1"
                    GGML_VK_DISABLE_COOPMAT2 = "1"
                    QWEN_CODE_SAMPLER = "host"
                }
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
        throw "unsupported host-agent schema_version '$($Command.schema_version)'; expected 2, restart the updated host agent"
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
    $voice = [string]$Command.voice
    if ($voice.Length -gt 128 -or $voice -match '[\x00-\x1f\x7f]') {
        throw "voice contains invalid characters"
    }
    if ([string]$Command.stage -eq "tts" -and [string]$Command.backend_id -eq "piper" -and
        @("thorsten", "libritts") -notcontains $voice) {
        throw "Piper voice '$voice' is not allowlisted"
    }    if ($Script:AllowedStages -notcontains [string]$Command.stage) {
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
        [int]$Pid = 0,
        [string]$ErrorMessage = ""
    )
    $result = [ordered]@{
        schema_version = $Script:SchemaVersion
        request_id = $RequestId
        state = $State
        pid = $Pid
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
                Write-CommandResult -RequestId $requestId -State "starting" -Pid ([int]$owned.pid)
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
    if ($vulkan.available) {
        foreach ($profile in $Script:Profiles.Values) {
            if (Test-Path -LiteralPath $profile.executable -PathType Leaf) {
                $availableProfiles += [string]$profile.name
            }
        }
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
        accelerators = if ($vulkan.available) { @("vulkan") } else { @() }
        profiles = @($availableProfiles | Sort-Object -Unique)
        processes = @($processes)
    }
    Write-JsonAtomic -Path $Script:StatusPath -Value $status
}

Load-OwnedState
Write-Host "[host-agent] data=$($Script:DataDir)"
Write-Host "[host-agent] models=$($Script:ModelsDir)"
Write-Host "[host-agent] Vulkan device index=$VulkanDevice"
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
