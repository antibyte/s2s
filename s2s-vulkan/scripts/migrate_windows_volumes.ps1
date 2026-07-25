# Non-destructively copy existing Docker named-volume contents into the
# Windows bind directories used by docker-compose.windows.yml.
#
# Existing destination files and the source volumes are never removed or
# overwritten. Review the copied data before changing your .env.

[CmdletBinding(SupportsShouldProcess = $true)]
param(
    [string]$ModelsVolume = "s2s-models",
    [string]$LabDataVolume = "s2s-lab-data",
    [string]$ModelsDir = "",
    [string]$DataDir = ""
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
if (-not $ModelsDir) {
    $ModelsDir = Join-Path $root "models"
}
if (-not $DataDir) {
    $DataDir = Join-Path $root "data"
}

function Copy-NamedVolumeMissingFiles {
    param(
        [Parameter(Mandatory = $true)][string]$VolumeName,
        [Parameter(Mandatory = $true)][string]$Destination
    )
    if ($VolumeName -notmatch '^[A-Za-z0-9][A-Za-z0-9_.-]*$') {
        throw "Unsafe Docker volume name '$VolumeName'"
    }
    docker volume inspect $VolumeName 2>$null | Out-Null
    if ($LASTEXITCODE -ne 0) {
        throw "Docker volume '$VolumeName' does not exist"
    }

    $absolute = [IO.Path]::GetFullPath($Destination)
    New-Item -ItemType Directory -Force -Path $absolute | Out-Null
    if ($PSCmdlet.ShouldProcess(
            $absolute,
            "copy missing files from Docker volume '$VolumeName'"
        )) {
        docker run --rm `
            -v "${VolumeName}:/source:ro" `
            -v "${absolute}:/target" `
            alpine:3.20 `
            sh -c "cp -a -n /source/. /target/"
        if ($LASTEXITCODE -ne 0) {
            throw "Copy from Docker volume '$VolumeName' failed"
        }
        Write-Host "Copied missing files: $VolumeName -> $absolute"
    }
}

Copy-NamedVolumeMissingFiles -VolumeName $ModelsVolume -Destination $ModelsDir
Copy-NamedVolumeMissingFiles -VolumeName $LabDataVolume -Destination $DataDir

Write-Host "Source volumes were retained. Set S2S_MODELS_HOST_DIR and S2S_DATA_HOST_DIR only after reviewing the targets."
