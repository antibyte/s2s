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
        # BusyBox `cp -a -n /source/. /target/` skips the complete source
        # directory when /target already exists. Create the directory tree
        # first, then copy each missing file/symlink independently.
        $copyMissing = @'
set -eu
cd /source
find . -type d -exec mkdir -p "/target/{}" \;
find . \( -type f -o -type l \) -exec sh -c '
    for path do
        target="/target/$path"
        if [ ! -e "$target" ] && [ ! -L "$target" ]; then
            cp -a "$path" "$target"
        fi
    done
' sh {} +
'@
        docker run --rm `
            -v "${VolumeName}:/source:ro" `
            -v "${absolute}:/target" `
            alpine:3.20 `
            sh -c $copyMissing
        if ($LASTEXITCODE -ne 0) {
            throw "Copy from Docker volume '$VolumeName' failed"
        }
        Write-Host "Copied missing files: $VolumeName -> $absolute"
    }
}

Copy-NamedVolumeMissingFiles -VolumeName $ModelsVolume -Destination $ModelsDir
Copy-NamedVolumeMissingFiles -VolumeName $LabDataVolume -Destination $DataDir

Write-Host "Source volumes were retained. Set S2S_MODELS_HOST_DIR and S2S_DATA_HOST_DIR only after reviewing the targets."
