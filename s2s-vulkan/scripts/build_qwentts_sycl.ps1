# Build qwentts.cpp with oneAPI SYCL for Intel Arc.
# Prerequisites:
#   1) Intel oneAPI Base Toolkit installed
#   2) VS 2022 Build Tools (cl/link for host objects)
#   3) In a shell AFTER setvars.bat:
#        cmd /c "\"C:\Program Files (x86)\Intel\oneAPI\setvars.bat\" intel64 vs2022 && powershell -File scripts\build_qwentts_sycl.ps1"
#
# Output: C:\b\qwentts-sycl\  (tts-server.exe, qwen-tts.exe, DLLs)

$ErrorActionPreference = "Stop"

$src = Join-Path $PSScriptRoot "..\third_party\qwentts.cpp" | Resolve-Path
$build = "D:\b\qwentts-sycl"
# After cmake, if link fails with vs_link_exe (__sycl_register_lib), re-link:
#   icx -fsycl ...\tts-server.cpp.obj -o tts-server.exe ... sycl8.lib
# (see install notes — pure -fsycl driver link, not /link)
$oneapi = $env:ONEAPI_ROOT
if (-not $oneapi) { $oneapi = $env:S2S_ONEAPI_ROOT }
if (-not $oneapi) {
    $candidates = @(
        "D:\Intel\oneAPI",
        "E:\Intel\oneAPI",
        "C:\Program Files (x86)\Intel\oneAPI",
        "C:\Program Files\Intel\oneAPI"
    )
    foreach ($c in $candidates) {
        if (Test-Path $c) { $oneapi = $c; break }
    }
}
if (-not $oneapi) {
    throw "ONEAPI_ROOT not set and oneAPI not found. Run setvars.bat or scripts\install_oneapi.ps1 first."
}

# Prefer compilers from PATH (after setvars)
$icx = Get-Command icx -ErrorAction SilentlyContinue
$icpx = Get-Command icpx -ErrorAction SilentlyContinue
if (-not $icx -or -not $icpx) {
    throw "icx/icpx not on PATH. Run: `"$oneapi\setvars.bat`" intel64 vs2022"
}

Write-Host "Source: $src"
Write-Host "Build:  $build"
Write-Host "icx:    $($icx.Source)"
Write-Host "sycl-ls:"
& sycl-ls 2>&1 | Select-Object -First 20

New-Item -ItemType Directory -Force -Path $build | Out-Null

# Short path + Ninja avoids Windows path length issues
$generator = "Ninja"
if (-not (Get-Command ninja -ErrorAction SilentlyContinue)) {
    $generator = "NMake Makefiles"
}

# Release + FP16 is the first SYCL perf experiment recommended by llama.cpp
# docs (GGML_SYCL_F16). DNN/GRAPH default ON in ggml-sycl CMake; keep explicit.
# Note: buildsycl.sh upstream only sets GGML_SYCL=ON (no Release/F16) — we do better.
$cmakeArgs = @(
    "-S", "$src",
    "-B", "$build",
    "-G", $generator,
    "-DCMAKE_BUILD_TYPE=Release",
    "-DGGML_SYCL=ON",
    "-DGGML_SYCL_F16=ON",
    "-DGGML_SYCL_DNN=ON",
    "-DGGML_SYCL_GRAPH=ON",
    "-DGGML_VULKAN=OFF",
    "-DCMAKE_C_COMPILER=icx",
    "-DCMAKE_CXX_COMPILER=icx"
)

if ($env:S2S_QWEN_SYCL_ARCH) {
    $cmakeArgs += "-DGGML_SYCL_DEVICE_ARCH=$env:S2S_QWEN_SYCL_ARCH"
}

Write-Host "cmake configure..."
& cmake @cmakeArgs
if ($LASTEXITCODE -ne 0) { throw "cmake configure failed" }

Write-Host "cmake build..."
& cmake --build $build --config Release -j $env:NUMBER_OF_PROCESSORS
# On Windows, cmake's vs_link_exe often fails with __sycl_register_lib even though
# objects/DLL are fine. Fall back to a pure `icx -fsycl` link for the servers.
if ($LASTEXITCODE -ne 0) {
    Write-Host "cmake build reported failure — attempting pure icx -fsycl link for tts-server/qwen-tts..."
    Push-Location $build
    try {
        $env:LIB = "$build;$build\ggml\src;$build\ggml\src\ggml-sycl;$build\vendor\cpp-httplib;$env:LIB"
        $common = @(
            "ggml\src\ggml.lib",
            "ggml\src\ggml-cpu.lib",
            "ggml\src\ggml-sycl\ggml-sycl.lib",
            "ggml\src\ggml-base.lib",
            "sycl8.lib"
        )
        if (Test-Path "CMakeFiles\tts-server.dir\tools\tts-server.cpp.obj") {
            & icx -fsycl -nologo `
                "CMakeFiles\tts-server.dir\tools\tts-server.cpp.obj" `
                -o tts-server.exe `
                qwen-core.lib `
                "vendor\cpp-httplib\httplib.lib" `
                yyjson.lib `
                @common `
                ws2_32.lib
            if ($LASTEXITCODE -ne 0) { throw "icx link tts-server failed" }
        }
        if (Test-Path "CMakeFiles\qwen-tts.dir\tools\qwen-tts.cpp.obj") {
            & icx -fsycl -nologo `
                "CMakeFiles\qwen-tts.dir\tools\qwen-tts.cpp.obj" `
                -o qwen-tts.exe `
                qwen-core.lib `
                @common
            if ($LASTEXITCODE -ne 0) { throw "icx link qwen-tts failed" }
        }
        if (-not (Test-Path "tts-server.exe")) { throw "tts-server.exe still missing after fallback link" }
    } finally {
        Pop-Location
    }
}

Write-Host "Artifacts:"
Get-ChildItem $build -Include tts-server.exe,qwen-tts.exe,*.dll -Recurse -ErrorAction SilentlyContinue |
    Select-Object FullName, Length | Format-Table -AutoSize

Write-Host "OK. Start with: .\scripts\start_qwen_sycl.ps1"
Write-Host "Expect log line:  GGML_SYCL_F16: yes"
