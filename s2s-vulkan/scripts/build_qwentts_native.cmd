@echo off
setlocal

set "ONEAPI_SETVARS=%S2S_ONEAPI_SETVARS%"
if not defined ONEAPI_SETVARS set "ONEAPI_SETVARS=D:\Intel\oneAPI\setvars.bat"

set "VCVARS64=%S2S_VCVARS64%"
if not defined VCVARS64 set "VCVARS64=C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat"

if not exist "%VCVARS64%" (
  echo [build] Visual Studio environment not found: %VCVARS64%
  exit /b 2
)
if not exist "%ONEAPI_SETVARS%" (
  echo [build] oneAPI environment not found: %ONEAPI_SETVARS%
  exit /b 2
)

call "%VCVARS64%" >nul
if errorlevel 1 exit /b %errorlevel%
call "%ONEAPI_SETVARS%" --force >nul
if errorlevel 1 exit /b %errorlevel%

set "QWENTTS_SOURCE=%S2S_QWENTTS_SOURCE%"
if not defined QWENTTS_SOURCE set "QWENTTS_SOURCE=%~dp0..\third_party\qwentts.cpp"

set "BUILD_DIR=%S2S_QWENTTS_BUILD_DIR%"
if not defined BUILD_DIR set "BUILD_DIR=%~dp0..\target\qwentts-sycl-native"

cmake -S "%QWENTTS_SOURCE%" -B "%BUILD_DIR%" -G Ninja ^
  -DCMAKE_BUILD_TYPE=Release ^
  -DCMAKE_C_COMPILER=icx ^
  -DCMAKE_CXX_COMPILER=icx ^
  -DGGML_SYCL=ON ^
  -DGGML_SYCL_F16=ON ^
  -DGGML_SYCL_DNN=ON ^
  -DGGML_SYCL_GRAPH=ON ^
  -DGGML_VULKAN=OFF ^
  -DGGML_NATIVE=OFF
if errorlevel 1 exit /b %errorlevel%

cmake --build "%BUILD_DIR%" --target test-abi-c qwen-tts tts-server -j 8
exit /b %errorlevel%
