@echo off
setlocal EnableDelayedExpansion
cd /d "%~dp0"

echo ============================================
echo   Building llama-cpp-python with CUDA
echo ============================================
echo.

:: Find Visual Studio 2022 (CUDA 12.x only supports VS 2017-2022)
set "VS_FOUND="
set "VCVARS="

:: Check Build Tools first, then Community/Professional/Enterprise
for %%p in (
    "%ProgramFiles(x86)%\Microsoft Visual Studio\2022\BuildTools"
    "%ProgramFiles%\Microsoft Visual Studio\2022\BuildTools"
    "%ProgramFiles(x86)%\Microsoft Visual Studio\2022\Community"
    "%ProgramFiles(x86)%\Microsoft Visual Studio\2022\Professional"
    "%ProgramFiles(x86)%\Microsoft Visual Studio\2022\Enterprise"
) do (
    if exist "%%~p\VC\Auxiliary\Build\vcvars64.bat" (
        set "VCVARS=%%~p\VC\Auxiliary\Build\vcvars64.bat"
        echo Found Visual Studio 2022: %%~p
        set "VS_FOUND=1"
        goto :vs_found
    )
)

:vs_found
if not defined VS_FOUND (
    echo [ERROR] Visual Studio 2022 not found.
    echo Please install Visual Studio Build Tools 2022 with C++ workload.
    pause
    exit /b 1
)

:: Load VS 2022 environment
echo Loading VS 2022 environment...
call "!VCVARS!" x64 >nul 2>&1

:: Find CUDA path
set "CUDA_FOUND="
if defined CUDA_PATH (
    echo Found CUDA_PATH from environment: %CUDA_PATH%
    set "CUDA_FOUND=1"
) else (
    :: Try to find nvcc in PATH
    for /f "delims=" %%i in ('where nvcc 2^>nul') do (
        set "NVCC_PATH=%%i"
        for %%j in ("!NVCC_PATH!") do set "CUDA_BIN=%%~dpj"
        for %%j in ("!CUDA_BIN:~0,-1!") do set "CUDA_PATH=%%~dpj"
        set "CUDA_PATH=!CUDA_PATH:~0,-1!"
        set "CUDA_FOUND=1"
        echo Found CUDA from nvcc: !CUDA_PATH!
        goto :set_env
    )
)

if not defined CUDA_FOUND (
    echo [ERROR] CUDA not found. Please install CUDA Toolkit 12.x.
    pause
    exit /b 1
)

:set_env
:: Set environment variables for Ninja build
set "CMAKE_ARGS=-DGGML_CUDA=on"
set "CMAKE_GENERATOR=Ninja"
set "CUDACXX=!CUDA_PATH!\bin\nvcc.exe"

echo.
echo Environment:
echo   CMAKE_ARGS=!CMAKE_ARGS!
echo   CMAKE_GENERATOR=!CMAKE_GENERATOR!
echo   CUDA_PATH=!CUDA_PATH!
echo   CUDACXX=!CUDACXX!
echo.

echo Uninstalling existing llama-cpp-python...
.venv\Scripts\pip.exe uninstall llama-cpp-python -y 2>nul

echo.
echo Building with CUDA support... (this may take 5-10 minutes)
echo.
.venv\Scripts\pip.exe install llama-cpp-python --no-cache-dir --force-reinstall

if !ERRORLEVEL! neq 0 (
    echo.
    echo [ERROR] Build failed.
    pause
    exit /b 1
)

echo.
echo ============================================
echo   Verifying GPU support
echo ============================================
.venv\Scripts\python.exe -c "import llama_cpp; print('GPU offload:', llama_cpp.llama_supports_gpu_offload())"

echo.
echo Done!
endlocal
pause
