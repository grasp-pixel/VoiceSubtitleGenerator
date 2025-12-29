@echo off
setlocal EnableDelayedExpansion

title Voice Subtitle Generator

:: Project root directory
set "PROJECT_DIR=%~dp0"
cd /d "%PROJECT_DIR%"

echo ============================================
echo   Voice Subtitle Generator
echo ============================================
echo.

:: Check if uv is installed
where uv >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo [1/4] uv not found. Installing...
    powershell -ExecutionPolicy Bypass -Command "irm https://astral.sh/uv/install.ps1 | iex"

    :: Refresh PATH
    set "PATH=%USERPROFILE%\.local\bin;%PATH%"

    :: Verify installation
    where uv >nul 2>&1
    if !ERRORLEVEL! neq 0 (
        echo.
        echo [ERROR] Failed to install uv.
        echo Please install manually: https://docs.astral.sh/uv/getting-started/installation/
        pause
        exit /b 1
    )
    echo uv installed successfully!
    echo.
) else (
    echo [1/4] uv found
)

:: Check if .venv exists
if not exist ".venv" (
    echo [2/4] Creating virtual environment and installing dependencies...
    echo       This may take 10-20 minutes on first run.
    echo.
    uv sync
    if !ERRORLEVEL! neq 0 (
        echo.
        echo [ERROR] Failed to install dependencies.
        pause
        exit /b 1
    )
    echo.
    echo Installation complete!
    echo.
) else (
    echo [2/4] Virtual environment found

    :: Quick sync to ensure dependencies are up to date
    uv sync --quiet
)

:: Check if NVIDIA GPU is available
where nvidia-smi >nul 2>&1
if %ERRORLEVEL% equ 0 (
    echo [3/4] Checking GPU support for translation model...

    :: Check if llama-cpp-python supports GPU
    .venv\Scripts\python.exe -c "import llama_cpp; exit(0 if llama_cpp.llama_supports_gpu_offload() else 1)" 2>nul
    if !ERRORLEVEL! neq 0 (
        echo       GPU detected but llama-cpp-python needs CUDA build.
        echo       Building with CUDA support... (this may take 5-10 minutes)
        echo.

        :: Find Visual Studio 2022 (CUDA 12.x only supports VS 2017-2022)
        set "VS_FOUND="
        set "VCVARS="
        for %%p in (
            "%ProgramFiles(x86)%\Microsoft Visual Studio\2022\BuildTools"
            "%ProgramFiles%\Microsoft Visual Studio\2022\BuildTools"
            "%ProgramFiles(x86)%\Microsoft Visual Studio\2022\Community"
            "%ProgramFiles(x86)%\Microsoft Visual Studio\2022\Professional"
            "%ProgramFiles(x86)%\Microsoft Visual Studio\2022\Enterprise"
        ) do (
            if exist "%%~p\VC\Auxiliary\Build\vcvars64.bat" (
                set "VCVARS=%%~p\VC\Auxiliary\Build\vcvars64.bat"
                set "VS_FOUND=1"
                goto :vs_found
            )
        )

        :vs_found
        if not defined VS_FOUND (
            echo [WARNING] Visual Studio 2022 not found. Using CPU mode.
            echo           Install VS Build Tools 2022 for GPU acceleration.
            goto :skip_cuda
        )

        :: Load VS 2022 environment
        call "!VCVARS!" x64 >nul 2>&1

        :: Find CUDA path
        set "CUDA_FOUND="
        if defined CUDA_PATH (
            set "CUDA_FOUND=1"
        ) else (
            for /f "delims=" %%i in ('where nvcc 2^>nul') do (
                set "NVCC_PATH=%%i"
                for %%j in ("!NVCC_PATH!") do set "CUDA_BIN=%%~dpj"
                for %%j in ("!CUDA_BIN:~0,-1!") do set "CUDA_PATH=%%~dpj"
                set "CUDA_PATH=!CUDA_PATH:~0,-1!"
                set "CUDA_FOUND=1"
                goto :cuda_found
            )
        )

        :cuda_found
        if defined CUDA_FOUND (
            :: Set environment variables for Ninja build
            set "CMAKE_ARGS=-DGGML_CUDA=on"
            set "CMAKE_GENERATOR=Ninja"
            set "CUDACXX=!CUDA_PATH!\bin\nvcc.exe"

            uv pip uninstall llama-cpp-python >nul 2>&1
            uv pip install llama-cpp-python --no-cache-dir --reinstall

            if !ERRORLEVEL! neq 0 (
                echo.
                echo [WARNING] CUDA build failed. Falling back to CPU mode.
                echo           Check README.md for prerequisites.
                uv pip install llama-cpp-python --no-cache-dir
            ) else (
                echo.
                echo       CUDA build complete!
            )
        ) else (
            echo.
            echo [WARNING] CUDA Toolkit not found. Using CPU mode.
            echo           Install CUDA Toolkit for GPU acceleration.
        )
        :skip_cuda
        echo.
    ) else (
        echo       GPU support confirmed
    )
) else (
    echo [3/4] No NVIDIA GPU detected, using CPU mode
)

:: Run the application
echo [4/4] Starting application...
echo.
uv run python main.py

:: If app exits with error, pause to show the message
if %ERRORLEVEL% neq 0 (
    echo.
    echo [ERROR] Application exited with code %ERRORLEVEL%
    pause
)

endlocal
