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
        echo       GPU detected but llama-cpp-python needs CUDA version.
        echo       Installing pre-built CUDA wheel... (downloading ~450MB)
        echo.

        :: Install pre-built CUDA wheel (cu124 for CUDA 12.x)
        .venv\Scripts\pip.exe install "https://github.com/abetlen/llama-cpp-python/releases/download/v0.3.4-cu124/llama_cpp_python-0.3.4-cp312-cp312-win_amd64.whl" --force-reinstall --quiet

        if !ERRORLEVEL! neq 0 (
            echo.
            echo [WARNING] CUDA wheel installation failed. Falling back to CPU mode.
        ) else (
            echo       CUDA installation complete!
        )
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
