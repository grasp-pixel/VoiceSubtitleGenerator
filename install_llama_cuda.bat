@echo off
cd /d "%~dp0"

echo Uninstalling existing llama-cpp-python...
.venv\Scripts\pip.exe uninstall llama-cpp-python -y

echo.
echo Building llama-cpp-python with CUDA support...
echo This may take several minutes...
echo.

set CMAKE_ARGS=-DGGML_CUDA=on
.venv\Scripts\pip.exe install llama-cpp-python --no-cache-dir --force-reinstall

echo.
echo Verifying GPU support...
.venv\Scripts\python.exe -c "import llama_cpp; print('GPU offload:', llama_cpp.llama_supports_gpu_offload())"

pause
