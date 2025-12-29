# Build llama-cpp-python with CUDA support
Set-Location $PSScriptRoot

Write-Host "============================================"
Write-Host "  Building llama-cpp-python with CUDA"
Write-Host "============================================"
Write-Host ""

# Find Visual Studio 2022 Build Tools (CUDA 12.4 only supports VS 2017-2022)
$vs2022Paths = @(
    "${env:ProgramFiles(x86)}\Microsoft Visual Studio\2022\BuildTools",
    "${env:ProgramFiles}\Microsoft Visual Studio\2022\BuildTools",
    "${env:ProgramFiles(x86)}\Microsoft Visual Studio\2022\Community",
    "${env:ProgramFiles}\Microsoft Visual Studio\2022\Community",
    "${env:ProgramFiles(x86)}\Microsoft Visual Studio\2022\Professional",
    "${env:ProgramFiles(x86)}\Microsoft Visual Studio\2022\Enterprise"
)

$vcvarsall = $null
foreach ($vsPath in $vs2022Paths) {
    $testPath = Join-Path $vsPath "VC\Auxiliary\Build\vcvars64.bat"
    if (Test-Path $testPath) {
        $vcvarsall = $testPath
        Write-Host "Found Visual Studio 2022: $vsPath"
        break
    }
}

if ($vcvarsall) {
    # Import VS 2022 environment
    cmd /c "`"$vcvarsall`" x64 && set" | ForEach-Object {
        if ($_ -match "^([^=]+)=(.*)$") {
            [System.Environment]::SetEnvironmentVariable($matches[1], $matches[2], "Process")
        }
    }
    Write-Host "VS 2022 environment loaded"
} else {
    Write-Host "[WARNING] Visual Studio 2022 not found. Build may fail."
    Write-Host "Please install Visual Studio Build Tools 2022 with C++ workload."
}

# Find CUDA path
$cudaPath = $env:CUDA_PATH
if (-not $cudaPath) {
    $nvccPath = (Get-Command nvcc -ErrorAction SilentlyContinue).Source
    if ($nvccPath) {
        $cudaPath = Split-Path (Split-Path $nvccPath -Parent) -Parent
        Write-Host "Found CUDA from nvcc: $cudaPath"
    } else {
        Write-Host "[ERROR] CUDA not found. Please install CUDA Toolkit."
        exit 1
    }
} else {
    Write-Host "Found CUDA_PATH from environment: $cudaPath"
}

# Set environment variables for Ninja build (bypasses VS CUDA integration issues)
$env:CMAKE_ARGS = "-DGGML_CUDA=on"
$env:CMAKE_GENERATOR = "Ninja"
$env:CUDACXX = "$cudaPath\bin\nvcc.exe"
$env:CUDA_PATH = $cudaPath
$env:CC = "cl"
$env:CXX = "cl"

Write-Host ""
Write-Host "Environment:"
Write-Host "  CMAKE_ARGS=$env:CMAKE_ARGS"
Write-Host "  CMAKE_GENERATOR=$env:CMAKE_GENERATOR"
Write-Host "  CUDA_PATH=$env:CUDA_PATH"
Write-Host "  CUDACXX=$env:CUDACXX"
Write-Host ""

Write-Host "Uninstalling existing llama-cpp-python..."
& .\.venv\Scripts\pip.exe uninstall llama-cpp-python -y 2>$null

Write-Host ""
Write-Host "Building with CUDA support... (5-10 minutes)"
Write-Host ""
& .\.venv\Scripts\pip.exe install llama-cpp-python --no-cache-dir --force-reinstall

if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "[ERROR] Build failed."
    exit 1
}

Write-Host ""
Write-Host "============================================"
Write-Host "  Verifying GPU support"
Write-Host "============================================"
& .\.venv\Scripts\python.exe -c "import llama_cpp; print('GPU offload:', llama_cpp.llama_supports_gpu_offload())"

Write-Host ""
Write-Host "Done!"
