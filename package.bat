@echo off
cd /d "%~dp0"

echo ============================================
echo   Creating Distribution ZIP
echo ============================================
echo.

if exist ".venv" (
    .venv\Scripts\python.exe scripts\package.py %*
) else (
    python scripts\package.py %*
)

echo.
pause
