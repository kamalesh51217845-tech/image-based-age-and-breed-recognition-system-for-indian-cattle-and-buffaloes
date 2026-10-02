@echo off
setlocal

cd /d "%~dp0"

set HOST=localhost
set PORT=8000

echo Starting server on http://localhost:%PORT% ...
echo (Close with Ctrl+C)
echo.

py -3.12 run.py
if errorlevel 1 (
  echo.
  echo Failed to run with "py -3.12". Trying "python"...
  echo.
  python run.py
)

echo.
pause
