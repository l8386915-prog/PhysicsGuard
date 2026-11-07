@echo off
title PhysicsGuard Dashboard Launcher

:: Change directory to script folder
cd /d "%~dp0"
set "APP_DIR=%~dp0"
set "APP_DIR=%APP_DIR:~0,-1%"

:: Check if port 8501 is already running
powershell -NoProfile -Command "if (-not (Get-NetTCPConnection -LocalPort 8501 -ErrorAction SilentlyContinue)) { exit 1 } else { exit 0 }" >nul 2>&1
if %ERRORLEVEL% equ 0 (
    echo [OK] PhysicsGuard Dashboard is already running.
) else (
    echo [*] Starting PhysicsGuard Dashboard server in background...
    powershell -NoProfile -WindowStyle Hidden -Command "Start-Process python -ArgumentList @('-m', 'streamlit', 'run', 'app.py', '--server.headless', 'true', '--server.port', '8501') -WorkingDirectory '%APP_DIR%' -WindowStyle Hidden"
    powershell -NoProfile -Command "Start-Sleep -Seconds 3"
)

echo [*] Opening browser to http://localhost:8501 ...
start http://localhost:8501
exit /b 0
