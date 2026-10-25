@echo off
title PhysicsGuard Dashboard Launcher

:: Check if port 8501 is already running
powershell -Command "if (-not (Get-NetTCPConnection -LocalPort 8501 -ErrorAction SilentlyContinue)) { exit 1 } else { exit 0 }" >nul 2>&1
if %ERRORLEVEL% equ 0 (
    echo [OK] PhysicsGuard Dashboard is already running.
) else (
    echo [*] Starting PhysicsGuard Dashboard server in background...
    powershell -WindowStyle Hidden -Command "Start-Process python -ArgumentList '-m streamlit run \"\"C:\Users\LokeshFix\Desktop\phase6.5\app.py\"\" --server.headless true --server.port 8501' -WindowStyle Hidden"
    timeout /t 3 /nobreak >nul
)

echo [*] Opening browser to http://localhost:8501 ...
start http://localhost:8501
exit /b 0
