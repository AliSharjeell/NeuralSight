@echo off
echo ====================================
echo  NeuralSight Max Launcher
echo ====================================
echo.

:: ── Terminal 1: OpenClaude Headless Server ─────────────────────────────────
echo [1/2] Starting OpenClaude gRPC Server...
start "NeuralSight: OpenClaude Server" powershell -NoExit -Command "cd 'C:\AppsNew\NeuralSight\openclaude'; .\start_openclaude_server.ps1"

:: Wait a moment for server to initialize
timeout /t 3 /nobreak >nul

:: ── Terminal 2: Voice Interface ────────────────────────────────────────────
echo [2/2] Starting Voice Terminal (Max)...
start "NeuralSight: Max (Voice)" cmd /k "cd /d C:\AppsNew\NeuralSight && python voice_terminal_pipeline.py"

echo.
echo ====================================
echo  NeuralSight is running!
echo  - Say 'Max' to activate voice
echo  - Say 'Max calibrate eye tracking' to calibrate
echo  - Say 'Max turn on eye tracking' to start tracking
echo ====================================
echo.
pause