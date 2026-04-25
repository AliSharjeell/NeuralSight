@echo off
echo ====================================
echo  NeuralSight Max Launcher
echo ====================================
echo.

:: ── Terminal 1: OpenClaude Headless Server ─────────────────────────────────
echo [1/4] Starting OpenClaude gRPC Server...
start "NeuralSight: OpenClaude Server" powershell -NoExit -Command "cd 'C:\AppsNew\NeuralSight\openclaude'; .\start_openclaude_server.ps1"

:: Wait for server to initialize
timeout /t 4 /nobreak >nul

:: ── Terminal 2: Eye Tracking Calibration ───────────────────────────────────
echo [2/4] Running Eye Tracker Calibration...
echo    (Say 'q' in calibration window to skip)
start "NeuralSight: Eye Calibration" cmd /k "cd /d C:\AppsNew\NeuralSight\Eye-Tracker && ..\..\AppData\Local\Programs\Python\Python311\python.exe calibrate.py"

:: Wait for calibration to complete (or user skips)
echo    Waiting for calibration to finish...
timeout /t 15 /nobreak >nul

:: ── Terminal 3: Eye Tracking (runs after calibration) ───────────────────────
echo [3/4] Starting Eye Tracker...
start "NeuralSight: Eye Tracker" cmd /k "cd /d C:\AppsNew\NeuralSight\Eye-Tracker && ..\..\AppData\Local\Programs\Python\Python311\python.exe track.py"

:: Short delay before voice UI
timeout /t 2 /nobreak >nul

:: ── Terminal 4: Voice Interface ────────────────────────────────────────────
echo [4/4] Starting Voice Terminal (Max)...
start "NeuralSight: Max (Voice)" cmd /k "cd /d C:\AppsNew\NeuralSight && python voice_terminal_pipeline.py"

echo.
echo ====================================
echo  NeuralSight is fully running!
echo  - Say 'Max' to activate voice
echo  - Say 'Max calibrate eye tracking' to recalibrate
echo  - Say 'Max turn on eye tracking' to restart tracking
echo ====================================
echo.
pause