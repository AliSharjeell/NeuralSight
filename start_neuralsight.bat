@echo off
setlocal enabledelayedexpansion

echo [NeuralSight] Starting Max (NeuralSight v2)...

:: 1. Cleanup Port 50051 (Headless Server)
echo [NeuralSight] Ensuring port 50051 is free...
for /f "tokens=5" %%a in ('netstat -aon ^| findstr :50051 ^| findstr LISTENING') do (
    taskkill /F /PID %%a >nul 2>&1
)

:: 2. Start OpenClaude gRPC Server
echo [NeuralSight] Launching OpenClaude Server...
start "NeuralSight Server" /b powershell -ExecutionPolicy Bypass -File "%~dp0\start_openclaude_server.ps1"

:: Wait for server tools to load
echo [NeuralSight] Waiting for tools to load (5s)...
timeout /t 5 /nobreak >nul

:: 3. Start Voice Pipeline UI
echo [NeuralSight] Launching Voice UI...
python "%~dp0\voice_terminal_pipeline.py"

echo [NeuralSight] Max has exited.
pause
