@echo off
setlocal enabledelayedexpansion

echo [NeuralSight] Starting Josh...

:: 1. Clean up port 50051 (OpenClaude gRPC)
echo [NeuralSight] Cleaning up port 50051...
for /f "tokens=5" %%a in ('netstat -aon ^| findstr :50051 ^| findstr LISTENING') do (
    echo [NeuralSight] Killing process %%a on port 50051...
    taskkill /F /PID %%a >nul 2>&1
)

:: 2. Ensure environment variables
if "%GROQ_API_KEY%"=="" (
    echo [WARNING] GROQ_API_KEY is not set. Wake-word detection will be slow.
)

:: 3. Start OpenClaude Server in background
echo [NeuralSight] Starting OpenClaude gRPC Server...
cd /d "%~dp0\openclaude"
start /b cmd /c "bun run dev:grpc"

:: Wait for server to initialize
echo [NeuralSight] Waiting for server...
timeout /t 5 /nobreak >nul

:: 4. Start Voice Pipeline
echo [NeuralSight] Starting Voice Pipeline UI...
cd /d "%~dp0"
python voice_terminal_pipeline.py

echo [NeuralSight] Josh has exited.
pause
