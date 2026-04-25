@echo off
echo Starting NeuralSight Suite...

:: Start the Headless Server
start "NeuralSight: Server" powershell -NoExit -Command "cd openclaude; .\start_openclaude_server.ps1"

:: Start the Voice Interface
start "NeuralSight: Max" python voice_terminal_pipeline.py

echo All components launched.
pause
