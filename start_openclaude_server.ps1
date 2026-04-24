# OpenClaude gRPC Server Launcher with MiniMax API
# ================================================
# Run this PowerShell script before starting NeuralSight
# to launch the OpenClaude gRPC server on localhost:50051
#
# Usage:
#   .\start_openclaude_server.ps1
#
# The server will keep running in the foreground.
# Press Ctrl+C to stop.

$ErrorActionPreference = "Stop"

# Keys should be set in your environment or a .env file.
# The script will use existing environment variables if they are set.
$env:API_TIMEOUT_MS               = "3000000"
$env:CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC = "1"
$env:ANTHROPIC_MODEL              = "MiniMax-M2.7"
$env:ANTHROPIC_SMALL_FAST_MODEL   = "MiniMax-M2.7"
$env:ANTHROPIC_DEFAULT_SONNET_MODEL = "MiniMax-M2.7"
$env:ANTHROPIC_DEFAULT_OPUS_MODEL   = "MiniMax-M2.7"
$env:ANTHROPIC_DEFAULT_HAIKU_MODEL  = "MiniMax-M2.7"
$env:CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS = "1"

# ── Debug Logging ────────────────────────────────────────────────────────────
$env:DEBUG                        = "1"
$env:CLAUDE_CODE_DEBUG_LOG_LEVEL  = "verbose"

# ── gRPC Server Binding ──────────────────────────────────────────────────────
$env:GRPC_PORT = "50051"
$env:GRPC_HOST = "127.0.0.1"

# ── Working Directory ────────────────────────────────────────────────────────
$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$openclaudeDir = Join-Path $repoRoot "openclaude"

if (-not (Test-Path $openclaudeDir)) {
    Write-Host "ERROR: openclaude source not found at $openclaudeDir" -ForegroundColor Red
    Write-Host "Clone it first: git clone https://github.com/Gitlawb/openclaude.git"
    exit 1
}

Set-Location $openclaudeDir

# ── Cleanup old gRPC server processes ────────────────────────────────────────
Write-Host "Cleaning up old gRPC server on port $env:GRPC_PORT..." -ForegroundColor Yellow
try {
    $connections = Get-NetTCPConnection -LocalPort $env:GRPC_PORT -ErrorAction SilentlyContinue
    foreach ($conn in $connections) {
        $pid = $conn.OwningProcess
        if ($pid -and $pid -ne 0) {
            Write-Host "  Killing PID $pid on port $env:GRPC_PORT" -ForegroundColor Yellow
            Stop-Process -Id $pid -Force -ErrorAction SilentlyContinue
        }
    }
    Start-Sleep -Seconds 1
} catch {
    # No process on port — that's fine
}

Write-Host "Starting OpenClaude gRPC Server..." -ForegroundColor Cyan
Write-Host "  Provider : MiniMax (via Anthropic-compatible endpoint)" -ForegroundColor Gray
Write-Host "  Model    : $env:ANTHROPIC_MODEL" -ForegroundColor Gray
Write-Host "  Endpoint : $env:ANTHROPIC_BASE_URL" -ForegroundColor Gray
Write-Host "  gRPC     : $env:GRPC_HOST`:$env:GRPC_PORT" -ForegroundColor Gray
Write-Host ""

# Run the gRPC server via bun
try {
    bun run dev:grpc
} catch {
    Write-Host "Failed to start server: $_" -ForegroundColor Red
    exit 1
}
