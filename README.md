# NeuralSight: Max

A high-speed, voice-controlled Windows assistant powered by **OpenClaude** and **Groq Whisper**.

## Features
- **Max Wake-Word**: Reliable wake-word detection using Groq's `whisper-large-v3-turbo`.
- **Interruptible Pipeline**: Say "Max" at any time to cancel the current task and start a new one.
- **Floating Pill UI**: A modern, translucent UI that follows your focus and shows real-time waveforms.
- **18+ Windows Tools**: Full control over apps, files, registry, UI elements, and PowerShell via `windows-mcp`.
- **Headless & Fast**: Optimized for low latency and Snappy voice interactions.

## Setup

1. **Prerequisites**:
   - [Bun](https://bun.sh) (for the gRPC server)
   - [Python 3.10+](https://python.org)
   - [uv](https://github.com/astral-sh/uv) (for MCP server management)

2. **Environment**:
   Create a `.env` file in the root:
   ```env
   GROQ_API_KEY=your_key_here
   MINIMAX_API_KEY=your_key_here
   ```

3. **Install Dependencies**:
   ```bash
   pip install customtkinter SpeechRecognition pyaudio groq python-dotenv grpcio grpcio-tools
   cd openclaude
   bun install
   ```

4. **Run**:
   Simply run the root batch file:
   ```bash
   .\start_neuralsight.bat
   ```

## Development & Testing
This repo includes a custom skill for validating the Windows MCP tools:
- `skills/windows-mcp-tool-tester/SKILL.md`: Use this to run automated QA on any of the 18 tools.

## Architecture
- `voice_terminal_pipeline.py`: Main entry point for voice capture and UI.
- `openclaude/`: Headless gRPC server re-exposing Claude Code internals.
- `.mcp.json`: Configuration for the `windows-mcp` server.
- `protos/`: gRPC service definitions.
