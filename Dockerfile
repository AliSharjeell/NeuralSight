# NeuralSight Max - Docker Container

FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    wget \
    sox \
    libsox-fmt-mp3 \
    portaudio19-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy NeuralSight Python files
COPY voice_terminal_pipeline.py .
COPY start.bat .
COPY start_neuralsight.bat .
COPY .env.example .env

# Copy OpenClaude (pre-built externally, see openclaude/Dockerfile for build)
COPY openclaude/ openclaude/

# Copy gRPC proto files
COPY protos/ protos/

# Build OpenClaude if not pre-built
# Note: Requires Bun to be installed in container
# For production, build openclaude externally and copy dist/cli.mjs
COPY openclaude/bin/openclaude openclaude/bin/openclaude
COPY openclaude/package.json openclaude/package.json
COPY openclaude/bun.lock openclaude/bun.lock

# Install Bun and build OpenClaude (only if dist doesn't exist)
RUN if [ ! -f "openclaude/dist/cli.mjs" ]; then \
    curl -fsSL https://bun.sh/install | bash && \
    export PATH="$HOME/.bun/bin:$PATH" && \
    cd openclaude && bun run build; \
    fi

# Copy Eye-Tracker (can run inside container or separately on host)
COPY Eye-Tracker/ Eye-Tracker/

# Expose ports for gRPC server
EXPOSE 50051

# Entry point - runs the voice interface
# Note: Camera/audio access requires container to run with --device flags
# Eye tracking and window control work best on host OS
CMD ["python", "voice_terminal_pipeline.py"]