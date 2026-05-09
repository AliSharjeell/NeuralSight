# NeuralSight Max - Docker Container

FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies (including build tools for Python packages)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    wget \
    sox \
    libsox-fmt-mp3 \
    portaudio19-dev \
    libportaudio2 \
    libasound2-dev \
    libssl-dev \
    libffi-dev \
    python3-dev \
    build-essential \
    cmake \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy NeuralSight Python files
COPY voice_terminal_pipeline.py .
COPY start.bat .
COPY start_neuralsight.bat .

# Copy OpenClaude
COPY openclaude/ openclaude/

# Copy gRPC proto files
COPY protos/ protos/

# Install Bun and build OpenClaude
RUN curl -fsSL https://bun.sh/install | bash
ENV PATH="/root/.bun/bin:$PATH"
WORKDIR /app/openclaude
RUN bun run build
WORKDIR /app

# Copy Eye-Tracker
COPY Eye-Tracker/ Eye-Tracker/

# Create .env from example (user should override with real keys)
COPY .env.example .env

# Expose ports for gRPC server
EXPOSE 50051

# Entry point - runs the voice interface
# Note: Camera/audio access requires container to run with --device flags
CMD ["python", "voice_terminal_pipeline.py"]