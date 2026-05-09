# NeuralSight Max - Docker Container
# Works on fresh PC with just Docker installed

# ---- Build Stage ----
FROM node:22-slim AS openclaude-build

# Install Bun
RUN npm install -g bun@1.3.11

# All build artifacts go under /app/openclaude
WORKDIR /app/openclaude

# Copy dependency manifests first for better layer caching
COPY openclaude/package.json openclaude/bun.lock ./

# Install all dependencies
RUN bun install --frozen-lockfile

# Copy OpenClaude source
COPY openclaude/src/ src/
COPY openclaude/scripts/ scripts/
COPY openclaude/bin/ bin/
COPY openclaude/tsconfig.json ./

# Build OpenClaude bundle
RUN bun run build

# ---- Final Stage ----
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    sox \
    libsox-fmt-mp3 \
    portaudio19-dev \
    libportaudio2 \
    libasound2-dev \
    libssl3 \
    libffi8 \
    python3-dev \
    build-essential \
    cmake \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy NeuralSight Python files
COPY voice_terminal_pipeline.py .
COPY start.bat .
COPY start_neuralsight.bat .

# Copy built OpenClaude from build stage
COPY --from=openclaude-build /app/openclaude/dist /app/openclaude/dist
COPY --from=openclaude-build /app/openclaude/bin /app/openclaude/bin
COPY --from=openclaude-build /app/openclaude/node_modules /app/openclaude/node_modules
COPY openclaude/package.json /app/openclaude/

# Copy gRPC proto files
COPY protos/ protos/

# Copy Eye-Tracker
COPY Eye-Tracker/ Eye-Tracker/

# Create .env from example
COPY .env.example .env

# Expose ports for gRPC server
EXPOSE 50051

# Entry point - runs the voice interface
CMD ["python", "voice_terminal_pipeline.py"]