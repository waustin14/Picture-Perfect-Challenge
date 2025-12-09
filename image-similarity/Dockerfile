# Image Similarity Service Dockerfile
# Multi-stage build for smaller final image

# ==============================================================================
# Stage 1: Base image with CUDA support
# ==============================================================================
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04 AS base

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install Python and system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-venv \
    python3-pip \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.10 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

# Create non-root user
RUN useradd -m -u 1000 appuser

# ==============================================================================
# Stage 2: Install dependencies
# ==============================================================================
FROM base AS dependencies

WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install PyTorch with CUDA support first (separate layer for caching)
RUN pip install --no-cache-dir \
    torch \
    torchvision \
    --index-url https://download.pytorch.org/whl/cu121

# Install other dependencies
RUN pip install --no-cache-dir -r requirements.txt

# ==============================================================================
# Stage 3: Final image
# ==============================================================================
FROM dependencies AS final

WORKDIR /app

# Create directories for model cache and app
RUN mkdir -p /app/models && chown -R appuser:appuser /app

# Copy application code
COPY --chown=appuser:appuser app/ ./app/
COPY --chown=appuser:appuser pytest.ini ./

# Switch to non-root user
USER appuser

# Environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEVICE=cuda \
    MODEL_TYPE=dino_vitb16 \
    MODEL_CACHE_DIR=/app/models \
    MAX_PAIRS_PER_REQUEST=128 \
    BATCH_SIZE_LIMIT=64

# Expose port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import requests; r = requests.get('http://localhost:8080/health'); exit(0 if r.status_code == 200 else 1)"

# Run the service
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]
