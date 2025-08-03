# Multi-stage Dockerfile for SNN-Fusion Production Deployment
# Optimized for neuromorphic computing with security best practices

# Build stage
FROM python:3.11-slim-bullseye AS builder

# Set build arguments
ARG BUILD_ENV=production
ARG CUDA_VERSION=11.8

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    ninja-build \
    pkg-config \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create build directory
WORKDIR /build

# Copy requirements first for better caching
COPY requirements.txt pyproject.toml setup.py ./
COPY src/snn_fusion/__init__.py src/snn_fusion/__init__.py

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Build package
RUN pip install --no-cache-dir .

# Runtime stage
FROM python:3.11-slim-bullseye AS runtime

# Set runtime arguments
ARG APP_USER=snnfusion
ARG APP_UID=1001
ARG APP_GID=1001

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Audio processing
    libasound2 \
    libportaudio2 \
    libsndfile1 \
    # Computer vision
    libopencv-dev \
    # Scientific computing
    libfftw3-3 \
    libhdf5-103 \
    libblas3 \
    liblapack3 \
    # Hardware interfaces
    libusb-1.0-0 \
    libhidapi-libusb0 \
    # System utilities
    tini \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user
RUN groupadd -g ${APP_GID} ${APP_USER} && \
    useradd -u ${APP_UID} -g ${APP_GID} -m -s /bin/bash ${APP_USER}

# Set up application directory
WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application files
COPY --chown=${APP_USER}:${APP_USER} src/ ./src/
COPY --chown=${APP_USER}:${APP_USER} configs/ ./configs/
COPY --chown=${APP_USER}:${APP_USER} scripts/ ./scripts/

# Create required directories
RUN mkdir -p data models outputs logs cache && \
    chown -R ${APP_USER}:${APP_USER} /app

# Set environment variables
ENV PYTHONPATH="/app/src:${PYTHONPATH}" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DATA_DIR=/app/data \
    MODEL_DIR=/app/models \
    OUTPUT_DIR=/app/outputs \
    LOG_LEVEL=INFO \
    API_HOST=0.0.0.0 \
    API_PORT=8080

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:${API_PORT}/health || exit 1

# Expose ports
EXPOSE 8080 8888 6006

# Switch to non-root user
USER ${APP_USER}

# Use tini as init system
ENTRYPOINT ["/usr/bin/tini", "--"]

# Default command
CMD ["python", "-m", "snn_fusion.api.app"]

# Labels for metadata
LABEL org.opencontainers.image.title="SNN-Fusion" \
      org.opencontainers.image.description="Neuromorphic multi-modal sensor fusion framework" \
      org.opencontainers.image.version="0.1.0" \
      org.opencontainers.image.vendor="Terragon Labs" \
      org.opencontainers.image.source="https://github.com/terragonlabs/Multi-Sensor-SNN-Fusion" \
      org.opencontainers.image.licenses="Apache-2.0"

# Production stage with optimizations
FROM runtime AS production

# Install production-only optimizations
USER root
RUN pip install --no-cache-dir gunicorn gevent && \
    # Remove development packages
    pip uninstall -y pip setuptools wheel && \
    # Clean up
    rm -rf /root/.cache /tmp/*

USER ${APP_USER}

# Production command with Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--workers", "4", "--worker-class", "gevent", "--timeout", "120", "snn_fusion.api.app:create_app()"]

# Development stage
FROM runtime AS development

# Install development dependencies
USER root
RUN pip install --no-cache-dir \
    pytest \
    pytest-cov \
    black \
    isort \
    flake8 \
    mypy \
    jupyter \
    ipywidgets \
    && rm -rf /root/.cache

USER ${APP_USER}

# Development command
CMD ["python", "-m", "snn_fusion.api.app"]