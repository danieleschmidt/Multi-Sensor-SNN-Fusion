# Multi-Sensor SNN-Fusion Production Dockerfile
# Global-first neuromorphic computing framework for production deployment

FROM python:3.11-slim as base

# Set global environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive \
    TZ=UTC

# Install system dependencies with multi-region support
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libhdf5-dev \
    pkg-config \
    locales \
    tzdata \
    && rm -rf /var/lib/apt/lists/*

# Configure locales for global deployment
RUN sed -i -e 's/# en_US.UTF-8 UTF-8/en_US.UTF-8 UTF-8/' /etc/locale.gen && \
    sed -i -e 's/# es_ES.UTF-8 UTF-8/es_ES.UTF-8 UTF-8/' /etc/locale.gen && \
    sed -i -e 's/# fr_FR.UTF-8 UTF-8/fr_FR.UTF-8 UTF-8/' /etc/locale.gen && \
    sed -i -e 's/# de_DE.UTF-8 UTF-8/de_DE.UTF-8 UTF-8/' /etc/locale.gen && \
    sed -i -e 's/# ja_JP.UTF-8 UTF-8/ja_JP.UTF-8 UTF-8/' /etc/locale.gen && \
    sed -i -e 's/# zh_CN.UTF-8 UTF-8/zh_CN.UTF-8 UTF-8/' /etc/locale.gen && \
    locale-gen

ENV LANG=en_US.UTF-8 \
    LANGUAGE=en_US:en \
    LC_ALL=en_US.UTF-8

# Create application user for security
RUN groupadd -r snnfusion && \
    useradd -r -g snnfusion -d /app -s /bin/bash snnfusion

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt ./
COPY pyproject.toml ./
COPY setup.py ./

# Install Python dependencies with optimization
RUN pip install --upgrade pip setuptools wheel && \
    pip install -r requirements.txt && \
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu && \
    pip cache purge

# Production stage
FROM base as production

# Copy application code
COPY --chown=snnfusion:snnfusion . .

# Install the package
RUN pip install -e . && \
    pip cache purge

# Create necessary directories
RUN mkdir -p /app/data /app/logs /app/models /app/config && \
    chown -R snnfusion:snnfusion /app

# Switch to non-root user
USER snnfusion

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import snn_fusion; print('OK')" || exit 1

# Expose ports
EXPOSE 8000 8080

# Default command
CMD ["python", "-m", "snn_fusion.api.app"]

# Development stage
FROM base as development

# Install development dependencies
RUN pip install pytest pytest-cov black isort flake8 mypy pre-commit jupyter

# Copy application code
COPY --chown=snnfusion:snnfusion . .

# Install in development mode
RUN pip install -e ."[dev]"

# Switch to non-root user
USER snnfusion

# Default command for development
CMD ["bash"]

# GPU-enabled stage
FROM nvidia/cuda:11.8-runtime-ubuntu20.04 as gpu

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEBIAN_FRONTEND=noninteractive \
    TZ=UTC

# Install Python and dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3-pip \
    build-essential \
    curl \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libhdf5-dev \
    && rm -rf /var/lib/apt/lists/*

# Create symlinks for python
RUN ln -s /usr/bin/python3.11 /usr/bin/python && \
    ln -s /usr/bin/python3.11 /usr/bin/python3

# Create application user
RUN groupadd -r snnfusion && \
    useradd -r -g snnfusion -d /app -s /bin/bash snnfusion

WORKDIR /app

# Copy and install requirements
COPY requirements.txt ./
COPY pyproject.toml ./
COPY setup.py ./

RUN pip install --upgrade pip setuptools wheel && \
    pip install -r requirements.txt && \
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Copy application
COPY --chown=snnfusion:snnfusion . .

# Install package
RUN pip install -e .

# Create directories
RUN mkdir -p /app/data /app/logs /app/models /app/config && \
    chown -R snnfusion:snnfusion /app

USER snnfusion

# GPU health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import torch; print('CUDA available:', torch.cuda.is_available())" || exit 1

EXPOSE 8000 8080

CMD ["python", "-m", "snn_fusion.api.app"]