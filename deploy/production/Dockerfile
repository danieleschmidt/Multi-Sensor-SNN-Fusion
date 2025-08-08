# Multi-stage Dockerfile for SNN-Fusion Production Deployment
FROM python:3.11-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    libffi-dev \
    libssl-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Create app user
RUN groupadd -r app && useradd -r -g app app

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt pyproject.toml ./

# Development stage
FROM base as development

# Install development dependencies
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir pytest pytest-cov black isort flake8 mypy

# Copy source code
COPY . .

# Install package in development mode
RUN pip install -e .

# Set ownership
RUN chown -R app:app /app
USER app

EXPOSE 8080

CMD ["python", "-m", "snn_fusion.api.app"]

# Production stage
FROM base as production

# Install production dependencies only
RUN pip install --no-cache-dir \
    torch>=2.1.0 \
    torchvision>=0.16.0 \
    torchaudio>=2.1.0 \
    numpy>=1.24.0 \
    scipy>=1.11.0 \
    pandas>=2.0.0 \
    fastapi>=0.95.0 \
    uvicorn[standard]>=0.22.0 \
    pydantic>=1.10.0 \
    sqlalchemy>=2.0.0 \
    alembic>=1.10.0 \
    redis>=4.5.0 \
    celery>=5.2.0 \
    prometheus-client>=0.16.0 \
    structlog>=23.1.0 \
    cryptography>=40.0.0 \
    psutil>=5.9.0 \
    pyyaml>=6.0 \
    click>=8.1.0 \
    gunicorn>=20.1.0

# Copy source code
COPY src/ ./src/
COPY pyproject.toml README.md LICENSE ./

# Install package
RUN pip install --no-cache-dir -e .

# Create necessary directories
RUN mkdir -p /app/logs /app/data /app/models /app/config && \
    chown -R app:app /app

# Copy configuration files
COPY deploy/production/config/ ./config/

# Set ownership and switch to app user
RUN chown -R app:app /app
USER app

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8080/health || exit 1

# Expose port
EXPOSE 8080

# Production command
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--workers", "4", "--worker-class", "uvicorn.workers.UvicornWorker", "--access-logfile", "-", "--error-logfile", "-", "--log-level", "info", "snn_fusion.api.app:app"]

# Kubernetes stage (optimized for k8s)
FROM production as kubernetes

# Install additional k8s tools
USER root
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Switch back to app user
USER app

# Copy k8s specific configs
COPY deploy/kubernetes/config/ ./k8s-config/

# Use different entrypoint for k8s
COPY deploy/kubernetes/entrypoint.sh ./entrypoint.sh
USER root
RUN chmod +x ./entrypoint.sh
USER app

ENTRYPOINT ["./entrypoint.sh"]