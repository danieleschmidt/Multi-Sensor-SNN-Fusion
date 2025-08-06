# SNN-Fusion Production Dockerfile
# Multi-stage build for optimized production deployment

# Build stage
FROM python:3.12-slim as builder

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create and activate virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy requirements and install dependencies
COPY requirements.txt /tmp/
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Production stage
FROM python:3.12-slim as production

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    SNN_FUSION_ENV=production

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder stage
COPY --from=builder /opt/venv /opt/venv

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash snn_user
USER snn_user
WORKDIR /home/snn_user

# Copy application code
COPY --chown=snn_user:snn_user . /home/snn_user/snn_fusion/

# Set Python path
ENV PYTHONPATH="/home/snn_user/snn_fusion/src:$PYTHONPATH"

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "from snn_fusion.monitoring.health import HealthMonitor; h = HealthMonitor(); print('healthy' if h.check_health()['status'] == 'healthy' else exit(1))"

# Expose port (configurable via environment)
EXPOSE ${SNN_FUSION_PORT:-8000}

# Set working directory to source
WORKDIR /home/snn_user/snn_fusion/src

# Default command
CMD ["python", "-m", "snn_fusion.main"]