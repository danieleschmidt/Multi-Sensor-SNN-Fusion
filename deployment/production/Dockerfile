# Production Dockerfile for SNN-Fusion
FROM python:3.11-slim AS base

# Set build arguments
ARG BUILD_DATE
ARG VERSION
ARG VCS_REF

# Add metadata
LABEL org.opencontainers.image.title="SNN-Fusion"
LABEL org.opencontainers.image.description="Production-ready Spiking Neural Network Fusion Platform"
LABEL org.opencontainers.image.version=${VERSION}
LABEL org.opencontainers.image.created=${BUILD_DATE}
LABEL org.opencontainers.image.revision=${VCS_REF}
LABEL org.opencontainers.image.vendor="Terragon Labs"

# System dependencies and security updates
RUN apt-get update && apt-get install -y \
    --no-install-recommends \
    build-essential \
    curl \
    ca-certificates \
    pkg-config \
    libhdf5-dev \
    libssl-dev \
    libffi-dev \
    libblas-dev \
    liblapack-dev \
    gfortran \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user for security
RUN groupadd -r snn && useradd -r -g snn snn

# Set working directory
WORKDIR /app

# Copy requirements first for better layer caching
COPY requirements.txt requirements-prod.txt ./

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements-prod.txt

# Copy application code
COPY . .

# Install SNN-Fusion package
RUN pip install --no-cache-dir -e .

# Create necessary directories
RUN mkdir -p /app/logs /app/models /app/data /app/tmp && \
    chown -R snn:snn /app

# Switch to non-root user
USER snn

# Expose ports
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command
CMD ["python", "-m", "snn_fusion.api.main"]