# Multi-stage production Dockerfile for Neuromorphic SNN Fusion
FROM python:3.11-slim as builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libc6-dev \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Production stage
FROM python:3.11-slim as production

# Create non-root user
RUN groupadd -r snnfusion && useradd -r -g snnfusion snnfusion

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Create application directory
WORKDIR /app

# Copy application code
COPY src/ src/
COPY examples/ examples/
COPY pyproject.toml .
COPY README.md .

# Install application
RUN pip install -e .

# Create directories for data and logs
RUN mkdir -p /app/data /app/logs && \
    chown -R snnfusion:snnfusion /app

# Switch to non-root user
USER snnfusion

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Expose port
EXPOSE 8080

# Set environment variables
ENV PYTHONPATH=/app/src
ENV SNN_FUSION_ENV=production
ENV SNN_FUSION_LOG_LEVEL=INFO

# Start application
CMD ["python", "-m", "snn_fusion.api.app"]
