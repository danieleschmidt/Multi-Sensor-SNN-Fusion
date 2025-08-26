# Multi-stage build for optimized production image
FROM python:3.11-slim as builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    make \
    cmake \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Production stage
FROM python:3.11-slim as production

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    libgomp1 \
    libblas3 \
    liblapack3 \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder stage
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Create non-root user
RUN useradd --create-home --shell /bin/bash terragon
USER terragon
WORKDIR /app

# Copy application code
COPY --chown=terragon:terragon src/ ./src/
COPY --chown=terragon:terragon *.py ./
COPY --chown=terragon:terragon config/ ./config/

# Set environment variables
ENV PYTHONPATH="/app/src"
ENV LOG_LEVEL=INFO
ENV ENABLE_QUANTUM_ACCELERATION=true
ENV BATCH_PROCESSING_SIZE=32
ENV PROMETHEUS_PORT=9090

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8081/health || exit 1

# Expose ports
EXPOSE 8080
EXPOSE 8081
EXPOSE 9090

# Create cache directories
RUN mkdir -p /app/cache /app/logs
VOLUME ["/app/cache", "/app/logs"]

# Start command
CMD ["python", "-m", "breakthrough_swarm_server"]
