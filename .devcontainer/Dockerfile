# Multi-Sensor SNN-Fusion Development Container
FROM mcr.microsoft.com/devcontainers/python:3.11-bullseye

# Install system dependencies for neuromorphic computing
RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
    && apt-get -y install --no-install-recommends \
    # Build tools
    build-essential \
    cmake \
    ninja-build \
    pkg-config \
    # Audio processing dependencies
    libasound2-dev \
    libportaudio2 \
    libportaudiocpp0 \
    portaudio19-dev \
    libsndfile1-dev \
    # Computer vision dependencies
    libopencv-dev \
    python3-opencv \
    # Scientific computing
    libfftw3-dev \
    libhdf5-dev \
    libblas-dev \
    liblapack-dev \
    # Hardware interface dependencies
    libusb-1.0-0-dev \
    libhidapi-dev \
    # Visualization dependencies
    graphviz \
    graphviz-dev \
    # Development tools
    git-lfs \
    curl \
    wget \
    vim \
    htop \
    && apt-get autoremove -y && apt-get clean -y && rm -rf /var/lib/apt/lists/*

# Install CUDA development tools (for GPU simulation)
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb \
    && dpkg -i cuda-keyring_1.0-1_all.deb \
    && apt-get update \
    && apt-get -y install cuda-toolkit-11-8 \
    || echo "CUDA installation failed - continuing without GPU support"

# Set environment variables
ENV PYTHONPATH="/workspace/src:$PYTHONPATH"
ENV CUDA_HOME="/usr/local/cuda"
ENV PATH="$CUDA_HOME/bin:$PATH"
ENV LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"

# Create workspace directory
WORKDIR /workspace

# Install Python dependencies early for better caching
COPY requirements.txt /tmp/
RUN pip install --no-cache-dir --upgrade pip setuptools wheel \
    && pip install --no-cache-dir -r /tmp/requirements.txt

# Install additional development dependencies
RUN pip install --no-cache-dir \
    # Development tools
    jupyter \
    jupyterlab \
    ipywidgets \
    # Testing framework
    pytest \
    pytest-cov \
    pytest-xdist \
    pytest-benchmark \
    # Code quality
    black \
    isort \
    flake8 \
    mypy \
    pre-commit \
    # Documentation
    sphinx \
    sphinx-rtd-theme \
    myst-parser \
    # Visualization
    matplotlib \
    seaborn \
    plotly \
    dash \
    tensorboard \
    wandb \
    # Profiling and debugging
    line_profiler \
    memory_profiler \
    py-spy

# Install neuromorphic hardware simulators (fallback versions)
RUN pip install --no-cache-dir \
    # SNN frameworks
    norse \
    snntorch \
    rockpool \
    # Event-based data processing
    dv-processing \
    || echo "Some neuromorphic packages unavailable - install manually if needed"

# Configure Jupyter Lab
RUN jupyter lab --generate-config \
    && echo "c.ServerApp.ip = '0.0.0.0'" >> ~/.jupyter/jupyter_lab_config.py \
    && echo "c.ServerApp.allow_root = True" >> ~/.jupyter/jupyter_lab_config.py \
    && echo "c.ServerApp.open_browser = False" >> ~/.jupyter/jupyter_lab_config.py

# Create cache directories
RUN mkdir -p /root/.cache/pip /root/.cache/torch

# Switch to non-root user
USER vscode

# Final workspace setup
WORKDIR /workspace