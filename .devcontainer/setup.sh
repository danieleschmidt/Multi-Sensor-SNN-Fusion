#!/bin/bash
# Post-create setup script for SNN-Fusion development container

set -e

echo "🚀 Setting up Multi-Sensor SNN-Fusion development environment..."

# Install the package in development mode
echo "📦 Installing snn-fusion package in development mode..."
pip install -e ".[dev,docs,hardware,viz]" || {
    echo "⚠️  Full installation failed, installing core package..."
    pip install -e .
}

# Setup pre-commit hooks
echo "🔧 Setting up pre-commit hooks..."
pre-commit install || echo "⚠️  Pre-commit setup failed"

# Create necessary directories
echo "📁 Creating project directories..."
mkdir -p data/{raw,processed,datasets}
mkdir -p models/{checkpoints,pretrained}
mkdir -p outputs/{logs,figures,results}
mkdir -p configs/{model,training,hardware}
mkdir -p notebooks/{exploratory,analysis,demos}

# Download sample datasets (if available)
echo "📊 Setting up sample data..."
python -c "
import os
import requests
from pathlib import Path

# Create sample configuration files
configs_dir = Path('configs')

# Model configurations
model_config = '''
model:
  modality_configs:
    audio:
      n_inputs: 128
      n_reservoir: 400
      connectivity: 0.12
      tau_mem: 20.0
      tau_adapt: 100.0
    vision:
      n_inputs: 1024
      n_reservoir: 500
      connectivity: 0.10
      tau_mem: 15.0
      tau_adapt: 80.0
    tactile:
      n_inputs: 6
      n_reservoir: 200
      connectivity: 0.15
      tau_mem: 25.0
      tau_adapt: 120.0
  
  n_outputs: 10
  fusion_type: attention
  fusion_config:
    hidden_dim: 256
    num_heads: 8
    dropout: 0.1
  
  global_reservoir_size: 600
'''

# Training configuration
training_config = '''
training:
  learning_rule: bptt
  temporal_window: 100
  batch_size: 32
  learning_rate: 1e-3
  optimizer: adamw
  epochs: 100
  early_stopping: 20
  gradient_clipping: 1.0
  
  regularization:
    weight_decay: 1e-4
    spike_regularization: 0.1
    
  scheduler:
    type: cosine_annealing
    T_max: 100
    eta_min: 1e-6

data:
  modalities: [audio, vision, tactile]
  sample_rate: 2000
  sequence_length: 1000
  augmentation: true
  validation_split: 0.2
  test_split: 0.1
'''

# Hardware deployment configuration
hardware_config = '''
hardware:
  loihi2:
    cores_used: 128
    neurons_per_core: 1024
    quantization_bits: 8
    optimization_level: 2
    
  akida:
    model_quantization: int8
    sparsity_target: 0.9
    batch_size: 1
    
  spinnaker:
    boards: 1
    chips_per_board: 48
    cores_per_chip: 18
    time_step: 1.0

deployment:
  target_latency_ms: 1.0
  power_budget_mw: 500
  accuracy_threshold: 0.90
'''

# Write configuration files
with open(configs_dir / 'model' / 'default.yaml', 'w') as f:
    f.write(model_config)

with open(configs_dir / 'training' / 'default.yaml', 'w') as f:
    f.write(training_config)
    
with open(configs_dir / 'hardware' / 'deployment.yaml', 'w') as f:
    f.write(hardware_config)

print('✅ Sample configurations created')
" || echo "⚠️  Configuration setup failed"

# Setup Git hooks and configurations
echo "🔧 Configuring Git..."
git config --global init.defaultBranch main
git config --global pull.rebase false
git config --global core.autocrlf input

# Create sample Jupyter notebooks
echo "📔 Creating sample notebooks..."
python -c "
import json
from pathlib import Path

# Create a sample exploratory notebook
notebook_content = {
    'cells': [
        {
            'cell_type': 'markdown',
            'metadata': {},
            'source': [
                '# Multi-Sensor SNN-Fusion Exploration\\n',
                '\\n',
                'This notebook demonstrates the core functionality of the neuromorphic sensor fusion framework.\\n'
            ]
        },
        {
            'cell_type': 'code',
            'execution_count': None,
            'metadata': {},
            'outputs': [],
            'source': [
                'import torch\\n',
                'import numpy as np\\n',
                'import matplotlib.pyplot as plt\\n',
                '\\n',
                'from snn_fusion.models import MultiModalLSM\\n',
                'from snn_fusion.preprocessing import AudioSpikeEncoder\\n',
                'from snn_fusion.training import SNNTrainer\\n',
                '\\n',
                'print(\"SNN-Fusion framework loaded successfully!\")\\n'
            ]
        },
        {
            'cell_type': 'markdown',
            'metadata': {},
            'source': [
                '## Model Initialization\\n',
                '\\n',
                'Create a multi-modal liquid state machine with audio, vision, and tactile processing.'
            ]
        },
        {
            'cell_type': 'code',
            'execution_count': None,
            'metadata': {},
            'outputs': [],
            'source': [
                '# Configure modalities\\n',
                'modality_configs = {\\n',
                '    \"audio\": {\"n_inputs\": 64, \"n_reservoir\": 300},\\n',
                '    \"vision\": {\"n_inputs\": 512, \"n_reservoir\": 400},\\n',
                '    \"tactile\": {\"n_inputs\": 6, \"n_reservoir\": 200}\\n',
                '}\\n',
                '\\n',
                '# Initialize model\\n',
                'model = MultiModalLSM(\\n',
                '    modality_configs=modality_configs,\\n',
                '    n_outputs=10,\\n',
                '    fusion_type=\"attention\"\\n',
                ')\\n',
                '\\n',
                'print(f\"Model created with {sum(p.numel() for p in model.parameters())} parameters\")\\n'
            ]
        }
    ],
    'metadata': {
        'kernelspec': {
            'display_name': 'Python 3',
            'language': 'python',
            'name': 'python3'
        },
        'language_info': {
            'name': 'python',
            'version': '3.11.0'
        }
    },
    'nbformat': 4,
    'nbformat_minor': 4
}

# Save notebook
notebooks_dir = Path('notebooks/exploratory')
with open(notebooks_dir / 'snn_fusion_intro.ipynb', 'w') as f:
    json.dump(notebook_content, f, indent=2)

print('✅ Sample notebook created')
" || echo "⚠️  Notebook creation failed"

# Setup development tools
echo "🛠️  Setting up development tools..."

# Create pytest configuration
cat > pytest.ini << 'EOF'
[tool:pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = 
    --strict-markers
    --strict-config
    --verbose
    --tb=short
    --cov=snn_fusion
    --cov-report=term-missing
    --cov-report=html:outputs/coverage
markers =
    slow: marks tests as slow (deselect with '-m "not slow"')
    gpu: marks tests that require GPU
    hardware: marks tests that require neuromorphic hardware
    integration: marks integration tests
EOF

# Create .env.example
cat > .env.example << 'EOF'
# Multi-Sensor SNN-Fusion Environment Configuration

# Data paths
DATA_DIR=./data
MODEL_DIR=./models
OUTPUT_DIR=./outputs

# Hardware configuration
CUDA_VISIBLE_DEVICES=0
LOIHI_PARTITION=
AKIDA_DEVICE=

# Logging
LOG_LEVEL=INFO
WANDB_PROJECT=snn-fusion
WANDB_ENTITY=

# Development
DEBUG=false
PROFILE=false

# Neuromorphic hardware URLs
LOIHI_SERVER_URL=
SPINNAKER_HOST=
AKIDA_RUNTIME_PATH=
EOF

# Create basic .gitignore additions
cat >> .gitignore << 'EOF'

# SNN-Fusion specific
data/raw/
data/processed/
models/checkpoints/
outputs/
*.h5
*.hdf5
*.spike

# Development container cache
.devcontainer/cache/
.devcontainer/pip-cache/

# Jupyter checkpoints
.ipynb_checkpoints/

# Environment files
.env

# Neuromorphic hardware files
*.nxlog
*.akida
*.spinn

# Profiling output
*.prof
*.lprof
EOF

echo "✅ Development environment setup complete!"
echo ""
echo "🎯 Quick start commands:"
echo "  snn-fusion init-config --config-template basic"
echo "  snn-fusion train --data-dir ./data --epochs 10"
echo "  jupyter lab --port 8888"
echo ""
echo "📖 Check out the sample notebook: notebooks/exploratory/snn_fusion_intro.ipynb"
echo "⚙️  Edit configurations in: configs/"
echo "📊 Monitor training in: outputs/"