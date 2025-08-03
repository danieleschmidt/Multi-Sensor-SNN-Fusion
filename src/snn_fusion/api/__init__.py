"""
REST API for SNN-Fusion

Provides HTTP endpoints for model training, inference, and management
with real-time monitoring and neuromorphic hardware integration.
"""

from .app import create_app
from .routes import (
    experiments_bp,
    models_bp,
    training_bp,
    inference_bp,
    hardware_bp,
    monitoring_bp,
)

__all__ = [
    "create_app",
    "experiments_bp",
    "models_bp", 
    "training_bp",
    "inference_bp",
    "hardware_bp",
    "monitoring_bp",
]