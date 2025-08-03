"""
Multi-Modal Datasets for SNN Training

Implements data loaders and preprocessing for synchronized multi-modal
sensor data including audio, event camera, and IMU recordings.
"""

from .maven_dataset import MAVENDataset, MAVENConfig
from .loaders import MultiModalDataLoader, create_dataloaders
from .synthetic import SyntheticMultiModalDataset
from .transforms import (
    SpikeTransform,
    TemporalJitter,
    ModalityDropout,
    SpikeMasking,
)

__all__ = [
    # Main datasets
    "MAVENDataset",
    "MAVENConfig",
    "SyntheticMultiModalDataset",
    # Data loading
    "MultiModalDataLoader",
    "create_dataloaders",
    # Transforms
    "SpikeTransform",
    "TemporalJitter",
    "ModalityDropout", 
    "SpikeMasking",
]