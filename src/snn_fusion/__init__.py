"""
Multi-Sensor SNN-Fusion: Neuromorphic computing framework for real-time multi-modal sensor fusion

This package provides a comprehensive framework for training and deploying spiking neural networks
that process multiple asynchronous sensory modalities using liquid state machines and cross-modal
fusion on neuromorphic hardware.

Key Components:
- models: Liquid State Machines and hierarchical fusion networks
- datasets: Multi-modal data loaders and preprocessing
- training: Spike-based learning algorithms and optimization
- hardware: Neuromorphic hardware deployment pipelines
- preprocessing: Signal processing for audio, vision, and tactile data
- visualization: Spike analysis and real-time monitoring tools
"""

__version__ = "0.1.0"
__author__ = "Terragon Labs"
__email__ = "research@terragonlabs.com"

from .models import (
    LiquidStateMachine,
    MultiModalLSM,
    HierarchicalFusionSNN,
    AdaptiveLIF,
)
from .datasets import (
    MAVENDataset,
    MultiModalDataLoader,
)
from .training import (
    SNNTrainer,
    STDPLearner,
    TemporalLoss,
)
from .preprocessing import (
    CochlearModel,
    BiauralProcessor,
    AudioSpikeEncoder,
    GaborFilters,
    EventEncoder,
    VisualSpikeEncoder,
    WaveletTransform,
    IMUEncoder,
    TactileSpikeEncoder,
    PopulationEncoder,
    TemporalEncoder,
    RateEncoder,
)

__all__ = [
    "__version__",
    "__author__", 
    "__email__",
    # Models
    "LiquidStateMachine",
    "MultiModalLSM", 
    "HierarchicalFusionSNN",
    "AdaptiveLIF",
    # Datasets
    "MAVENDataset",
    "MultiModalDataLoader",
    # Training
    "SNNTrainer",
    "STDPLearner", 
    "TemporalLoss",
    # Preprocessing
    "CochlearModel",
    "BiauralProcessor",
    "AudioSpikeEncoder",
    "GaborFilters",
    "EventEncoder",
    "VisualSpikeEncoder",
    "WaveletTransform",
    "IMUEncoder",
    "TactileSpikeEncoder",
    "PopulationEncoder",
    "TemporalEncoder",
    "RateEncoder",
]