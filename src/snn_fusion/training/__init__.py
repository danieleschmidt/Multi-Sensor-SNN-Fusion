"""
Training Infrastructure for Spiking Neural Networks

Implements specialized training algorithms, loss functions, and optimization
techniques for neuromorphic multi-modal sensor fusion systems.
"""

from .trainer import SNNTrainer, MultiModalTrainer
from .losses import TemporalLoss, SpikeLoss, CrossModalLoss
from .plasticity import STDPLearner, RewardModulatedSTDP
from .optimizers import SpikingOptimizer, NeuromorphicOptimizer
from .schedulers import AdaptiveThresholdScheduler, PlasticityScheduler

__all__ = [
    # Training coordinators
    "SNNTrainer",
    "MultiModalTrainer",
    # Loss functions
    "TemporalLoss",
    "SpikeLoss", 
    "CrossModalLoss",
    # Learning rules
    "STDPLearner",
    "RewardModulatedSTDP",
    # Optimizers
    "SpikingOptimizer",
    "NeuromorphicOptimizer",
    # Schedulers
    "AdaptiveThresholdScheduler",
    "PlasticityScheduler",
]