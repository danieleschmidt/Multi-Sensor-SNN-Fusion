"""
Neuromorphic Models Module

Implements core spiking neural network architectures for multi-modal sensor fusion:
- Liquid State Machines with reservoir computing
- Multi-modal fusion networks with attention
- Hierarchical processing architectures
- Adaptive neuron models optimized for neuromorphic hardware
"""

from .lsm import LiquidStateMachine
from .multimodal_lsm import MultiModalLSM
from .hierarchical_fusion import HierarchicalFusionSNN
from .neurons import AdaptiveLIF, SpikingNeuron
from .readouts import LinearReadout, TemporalReadout
from .attention import CrossModalAttention, TemporalAttention

__all__ = [
    "LiquidStateMachine",
    "MultiModalLSM",
    "HierarchicalFusionSNN", 
    "AdaptiveLIF",
    "SpikingNeuron",
    "LinearReadout",
    "TemporalReadout",
    "CrossModalAttention",
    "TemporalAttention",
]