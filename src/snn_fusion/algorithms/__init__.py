"""
Advanced Neuromorphic Algorithms Module

Implements business logic and core algorithms for neuromorphic computing,
including optimization, adaptation, and hardware-specific implementations.
"""

from .optimization import (
    NeuromorphicOptimizer,
    SpikingGradientDescent,
    STDPOptimizer,
    HardwareAwareOptimizer,
)
from .adaptation import (
    AdaptiveController,
    PlasticityManager,
    ThresholdAdaptation,
    SynapticScaling,
)
from .encoding import (
    SpikeEncoder,
    CochlearEncoder,
    DVSEncoder,
    TactileEncoder,
    MultiModalEncoder,
)
from .fusion import (
    CrossModalFusion,
    AttentionMechanism,
    TemporalFusion,
    SpatioTemporalFusion,
)

__all__ = [
    'NeuromorphicOptimizer',
    'SpikingGradientDescent', 
    'STDPOptimizer',
    'HardwareAwareOptimizer',
    'AdaptiveController',
    'PlasticityManager',
    'ThresholdAdaptation',
    'SynapticScaling',
    'SpikeEncoder',
    'CochlearEncoder',
    'DVSEncoder',
    'TactileEncoder',
    'MultiModalEncoder',
    'CrossModalFusion',
    'AttentionMechanism',
    'TemporalFusion',
    'SpatioTemporalFusion',
]