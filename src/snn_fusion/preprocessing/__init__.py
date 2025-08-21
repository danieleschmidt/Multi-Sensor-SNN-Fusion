"""
Preprocessing Module for Multi-Modal Sensor Data

Implements signal processing pipelines for converting raw sensor data
into spike-based representations suitable for neuromorphic processing.
"""

from .audio import CochlearModel, BiauralProcessor, AudioSpikeEncoder
from .vision import GaborFilters, EventEncoder, VisualSpikeEncoder, DVSEventProcessor
from .tactile import WaveletTransform, IMUEncoder, TactileSpikeEncoder, VibrotactileSensor
from .spike_encoders import (
    PopulationEncoder,
    TemporalEncoder,
    RateEncoder,
    LatencyEncoder,
    DeltaEncoder,
    AdaptiveEncoder,
)

__all__ = [
    # Audio processing
    "CochlearModel",
    "BiauralProcessor", 
    "AudioSpikeEncoder",
    # Vision processing
    "GaborFilters",
    "EventEncoder",
    "VisualSpikeEncoder",
    "DVSEventProcessor",
    # Tactile processing
    "WaveletTransform",
    "IMUEncoder",
    "TactileSpikeEncoder",
    "VibrotactileSensor",
    # Spike encoders
    "PopulationEncoder",
    "TemporalEncoder",
    "RateEncoder",
    "LatencyEncoder",
    "DeltaEncoder",
    "AdaptiveEncoder",
]