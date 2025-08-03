"""
Preprocessing Module for Multi-Modal Sensor Data

Implements signal processing pipelines for converting raw sensor data
into spike-based representations suitable for neuromorphic processing.
"""

from .audio import CochlearModel, BiauralProcessor, AudioSpikeEncoder
from .vision import GaborFilters, EventEncoder, VisualSpikeEncoder
from .tactile import WaveletTransform, IMUEncoder, TactileSpikeEncoder
from .spike_encoders import (
    PopulationEncoder,
    TemporalEncoder, 
    RateEncoder,
    LatencyEncoder,
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
    # Tactile processing
    "WaveletTransform",
    "IMUEncoder",
    "TactileSpikeEncoder",
    # Spike encoding
    "PopulationEncoder",
    "TemporalEncoder",
    "RateEncoder", 
    "LatencyEncoder",
]