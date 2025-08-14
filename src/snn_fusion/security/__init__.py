"""
SNN-Fusion Security Module

Comprehensive security framework for neuromorphic computing systems,
including spike train validation, temporal integrity checking, and
hardware-specific security controls.
"""

from .neuromorphic_security import (
    SecurityThreat,
    SecurityEvent,
    SpikeTrainValidator,
    TemporalIntegrityChecker,
    NeuromorphicSecurityManager,
    neuromorphic_secure,
    SecurityError,
)

__all__ = [
    'SecurityThreat',
    'SecurityEvent',
    'SpikeTrainValidator', 
    'TemporalIntegrityChecker',
    'NeuromorphicSecurityManager',
    'neuromorphic_secure',
    'SecurityError',
]