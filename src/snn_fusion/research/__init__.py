"""
SNN-Fusion Research Module

Advanced research implementations and validation frameworks for neuromorphic
computing and multi-modal sensor fusion research.

Research Components:
- Temporal Spike Attention (TSA) algorithm validation
- Neuromorphic hardware benchmarking suite
- Statistical significance testing frameworks
- Cross-modal synchrony analysis tools
- Energy efficiency optimization studies
"""

from .neuromorphic_benchmarks import (
    NeuromorphicBenchmarkSuite,
    BenchmarkConfig,
    BenchmarkResult,
    validate_tsa_performance,
)

__all__ = [
    'NeuromorphicBenchmarkSuite',
    'BenchmarkConfig', 
    'BenchmarkResult',
    'validate_tsa_performance',
]