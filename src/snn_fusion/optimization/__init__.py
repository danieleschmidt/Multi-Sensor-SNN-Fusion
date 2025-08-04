"""
Optimization Module for SNN-Fusion

This module provides performance optimization utilities for multi-modal
spiking neural networks, including memory optimization, computation
acceleration, and distributed training support.
"""

from .memory import (
    MemoryOptimizer,
    GradientCheckpointing,
    ModelSharding,
    optimize_memory_usage
)
from .compute import (
    ComputeOptimizer,
    TensorOptimizer,
    BatchProcessor,
    ParallelProcessor
)
from .distributed import (
    DistributedTrainer,
    DataParallel,
    ModelParallel,
    setup_distributed_training
)
from .profiling import (
    ProfilerContext,
    ModelProfiler,
    MemoryProfiler,
    profile_model_performance
)

__all__ = [
    # Memory optimization
    "MemoryOptimizer",
    "GradientCheckpointing", 
    "ModelSharding",
    "optimize_memory_usage",
    
    # Compute optimization
    "ComputeOptimizer",
    "TensorOptimizer",
    "BatchProcessor",
    "ParallelProcessor",
    
    # Distributed training
    "DistributedTrainer",
    "DataParallel",
    "ModelParallel", 
    "setup_distributed_training",
    
    # Profiling
    "ProfilerContext",
    "ModelProfiler",
    "MemoryProfiler",
    "profile_model_performance",
]