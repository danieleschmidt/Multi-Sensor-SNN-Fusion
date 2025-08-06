"""
Scaling and Load Balancing Module for SNN-Fusion

This module provides comprehensive scaling capabilities including:
- Advanced load balancing with multiple strategies
- Concurrent and parallel processing
- Distributed task execution
- Resource-aware scheduling
- Dynamic scaling and fault tolerance

Key Components:
- LoadBalancer: Distributes tasks across multiple worker nodes
- ConcurrentProcessor: Handles parallel and asynchronous processing
- SpikeProcessingPipeline: Specialized pipeline for SNN operations

Example usage:
    from snn_fusion.scaling import LoadBalancer, ConcurrentProcessor
    from snn_fusion.scaling import LoadBalancingStrategy, ProcessingMode
    
    # Create load balancer
    lb = LoadBalancer(strategy=LoadBalancingStrategy.RESOURCE_AWARE)
    
    # Create concurrent processor  
    config = ProcessingConfig(mode=ProcessingMode.HYBRID)
    processor = ConcurrentProcessor(config)
"""

from .load_balancer import (
    LoadBalancer,
    WorkerNode,
    Task,
    LoadBalancingStrategy,
    NodeStatus
)

from .concurrent_processing import (
    ConcurrentProcessor,
    SpikeProcessingPipeline,
    ProcessingTask,
    ProcessingConfig,
    ProcessingMode,
    TaskPriority
)

__all__ = [
    # Load Balancing
    'LoadBalancer',
    'WorkerNode', 
    'Task',
    'LoadBalancingStrategy',
    'NodeStatus',
    
    # Concurrent Processing
    'ConcurrentProcessor',
    'SpikeProcessingPipeline',
    'ProcessingTask',
    'ProcessingConfig',
    'ProcessingMode',
    'TaskPriority'
]