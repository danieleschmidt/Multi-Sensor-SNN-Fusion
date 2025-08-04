"""
Utilities Module for SNN-Fusion

This module provides utility functions, validation, logging, and error handling
for the multi-modal spiking neural network framework.
"""

from .validation import (
    validate_tensor_shape,
    validate_modality_data,
    validate_configuration,
    ValidationError
)
from .logging import (
    setup_logging,
    get_logger,
    log_performance_metrics,
    LogLevel
)
from .security import (
    sanitize_input,
    validate_file_path,
    check_resource_limits,
    SecurityError
)
from .monitoring import (
    PerformanceMonitor,
    ResourceMonitor,
    SystemMonitor
)

__all__ = [
    # Validation
    "validate_tensor_shape",
    "validate_modality_data", 
    "validate_configuration",
    "ValidationError",
    
    # Logging
    "setup_logging",
    "get_logger",
    "log_performance_metrics",
    "LogLevel",
    
    # Security
    "sanitize_input",
    "validate_file_path",
    "check_resource_limits",
    "SecurityError",
    
    # Monitoring
    "PerformanceMonitor",
    "ResourceMonitor",
    "SystemMonitor",
]