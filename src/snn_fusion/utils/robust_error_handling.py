"""
Robust Error Handling for SNN-Fusion

Implements comprehensive error handling, recovery mechanisms, and graceful degradation
for neuromorphic computing systems.
"""

import torch
import torch.nn as nn
import functools
import traceback
import logging
import time
import threading
from typing import Any, Callable, Optional, Dict, List, Union, Type
from contextlib import contextmanager
from enum import Enum


class ErrorSeverity(Enum):
    """Error severity levels for prioritized handling."""
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    INFO = "INFO"


class RecoveryStrategy(Enum):
    """Recovery strategies for different error types."""
    RETRY = "RETRY"
    FALLBACK = "FALLBACK"
    GRACEFUL_DEGRADATION = "GRACEFUL_DEGRADATION"
    CIRCUIT_BREAKER = "CIRCUIT_BREAKER"
    FAIL_FAST = "FAIL_FAST"


class SNNFusionError(Exception):
    """Base exception for SNN-Fusion framework."""
    
    def __init__(
        self,
        message: str,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        recovery_strategy: RecoveryStrategy = RecoveryStrategy.RETRY,
        context: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message)
        self.message = message
        self.severity = severity
        self.recovery_strategy = recovery_strategy
        self.context = context or {}
        self.timestamp = time.time()


def retry_with_exponential_backoff(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    backoff_factor: float = 2.0,
    exceptions: tuple = (Exception,)
):
    """Decorator for retrying operations with exponential backoff."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    
                    if attempt == max_retries:
                        break
                    
                    delay = min(base_delay * (backoff_factor ** attempt), max_delay)
                    time.sleep(delay)
            
            raise SNNFusionError(
                f"All {max_retries} retry attempts failed for {func.__name__}",
                severity=ErrorSeverity.HIGH,
                recovery_strategy=RecoveryStrategy.FALLBACK,
                context={'last_exception': str(last_exception)}
            )
        
        return wrapper
    return decorator


class InputValidator:
    """Comprehensive input validation for neuromorphic data processing."""
    
    @staticmethod
    def validate_tensor(
        tensor: torch.Tensor,
        expected_shape: Optional[tuple] = None,
        expected_dtype: Optional[torch.dtype] = None,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
        allow_nan: bool = False,
        allow_inf: bool = False,
        name: str = "tensor"
    ) -> torch.Tensor:
        """Validate tensor properties and values."""
        
        if not isinstance(tensor, torch.Tensor):
            raise ValueError(f"{name} must be a torch.Tensor, got {type(tensor)}")
        
        if expected_shape is not None and tensor.shape != expected_shape:
            if len(expected_shape) > 0 and expected_shape[0] == -1:
                expected_shape = (tensor.shape[0],) + expected_shape[1:]
            
            if tensor.shape != expected_shape:
                raise ValueError(f"{name} shape mismatch: expected {expected_shape}, got {tensor.shape}")
        
        if expected_dtype is not None and tensor.dtype != expected_dtype:
            logging.warning(f"{name} dtype mismatch: expected {expected_dtype}, got {tensor.dtype}")
        
        if not allow_nan and torch.isnan(tensor).any():
            raise ValueError(f"{name} contains NaN values")
        
        if not allow_inf and torch.isinf(tensor).any():
            raise ValueError(f"{name} contains infinite values")
        
        if min_value is not None and tensor.min().item() < min_value:
            logging.warning(f"{name} contains values below minimum {min_value}")
        
        if max_value is not None and tensor.max().item() > max_value:
            logging.warning(f"{name} contains values above maximum {max_value}")
        
        return tensor
    
    @staticmethod
    def validate_spike_data(
        spikes: torch.Tensor,
        time_steps: Optional[int] = None,
        num_neurons: Optional[int] = None,
        name: str = "spikes"
    ) -> torch.Tensor:
        """Validate spike train data."""
        
        spikes = InputValidator.validate_tensor(
            spikes,
            expected_dtype=torch.float32,
            min_value=0.0,
            max_value=1.0,
            name=name
        )
        
        if spikes.dim() < 2:
            raise ValueError(f"{name} must have at least 2 dimensions (neurons, time)")
        
        if time_steps is not None and spikes.shape[-1] != time_steps:
            logging.warning(f"{name} time dimension mismatch: expected {time_steps}, got {spikes.shape[-1]}")
        
        return spikes


@contextmanager
def error_context(operation_name: str, fallback_value: Any = None):
    """Context manager for handling errors with fallback."""
    try:
        yield
    except Exception as e:
        logging.error(f"Error in {operation_name}: {str(e)}")
        if fallback_value is not None:
            return fallback_value
        else:
            raise


class SecurityValidator:
    """Security validation for neuromorphic computing systems."""
    
    @staticmethod
    def validate_input_safety(data: torch.Tensor, max_memory_gb: float = 4.0) -> bool:
        """Validate input data for memory safety."""
        memory_bytes = data.numel() * data.element_size()
        memory_gb = memory_bytes / (1024 ** 3)
        
        if memory_gb > max_memory_gb:
            raise ValueError(f"Input data too large: {memory_gb:.2f}GB > {max_memory_gb}GB")
        
        return True
    
    @staticmethod
    def sanitize_file_path(path: str) -> str:
        """Sanitize file paths to prevent directory traversal."""
        import os
        path = os.path.normpath(path)
        if '..' in path or path.startswith('/'):
            raise ValueError(f"Unsafe file path: {path}")
        return path


class PerformanceMonitor:
    """Monitor system performance and resource usage."""
    
    def __init__(self):
        self.metrics = {}
        self.start_time = time.time()
    
    def start_operation(self, name: str):
        """Start timing an operation."""
        self.metrics[name] = {'start': time.time()}
    
    def end_operation(self, name: str):
        """End timing an operation."""
        if name in self.metrics:
            self.metrics[name]['duration'] = time.time() - self.metrics[name]['start']
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage."""
        if torch.cuda.is_available():
            return {
                'gpu_allocated': torch.cuda.memory_allocated() / 1024**3,
                'gpu_cached': torch.cuda.memory_reserved() / 1024**3,
            }
        return {'gpu_allocated': 0.0, 'gpu_cached': 0.0}