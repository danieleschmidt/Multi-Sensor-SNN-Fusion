"""
Logging Utilities for SNN-Fusion

This module provides comprehensive logging functionality for the multi-modal
spiking neural network framework, including performance monitoring, error tracking,
and structured logging for debugging and production monitoring.
"""

import logging
import sys
import json
import time
import traceback
import threading
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Callable
from datetime import datetime
from enum import Enum
import os


class LogLevel(Enum):
    """Log levels for SNN-Fusion."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class StructuredFormatter(logging.Formatter):
    """
    Custom formatter for structured logging.
    
    Outputs logs in JSON format with consistent structure for easy parsing
    and analysis in log aggregation systems.
    """
    
    def __init__(self, include_extra: bool = True):
        super().__init__()
        self.include_extra = include_extra
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as structured JSON."""
        log_data = {
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
        }
        
        # Add process/thread info
        log_data['process_id'] = os.getpid()
        log_data['thread_id'] = threading.get_ident()
        
        # Add exception info if present
        if record.exc_info:
            log_data['exception'] = {
                'type': record.exc_info[0].__name__,
                'message': str(record.exc_info[1]),
                'traceback': traceback.format_exception(*record.exc_info)
            }
        
        # Add extra fields
        if self.include_extra:
            extra_fields = {}
            for key, value in record.__dict__.items():
                if key not in ['name', 'msg', 'args', 'levelname', 'levelno', 
                              'pathname', 'filename', 'module', 'lineno', 
                              'funcName', 'created', 'msecs', 'relativeCreated',
                              'thread', 'threadName', 'processName', 'process',
                              'exc_info', 'exc_text', 'stack_info', 'message']:
                    extra_fields[key] = value
            
            if extra_fields:
                log_data['extra'] = extra_fields
        
        return json.dumps(log_data, default=str, ensure_ascii=False)


class PerformanceLogger:
    """
    Logger for performance metrics and timing information.
    """
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.timers = {}
        self.counters = {}
        self.metrics = {}
    
    def start_timer(self, name: str) -> None:
        """Start a named timer."""
        self.timers[name] = time.time()
    
    def end_timer(self, name: str) -> float:
        """End a named timer and log the duration."""
        if name not in self.timers:
            self.logger.warning(f"Timer '{name}' was not started")
            return 0.0
        
        duration = time.time() - self.timers[name]
        del self.timers[name]
        
        self.logger.info(
            f"Timer completed: {name}",
            extra={
                'timer_name': name,
                'duration_seconds': duration,
                'metric_type': 'timer'
            }
        )
        
        return duration
    
    def time_function(self, func: Callable, *args, **kwargs) -> tuple:
        """Time a function call and return (result, duration)."""
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            duration = time.time() - start_time
            
            self.logger.info(
                f"Function timed: {func.__name__}",
                extra={
                    'function_name': func.__name__,
                    'duration_seconds': duration,
                    'success': True,
                    'metric_type': 'function_timer'
                }
            )
            
            return result, duration
        
        except Exception as e:
            duration = time.time() - start_time
            
            self.logger.error(
                f"Function failed: {func.__name__}",
                extra={
                    'function_name': func.__name__,
                    'duration_seconds': duration,
                    'success': False,
                    'error': str(e),
                    'metric_type': 'function_timer'
                },
                exc_info=True
            )
            
            raise
    
    def increment_counter(self, name: str, value: int = 1) -> None:
        """Increment a named counter."""
        self.counters[name] = self.counters.get(name, 0) + value
        
        self.logger.debug(
            f"Counter incremented: {name}",
            extra={
                'counter_name': name,
                'increment': value,
                'total': self.counters[name],
                'metric_type': 'counter'
            }
        )
    
    def record_metric(self, name: str, value: float, unit: str = None) -> None:
        """Record a named metric value."""
        self.metrics[name] = value
        
        log_extra = {
            'metric_name': name,
            'value': value,
            'metric_type': 'gauge'
        }
        
        if unit:
            log_extra['unit'] = unit
        
        self.logger.info(
            f"Metric recorded: {name} = {value}",
            extra=log_extra
        )
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of all metrics."""
        return {
            'counters': self.counters.copy(),
            'metrics': self.metrics.copy(),
            'active_timers': list(self.timers.keys())
        }


def setup_logging(
    log_level: Union[str, LogLevel] = LogLevel.INFO,
    log_file: Optional[Union[str, Path]] = None,
    enable_console: bool = True,
    structured_format: bool = False,
    max_file_size: int = 100 * 1024 * 1024,  # 100MB
    backup_count: int = 5
) -> logging.Logger:
    """
    Set up comprehensive logging for SNN-Fusion.
    
    Args:
        log_level: Logging level
        log_file: Optional log file path
        enable_console: Whether to enable console logging
        structured_format: Whether to use structured JSON logging
        max_file_size: Maximum size of log file before rotation
        backup_count: Number of backup files to keep
        
    Returns:
        Configured logger instance
    """
    # Convert log level
    if isinstance(log_level, LogLevel):
        log_level = log_level.value
    
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    
    # Create main logger
    logger = logging.getLogger('snn_fusion')
    logger.setLevel(numeric_level)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Choose formatter
    if structured_format:
        formatter = StructuredFormatter()
    else:
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    
    # Console handler
    if enable_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(numeric_level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # File handler with rotation
    if log_file:
        from logging.handlers import RotatingFileHandler
        
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = RotatingFileHandler(
            log_path,
            maxBytes=max_file_size,
            backupCount=backup_count,
            encoding='utf-8'
        )
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    # Log initial setup message
    logger.info(
        "Logging initialized",
        extra={
            'log_level': log_level,
            'log_file': str(log_file) if log_file else None,
            'structured_format': structured_format,
            'console_enabled': enable_console
        }
    )
    
    return logger


def get_logger(name: str = None) -> logging.Logger:
    """
    Get logger instance for a specific module.
    
    Args:
        name: Logger name (defaults to calling module)
        
    Returns:
        Logger instance
    """
    if name is None:
        # Get calling module name
        frame = sys._getframe(1)
        name = frame.f_globals.get('__name__', 'unknown')
    
    # Ensure it's under the snn_fusion hierarchy
    if not name.startswith('snn_fusion'):
        name = f'snn_fusion.{name}'
    
    return logging.getLogger(name)


def log_performance_metrics(
    logger: logging.Logger,
    metrics: Dict[str, Any],
    context: str = "performance"
) -> None:
    """
    Log performance metrics in a structured format.
    
    Args:
        logger: Logger instance
        metrics: Dictionary of metrics to log
        context: Context string for the metrics
    """
    logger.info(
        f"Performance metrics: {context}",
        extra={
            'metrics': metrics,
            'context': context,
            'metric_type': 'performance_summary'
        }
    )


def log_model_info(
    logger: logging.Logger,
    model: Any,
    model_name: str = "model"
) -> None:
    """
    Log detailed model information.
    
    Args:
        logger: Logger instance
        model: Model instance
        model_name: Name of the model
    """
    try:
        # Count parameters
        if hasattr(model, 'parameters'):
            param_count = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        else:
            param_count = 0
            trainable_params = 0
        
        # Get model info
        model_info = {
            'model_name': model_name,
            'model_type': type(model).__name__,
            'total_parameters': param_count,
            'trainable_parameters': trainable_params,
            'model_size_mb': param_count * 4 / (1024 * 1024),  # Assuming 4-byte floats
        }
        
        # Add device info if available
        if hasattr(model, 'device'):
            model_info['device'] = str(model.device)
        
        logger.info(
            f"Model information: {model_name}",
            extra={
                'model_info': model_info,
                'metric_type': 'model_info'
            }
        )
    
    except Exception as e:
        logger.warning(
            f"Could not log model info for {model_name}: {e}",
            exc_info=True
        )


def log_dataset_info(
    logger: logging.Logger,
    dataset: Any,
    dataset_name: str = "dataset"
) -> None:
    """
    Log dataset information.
    
    Args:
        logger: Logger instance
        dataset: Dataset instance
        dataset_name: Name of the dataset
    """
    try:
        dataset_info = {
            'dataset_name': dataset_name,
            'dataset_type': type(dataset).__name__,
        }
        
        # Add size if available
        if hasattr(dataset, '__len__'):
            dataset_info['size'] = len(dataset)
        
        # Add configuration if available
        if hasattr(dataset, 'config'):
            config = dataset.config
            if hasattr(config, '__dict__'):
                dataset_info['config'] = {
                    k: v for k, v in config.__dict__.items()
                    if not k.startswith('_')
                }
        
        # Add modalities if available
        if hasattr(dataset, 'modalities'):
            dataset_info['modalities'] = dataset.modalities
        
        logger.info(
            f"Dataset information: {dataset_name}",
            extra={
                'dataset_info': dataset_info,
                'metric_type': 'dataset_info'
            }
        )
    
    except Exception as e:
        logger.warning(
            f"Could not log dataset info for {dataset_name}: {e}",
            exc_info=True
        )


def log_training_progress(
    logger: logging.Logger,
    epoch: int,
    batch_idx: int,
    loss: float,
    accuracy: Optional[float] = None,
    learning_rate: Optional[float] = None,
    **kwargs
) -> None:
    """
    Log training progress information.
    
    Args:
        logger: Logger instance
        epoch: Current epoch
        batch_idx: Current batch index
        loss: Current loss value
        accuracy: Current accuracy (optional)
        learning_rate: Current learning rate (optional)
        **kwargs: Additional metrics
    """
    progress_info = {
        'epoch': epoch,
        'batch_idx': batch_idx,
        'loss': loss,
        'metric_type': 'training_progress'
    }
    
    if accuracy is not None:
        progress_info['accuracy'] = accuracy
    
    if learning_rate is not None:
        progress_info['learning_rate'] = learning_rate
    
    # Add any additional metrics
    progress_info.update(kwargs)
    
    logger.info(
        f"Training progress - Epoch {epoch}, Batch {batch_idx}, Loss {loss:.6f}",
        extra=progress_info
    )


class TimerContext:
    """Context manager for timing code blocks."""
    
    def __init__(self, logger: logging.Logger, name: str, log_level: int = logging.INFO):
        self.logger = logger
        self.name = name
        self.log_level = log_level
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        self.logger.log(self.log_level, f"Starting timer: {self.name}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self.start_time
        
        if exc_type is None:
            self.logger.log(
                self.log_level,
                f"Timer completed: {self.name} ({duration:.3f}s)",
                extra={
                    'timer_name': self.name,
                    'duration_seconds': duration,
                    'success': True,
                    'metric_type': 'timer'
                }
            )
        else:
            self.logger.error(
                f"Timer failed: {self.name} ({duration:.3f}s)",
                extra={
                    'timer_name': self.name,
                    'duration_seconds': duration,
                    'success': False,
                    'exception_type': exc_type.__name__ if exc_type else None,
                    'metric_type': 'timer'
                },
                exc_info=True
            )


def timer(name: str = None, logger: logging.Logger = None):
    """
    Decorator for timing function execution.
    
    Args:
        name: Timer name (defaults to function name)
        logger: Logger instance (defaults to module logger)
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            timer_name = name or func.__name__
            timer_logger = logger or get_logger(func.__module__)
            
            with TimerContext(timer_logger, timer_name):
                return func(*args, **kwargs)
        
        return wrapper
    return decorator


# Error tracking utilities

def log_exception(
    logger: logging.Logger,
    exception: Exception,
    context: str = None,
    **extra_data
) -> None:
    """
    Log exception with full context information.
    
    Args:
        logger: Logger instance
        exception: Exception to log
        context: Additional context string
        **extra_data: Additional data to include
    """
    error_info = {
        'exception_type': type(exception).__name__,
        'exception_message': str(exception),
        'metric_type': 'error'
    }
    
    if context:
        error_info['context'] = context
    
    error_info.update(extra_data)
    
    message = f"Exception occurred: {type(exception).__name__}"
    if context:
        message += f" in {context}"
    
    logger.error(message, extra=error_info, exc_info=True)


# Example usage and testing
if __name__ == "__main__":
    print("Testing logging functionality...")
    
    # Set up logging
    logger = setup_logging(
        log_level=LogLevel.DEBUG,
        log_file="test_snn_fusion.log",
        structured_format=True
    )
    
    # Test basic logging
    logger.info("Test info message")
    logger.warning("Test warning message")
    logger.error("Test error message")
    
    # Test performance logging
    perf_logger = PerformanceLogger(logger)
    
    perf_logger.start_timer("test_operation")
    time.sleep(0.1)
    duration = perf_logger.end_timer("test_operation")
    
    perf_logger.increment_counter("test_counter", 5)
    perf_logger.record_metric("test_metric", 42.5, "units")
    
    # Test timer context
    with TimerContext(logger, "context_test"):
        time.sleep(0.05)
    
    # Test timer decorator
    @timer("decorated_function", logger)
    def test_function():
        time.sleep(0.02)
        return "result"
    
    result = test_function()
    
    # Test exception logging
    try:
        raise ValueError("Test exception")
    except ValueError as e:
        log_exception(logger, e, "testing", test_data="example")
    
    print("Logging tests completed!")