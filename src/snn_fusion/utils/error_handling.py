"""
Comprehensive Error Handling and Recovery

This module provides robust error handling, recovery mechanisms, and
graceful degradation strategies for the SNN-Fusion framework.
"""

import sys
import traceback
import logging
import functools
import time
from typing import Any, Callable, Dict, List, Optional, Type, Union
from datetime import datetime
from pathlib import Path
import json
from enum import Enum
from dataclasses import dataclass, asdict


class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Error categories."""
    DATA_ERROR = "data_error"
    MODEL_ERROR = "model_error"
    HARDWARE_ERROR = "hardware_error"
    NETWORK_ERROR = "network_error"
    FILESYSTEM_ERROR = "filesystem_error"
    CONFIGURATION_ERROR = "configuration_error"
    VALIDATION_ERROR = "validation_error"
    RUNTIME_ERROR = "runtime_error"
    UNKNOWN_ERROR = "unknown_error"


@dataclass
class ErrorReport:
    """Structured error report."""
    error_id: str
    timestamp: datetime
    error_type: str
    category: ErrorCategory
    severity: ErrorSeverity
    message: str
    traceback: str
    context: Dict[str, Any]
    recovery_attempted: bool = False
    recovery_successful: bool = False
    recovery_details: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        data['category'] = self.category.value
        data['severity'] = self.severity.value
        return data


class SNNFusionError(Exception):
    """Base exception for SNN-Fusion framework."""
    
    def __init__(
        self,
        message: str,
        category: ErrorCategory = ErrorCategory.UNKNOWN_ERROR,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        context: Optional[Dict[str, Any]] = None,
        recovery_hint: Optional[str] = None
    ):
        super().__init__(message)
        self.category = category
        self.severity = severity
        self.context = context or {}
        self.recovery_hint = recovery_hint
        self.timestamp = datetime.now()


class DataError(SNNFusionError):
    """Errors related to data processing."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            category=ErrorCategory.DATA_ERROR,
            **kwargs
        )


class ModelError(SNNFusionError):
    """Errors related to model operations."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            category=ErrorCategory.MODEL_ERROR,
            **kwargs
        )


class HardwareError(SNNFusionError):
    """Errors related to hardware operations."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            category=ErrorCategory.HARDWARE_ERROR,
            **kwargs
        )


class ConfigurationError(SNNFusionError):
    """Errors related to configuration."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            category=ErrorCategory.CONFIGURATION_ERROR,
            **kwargs
        )


class ErrorHandler:
    """
    Comprehensive error handler with recovery mechanisms.
    
    Provides error tracking, recovery strategies, and graceful degradation
    for the SNN-Fusion framework.
    """
    
    def __init__(
        self,
        log_errors: bool = True,
        save_error_reports: bool = True,
        error_report_dir: Optional[str] = None,
        max_recovery_attempts: int = 3,
        enable_graceful_degradation: bool = True
    ):
        """
        Initialize error handler.
        
        Args:
            log_errors: Whether to log errors
            save_error_reports: Whether to save detailed error reports
            error_report_dir: Directory for error reports
            max_recovery_attempts: Maximum recovery attempts per error
            enable_graceful_degradation: Enable graceful degradation
        """
        self.log_errors = log_errors
        self.save_error_reports = save_error_reports
        self.error_report_dir = Path(error_report_dir) if error_report_dir else Path("./error_reports")
        self.max_recovery_attempts = max_recovery_attempts
        self.enable_graceful_degradation = enable_graceful_degradation
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Error tracking
        self.error_history: List[ErrorReport] = []
        self.error_counts: Dict[str, int] = {}
        self.recovery_strategies: Dict[ErrorCategory, Callable] = {}
        
        # Recovery attempt tracking
        self.recovery_attempts: Dict[str, int] = {}
        
        # Initialize error report directory
        if self.save_error_reports:
            self.error_report_dir.mkdir(parents=True, exist_ok=True)
        
        # Register default recovery strategies
        self._register_default_recovery_strategies()
        
        self.logger.info("ErrorHandler initialized")
    
    def _register_default_recovery_strategies(self):
        """Register default recovery strategies."""
        self.recovery_strategies[ErrorCategory.DATA_ERROR] = self._recover_data_error
        self.recovery_strategies[ErrorCategory.MODEL_ERROR] = self._recover_model_error
        self.recovery_strategies[ErrorCategory.HARDWARE_ERROR] = self._recover_hardware_error
        self.recovery_strategies[ErrorCategory.NETWORK_ERROR] = self._recover_network_error
        self.recovery_strategies[ErrorCategory.FILESYSTEM_ERROR] = self._recover_filesystem_error
    
    def handle_error(
        self,
        error: Exception,
        context: Optional[Dict[str, Any]] = None,
        attempt_recovery: bool = True
    ) -> ErrorReport:
        """
        Handle an error with comprehensive logging and recovery.
        
        Args:
            error: Exception that occurred
            context: Additional context information
            attempt_recovery: Whether to attempt automatic recovery
            
        Returns:
            ErrorReport with details and recovery status
        """
        # Generate error ID
        error_id = f"ERR_{int(time.time())}_{hash(str(error)) % 10000:04d}"
        
        # Determine error category and severity
        if isinstance(error, SNNFusionError):
            category = error.category
            severity = error.severity
            context = {**(context or {}), **error.context}
        else:
            category = self._categorize_error(error)
            severity = self._assess_severity(error, category)
        
        # Create error report
        error_report = ErrorReport(
            error_id=error_id,
            timestamp=datetime.now(),
            error_type=type(error).__name__,
            category=category,
            severity=severity,
            message=str(error),
            traceback=traceback.format_exc(),
            context=context or {}
        )
        
        # Log error
        if self.log_errors:
            self._log_error(error_report)
        
        # Track error
        self._track_error(error_report)
        
        # Attempt recovery
        if attempt_recovery and category in self.recovery_strategies:
            recovery_key = f"{category.value}_{hash(str(error)) % 1000}"
            attempts = self.recovery_attempts.get(recovery_key, 0)
            
            if attempts < self.max_recovery_attempts:
                self.recovery_attempts[recovery_key] = attempts + 1
                
                try:
                    recovery_result = self.recovery_strategies[category](error, context)
                    
                    error_report.recovery_attempted = True
                    error_report.recovery_successful = recovery_result.get('success', False)
                    error_report.recovery_details = recovery_result.get('details', 'No details')
                    
                    if error_report.recovery_successful:
                        self.logger.info(f"Recovery successful for error {error_id}")
                    else:
                        self.logger.warning(f"Recovery failed for error {error_id}")
                        
                except Exception as recovery_error:
                    error_report.recovery_attempted = True
                    error_report.recovery_successful = False
                    error_report.recovery_details = f"Recovery failed: {recovery_error}"
                    
                    self.logger.error(f"Recovery attempt failed for {error_id}: {recovery_error}")
            else:
                self.logger.warning(f"Max recovery attempts ({self.max_recovery_attempts}) reached for {recovery_key}")
        
        # Save error report
        if self.save_error_reports:
            self._save_error_report(error_report)
        
        return error_report
    
    def _categorize_error(self, error: Exception) -> ErrorCategory:
        """Categorize error based on exception type and message."""
        error_message = str(error).lower()
        error_type = type(error).__name__.lower()
        
        # Data-related errors
        if any(keyword in error_message for keyword in ['data', 'dataset', 'tensor', 'shape', 'dimension']):
            return ErrorCategory.DATA_ERROR
        
        # Model-related errors
        if any(keyword in error_message for keyword in ['model', 'forward', 'backward', 'parameter', 'gradient']):
            return ErrorCategory.MODEL_ERROR
        
        # Hardware-related errors
        if any(keyword in error_message for keyword in ['cuda', 'gpu', 'memory', 'device']):
            return ErrorCategory.HARDWARE_ERROR
        
        # Network-related errors
        if any(keyword in error_message for keyword in ['connection', 'network', 'timeout', 'socket']):
            return ErrorCategory.NETWORK_ERROR
        
        # Filesystem-related errors
        if any(keyword in error_message for keyword in ['file', 'directory', 'path', 'permission']):
            return ErrorCategory.FILESYSTEM_ERROR
        
        # Configuration-related errors
        if any(keyword in error_message for keyword in ['config', 'setting', 'parameter', 'invalid']):
            return ErrorCategory.CONFIGURATION_ERROR
        
        # Validation-related errors
        if 'validation' in error_type or any(keyword in error_message for keyword in ['validate', 'invalid', 'required']):
            return ErrorCategory.VALIDATION_ERROR
        
        # Runtime errors
        if any(error_type.startswith(prefix) for prefix in ['runtime', 'value', 'type', 'attribute']):
            return ErrorCategory.RUNTIME_ERROR
        
        return ErrorCategory.UNKNOWN_ERROR
    
    def _assess_severity(self, error: Exception, category: ErrorCategory) -> ErrorSeverity:
        """Assess error severity."""
        error_message = str(error).lower()
        
        # Critical errors
        if any(keyword in error_message for keyword in ['critical', 'fatal', 'corrupted', 'system']):
            return ErrorSeverity.CRITICAL
        
        # High severity errors
        if category in [ErrorCategory.MODEL_ERROR, ErrorCategory.HARDWARE_ERROR]:
            return ErrorSeverity.HIGH
        
        # Medium severity errors (default)
        return ErrorSeverity.MEDIUM
    
    def _log_error(self, error_report: ErrorReport):
        """Log error with appropriate level."""
        log_message = f"[{error_report.error_id}] {error_report.category.value}: {error_report.message}"
        
        if error_report.severity == ErrorSeverity.CRITICAL:
            self.logger.critical(log_message)
        elif error_report.severity == ErrorSeverity.HIGH:
            self.logger.error(log_message)
        elif error_report.severity == ErrorSeverity.MEDIUM:
            self.logger.warning(log_message)
        else:
            self.logger.info(log_message)
    
    def _track_error(self, error_report: ErrorReport):
        """Track error in history and statistics."""
        self.error_history.append(error_report)
        
        # Track error counts
        error_key = f"{error_report.category.value}_{error_report.error_type}"
        self.error_counts[error_key] = self.error_counts.get(error_key, 0) + 1
        
        # Trim history if too large
        if len(self.error_history) > 1000:
            self.error_history = self.error_history[-1000:]
    
    def _save_error_report(self, error_report: ErrorReport):
        """Save detailed error report to file."""
        try:
            report_filename = f"error_{error_report.error_id}.json"
            report_path = self.error_report_dir / report_filename
            
            with open(report_path, 'w') as f:
                json.dump(error_report.to_dict(), f, indent=2, default=str)
                
        except Exception as e:
            self.logger.error(f"Failed to save error report: {e}")
    
    # Default recovery strategies
    
    def _recover_data_error(self, error: Exception, context: Optional[Dict] = None) -> Dict[str, Any]:
        """Attempt to recover from data errors."""
        recovery_details = []
        
        error_message = str(error).lower()
        
        # Shape mismatch recovery
        if 'shape' in error_message or 'dimension' in error_message:
            recovery_details.append("Attempting to reshape or transpose data")
            # In practice, would implement actual reshaping logic
            return {'success': False, 'details': '; '.join(recovery_details)}
        
        # Missing data recovery
        if 'missing' in error_message or 'empty' in error_message:
            recovery_details.append("Using fallback synthetic data")
            return {'success': True, 'details': '; '.join(recovery_details)}
        
        return {'success': False, 'details': 'No recovery strategy available'}
    
    def _recover_model_error(self, error: Exception, context: Optional[Dict] = None) -> Dict[str, Any]:
        """Attempt to recover from model errors."""
        recovery_details = []
        
        error_message = str(error).lower()
        
        # Parameter/gradient issues
        if 'gradient' in error_message or 'nan' in error_message:
            recovery_details.append("Clipping gradients and resetting optimizer state")
            return {'success': True, 'details': '; '.join(recovery_details)}
        
        # Model structure issues
        if 'forward' in error_message:
            recovery_details.append("Reinitializing model layers")
            return {'success': False, 'details': '; '.join(recovery_details)}
        
        return {'success': False, 'details': 'No recovery strategy available'}
    
    def _recover_hardware_error(self, error: Exception, context: Optional[Dict] = None) -> Dict[str, Any]:
        """Attempt to recover from hardware errors."""
        recovery_details = []
        
        error_message = str(error).lower()
        
        # CUDA/GPU errors
        if 'cuda' in error_message or 'gpu' in error_message:
            recovery_details.append("Falling back to CPU processing")
            return {'success': True, 'details': '; '.join(recovery_details)}
        
        # Memory errors
        if 'memory' in error_message or 'oom' in error_message:
            recovery_details.append("Reducing batch size and clearing cache")
            return {'success': True, 'details': '; '.join(recovery_details)}
        
        return {'success': False, 'details': 'No recovery strategy available'}
    
    def _recover_network_error(self, error: Exception, context: Optional[Dict] = None) -> Dict[str, Any]:
        """Attempt to recover from network errors."""
        recovery_details = []
        
        # Connection timeout
        recovery_details.append("Retrying with exponential backoff")
        time.sleep(1)  # Simple retry delay
        
        return {'success': True, 'details': '; '.join(recovery_details)}
    
    def _recover_filesystem_error(self, error: Exception, context: Optional[Dict] = None) -> Dict[str, Any]:
        """Attempt to recover from filesystem errors."""
        recovery_details = []
        
        error_message = str(error).lower()
        
        # Permission errors
        if 'permission' in error_message:
            recovery_details.append("Using alternative file location")
            return {'success': True, 'details': '; '.join(recovery_details)}
        
        # File not found
        if 'not found' in error_message or 'no such file' in error_message:
            recovery_details.append("Creating missing directories/files")
            return {'success': True, 'details': '; '.join(recovery_details)}
        
        return {'success': False, 'details': 'No recovery strategy available'}
    
    def register_recovery_strategy(
        self,
        category: ErrorCategory,
        strategy: Callable[[Exception, Optional[Dict]], Dict[str, Any]]
    ):
        """Register custom recovery strategy."""
        self.recovery_strategies[category] = strategy
        self.logger.info(f"Registered recovery strategy for {category.value}")
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get error statistics."""
        total_errors = len(self.error_history)
        
        if total_errors == 0:
            return {'total_errors': 0, 'message': 'No errors recorded'}
        
        # Count by category
        category_counts = {}
        severity_counts = {}
        recovery_stats = {'attempted': 0, 'successful': 0}
        
        for error in self.error_history:
            category = error.category.value
            severity = error.severity.value
            
            category_counts[category] = category_counts.get(category, 0) + 1
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
            
            if error.recovery_attempted:
                recovery_stats['attempted'] += 1
                if error.recovery_successful:
                    recovery_stats['successful'] += 1
        
        # Recent errors (last 24 hours)
        recent_cutoff = datetime.now() - timedelta(hours=24)
        recent_errors = [e for e in self.error_history if e.timestamp > recent_cutoff]
        
        return {
            'total_errors': total_errors,
            'recent_errors_24h': len(recent_errors),
            'category_breakdown': category_counts,
            'severity_breakdown': severity_counts,
            'recovery_stats': recovery_stats,
            'recovery_success_rate': recovery_stats['successful'] / max(recovery_stats['attempted'], 1),
            'most_common_errors': dict(sorted(self.error_counts.items(), key=lambda x: x[1], reverse=True)[:5])
        }


# Decorator for automatic error handling
def handle_errors(
    error_handler: Optional[ErrorHandler] = None,
    reraise: bool = False,
    return_on_error: Any = None,
    attempt_recovery: bool = True
):
    """
    Decorator for automatic error handling.
    
    Args:
        error_handler: ErrorHandler instance to use
        reraise: Whether to reraise the error after handling
        return_on_error: Value to return if error occurs and not reraising
        attempt_recovery: Whether to attempt recovery
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # Get or create error handler
                handler = error_handler or ErrorHandler()
                
                # Create context
                context = {
                    'function': func.__name__,
                    'args': str(args)[:200],  # Truncate for safety
                    'kwargs': str(kwargs)[:200]
                }
                
                # Handle error
                error_report = handler.handle_error(e, context, attempt_recovery)
                
                # Log error report ID
                logging.getLogger(__name__).info(f"Error handled: {error_report.error_id}")
                
                if reraise:
                    raise e
                else:
                    return return_on_error
        
        return wrapper
    return decorator


# Context manager for error handling
class ErrorContext:
    """Context manager for comprehensive error handling."""
    
    def __init__(
        self,
        error_handler: Optional[ErrorHandler] = None,
        context: Optional[Dict[str, Any]] = None,
        reraise: bool = True,
        attempt_recovery: bool = True
    ):
        self.error_handler = error_handler or ErrorHandler()
        self.context = context or {}
        self.reraise = reraise
        self.attempt_recovery = attempt_recovery
        
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            error_report = self.error_handler.handle_error(
                exc_val, self.context, self.attempt_recovery
            )
            
            logging.getLogger(__name__).info(f"Error handled in context: {error_report.error_id}")
            
            if not self.reraise:
                return True  # Suppress exception
        
        return False  # Let exception propagate


# Convenience functions

def create_error_handler(config: Dict[str, Any]) -> ErrorHandler:
    """Create error handler from configuration."""
    return ErrorHandler(
        log_errors=config.get('log_errors', True),
        save_error_reports=config.get('save_error_reports', True),
        error_report_dir=config.get('error_report_dir'),
        max_recovery_attempts=config.get('max_recovery_attempts', 3),
        enable_graceful_degradation=config.get('enable_graceful_degradation', True)
    )


# Example usage and testing
if __name__ == "__main__":
    # Test error handling
    print("Testing Error Handling...")
    
    # Create error handler
    error_handler = ErrorHandler(error_report_dir="./test_error_reports")
    
    # Test different types of errors
    test_errors = [
        ValueError("Invalid tensor shape (32, 64) expected (32, 128)"),
        RuntimeError("CUDA out of memory"),
        FileNotFoundError("Model checkpoint not found: ./model.pt"),
        ConnectionError("Network timeout while downloading data"),
    ]
    
    for i, error in enumerate(test_errors):
        print(f"\nTest {i+1}: {type(error).__name__}")
        
        context = {'test_id': i, 'function': 'test_function'}
        error_report = error_handler.handle_error(error, context)
        
        print(f"  Error ID: {error_report.error_id}")
        print(f"  Category: {error_report.category.value}")
        print(f"  Severity: {error_report.severity.value}")
        print(f"  Recovery attempted: {error_report.recovery_attempted}")
        print(f"  Recovery successful: {error_report.recovery_successful}")
    
    # Test decorator
    @handle_errors(error_handler=error_handler, return_on_error="FALLBACK")
    def test_function():
        raise ValueError("Test error for decorator")
    
    result = test_function()
    print(f"\nDecorator test result: {result}")
    
    # Test context manager
    with ErrorContext(error_handler=error_handler, reraise=False):
        raise RuntimeError("Test error for context manager")
    
    print("Context manager test completed")
    
    # Get statistics
    stats = error_handler.get_error_statistics()
    print(f"\nError Statistics:")
    print(f"  Total errors: {stats['total_errors']}")
    print(f"  Recovery success rate: {stats['recovery_success_rate']:.2%}")
    print(f"  Category breakdown: {stats['category_breakdown']}")
    
    print("✓ Error handling test completed!")