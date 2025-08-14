"""
Robust Error Handling and Recovery System

Comprehensive error handling, graceful degradation, and recovery mechanisms
for neuromorphic multi-modal fusion systems. Ensures reliability and fault
tolerance in production neuromorphic deployments.
"""

import sys
import traceback
import logging
import time
import functools
from typing import Any, Callable, Dict, List, Optional, Union, Type, Tuple
from enum import Enum
from dataclasses import dataclass, field
from contextlib import contextmanager
import threading
from pathlib import Path
import json

import numpy as np


class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Error categories for classification."""
    DATA_CORRUPTION = "data_corruption"
    HARDWARE_FAILURE = "hardware_failure"
    MEMORY_ERROR = "memory_error"
    NETWORK_ERROR = "network_error"
    ALGORITHM_ERROR = "algorithm_error"
    VALIDATION_ERROR = "validation_error"
    TIMEOUT_ERROR = "timeout_error"
    RESOURCE_ERROR = "resource_error"


@dataclass
class ErrorReport:
    """Comprehensive error report structure."""
    timestamp: float
    error_type: str
    error_message: str
    severity: ErrorSeverity
    category: ErrorCategory
    function_name: str
    module_name: str
    traceback_info: str
    context_data: Dict[str, Any] = field(default_factory=dict)
    recovery_attempted: bool = False
    recovery_successful: bool = False
    recovery_strategy: Optional[str] = None
    impact_assessment: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'timestamp': self.timestamp,
            'error_type': self.error_type,
            'error_message': self.error_message,
            'severity': self.severity.value,
            'category': self.category.value,
            'function_name': self.function_name,
            'module_name': self.module_name,
            'traceback_info': self.traceback_info,
            'context_data': self.context_data,
            'recovery_attempted': self.recovery_attempted,
            'recovery_successful': self.recovery_successful,
            'recovery_strategy': self.recovery_strategy,
            'impact_assessment': self.impact_assessment,
        }


class RecoveryStrategy:
    """Base class for recovery strategies."""
    
    def __init__(self, name: str, max_retries: int = 3):
        self.name = name
        self.max_retries = max_retries
        self.retry_count = 0
        
    def can_recover(self, error: Exception, context: Dict[str, Any]) -> bool:
        """Check if this strategy can handle the error."""
        return True
    
    def attempt_recovery(self, error: Exception, context: Dict[str, Any]) -> bool:
        """Attempt recovery. Returns True if successful."""
        self.retry_count += 1
        return False
    
    def reset(self) -> None:
        """Reset recovery state."""
        self.retry_count = 0


class RetryRecoveryStrategy(RecoveryStrategy):
    """Simple retry strategy with exponential backoff."""
    
    def __init__(self, max_retries: int = 3, base_delay: float = 0.1):
        super().__init__("retry", max_retries)
        self.base_delay = base_delay
        
    def attempt_recovery(self, error: Exception, context: Dict[str, Any]) -> bool:
        """Attempt recovery with exponential backoff."""
        if self.retry_count >= self.max_retries:
            return False
            
        # Exponential backoff
        delay = self.base_delay * (2 ** self.retry_count)
        time.sleep(delay)
        
        self.retry_count += 1
        return True


class FallbackRecoveryStrategy(RecoveryStrategy):
    """Fallback to alternative implementation."""
    
    def __init__(self, fallback_function: Callable):
        super().__init__("fallback", max_retries=1)
        self.fallback_function = fallback_function
        
    def attempt_recovery(self, error: Exception, context: Dict[str, Any]) -> bool:
        """Attempt fallback recovery."""
        if self.retry_count >= self.max_retries:
            return False
            
        try:
            # Try fallback function
            args = context.get('args', ())
            kwargs = context.get('kwargs', {})
            result = self.fallback_function(*args, **kwargs)
            
            # Store result for retrieval
            context['recovery_result'] = result
            self.retry_count += 1
            return True
            
        except Exception:
            return False


class GracefulDegradationStrategy(RecoveryStrategy):
    """Graceful degradation with reduced functionality."""
    
    def __init__(self, degraded_mode_function: Callable):
        super().__init__("graceful_degradation", max_retries=1)
        self.degraded_mode_function = degraded_mode_function
        
    def attempt_recovery(self, error: Exception, context: Dict[str, Any]) -> bool:
        """Attempt graceful degradation."""
        if self.retry_count >= self.max_retries:
            return False
            
        try:
            args = context.get('args', ())
            kwargs = context.get('kwargs', {})
            
            # Run in degraded mode
            result = self.degraded_mode_function(*args, **kwargs)
            context['recovery_result'] = result
            context['degraded_mode'] = True
            
            self.retry_count += 1
            return True
            
        except Exception:
            return False


class RobustErrorHandler:
    """
    Comprehensive error handling system with recovery strategies.
    
    Features:
    - Automatic error classification and reporting
    - Multiple recovery strategies
    - Error analytics and monitoring
    - Graceful degradation capabilities
    - Production-ready logging and alerting
    """
    
    def __init__(
        self,
        log_file: Optional[str] = None,
        enable_analytics: bool = True,
        max_error_history: int = 1000,
    ):
        """
        Initialize robust error handler.
        
        Args:
            log_file: Optional log file path
            enable_analytics: Enable error analytics
            max_error_history: Maximum errors to keep in history
        """
        # Setup logging
        self.logger = self._setup_logging(log_file)
        
        # Error tracking
        self.error_history: List[ErrorReport] = []
        self.max_error_history = max_error_history
        self.enable_analytics = enable_analytics
        
        # Recovery strategies
        self.recovery_strategies: Dict[str, List[RecoveryStrategy]] = {}
        self.default_strategies = [
            RetryRecoveryStrategy(max_retries=3),
        ]
        
        # Error statistics
        self.error_stats = {
            'total_errors': 0,
            'recovered_errors': 0,
            'critical_errors': 0,
            'error_categories': {cat.value: 0 for cat in ErrorCategory},
            'error_severities': {sev.value: 0 for sev in ErrorSeverity},
        }
        
        # Thread safety
        self._lock = threading.RLock()
        
        self.logger.info("RobustErrorHandler initialized")
    
    def _setup_logging(self, log_file: Optional[str]) -> logging.Logger:
        """Setup comprehensive logging."""
        logger = logging.getLogger(f"{__name__}.RobustErrorHandler")
        logger.setLevel(logging.INFO)
        
        # Clear existing handlers
        logger.handlers.clear()
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
        
        # File handler (if specified)
        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
            )
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)
        
        return logger
    
    def register_recovery_strategy(
        self,
        function_name: str,
        strategy: RecoveryStrategy,
    ) -> None:
        """Register recovery strategy for specific function."""
        with self._lock:
            if function_name not in self.recovery_strategies:
                self.recovery_strategies[function_name] = []
            self.recovery_strategies[function_name].append(strategy)
        
        self.logger.info(f"Registered recovery strategy '{strategy.name}' for {function_name}")
    
    def classify_error(self, error: Exception, context: Dict[str, Any]) -> Tuple[ErrorSeverity, ErrorCategory]:
        """Classify error by severity and category."""
        error_type = type(error).__name__
        error_msg = str(error).lower()
        
        # Classify by category
        category = ErrorCategory.ALGORITHM_ERROR  # Default
        
        if any(term in error_msg for term in ['memory', 'out of memory', 'memoryerror']):
            category = ErrorCategory.MEMORY_ERROR
        elif any(term in error_msg for term in ['timeout', 'time', 'deadline']):
            category = ErrorCategory.TIMEOUT_ERROR
        elif any(term in error_msg for term in ['network', 'connection', 'socket']):
            category = ErrorCategory.NETWORK_ERROR
        elif any(term in error_msg for term in ['hardware', 'device', 'cuda', 'gpu']):
            category = ErrorCategory.HARDWARE_FAILURE
        elif any(term in error_msg for term in ['validation', 'invalid', 'assert']):
            category = ErrorCategory.VALIDATION_ERROR
        elif any(term in error_msg for term in ['resource', 'file', 'permission']):
            category = ErrorCategory.RESOURCE_ERROR
        elif any(term in error_msg for term in ['corrupt', 'checksum', 'integrity']):
            category = ErrorCategory.DATA_CORRUPTION
        
        # Classify by severity
        severity = ErrorSeverity.MEDIUM  # Default
        
        if error_type in ['SystemExit', 'KeyboardInterrupt']:
            severity = ErrorSeverity.CRITICAL
        elif error_type in ['MemoryError', 'RuntimeError'] or 'critical' in error_msg:
            severity = ErrorSeverity.HIGH
        elif error_type in ['AssertionError', 'ValueError', 'TypeError']:
            severity = ErrorSeverity.MEDIUM
        elif error_type in ['Warning', 'UserWarning']:
            severity = ErrorSeverity.LOW
        
        # Context-based severity adjustment
        if context.get('critical_path', False):
            if severity == ErrorSeverity.LOW:
                severity = ErrorSeverity.MEDIUM
            elif severity == ErrorSeverity.MEDIUM:
                severity = ErrorSeverity.HIGH
        
        return severity, category
    
    def create_error_report(
        self,
        error: Exception,
        function_name: str,
        context: Dict[str, Any],
    ) -> ErrorReport:
        """Create comprehensive error report."""
        severity, category = self.classify_error(error, context)
        
        # Get traceback information
        tb_lines = traceback.format_exception(type(error), error, error.__traceback__)
        traceback_info = ''.join(tb_lines)
        
        # Get module information
        frame = sys._getframe(2)  # Go up to calling function
        module_name = frame.f_globals.get('__name__', 'unknown')
        
        report = ErrorReport(
            timestamp=time.time(),
            error_type=type(error).__name__,
            error_message=str(error),
            severity=severity,
            category=category,
            function_name=function_name,
            module_name=module_name,
            traceback_info=traceback_info,
            context_data=context.copy(),
        )
        
        return report
    
    def attempt_recovery(
        self,
        error: Exception,
        function_name: str,
        context: Dict[str, Any],
    ) -> Tuple[bool, Any]:
        """
        Attempt error recovery using registered strategies.
        
        Returns:
            (success, result) tuple
        """
        # Get recovery strategies for this function
        strategies = self.recovery_strategies.get(function_name, self.default_strategies)
        
        for strategy in strategies:
            if strategy.can_recover(error, context):
                self.logger.info(f"Attempting recovery with strategy: {strategy.name}")
                
                try:
                    success = strategy.attempt_recovery(error, context)
                    
                    if success:
                        self.logger.info(f"Recovery successful with strategy: {strategy.name}")
                        result = context.get('recovery_result')
                        return True, result
                        
                except Exception as recovery_error:
                    self.logger.warning(f"Recovery strategy {strategy.name} failed: {recovery_error}")
                    continue
        
        self.logger.error(f"All recovery strategies failed for {function_name}")
        return False, None
    
    def handle_error(
        self,
        error: Exception,
        function_name: str,
        context: Optional[Dict[str, Any]] = None,
        attempt_recovery: bool = True,
    ) -> Tuple[bool, Any]:
        """
        Handle error with comprehensive reporting and recovery.
        
        Args:
            error: The exception to handle
            function_name: Name of the function where error occurred
            context: Additional context information
            attempt_recovery: Whether to attempt recovery
            
        Returns:
            (recovered, result) tuple
        """
        if context is None:
            context = {}
        
        with self._lock:
            # Create error report
            report = self.create_error_report(error, function_name, context)
            
            # Update statistics
            self.error_stats['total_errors'] += 1
            self.error_stats['error_categories'][report.category.value] += 1
            self.error_stats['error_severities'][report.severity.value] += 1
            
            if report.severity == ErrorSeverity.CRITICAL:
                self.error_stats['critical_errors'] += 1
            
            # Attempt recovery if enabled
            recovered = False
            result = None
            
            if attempt_recovery and report.severity != ErrorSeverity.CRITICAL:
                recovered, result = self.attempt_recovery(error, function_name, context)
                
                if recovered:
                    report.recovery_attempted = True
                    report.recovery_successful = True
                    self.error_stats['recovered_errors'] += 1
                else:
                    report.recovery_attempted = True
                    report.recovery_successful = False
            
            # Add to error history
            self.error_history.append(report)
            if len(self.error_history) > self.max_error_history:
                self.error_history.pop(0)
            
            # Log error
            log_level = self._get_log_level(report.severity)
            self.logger.log(
                log_level,
                f"Error in {function_name}: {report.error_message} "
                f"[{report.severity.value}|{report.category.value}]"
            )
            
            # Log recovery status
            if attempt_recovery:
                if recovered:
                    self.logger.info(f"Error recovered successfully in {function_name}")
                else:
                    self.logger.error(f"Error recovery failed in {function_name}")
            
            return recovered, result
    
    def _get_log_level(self, severity: ErrorSeverity) -> int:
        """Get appropriate log level for severity."""
        mapping = {
            ErrorSeverity.LOW: logging.INFO,
            ErrorSeverity.MEDIUM: logging.WARNING,
            ErrorSeverity.HIGH: logging.ERROR,
            ErrorSeverity.CRITICAL: logging.CRITICAL,
        }
        return mapping[severity]
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get comprehensive error statistics."""
        with self._lock:
            recent_errors = [
                report for report in self.error_history
                if time.time() - report.timestamp < 3600  # Last hour
            ]
            
            stats = self.error_stats.copy()
            stats.update({
                'recent_error_count': len(recent_errors),
                'recovery_rate': (
                    self.error_stats['recovered_errors'] / max(1, self.error_stats['total_errors'])
                ),
                'critical_error_rate': (
                    self.error_stats['critical_errors'] / max(1, self.error_stats['total_errors'])
                ),
                'error_history_size': len(self.error_history),
            })
            
            return stats
    
    def export_error_reports(self, filepath: str) -> None:
        """Export error reports to JSON file."""
        with self._lock:
            reports_data = [report.to_dict() for report in self.error_history]
            
            export_data = {
                'timestamp': time.time(),
                'statistics': self.get_error_statistics(),
                'error_reports': reports_data,
            }
            
            with open(filepath, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            self.logger.info(f"Error reports exported to {filepath}")
    
    @contextmanager
    def error_context(
        self,
        function_name: str,
        context: Optional[Dict[str, Any]] = None,
        attempt_recovery: bool = True,
        reraise_on_failure: bool = True,
    ):
        """Context manager for automatic error handling."""
        if context is None:
            context = {}
            
        try:
            yield context
        except Exception as error:
            recovered, result = self.handle_error(error, function_name, context, attempt_recovery)
            
            if recovered:
                # Store result in context for access
                context['_recovery_result'] = result
            elif reraise_on_failure:
                raise
            else:
                context['_error_handled'] = True


# Global error handler instance
_global_error_handler = None


def get_global_error_handler() -> RobustErrorHandler:
    """Get or create global error handler."""
    global _global_error_handler
    if _global_error_handler is None:
        _global_error_handler = RobustErrorHandler()
    return _global_error_handler


def robust_function(
    recovery_strategies: Optional[List[RecoveryStrategy]] = None,
    attempt_recovery: bool = True,
    reraise_on_failure: bool = True,
    critical_path: bool = False,
) -> Callable:
    """
    Decorator for robust function execution with error handling and recovery.
    
    Args:
        recovery_strategies: Custom recovery strategies
        attempt_recovery: Whether to attempt recovery
        reraise_on_failure: Whether to reraise on recovery failure
        critical_path: Mark as critical path (affects severity)
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            handler = get_global_error_handler()
            
            # Register recovery strategies if provided
            if recovery_strategies:
                for strategy in recovery_strategies:
                    handler.register_recovery_strategy(func.__name__, strategy)
            
            # Prepare context
            context = {
                'args': args,
                'kwargs': kwargs,
                'critical_path': critical_path,
            }
            
            try:
                return func(*args, **kwargs)
            except Exception as error:
                recovered, result = handler.handle_error(
                    error, func.__name__, context, attempt_recovery
                )
                
                if recovered:
                    return result
                elif reraise_on_failure:
                    raise
                else:
                    return None
        
        return wrapper
    return decorator


# Specialized error handling decorators
def neuromorphic_robust(func: Callable) -> Callable:
    """Specialized decorator for neuromorphic computing functions."""
    # Define neuromorphic-specific recovery strategies
    fallback_strategy = FallbackRecoveryStrategy(
        lambda *args, **kwargs: {'status': 'degraded', 'result': None}
    )
    
    return robust_function(
        recovery_strategies=[
            RetryRecoveryStrategy(max_retries=2, base_delay=0.01),  # Fast retry for real-time
            fallback_strategy,
        ],
        critical_path=True,
    )(func)


def fusion_robust(func: Callable) -> Callable:
    """Specialized decorator for multi-modal fusion functions."""
    def degraded_fusion(*args, **kwargs):
        """Degraded fusion mode - use single modality if available."""
        # Simple fallback: return first available modality data
        if args and hasattr(args[0], 'items'):
            modality_data = args[0]
            for modality, data in modality_data.items():
                return {
                    'fused_spikes': np.column_stack([data.spike_times, data.neuron_ids]),
                    'fusion_weights': {modality: 1.0},
                    'confidence_scores': {modality: 0.5},  # Reduced confidence
                    'degraded_mode': True,
                }
        return {'error': 'No fallback possible'}
    
    degradation_strategy = GracefulDegradationStrategy(degraded_fusion)
    
    return robust_function(
        recovery_strategies=[
            RetryRecoveryStrategy(max_retries=1),
            degradation_strategy,
        ],
        critical_path=True,
    )(func)