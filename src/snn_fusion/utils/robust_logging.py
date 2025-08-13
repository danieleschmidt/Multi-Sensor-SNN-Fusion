"""
Robust Logging System

Implements comprehensive logging with structured output, rotation, 
security features, and performance monitoring.
"""

import logging
import logging.handlers
import json
import os
import sys
import traceback
import threading
import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Union, List
from pathlib import Path
from enum import Enum
import hashlib
import gzip
import shutil


class LogLevel(Enum):
    """Extended log levels with security context."""
    TRACE = 5
    DEBUG = 10
    INFO = 20
    WARNING = 30
    ERROR = 40
    CRITICAL = 50
    SECURITY = 60  # Special level for security events


class LogCategory(Enum):
    """Log categories for structured logging."""
    SYSTEM = "system"
    SECURITY = "security"
    PERFORMANCE = "performance"
    MODEL = "model"
    DATA = "data"
    NETWORK = "network"
    USER = "user"
    AUDIT = "audit"


class StructuredFormatter(logging.Formatter):
    """
    Structured JSON formatter with enhanced metadata.
    """
    
    def __init__(self, include_extra: bool = True, 
                 sensitive_fields: Optional[List[str]] = None):
        super().__init__()
        self.include_extra = include_extra
        self.sensitive_fields = sensitive_fields or [
            'password', 'token', 'key', 'secret', 'api_key', 'auth'
        ]
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as structured JSON."""
        # Base log data
        log_data = {
            'timestamp': datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
            'thread': threading.current_thread().name,
            'process': os.getpid()
        }
        
        # Add exception information if present
        if record.exc_info:
            log_data['exception'] = {
                'type': record.exc_info[0].__name__,
                'message': str(record.exc_info[1]),
                'traceback': traceback.format_exception(*record.exc_info)
            }
        
        # Add extra fields if enabled
        if self.include_extra:
            extra_data = {}
            for key, value in record.__dict__.items():
                if key not in ['name', 'msg', 'args', 'levelname', 'levelno', 
                              'pathname', 'filename', 'module', 'lineno', 
                              'funcName', 'created', 'msecs', 'relativeCreated',
                              'thread', 'threadName', 'processName', 'process',
                              'exc_info', 'exc_text', 'stack_info']:
                    
                    # Filter sensitive data
                    if any(sensitive in key.lower() for sensitive in self.sensitive_fields):
                        extra_data[key] = "[REDACTED]"
                    else:
                        extra_data[key] = self._serialize_value(value)
            
            if extra_data:
                log_data['extra'] = extra_data
        
        # Add security context if available
        if hasattr(record, 'security_event'):
            log_data['security'] = record.security_event
        
        # Add performance context if available
        if hasattr(record, 'performance_data'):
            log_data['performance'] = record.performance_data
        
        return json.dumps(log_data, default=str, ensure_ascii=False)
    
    def _serialize_value(self, value: Any) -> Any:
        """Safely serialize values for JSON."""
        try:
            json.dumps(value)
            return value
        except (TypeError, ValueError):
            return str(value)


class SecurityAuditHandler(logging.Handler):
    """
    Special handler for security audit logs with integrity protection.
    """
    
    def __init__(self, filepath: str, secret_key: Optional[str] = None):
        super().__init__()
        self.filepath = Path(filepath)
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        self.secret_key = secret_key or os.environ.get('LOG_SECRET_KEY', 'default_key')
        
        # Create lock for thread safety
        self._lock = threading.Lock()
    
    def emit(self, record: logging.LogRecord):
        """Emit security log with integrity hash."""
        try:
            with self._lock:
                # Format the record
                formatted_msg = self.format(record)
                
                # Create integrity hash
                integrity_hash = hashlib.sha256(
                    (formatted_msg + self.secret_key).encode()
                ).hexdigest()
                
                # Create audit entry
                audit_entry = {
                    'log_data': json.loads(formatted_msg),
                    'integrity_hash': integrity_hash,
                    'audit_timestamp': datetime.utcnow().isoformat()
                }
                
                # Write to file
                with open(self.filepath, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(audit_entry) + '\n')
                    f.flush()
        
        except Exception:
            self.handleError(record)
    
    def verify_integrity(self, line: str) -> bool:
        """Verify integrity of a log line."""
        try:
            audit_entry = json.loads(line)
            log_data = json.dumps(audit_entry['log_data'])
            expected_hash = hashlib.sha256(
                (log_data + self.secret_key).encode()
            ).hexdigest()
            return audit_entry['integrity_hash'] == expected_hash
        except Exception:
            return False


class PerformanceLogHandler(logging.Handler):
    """
    Handler for performance metrics logging.
    """
    
    def __init__(self, filepath: str, buffer_size: int = 100):
        super().__init__()
        self.filepath = Path(filepath)
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        self.buffer_size = buffer_size
        self.buffer = []
        self._lock = threading.Lock()
        
        # Setup periodic flush
        self._setup_periodic_flush()
    
    def emit(self, record: logging.LogRecord):
        """Buffer performance logs for batch writing."""
        if hasattr(record, 'performance_data'):
            with self._lock:
                self.buffer.append({
                    'timestamp': datetime.fromtimestamp(record.created).isoformat(),
                    'logger': record.name,
                    'performance': record.performance_data
                })
                
                if len(self.buffer) >= self.buffer_size:
                    self._flush_buffer()
    
    def _flush_buffer(self):
        """Flush buffer to file."""
        if not self.buffer:
            return
        
        try:
            with open(self.filepath, 'a', encoding='utf-8') as f:
                for entry in self.buffer:
                    f.write(json.dumps(entry) + '\n')
                f.flush()
            self.buffer.clear()
        except Exception as e:
            print(f"Failed to flush performance log buffer: {e}", file=sys.stderr)
    
    def _setup_periodic_flush(self):
        """Setup periodic buffer flush."""
        def flush_periodically():
            while True:
                time.sleep(30)  # Flush every 30 seconds
                with self._lock:
                    self._flush_buffer()
        
        thread = threading.Thread(target=flush_periodically, daemon=True)
        thread.start()
    
    def close(self):
        """Close handler and flush remaining buffer."""
        with self._lock:
            self._flush_buffer()
        super().close()


class RobustLogger:
    """
    Comprehensive logging system with multiple handlers and features.
    """
    
    def __init__(self, name: str, log_dir: str = "./logs", 
                 enable_security_audit: bool = True,
                 enable_performance_logging: bool = True,
                 log_level: Union[int, str] = logging.INFO,
                 max_file_size: int = 100 * 1024 * 1024,  # 100MB
                 backup_count: int = 5,
                 compression: bool = True):
        """
        Initialize robust logger.
        
        Args:
            name: Logger name
            log_dir: Directory for log files
            enable_security_audit: Enable security audit logging
            enable_performance_logging: Enable performance logging
            log_level: Base logging level
            max_file_size: Maximum size per log file
            backup_count: Number of backup files to keep
            compression: Whether to compress rotated files
        """
        self.name = name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(log_level)
        
        # Clear existing handlers to avoid duplicates
        self.logger.handlers.clear()
        
        # Setup handlers
        self._setup_console_handler()
        self._setup_file_handler(max_file_size, backup_count, compression)
        
        if enable_security_audit:
            self._setup_security_handler()
        
        if enable_performance_logging:
            self._setup_performance_handler()
        
        # Setup custom log levels
        self._setup_custom_levels()
        
        # Performance tracking
        self.operation_times = {}
        self.operation_counts = {}
        
        self.logger.info(f"RobustLogger initialized for {name}")
    
    def _setup_console_handler(self):
        """Setup console handler with colored output."""
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        
        # Simple formatter for console
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(console_formatter)
        
        self.logger.addHandler(console_handler)
    
    def _setup_file_handler(self, max_size: int, backup_count: int, compression: bool):
        """Setup rotating file handler."""
        log_file = self.log_dir / f"{self.name}.log"
        
        # Custom rotating handler with compression
        if compression:
            file_handler = CompressingRotatingFileHandler(
                str(log_file), maxBytes=max_size, backupCount=backup_count
            )
        else:
            file_handler = logging.handlers.RotatingFileHandler(
                str(log_file), maxBytes=max_size, backupCount=backup_count
            )
        
        file_handler.setLevel(logging.DEBUG)
        
        # Structured formatter for files
        structured_formatter = StructuredFormatter()
        file_handler.setFormatter(structured_formatter)
        
        self.logger.addHandler(file_handler)
    
    def _setup_security_handler(self):
        """Setup security audit handler."""
        security_file = self.log_dir / f"{self.name}_security.log"
        security_handler = SecurityAuditHandler(str(security_file))
        security_handler.setLevel(LogLevel.SECURITY.value)
        
        # Security-specific formatter
        security_formatter = StructuredFormatter(include_extra=True)
        security_handler.setFormatter(security_formatter)
        
        self.logger.addHandler(security_handler)
    
    def _setup_performance_handler(self):
        """Setup performance logging handler."""
        performance_file = self.log_dir / f"{self.name}_performance.log"
        performance_handler = PerformanceLogHandler(str(performance_file))
        performance_handler.setLevel(logging.DEBUG)
        
        # Performance formatter
        performance_formatter = StructuredFormatter()
        performance_handler.setFormatter(performance_formatter)
        
        self.logger.addHandler(performance_handler)
    
    def _setup_custom_levels(self):
        """Setup custom log levels."""
        # Add TRACE level
        logging.addLevelName(LogLevel.TRACE.value, "TRACE")
        def trace(self, message, *args, **kwargs):
            if self.isEnabledFor(LogLevel.TRACE.value):
                self._log(LogLevel.TRACE.value, message, args, **kwargs)
        logging.Logger.trace = trace
        
        # Add SECURITY level
        logging.addLevelName(LogLevel.SECURITY.value, "SECURITY")
        def security(self, message, *args, **kwargs):
            if self.isEnabledFor(LogLevel.SECURITY.value):
                self._log(LogLevel.SECURITY.value, message, args, **kwargs)
        logging.Logger.security = security
    
    def log_security_event(self, event_type: str, message: str, 
                          severity: str = "medium", 
                          metadata: Optional[Dict] = None):
        """Log security event with special handling."""
        security_data = {
            'event_type': event_type,
            'severity': severity,
            'metadata': metadata or {}
        }
        
        self.logger.security(message, extra={'security_event': security_data})
    
    def log_performance(self, operation: str, duration: float, 
                       metadata: Optional[Dict] = None):
        """Log performance metrics."""
        # Update statistics
        if operation not in self.operation_times:
            self.operation_times[operation] = []
            self.operation_counts[operation] = 0
        
        self.operation_times[operation].append(duration)
        self.operation_counts[operation] += 1
        
        # Keep only recent measurements (last 1000)
        if len(self.operation_times[operation]) > 1000:
            self.operation_times[operation] = self.operation_times[operation][-1000:]
        
        performance_data = {
            'operation': operation,
            'duration': duration,
            'count': self.operation_counts[operation],
            'avg_duration': sum(self.operation_times[operation]) / len(self.operation_times[operation]),
            'metadata': metadata or {}
        }
        
        self.logger.debug("Performance metric recorded", 
                         extra={'performance_data': performance_data})
    
    def log_model_event(self, event_type: str, model_name: str, 
                       message: str, metadata: Optional[Dict] = None):
        """Log model-related events."""
        model_data = {
            'event_type': event_type,
            'model_name': model_name,
            'metadata': metadata or {}
        }
        
        self.logger.info(message, extra={
            'category': LogCategory.MODEL.value,
            'model_event': model_data
        })
    
    def log_data_event(self, event_type: str, dataset_name: str,
                      message: str, metadata: Optional[Dict] = None):
        """Log data-related events."""
        data_event = {
            'event_type': event_type,
            'dataset_name': dataset_name,
            'metadata': metadata or {}
        }
        
        self.logger.info(message, extra={
            'category': LogCategory.DATA.value,
            'data_event': data_event
        })
    
    def time_operation(self, operation_name: str):
        """Context manager for timing operations."""
        return OperationTimer(self, operation_name)
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary statistics."""
        summary = {}
        
        for operation, times in self.operation_times.items():
            if times:
                summary[operation] = {
                    'count': self.operation_counts[operation],
                    'avg_duration': sum(times) / len(times),
                    'min_duration': min(times),
                    'max_duration': max(times),
                    'total_duration': sum(times)
                }
        
        return summary
    
    def emergency_shutdown(self, reason: str):
        """Emergency shutdown with critical logging."""
        self.logger.critical(f"EMERGENCY SHUTDOWN: {reason}", extra={
            'emergency': True,
            'shutdown_reason': reason,
            'timestamp': datetime.utcnow().isoformat()
        })
        
        # Flush all handlers
        for handler in self.logger.handlers:
            handler.flush()
            if hasattr(handler, 'close'):
                handler.close()


class CompressingRotatingFileHandler(logging.handlers.RotatingFileHandler):
    """
    Rotating file handler that compresses old log files.
    """
    
    def doRollover(self):
        """Override to add compression."""
        if self.stream:
            self.stream.close()
            self.stream = None
        
        if self.backupCount > 0:
            for i in range(self.backupCount - 1, 0, -1):
                sfn = f"{self.baseFilename}.{i}.gz"
                dfn = f"{self.baseFilename}.{i + 1}.gz"
                if os.path.exists(sfn):
                    if os.path.exists(dfn):
                        os.remove(dfn)
                    os.rename(sfn, dfn)
            
            # Compress the current file
            dfn = f"{self.baseFilename}.1.gz"
            if os.path.exists(dfn):
                os.remove(dfn)
            
            with open(self.baseFilename, 'rb') as f_in:
                with gzip.open(dfn, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            
            os.remove(self.baseFilename)
        
        if not self.delay:
            self.stream = self._open()


class OperationTimer:
    """Context manager for timing operations."""
    
    def __init__(self, logger: RobustLogger, operation_name: str, 
                 metadata: Optional[Dict] = None):
        self.logger = logger
        self.operation_name = operation_name
        self.metadata = metadata or {}
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time is not None:
            duration = time.time() - self.start_time
            
            # Add exception info if occurred
            if exc_type is not None:
                self.metadata['exception'] = {
                    'type': exc_type.__name__,
                    'message': str(exc_val)
                }
            
            self.logger.log_performance(self.operation_name, duration, self.metadata)


# Global logger instance
default_logger = None


def get_logger(name: str = "snn_fusion", **kwargs) -> RobustLogger:
    """Get or create robust logger instance."""
    global default_logger
    
    if default_logger is None or default_logger.name != name:
        default_logger = RobustLogger(name, **kwargs)
    
    return default_logger


def log_security(event_type: str, message: str, severity: str = "medium",
                metadata: Optional[Dict] = None):
    """Convenience function for security logging."""
    logger = get_logger()
    logger.log_security_event(event_type, message, severity, metadata)


def log_performance(operation: str, duration: float, 
                   metadata: Optional[Dict] = None):
    """Convenience function for performance logging."""
    logger = get_logger()
    logger.log_performance(operation, duration, metadata)


# Example usage and testing
if __name__ == "__main__":
    print("Testing Robust Logging System...")
    
    # Create logger
    logger = RobustLogger("test_logger", log_dir="./test_logs")
    
    # Test basic logging
    logger.logger.info("Test info message")
    logger.logger.warning("Test warning message")
    logger.logger.error("Test error message")
    
    # Test security logging
    logger.log_security_event(
        "authentication_failure",
        "Failed login attempt",
        severity="high",
        metadata={'user': 'test_user', 'ip': '192.168.1.1'}
    )
    
    # Test performance logging
    with logger.time_operation("test_operation"):
        time.sleep(0.1)  # Simulate work
    
    # Test model event logging
    logger.log_model_event(
        "training_started",
        "multimodal_lsm",
        "Training started for MultiModal LSM",
        metadata={'epochs': 100, 'batch_size': 32}
    )
    
    # Test exception logging
    try:
        raise ValueError("Test exception")
    except Exception as e:
        logger.logger.exception("Exception occurred during testing")
    
    # Get performance summary
    perf_summary = logger.get_performance_summary()
    print(f"Performance summary: {perf_summary}")
    
    print("✓ Robust logging test completed!")
    print(f"Log files created in: {logger.log_dir}")