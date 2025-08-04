"""
Monitoring and Performance Utilities for SNN-Fusion

This module provides comprehensive monitoring capabilities including performance
tracking, resource monitoring, and system health checks for production deployment.
"""

import time
import psutil
import threading
import queue
from collections import defaultdict, deque
from typing import Dict, List, Optional, Union, Any, Callable
import json
import warnings
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import gc


@dataclass
class PerformanceMetrics:
    """Data class for performance metrics."""
    timestamp: float
    cpu_percent: float
    memory_used: int
    memory_percent: float
    gpu_memory_used: Optional[int] = None
    gpu_utilization: Optional[float] = None
    disk_io_read: int = 0
    disk_io_write: int = 0
    network_sent: int = 0
    network_recv: int = 0


@dataclass
class ModelMetrics:
    """Data class for model-specific metrics."""
    timestamp: float
    forward_time: float
    backward_time: Optional[float] = None
    batch_size: int = 1
    sequence_length: int = 1
    throughput: float = 0.0  # samples/second
    memory_allocated: int = 0
    memory_cached: int = 0
    gradient_norm: Optional[float] = None
    loss_value: Optional[float] = None


class PerformanceMonitor:
    """
    Monitor for tracking model and system performance metrics.
    
    Provides real-time monitoring of computational performance with
    minimal overhead and automatic alerting for performance issues.
    """
    
    def __init__(
        self,
        history_size: int = 1000,
        sampling_interval: float = 1.0,
        enable_gpu_monitoring: bool = True
    ):
        """
        Initialize performance monitor.
        
        Args:
            history_size: Number of historical metrics to keep
            sampling_interval: Sampling interval in seconds
            enable_gpu_monitoring: Whether to monitor GPU metrics
        """
        self.history_size = history_size
        self.sampling_interval = sampling_interval
        self.enable_gpu_monitoring = enable_gpu_monitoring
        
        # Metric storage
        self.system_metrics = deque(maxlen=history_size)
        self.model_metrics = deque(maxlen=history_size)
        self.custom_metrics = defaultdict(lambda: deque(maxlen=history_size))
        
        # Performance counters
        self.operation_times = defaultdict(list)
        self.operation_counts = defaultdict(int)
        
        # GPU monitoring setup
        self.gpu_available = False
        if enable_gpu_monitoring:
            try:
                import torch
                self.gpu_available = torch.cuda.is_available()
                if self.gpu_available:
                    self.device_count = torch.cuda.device_count()
            except ImportError:
                self.gpu_available = False
        
        # Monitoring state
        self.monitoring_active = False
        self.monitor_thread = None
        self.stop_event = threading.Event()
        
        # Alert thresholds
        self.cpu_threshold = 90.0  # %
        self.memory_threshold = 90.0  # %
        self.gpu_memory_threshold = 90.0  # %
        self.alert_callbacks = []
    
    def start_monitoring(self) -> None:
        """Start background monitoring thread."""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.stop_event.clear()
        self.monitor_thread = threading.Thread(target=self._monitoring_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
    
    def stop_monitoring(self) -> None:
        """Stop background monitoring thread."""
        if not self.monitoring_active:
            return
        
        self.monitoring_active = False
        self.stop_event.set()
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5.0)
    
    def _monitoring_loop(self) -> None:
        """Main monitoring loop running in background thread."""
        while not self.stop_event.wait(self.sampling_interval):
            try:
                metrics = self._collect_system_metrics()
                self.system_metrics.append(metrics)
                self._check_alerts(metrics)
            except Exception as e:
                warnings.warn(f"Error in monitoring loop: {e}")
    
    def _collect_system_metrics(self) -> PerformanceMetrics:
        """Collect current system performance metrics."""
        # CPU and memory
        cpu_percent = psutil.cpu_percent()
        memory = psutil.virtual_memory()
        
        # Disk I/O
        disk_io = psutil.disk_io_counters()
        disk_read = disk_io.read_bytes if disk_io else 0
        disk_write = disk_io.write_bytes if disk_io else 0
        
        # Network I/O
        network_io = psutil.net_io_counters()
        network_sent = network_io.bytes_sent if network_io else 0
        network_recv = network_io.bytes_recv if network_io else 0
        
        # GPU metrics
        gpu_memory_used = None
        gpu_utilization = None
        
        if self.gpu_available:
            try:
                import torch
                if torch.cuda.is_available():
                    gpu_memory_used = torch.cuda.memory_allocated(0)
                    # GPU utilization would require nvidia-ml-py
            except Exception:
                pass
        
        return PerformanceMetrics(
            timestamp=time.time(),
            cpu_percent=cpu_percent,
            memory_used=memory.used,
            memory_percent=memory.percent,
            gpu_memory_used=gpu_memory_used,
            gpu_utilization=gpu_utilization,
            disk_io_read=disk_read,
            disk_io_write=disk_write,
            network_sent=network_sent,
            network_recv=network_recv
        )
    
    def record_model_metrics(
        self,
        forward_time: float,
        backward_time: Optional[float] = None,
        batch_size: int = 1,
        sequence_length: int = 1,
        **kwargs
    ) -> None:
        """
        Record model-specific performance metrics.
        
        Args:
            forward_time: Forward pass time in seconds
            backward_time: Backward pass time in seconds
            batch_size: Batch size
            sequence_length: Sequence length
            **kwargs: Additional metrics
        """
        # Calculate throughput
        total_time = forward_time + (backward_time or 0)
        throughput = (batch_size * sequence_length) / total_time if total_time > 0 else 0
        
        # GPU memory info
        memory_allocated = 0
        memory_cached = 0
        
        if self.gpu_available:
            try:
                import torch
                if torch.cuda.is_available():
                    memory_allocated = torch.cuda.memory_allocated(0)
                    memory_cached = torch.cuda.memory_cached(0)
            except Exception:
                pass
        
        metrics = ModelMetrics(
            timestamp=time.time(),
            forward_time=forward_time,
            backward_time=backward_time,
            batch_size=batch_size,
            sequence_length=sequence_length,
            throughput=throughput,
            memory_allocated=memory_allocated,
            memory_cached=memory_cached,
            **kwargs
        )
        
        self.model_metrics.append(metrics)
    
    def record_operation_time(self, operation_name: str, duration: float) -> None:
        """Record timing for a named operation."""
        self.operation_times[operation_name].append(duration)
        self.operation_counts[operation_name] += 1
        
        # Keep only recent times to avoid memory growth
        if len(self.operation_times[operation_name]) > 100:
            self.operation_times[operation_name] = self.operation_times[operation_name][-100:]
    
    def record_custom_metric(self, name: str, value: float, timestamp: Optional[float] = None) -> None:
        """Record a custom metric value."""
        if timestamp is None:
            timestamp = time.time()
        
        self.custom_metrics[name].append((timestamp, value))
    
    def get_performance_summary(self, last_n_seconds: int = 60) -> Dict[str, Any]:
        """
        Get performance summary for the last N seconds.
        
        Args:
            last_n_seconds: Time window for summary
            
        Returns:
            Performance summary dictionary
        """
        cutoff_time = time.time() - last_n_seconds
        
        # Filter recent system metrics
        recent_system = [m for m in self.system_metrics if m.timestamp >= cutoff_time]
        recent_model = [m for m in self.model_metrics if m.timestamp >= cutoff_time]
        
        summary = {
            'time_window': last_n_seconds,
            'timestamp': time.time(),
            'system_metrics': {},
            'model_metrics': {},
            'operation_stats': {}
        }
        
        # System metrics summary
        if recent_system:
            summary['system_metrics'] = {
                'cpu_percent': {
                    'avg': sum(m.cpu_percent for m in recent_system) / len(recent_system),
                    'max': max(m.cpu_percent for m in recent_system),
                    'min': min(m.cpu_percent for m in recent_system)
                },
                'memory_percent': {
                    'avg': sum(m.memory_percent for m in recent_system) / len(recent_system),
                    'max': max(m.memory_percent for m in recent_system),
                    'min': min(m.memory_percent for m in recent_system)
                }
            }
            
            # GPU metrics if available
            gpu_memory_values = [m.gpu_memory_used for m in recent_system if m.gpu_memory_used is not None]
            if gpu_memory_values:
                summary['system_metrics']['gpu_memory_mb'] = {
                    'avg': sum(gpu_memory_values) / len(gpu_memory_values) / (1024 * 1024),
                    'max': max(gpu_memory_values) / (1024 * 1024),
                    'min': min(gpu_memory_values) / (1024 * 1024)
                }
        
        # Model metrics summary
        if recent_model:
            forward_times = [m.forward_time for m in recent_model]
            throughputs = [m.throughput for m in recent_model if m.throughput > 0]
            
            summary['model_metrics'] = {
                'forward_time_ms': {
                    'avg': sum(forward_times) / len(forward_times) * 1000,
                    'max': max(forward_times) * 1000,
                    'min': min(forward_times) * 1000
                }
            }
            
            if throughputs:
                summary['model_metrics']['throughput_samples_per_sec'] = {
                    'avg': sum(throughputs) / len(throughputs),
                    'max': max(throughputs),
                    'min': min(throughputs)
                }
        
        # Operation statistics
        for op_name, times in self.operation_times.items():
            if times:
                summary['operation_stats'][op_name] = {
                    'count': self.operation_counts[op_name],
                    'avg_time_ms': sum(times) / len(times) * 1000,
                    'total_time_s': sum(times)
                }
        
        return summary
    
    def _check_alerts(self, metrics: PerformanceMetrics) -> None:
        """Check metrics against alert thresholds."""
        alerts = []
        
        # CPU threshold
        if metrics.cpu_percent > self.cpu_threshold:
            alerts.append({
                'type': 'cpu_high',
                'value': metrics.cpu_percent,
                'threshold': self.cpu_threshold,
                'severity': 'warning'
            })
        
        # Memory threshold
        if metrics.memory_percent > self.memory_threshold:
            alerts.append({
                'type': 'memory_high',
                'value': metrics.memory_percent,
                'threshold': self.memory_threshold,
                'severity': 'warning'
            })
        
        # GPU memory threshold
        if (metrics.gpu_memory_used is not None and 
            self.gpu_available):
            try:
                import torch
                total_memory = torch.cuda.get_device_properties(0).total_memory
                gpu_percent = (metrics.gpu_memory_used / total_memory) * 100
                
                if gpu_percent > self.gpu_memory_threshold:
                    alerts.append({
                        'type': 'gpu_memory_high',
                        'value': gpu_percent,
                        'threshold': self.gpu_memory_threshold,
                        'severity': 'warning'
                    })
            except Exception:
                pass
        
        # Trigger alert callbacks
        for alert in alerts:
            for callback in self.alert_callbacks:
                try:
                    callback(alert)
                except Exception as e:
                    warnings.warn(f"Error in alert callback: {e}")
    
    def add_alert_callback(self, callback: Callable) -> None:
        """Add callback function for performance alerts."""
        self.alert_callbacks.append(callback)
    
    def export_metrics(self, file_path: str, format: str = 'json') -> None:
        """
        Export collected metrics to file.
        
        Args:
            file_path: Output file path
            format: Export format ('json', 'csv')
        """
        if format == 'json':
            data = {
                'system_metrics': [asdict(m) for m in self.system_metrics],
                'model_metrics': [asdict(m) for m in self.model_metrics],
                'custom_metrics': {k: list(v) for k, v in self.custom_metrics.items()},
                'operation_stats': dict(self.operation_times)
            }
            
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2, default=str)
        
        elif format == 'csv':
            import csv
            
            with open(file_path, 'w', newline='') as f:
                writer = csv.writer(f)
                
                # System metrics
                writer.writerow(['timestamp', 'cpu_percent', 'memory_percent', 'gpu_memory_mb'])
                for m in self.system_metrics:
                    gpu_mb = m.gpu_memory_used / (1024 * 1024) if m.gpu_memory_used else None
                    writer.writerow([m.timestamp, m.cpu_percent, m.memory_percent, gpu_mb])
        else:
            raise ValueError(f"Unsupported export format: {format}")


class ResourceMonitor:
    """
    Monitor system resources and detect potential issues.
    
    Provides early warning for resource exhaustion and automatic
    cleanup when needed.
    """
    
    def __init__(self):
        self.process = psutil.Process()
        self.initial_memory = self.process.memory_info().rss
        self.peak_memory = self.initial_memory
        self.gc_collections = 0
    
    def check_memory_usage(self, threshold_gb: float = 8.0) -> Dict[str, Any]:
        """
        Check current memory usage against threshold.
        
        Args:
            threshold_gb: Memory threshold in GB
            
        Returns:
            Memory usage information
        """
        current_memory = self.process.memory_info().rss
        self.peak_memory = max(self.peak_memory, current_memory)
        
        memory_gb = current_memory / (1024 ** 3)
        peak_gb = self.peak_memory / (1024 ** 3)
        
        usage_info = {
            'current_memory_gb': memory_gb,
            'peak_memory_gb': peak_gb,
            'threshold_gb': threshold_gb,
            'within_limit': memory_gb < threshold_gb,
            'memory_growth_gb': (current_memory - self.initial_memory) / (1024 ** 3)
        }
        
        # Trigger garbage collection if approaching limit
        if memory_gb > threshold_gb * 0.8:
            collected = gc.collect()
            self.gc_collections += 1
            usage_info['gc_triggered'] = True
            usage_info['objects_collected'] = collected
        
        return usage_info
    
    def check_disk_space(self, path: str = '.', threshold_gb: float = 1.0) -> Dict[str, Any]:
        """
        Check available disk space.
        
        Args:
            path: Path to check disk space for
            threshold_gb: Minimum free space threshold in GB
            
        Returns:
            Disk space information
        """
        disk_usage = psutil.disk_usage(path)
        
        free_gb = disk_usage.free / (1024 ** 3)
        total_gb = disk_usage.total / (1024 ** 3)
        used_percent = (disk_usage.used / disk_usage.total) * 100
        
        return {
            'free_space_gb': free_gb,
            'total_space_gb': total_gb,
            'used_percent': used_percent,
            'threshold_gb': threshold_gb,
            'sufficient_space': free_gb > threshold_gb
        }
    
    def get_system_limits(self) -> Dict[str, Any]:
        """Get system resource limits."""
        try:
            import resource
            
            limits = {}
            
            # Memory limit
            mem_limit = resource.getrlimit(resource.RLIMIT_AS)
            limits['memory_limit'] = {
                'soft': mem_limit[0] if mem_limit[0] != resource.RLIM_INFINITY else None,
                'hard': mem_limit[1] if mem_limit[1] != resource.RLIM_INFINITY else None
            }
            
            # File descriptor limit
            fd_limit = resource.getrlimit(resource.RLIMIT_NOFILE)
            limits['file_descriptors'] = {
                'soft': fd_limit[0],
                'hard': fd_limit[1]
            }
            
            # CPU time limit
            cpu_limit = resource.getrlimit(resource.RLIMIT_CPU)
            limits['cpu_time'] = {
                'soft': cpu_limit[0] if cpu_limit[0] != resource.RLIM_INFINITY else None,
                'hard': cpu_limit[1] if cpu_limit[1] != resource.RLIM_INFINITY else None
            }
            
            return limits
        
        except ImportError:
            return {'error': 'resource module not available'}


class SystemMonitor:
    """
    Comprehensive system monitoring combining performance and resource monitoring.
    
    Provides unified interface for all monitoring capabilities with automatic
    reporting and alerting.
    """
    
    def __init__(self, enable_background_monitoring: bool = True):
        self.performance_monitor = PerformanceMonitor()
        self.resource_monitor = ResourceMonitor()
        
        # Health check registry
        self.health_checks = {}
        
        # Monitoring configuration
        self.enable_background_monitoring = enable_background_monitoring
        
        if enable_background_monitoring:
            self.performance_monitor.start_monitoring()
    
    def register_health_check(self, name: str, check_function: Callable) -> None:
        """
        Register a health check function.
        
        Args:
            name: Name of the health check
            check_function: Function that returns dict with health status
        """
        self.health_checks[name] = check_function
    
    def run_health_checks(self) -> Dict[str, Any]:
        """
        Run all registered health checks.
        
        Returns:
            Dictionary of health check results
        """
        results = {
            'timestamp': time.time(),
            'overall_healthy': True,
            'checks': {}
        }
        
        for name, check_func in self.health_checks.items():
            try:
                check_result = check_func()
                results['checks'][name] = {
                    'status': 'healthy' if check_result.get('healthy', True) else 'unhealthy',
                    'details': check_result
                }
                
                if not check_result.get('healthy', True):
                    results['overall_healthy'] = False
            
            except Exception as e:
                results['checks'][name] = {
                    'status': 'error',
                    'error': str(e)
                }
                results['overall_healthy'] = False
        
        return results
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        status = {
            'timestamp': time.time(),
            'performance': self.performance_monitor.get_performance_summary(),
            'resources': {
                'memory': self.resource_monitor.check_memory_usage(),
                'disk': self.resource_monitor.check_disk_space(),
                'limits': self.resource_monitor.get_system_limits()
            },
            'health_checks': self.run_health_checks()
        }
        
        return status
    
    def export_status_report(self, file_path: str) -> None:
        """Export comprehensive status report."""
        status = self.get_system_status()
        
        with open(file_path, 'w') as f:
            json.dump(status, f, indent=2, default=str)
    
    def shutdown(self) -> None:
        """Shutdown monitoring and cleanup resources."""
        if self.enable_background_monitoring:
            self.performance_monitor.stop_monitoring()


# Utility functions for monitoring

def create_memory_health_check(threshold_gb: float = 8.0) -> Callable:
    """Create memory usage health check function."""
    def check_memory():
        process = psutil.Process()
        memory_gb = process.memory_info().rss / (1024 ** 3)
        
        return {
            'healthy': memory_gb < threshold_gb,
            'memory_usage_gb': memory_gb,
            'threshold_gb': threshold_gb
        }
    
    return check_memory


def create_gpu_health_check() -> Callable:
    """Create GPU health check function."""
    def check_gpu():
        try:
            import torch
            
            if not torch.cuda.is_available():
                return {'healthy': True, 'message': 'No GPU available'}
            
            # Test GPU accessibility
            device = torch.cuda.current_device()
            memory_allocated = torch.cuda.memory_allocated(device)
            memory_cached = torch.cuda.memory_cached(device)
            
            return {
                'healthy': True,
                'device_count': torch.cuda.device_count(),
                'current_device': device,
                'memory_allocated_mb': memory_allocated / (1024 * 1024),
                'memory_cached_mb': memory_cached / (1024 * 1024)
            }
        
        except Exception as e:
            return {
                'healthy': False,
                'error': str(e)
            }
    
    return check_gpu


# Example usage and testing
if __name__ == "__main__":
    print("Testing monitoring functionality...")
    
    # Create system monitor
    monitor = SystemMonitor(enable_background_monitoring=False)
    
    # Register health checks
    monitor.register_health_check('memory', create_memory_health_check())
    monitor.register_health_check('gpu', create_gpu_health_check())
    
    # Test performance monitoring
    perf_monitor = monitor.performance_monitor
    
    # Simulate some operations
    perf_monitor.record_operation_time('test_operation', 0.1)
    perf_monitor.record_custom_metric('test_metric', 42.0)
    
    # Record model metrics
    perf_monitor.record_model_metrics(
        forward_time=0.05,
        backward_time=0.03,
        batch_size=32,
        sequence_length=100
    )
    
    # Get status
    status = monitor.get_system_status()
    print("System status keys:", list(status.keys()))
    
    # Test resource monitoring
    resource_info = monitor.resource_monitor.check_memory_usage()
    print(f"Memory usage: {resource_info['current_memory_gb']:.2f} GB")
    
    # Get performance summary
    summary = perf_monitor.get_performance_summary()
    print("Performance summary keys:", list(summary.keys()))
    
    print("Monitoring tests completed!")