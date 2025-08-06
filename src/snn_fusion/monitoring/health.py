"""
Health Monitoring and System Status

This module provides comprehensive health monitoring capabilities for the
SNN-Fusion system, including system status checks, resource monitoring,
and failure detection.
"""

import os
import time
import psutil
import logging
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass, asdict
from enum import Enum
import threading
import json


class HealthStatus(Enum):
    """Health status levels."""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


@dataclass
class HealthMetric:
    """Individual health metric."""
    name: str
    value: Any
    status: HealthStatus
    message: str
    timestamp: datetime
    threshold_warning: Optional[float] = None
    threshold_critical: Optional[float] = None
    unit: Optional[str] = None


@dataclass
class SystemHealth:
    """Overall system health summary."""
    status: HealthStatus
    timestamp: datetime
    metrics: List[HealthMetric]
    uptime_seconds: float
    issues: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'status': self.status.value,
            'timestamp': self.timestamp.isoformat(),
            'metrics': [asdict(m) for m in self.metrics],
            'uptime_seconds': self.uptime_seconds,
            'issues': self.issues
        }


class HealthMonitor:
    """
    Comprehensive system health monitor.
    
    Monitors system resources, model performance, and service availability
    with configurable thresholds and alerting capabilities.
    """
    
    def __init__(
        self,
        check_interval: int = 30,  # seconds
        enable_continuous_monitoring: bool = False,
        log_level: str = "INFO"
    ):
        """
        Initialize health monitor.
        
        Args:
            check_interval: Interval between health checks in seconds
            enable_continuous_monitoring: Whether to run continuous monitoring
            log_level: Logging level
        """
        self.check_interval = check_interval
        self.enable_continuous_monitoring = enable_continuous_monitoring
        self.start_time = time.time()
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(getattr(logging, log_level.upper()))
        
        # Health checkers
        self.health_checkers: Dict[str, Callable[[], HealthMetric]] = {}
        self.register_default_checkers()
        
        # Monitoring thread
        self.monitoring_thread: Optional[threading.Thread] = None
        self.stop_monitoring = threading.Event()
        
        # Health history
        self.health_history: List[SystemHealth] = []
        self.max_history_size = 1000
        
        self.logger.info("HealthMonitor initialized")
    
    def register_default_checkers(self):
        """Register default system health checkers."""
        self.register_checker("cpu_usage", self._check_cpu_usage)
        self.register_checker("memory_usage", self._check_memory_usage)
        self.register_checker("disk_usage", self._check_disk_usage)
        self.register_checker("gpu_usage", self._check_gpu_usage)
        self.register_checker("process_count", self._check_process_count)
        self.register_checker("network_connectivity", self._check_network_connectivity)
        self.register_checker("disk_io", self._check_disk_io)
        
    def register_checker(self, name: str, checker: Callable[[], HealthMetric]):
        """Register a custom health checker."""
        self.health_checkers[name] = checker
        self.logger.debug(f"Registered health checker: {name}")
    
    def unregister_checker(self, name: str):
        """Unregister a health checker."""
        if name in self.health_checkers:
            del self.health_checkers[name]
            self.logger.debug(f"Unregistered health checker: {name}")
    
    def check_health(self) -> SystemHealth:
        """Perform comprehensive health check."""
        start_time = time.time()
        metrics = []
        issues = []
        overall_status = HealthStatus.HEALTHY
        
        self.logger.debug("Performing health check...")
        
        # Run all health checkers
        for name, checker in self.health_checkers.items():
            try:
                metric = checker()
                metrics.append(metric)
                
                # Track issues
                if metric.status == HealthStatus.WARNING:
                    issues.append(f"{name}: {metric.message}")
                    if overall_status == HealthStatus.HEALTHY:
                        overall_status = HealthStatus.WARNING
                elif metric.status == HealthStatus.CRITICAL:
                    issues.append(f"CRITICAL - {name}: {metric.message}")
                    overall_status = HealthStatus.CRITICAL
                    
            except Exception as e:
                self.logger.error(f"Health checker {name} failed: {e}")
                error_metric = HealthMetric(
                    name=name,
                    value="ERROR",
                    status=HealthStatus.CRITICAL,
                    message=f"Health checker failed: {e}",
                    timestamp=datetime.now()
                )
                metrics.append(error_metric)
                issues.append(f"CRITICAL - {name}: Health checker failed")
                overall_status = HealthStatus.CRITICAL
        
        # Calculate uptime
        uptime_seconds = time.time() - self.start_time
        
        # Create health summary
        health = SystemHealth(
            status=overall_status,
            timestamp=datetime.now(),
            metrics=metrics,
            uptime_seconds=uptime_seconds,
            issues=issues
        )
        
        # Add to history
        self._add_to_history(health)
        
        check_duration = time.time() - start_time
        self.logger.debug(f"Health check completed in {check_duration:.2f}s")
        
        if issues:
            self.logger.warning(f"Health issues detected: {len(issues)} issues")
            for issue in issues:
                self.logger.warning(f"  - {issue}")
        
        return health
    
    def _add_to_history(self, health: SystemHealth):
        """Add health check to history."""
        self.health_history.append(health)
        
        # Trim history if too large
        if len(self.health_history) > self.max_history_size:
            self.health_history = self.health_history[-self.max_history_size:]
    
    def start_continuous_monitoring(self):
        """Start continuous health monitoring in background thread."""
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.logger.warning("Monitoring already running")
            return
        
        self.stop_monitoring.clear()
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True
        )
        self.monitoring_thread.start()
        self.logger.info("Started continuous health monitoring")
    
    def stop_continuous_monitoring(self):
        """Stop continuous health monitoring."""
        self.stop_monitoring.set()
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        self.logger.info("Stopped continuous health monitoring")
    
    def _monitoring_loop(self):
        """Continuous monitoring loop."""
        while not self.stop_monitoring.is_set():
            try:
                health = self.check_health()
                
                # Log critical issues
                if health.status == HealthStatus.CRITICAL:
                    self.logger.critical(f"System health critical: {len(health.issues)} issues")
                elif health.status == HealthStatus.WARNING:
                    self.logger.warning(f"System health warning: {len(health.issues)} issues")
                
                # Wait for next check
                if self.stop_monitoring.wait(self.check_interval):
                    break
                    
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                if self.stop_monitoring.wait(self.check_interval):
                    break
    
    def get_health_summary(self, last_n_checks: Optional[int] = None) -> Dict[str, Any]:
        """Get health summary statistics."""
        if not self.health_history:
            return {"status": "no_data", "message": "No health checks performed"}
        
        history = self.health_history[-last_n_checks:] if last_n_checks else self.health_history
        
        # Calculate status distribution
        status_counts = {}
        for health in history:
            status = health.status.value
            status_counts[status] = status_counts.get(status, 0) + 1
        
        # Get latest health
        latest_health = history[-1]
        
        # Calculate average metrics
        metric_averages = {}
        for health in history:
            for metric in health.metrics:
                if isinstance(metric.value, (int, float)):
                    if metric.name not in metric_averages:
                        metric_averages[metric.name] = []
                    metric_averages[metric.name].append(metric.value)
        
        for metric_name, values in metric_averages.items():
            metric_averages[metric_name] = {
                'average': sum(values) / len(values),
                'min': min(values),
                'max': max(values),
                'count': len(values)
            }
        
        return {
            'current_status': latest_health.status.value,
            'uptime_seconds': latest_health.uptime_seconds,
            'total_checks': len(history),
            'status_distribution': status_counts,
            'metric_averages': metric_averages,
            'recent_issues': latest_health.issues,
            'last_check': latest_health.timestamp.isoformat()
        }
    
    def save_health_report(self, filepath: str):
        """Save health report to file."""
        report = {
            'timestamp': datetime.now().isoformat(),
            'summary': self.get_health_summary(),
            'latest_health': self.health_history[-1].to_dict() if self.health_history else None,
            'system_info': self._get_system_info()
        }
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        self.logger.info(f"Health report saved to {filepath}")
    
    def _get_system_info(self) -> Dict[str, Any]:
        """Get basic system information."""
        try:
            import platform
            return {
                'platform': platform.platform(),
                'python_version': platform.python_version(),
                'cpu_count': psutil.cpu_count(),
                'total_memory_gb': psutil.virtual_memory().total / (1024**3),
                'total_disk_gb': psutil.disk_usage('/').total / (1024**3),
            }
        except Exception as e:
            return {'error': str(e)}
    
    # Default health checkers
    
    def _check_cpu_usage(self) -> HealthMetric:
        """Check CPU usage percentage."""
        cpu_percent = psutil.cpu_percent(interval=1)
        
        if cpu_percent > 90:
            status = HealthStatus.CRITICAL
            message = f"Very high CPU usage: {cpu_percent:.1f}%"
        elif cpu_percent > 75:
            status = HealthStatus.WARNING
            message = f"High CPU usage: {cpu_percent:.1f}%"
        else:
            status = HealthStatus.HEALTHY
            message = f"CPU usage normal: {cpu_percent:.1f}%"
        
        return HealthMetric(
            name="cpu_usage",
            value=cpu_percent,
            status=status,
            message=message,
            timestamp=datetime.now(),
            threshold_warning=75.0,
            threshold_critical=90.0,
            unit="percent"
        )
    
    def _check_memory_usage(self) -> HealthMetric:
        """Check memory usage percentage."""
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        
        if memory_percent > 95:
            status = HealthStatus.CRITICAL
            message = f"Very high memory usage: {memory_percent:.1f}%"
        elif memory_percent > 85:
            status = HealthStatus.WARNING
            message = f"High memory usage: {memory_percent:.1f}%"
        else:
            status = HealthStatus.HEALTHY
            message = f"Memory usage normal: {memory_percent:.1f}%"
        
        return HealthMetric(
            name="memory_usage",
            value=memory_percent,
            status=status,
            message=message,
            timestamp=datetime.now(),
            threshold_warning=85.0,
            threshold_critical=95.0,
            unit="percent"
        )
    
    def _check_disk_usage(self) -> HealthMetric:
        """Check disk usage percentage."""
        disk = psutil.disk_usage('/')
        disk_percent = (disk.used / disk.total) * 100
        
        if disk_percent > 95:
            status = HealthStatus.CRITICAL
            message = f"Very high disk usage: {disk_percent:.1f}%"
        elif disk_percent > 85:
            status = HealthStatus.WARNING
            message = f"High disk usage: {disk_percent:.1f}%"
        else:
            status = HealthStatus.HEALTHY
            message = f"Disk usage normal: {disk_percent:.1f}%"
        
        return HealthMetric(
            name="disk_usage",
            value=disk_percent,
            status=status,
            message=message,
            timestamp=datetime.now(),
            threshold_warning=85.0,
            threshold_critical=95.0,
            unit="percent"
        )
    
    def _check_gpu_usage(self) -> HealthMetric:
        """Check GPU usage (if available)."""
        try:
            import GPUtil
            gpus = GPUtil.getGPUs()
            
            if not gpus:
                return HealthMetric(
                    name="gpu_usage",
                    value=0,
                    status=HealthStatus.HEALTHY,
                    message="No GPU detected",
                    timestamp=datetime.now(),
                    unit="percent"
                )
            
            # Use first GPU
            gpu = gpus[0]
            gpu_percent = gpu.load * 100
            
            if gpu_percent > 95:
                status = HealthStatus.WARNING  # GPU high usage is often expected
                message = f"Very high GPU usage: {gpu_percent:.1f}%"
            else:
                status = HealthStatus.HEALTHY
                message = f"GPU usage: {gpu_percent:.1f}%"
            
            return HealthMetric(
                name="gpu_usage",
                value=gpu_percent,
                status=status,
                message=message,
                timestamp=datetime.now(),
                threshold_warning=95.0,
                unit="percent"
            )
            
        except ImportError:
            return HealthMetric(
                name="gpu_usage",
                value="N/A",
                status=HealthStatus.UNKNOWN,
                message="GPUtil not available",
                timestamp=datetime.now()
            )
        except Exception as e:
            return HealthMetric(
                name="gpu_usage",
                value="ERROR",
                status=HealthStatus.WARNING,
                message=f"GPU check failed: {e}",
                timestamp=datetime.now()
            )
    
    def _check_process_count(self) -> HealthMetric:
        """Check number of running processes."""
        process_count = len(psutil.pids())
        
        if process_count > 1000:
            status = HealthStatus.WARNING
            message = f"High process count: {process_count}"
        else:
            status = HealthStatus.HEALTHY
            message = f"Process count normal: {process_count}"
        
        return HealthMetric(
            name="process_count",
            value=process_count,
            status=status,
            message=message,
            timestamp=datetime.now(),
            threshold_warning=1000,
            unit="processes"
        )
    
    def _check_network_connectivity(self) -> HealthMetric:
        """Check basic network connectivity."""
        try:
            import socket
            socket.create_connection(("8.8.8.8", 53), timeout=5)
            
            return HealthMetric(
                name="network_connectivity",
                value=True,
                status=HealthStatus.HEALTHY,
                message="Network connectivity OK",
                timestamp=datetime.now()
            )
            
        except Exception as e:
            return HealthMetric(
                name="network_connectivity",
                value=False,
                status=HealthStatus.WARNING,
                message=f"Network connectivity issue: {e}",
                timestamp=datetime.now()
            )
    
    def _check_disk_io(self) -> HealthMetric:
        """Check disk I/O statistics."""
        try:
            # Get disk I/O stats
            disk_io_1 = psutil.disk_io_counters()
            time.sleep(1)
            disk_io_2 = psutil.disk_io_counters()
            
            # Calculate reads/writes per second
            reads_per_sec = disk_io_2.read_count - disk_io_1.read_count
            writes_per_sec = disk_io_2.write_count - disk_io_1.write_count
            total_io = reads_per_sec + writes_per_sec
            
            if total_io > 1000:
                status = HealthStatus.WARNING
                message = f"High disk I/O: {total_io} ops/sec"
            else:
                status = HealthStatus.HEALTHY
                message = f"Disk I/O normal: {total_io} ops/sec"
            
            return HealthMetric(
                name="disk_io",
                value=total_io,
                status=status,
                message=message,
                timestamp=datetime.now(),
                threshold_warning=1000,
                unit="ops/sec"
            )
            
        except Exception as e:
            return HealthMetric(
                name="disk_io",
                value="ERROR",
                status=HealthStatus.WARNING,
                message=f"Disk I/O check failed: {e}",
                timestamp=datetime.now()
            )


class ModelHealthChecker:
    """Health checker specific to SNN models."""
    
    def __init__(self, model_name: str = "SNN_Model"):
        self.model_name = model_name
        self.logger = logging.getLogger(__name__)
        
    def check_model_health(self, model, sample_input=None) -> HealthMetric:
        """Check model health and inference capability."""
        try:
            # Check if model exists and has parameters
            if model is None:
                return HealthMetric(
                    name=f"{self.model_name}_health",
                    value=False,
                    status=HealthStatus.CRITICAL,
                    message="Model is None",
                    timestamp=datetime.now()
                )
            
            # Count parameters
            try:
                param_count = sum(p.numel() for p in model.parameters())
                
                if param_count == 0:
                    status = HealthStatus.WARNING
                    message = f"Model has no parameters"
                else:
                    status = HealthStatus.HEALTHY
                    message = f"Model loaded with {param_count:,} parameters"
                
            except Exception:
                # Fallback for non-PyTorch models
                param_count = "Unknown"
                status = HealthStatus.HEALTHY
                message = "Model structure appears valid"
            
            # Test inference if sample input provided
            if sample_input is not None:
                try:
                    start_time = time.time()
                    with torch.no_grad():
                        output = model(sample_input)
                    inference_time = time.time() - start_time
                    
                    message += f", inference time: {inference_time:.3f}s"
                    
                    # Check for NaN outputs
                    if hasattr(output, 'isnan') and output.isnan().any():
                        status = HealthStatus.CRITICAL
                        message += " - NaN values detected in output"
                        
                except Exception as e:
                    status = HealthStatus.CRITICAL
                    message = f"Inference failed: {e}"
            
            return HealthMetric(
                name=f"{self.model_name}_health",
                value=param_count,
                status=status,
                message=message,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            return HealthMetric(
                name=f"{self.model_name}_health",
                value="ERROR",
                status=HealthStatus.CRITICAL,
                message=f"Model health check failed: {e}",
                timestamp=datetime.now()
            )


# Convenience functions

def create_health_monitor(config: Dict[str, Any]) -> HealthMonitor:
    """Create health monitor from configuration."""
    return HealthMonitor(
        check_interval=config.get('check_interval', 30),
        enable_continuous_monitoring=config.get('enable_continuous_monitoring', False),
        log_level=config.get('log_level', 'INFO')
    )


def quick_health_check() -> SystemHealth:
    """Perform a quick health check."""
    monitor = HealthMonitor()
    return monitor.check_health()


# Example usage
if __name__ == "__main__":
    # Test health monitoring
    print("Testing Health Monitor...")
    
    # Create monitor
    monitor = HealthMonitor(check_interval=5)
    
    # Perform health check
    health = monitor.check_health()
    
    print(f"System Status: {health.status.value}")
    print(f"Uptime: {health.uptime_seconds:.1f} seconds")
    print(f"Issues: {len(health.issues)}")
    
    for metric in health.metrics:
        print(f"  {metric.name}: {metric.value} {metric.unit or ''} ({metric.status.value})")
    
    if health.issues:
        print("Issues found:")
        for issue in health.issues:
            print(f"  - {issue}")
    
    # Get summary
    summary = monitor.get_health_summary()
    print(f"\nHealth Summary:")
    print(f"  Total checks: {summary['total_checks']}")
    print(f"  Current status: {summary['current_status']}")
    
    # Save report
    monitor.save_health_report("health_report.json")
    print("Health report saved to health_report.json")
    
    print("✓ Health monitoring test completed!")