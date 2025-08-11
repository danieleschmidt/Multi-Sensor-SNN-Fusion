"""
Comprehensive Health Monitoring and System Diagnostics

Implements real-time health monitoring, performance tracking, and diagnostics
for neuromorphic SNN fusion systems with automated alert generation.
"""

import psutil
import time
import threading
import logging
from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
from collections import deque, defaultdict
import json
from pathlib import Path
import warnings
import statistics


class HealthStatus(Enum):
    """System health status levels."""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    FAILED = "failed"


class MetricType(Enum):
    """Types of health metrics."""
    SYSTEM = "system"
    MODEL = "model"
    DATA = "data"
    NETWORK = "network"
    MEMORY = "memory"
    STORAGE = "storage"
    CUSTOM = "custom"


@dataclass
class HealthMetric:
    """Individual health metric data."""
    name: str
    value: float
    unit: str
    timestamp: float
    status: HealthStatus
    metric_type: MetricType
    threshold_warning: Optional[float] = None
    threshold_critical: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class HealthAlert:
    """Health alert notification."""
    alert_id: str
    timestamp: float
    severity: HealthStatus
    metric_name: str
    current_value: float
    threshold_value: float
    message: str
    auto_resolved: bool = False
    resolved_timestamp: Optional[float] = None


@dataclass
class SystemSnapshot:
    """Complete system health snapshot."""
    timestamp: float
    overall_status: HealthStatus
    metrics: List[HealthMetric]
    alerts: List[HealthAlert]
    summary: Dict[str, Any]


class HealthMonitor:
    """
    Comprehensive health monitoring system for SNN Fusion framework.
    
    Monitors system resources, model performance, data pipeline health,
    and provides real-time alerts and diagnostics.
    """
    
    def __init__(
        self,
        monitoring_interval: float = 30.0,
        history_size: int = 1000,
        enable_alerts: bool = True,
        alert_callbacks: Optional[List[Callable]] = None
    ):
        """
        Initialize health monitor.
        
        Args:
            monitoring_interval: Monitoring interval in seconds
            history_size: Maximum number of historical snapshots
            enable_alerts: Enable alert generation
            alert_callbacks: List of functions to call when alerts are generated
        """
        self.monitoring_interval = monitoring_interval
        self.history_size = history_size
        self.enable_alerts = enable_alerts
        self.alert_callbacks = alert_callbacks or []
        
        self.logger = logging.getLogger(__name__)
        
        # Monitoring state
        self._monitoring = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._lock = threading.RLock()
        
        # Historical data
        self.health_history: deque = deque(maxlen=history_size)
        self.metric_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        
        # Active alerts
        self.active_alerts: Dict[str, HealthAlert] = {}
        self.alert_history: deque = deque(maxlen=500)
        
        # Custom metrics
        self.custom_metrics: Dict[str, Callable[[], float]] = {}
        
        # Default thresholds
        self.default_thresholds = {
            'cpu_usage': {'warning': 80.0, 'critical': 95.0},
            'memory_usage': {'warning': 85.0, 'critical': 95.0},
            'disk_usage': {'warning': 80.0, 'critical': 90.0},
            'gpu_usage': {'warning': 90.0, 'critical': 98.0},
            'gpu_memory': {'warning': 85.0, 'critical': 95.0},
            'temperature': {'warning': 70.0, 'critical': 85.0},
            'latency': {'warning': 100.0, 'critical': 500.0},  # ms
            'error_rate': {'warning': 0.05, 'critical': 0.1},  # percentage
        }
        
        self.logger.info("Health monitor initialized")
    
    def start_monitoring(self) -> None:
        """Start continuous health monitoring."""
        if self._monitoring:
            self.logger.warning("Monitoring already started")
            return
        
        self._monitoring = True
        self._monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self._monitor_thread.start()
        
        self.logger.info("Health monitoring started")
    
    def stop_monitoring(self) -> None:
        """Stop continuous health monitoring."""
        if not self._monitoring:
            return
        
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5.0)
        
        self.logger.info("Health monitoring stopped")
    
    def _monitoring_loop(self) -> None:
        """Main monitoring loop."""
        while self._monitoring:
            try:
                snapshot = self.take_snapshot()
                
                with self._lock:
                    self.health_history.append(snapshot)
                    
                    # Update metric history
                    for metric in snapshot.metrics:
                        self.metric_history[metric.name].append(
                            (metric.timestamp, metric.value, metric.status.value)
                        )
                    
                    # Process alerts
                    if self.enable_alerts:
                        self._process_alerts(snapshot)
                
                time.sleep(self.monitoring_interval)
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                time.sleep(self.monitoring_interval)
    
    def take_snapshot(self) -> SystemSnapshot:
        """Take a complete system health snapshot."""
        timestamp = time.time()
        metrics = []
        
        # System metrics
        metrics.extend(self._collect_system_metrics())
        
        # Model metrics (if available)
        metrics.extend(self._collect_model_metrics())
        
        # Data pipeline metrics
        metrics.extend(self._collect_data_metrics())
        
        # Network metrics
        metrics.extend(self._collect_network_metrics())
        
        # Custom metrics
        metrics.extend(self._collect_custom_metrics())
        
        # Determine overall status
        overall_status = self._determine_overall_status(metrics)
        
        # Get current alerts
        with self._lock:
            current_alerts = list(self.active_alerts.values())
        
        # Create summary
        summary = self._create_summary(metrics)
        
        snapshot = SystemSnapshot(
            timestamp=timestamp,
            overall_status=overall_status,
            metrics=metrics,
            alerts=current_alerts,
            summary=summary
        )
        
        return snapshot
    
    def _collect_system_metrics(self) -> List[HealthMetric]:
        """Collect system resource metrics."""
        metrics = []
        timestamp = time.time()
        
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=0.1)
            metrics.append(HealthMetric(
                name='cpu_usage',
                value=cpu_percent,
                unit='%',
                timestamp=timestamp,
                status=self._get_status_for_metric('cpu_usage', cpu_percent),
                metric_type=MetricType.SYSTEM,
                threshold_warning=self.default_thresholds['cpu_usage']['warning'],
                threshold_critical=self.default_thresholds['cpu_usage']['critical']
            ))
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            metrics.append(HealthMetric(
                name='memory_usage',
                value=memory_percent,
                unit='%',
                timestamp=timestamp,
                status=self._get_status_for_metric('memory_usage', memory_percent),
                metric_type=MetricType.MEMORY,
                threshold_warning=self.default_thresholds['memory_usage']['warning'],
                threshold_critical=self.default_thresholds['memory_usage']['critical'],
                metadata={
                    'total_gb': memory.total / (1024**3),
                    'available_gb': memory.available / (1024**3),
                    'used_gb': memory.used / (1024**3)
                }
            ))
            
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            metrics.append(HealthMetric(
                name='disk_usage',
                value=disk_percent,
                unit='%',
                timestamp=timestamp,
                status=self._get_status_for_metric('disk_usage', disk_percent),
                metric_type=MetricType.STORAGE,
                threshold_warning=self.default_thresholds['disk_usage']['warning'],
                threshold_critical=self.default_thresholds['disk_usage']['critical'],
                metadata={
                    'total_gb': disk.total / (1024**3),
                    'free_gb': disk.free / (1024**3),
                    'used_gb': disk.used / (1024**3)
                }
            ))
            
            # System load
            load_avg = psutil.getloadavg()[0] if hasattr(psutil, 'getloadavg') else 0
            cpu_count = psutil.cpu_count()
            load_percent = (load_avg / cpu_count) * 100 if cpu_count > 0 else 0
            metrics.append(HealthMetric(
                name='system_load',
                value=load_percent,
                unit='%',
                timestamp=timestamp,
                status=self._get_status_for_metric('cpu_usage', load_percent),  # Use CPU thresholds
                metric_type=MetricType.SYSTEM,
                metadata={
                    'load_avg': load_avg,
                    'cpu_count': cpu_count
                }
            ))
            
            # Temperature (if available)
            try:
                temps = psutil.sensors_temperatures()
                if temps:
                    for name, entries in temps.items():
                        for entry in entries:
                            if entry.current:
                                metrics.append(HealthMetric(
                                    name=f'temperature_{name}',
                                    value=entry.current,
                                    unit='°C',
                                    timestamp=timestamp,
                                    status=self._get_status_for_metric('temperature', entry.current),
                                    metric_type=MetricType.SYSTEM,
                                    threshold_warning=self.default_thresholds['temperature']['warning'],
                                    threshold_critical=self.default_thresholds['temperature']['critical']
                                ))
                                break  # Only take first sensor per category
            except (AttributeError, OSError):
                pass  # Temperature monitoring not available
            
        except Exception as e:
            self.logger.error(f"Error collecting system metrics: {e}")
        
        return metrics
    
    def _collect_model_metrics(self) -> List[HealthMetric]:
        """Collect model performance metrics."""
        metrics = []
        timestamp = time.time()
        
        try:
            # GPU metrics (if available)
            try:
                import torch
                if torch.cuda.is_available():
                    for i in range(torch.cuda.device_count()):
                        # GPU memory
                        memory_reserved = torch.cuda.memory_reserved(i)
                        memory_allocated = torch.cuda.memory_allocated(i)
                        total_memory = torch.cuda.get_device_properties(i).total_memory
                        
                        memory_percent = (memory_allocated / total_memory) * 100
                        
                        metrics.append(HealthMetric(
                            name=f'gpu_{i}_memory',
                            value=memory_percent,
                            unit='%',
                            timestamp=timestamp,
                            status=self._get_status_for_metric('gpu_memory', memory_percent),
                            metric_type=MetricType.MEMORY,
                            threshold_warning=self.default_thresholds['gpu_memory']['warning'],
                            threshold_critical=self.default_thresholds['gpu_memory']['critical'],
                            metadata={
                                'allocated_gb': memory_allocated / (1024**3),
                                'reserved_gb': memory_reserved / (1024**3),
                                'total_gb': total_memory / (1024**3)
                            }
                        ))
                        
                        # GPU utilization (if nvidia-ml-py available)
                        try:
                            import pynvml
                            pynvml.nvmlInit()
                            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                            
                            metrics.append(HealthMetric(
                                name=f'gpu_{i}_usage',
                                value=util.gpu,
                                unit='%',
                                timestamp=timestamp,
                                status=self._get_status_for_metric('gpu_usage', util.gpu),
                                metric_type=MetricType.SYSTEM,
                                threshold_warning=self.default_thresholds['gpu_usage']['warning'],
                                threshold_critical=self.default_thresholds['gpu_usage']['critical']
                            ))
                            
                            # GPU temperature
                            temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                            metrics.append(HealthMetric(
                                name=f'gpu_{i}_temperature',
                                value=temp,
                                unit='°C',
                                timestamp=timestamp,
                                status=self._get_status_for_metric('temperature', temp),
                                metric_type=MetricType.SYSTEM,
                                threshold_warning=self.default_thresholds['temperature']['warning'],
                                threshold_critical=self.default_thresholds['temperature']['critical']
                            ))
                            
                        except ImportError:
                            pass  # pynvml not available
                        
            except ImportError:
                pass  # PyTorch not available
                
        except Exception as e:
            self.logger.error(f"Error collecting model metrics: {e}")
        
        return metrics
    
    def _collect_data_metrics(self) -> List[HealthMetric]:
        """Collect data pipeline metrics."""
        metrics = []
        timestamp = time.time()
        
        # These would typically be populated by the data pipeline
        # For now, return empty list
        return metrics
    
    def _collect_network_metrics(self) -> List[HealthMetric]:
        """Collect network metrics."""
        metrics = []
        timestamp = time.time()
        
        try:
            # Network I/O
            net_io = psutil.net_io_counters()
            if net_io:
                # Calculate bytes per second if we have previous data
                if hasattr(self, '_prev_net_io') and hasattr(self, '_prev_net_time'):
                    time_delta = timestamp - self._prev_net_time
                    if time_delta > 0:
                        bytes_sent_rate = (net_io.bytes_sent - self._prev_net_io.bytes_sent) / time_delta
                        bytes_recv_rate = (net_io.bytes_recv - self._prev_net_io.bytes_recv) / time_delta
                        
                        metrics.append(HealthMetric(
                            name='network_sent_rate',
                            value=bytes_sent_rate / (1024 * 1024),  # MB/s
                            unit='MB/s',
                            timestamp=timestamp,
                            status=HealthStatus.HEALTHY,
                            metric_type=MetricType.NETWORK
                        ))
                        
                        metrics.append(HealthMetric(
                            name='network_recv_rate',
                            value=bytes_recv_rate / (1024 * 1024),  # MB/s
                            unit='MB/s',
                            timestamp=timestamp,
                            status=HealthStatus.HEALTHY,
                            metric_type=MetricType.NETWORK
                        ))
                
                # Store for next calculation
                self._prev_net_io = net_io
                self._prev_net_time = timestamp
                
        except Exception as e:
            self.logger.error(f"Error collecting network metrics: {e}")
        
        return metrics
    
    def _collect_custom_metrics(self) -> List[HealthMetric]:
        """Collect custom metrics."""
        metrics = []
        timestamp = time.time()
        
        for name, metric_func in self.custom_metrics.items():
            try:
                value = metric_func()
                if isinstance(value, (int, float)):
                    metrics.append(HealthMetric(
                        name=name,
                        value=value,
                        unit='',
                        timestamp=timestamp,
                        status=HealthStatus.HEALTHY,
                        metric_type=MetricType.CUSTOM
                    ))
            except Exception as e:
                self.logger.error(f"Error collecting custom metric '{name}': {e}")
        
        return metrics
    
    def _get_status_for_metric(self, metric_name: str, value: float) -> HealthStatus:
        """Determine health status for a metric value."""
        thresholds = self.default_thresholds.get(metric_name, {})
        
        critical_threshold = thresholds.get('critical')
        warning_threshold = thresholds.get('warning')
        
        if critical_threshold is not None and value >= critical_threshold:
            return HealthStatus.CRITICAL
        elif warning_threshold is not None and value >= warning_threshold:
            return HealthStatus.WARNING
        else:
            return HealthStatus.HEALTHY
    
    def _determine_overall_status(self, metrics: List[HealthMetric]) -> HealthStatus:
        """Determine overall system health status."""
        if not metrics:
            return HealthStatus.HEALTHY
        
        status_counts = {status: 0 for status in HealthStatus}
        for metric in metrics:
            status_counts[metric.status] += 1
        
        # Priority: FAILED > CRITICAL > WARNING > HEALTHY
        if status_counts[HealthStatus.FAILED] > 0:
            return HealthStatus.FAILED
        elif status_counts[HealthStatus.CRITICAL] > 0:
            return HealthStatus.CRITICAL
        elif status_counts[HealthStatus.WARNING] > 0:
            return HealthStatus.WARNING
        else:
            return HealthStatus.HEALTHY
    
    def _create_summary(self, metrics: List[HealthMetric]) -> Dict[str, Any]:
        """Create a summary of current metrics."""
        summary = {
            'total_metrics': len(metrics),
            'status_counts': {status.value: 0 for status in HealthStatus},
            'metric_types': {mtype.value: 0 for mtype in MetricType},
        }
        
        for metric in metrics:
            summary['status_counts'][metric.status.value] += 1
            summary['metric_types'][metric.metric_type.value] += 1
        
        return summary
    
    def _process_alerts(self, snapshot: SystemSnapshot) -> None:
        """Process alerts based on snapshot."""
        current_time = snapshot.timestamp
        
        for metric in snapshot.metrics:
            alert_key = f"{metric.name}_{metric.status.value}"
            
            # Generate alert if metric is in warning or critical state
            if metric.status in [HealthStatus.WARNING, HealthStatus.CRITICAL]:
                if alert_key not in self.active_alerts:
                    # New alert
                    alert = HealthAlert(
                        alert_id=f"ALERT_{int(current_time)}_{len(self.active_alerts)}",
                        timestamp=current_time,
                        severity=metric.status,
                        metric_name=metric.name,
                        current_value=metric.value,
                        threshold_value=metric.threshold_warning or metric.threshold_critical,
                        message=f"{metric.name} is {metric.status.value}: {metric.value}{metric.unit}"
                    )
                    
                    self.active_alerts[alert_key] = alert
                    self.alert_history.append(alert)
                    
                    # Call alert callbacks
                    for callback in self.alert_callbacks:
                        try:
                            callback(alert)
                        except Exception as e:
                            self.logger.error(f"Error in alert callback: {e}")
                    
                    self.logger.warning(f"ALERT: {alert.message}")
            
            else:
                # Resolve alert if it exists
                if alert_key in self.active_alerts:
                    alert = self.active_alerts[alert_key]
                    alert.auto_resolved = True
                    alert.resolved_timestamp = current_time
                    
                    del self.active_alerts[alert_key]
                    
                    self.logger.info(f"RESOLVED: {alert.message}")
    
    def register_custom_metric(self, name: str, metric_func: Callable[[], float]) -> None:
        """Register a custom metric function."""
        self.custom_metrics[name] = metric_func
        self.logger.info(f"Registered custom metric: {name}")
    
    def set_threshold(
        self, 
        metric_name: str, 
        warning: Optional[float] = None,
        critical: Optional[float] = None
    ) -> None:
        """Set custom thresholds for a metric."""
        if metric_name not in self.default_thresholds:
            self.default_thresholds[metric_name] = {}
        
        if warning is not None:
            self.default_thresholds[metric_name]['warning'] = warning
        if critical is not None:
            self.default_thresholds[metric_name]['critical'] = critical
        
        self.logger.info(f"Set thresholds for {metric_name}: warning={warning}, critical={critical}")
    
    def get_current_status(self) -> Dict[str, Any]:
        """Get current system health status."""
        if not self.health_history:
            return {'status': 'no_data', 'message': 'No health data available'}
        
        with self._lock:
            latest_snapshot = self.health_history[-1]
        
        return {
            'overall_status': latest_snapshot.overall_status.value,
            'timestamp': latest_snapshot.timestamp,
            'active_alerts': len(self.active_alerts),
            'total_metrics': len(latest_snapshot.metrics),
            'summary': latest_snapshot.summary
        }
    
    def get_metric_trends(
        self, 
        metric_name: str, 
        time_window: int = 3600
    ) -> Dict[str, Any]:
        """Get trend analysis for a specific metric."""
        with self._lock:
            history = self.metric_history.get(metric_name, deque())
        
        if not history:
            return {'error': f'No data available for metric {metric_name}'}
        
        current_time = time.time()
        recent_data = [(timestamp, value, status) for timestamp, value, status in history
                      if current_time - timestamp <= time_window]
        
        if not recent_data:
            return {'error': f'No recent data for metric {metric_name}'}
        
        values = [value for _, value, _ in recent_data]
        
        return {
            'metric_name': metric_name,
            'data_points': len(values),
            'current_value': values[-1],
            'average': statistics.mean(values),
            'min_value': min(values),
            'max_value': max(values),
            'trend': self._calculate_trend(values),
            'volatility': statistics.stdev(values) if len(values) > 1 else 0
        }
    
    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction for a series of values."""
        if len(values) < 2:
            return 'insufficient_data'
        
        # Simple linear trend
        n = len(values)
        x = list(range(n))
        y = values
        
        # Calculate slope
        x_mean = sum(x) / n
        y_mean = sum(y) / n
        
        numerator = sum((x[i] - x_mean) * (y[i] - y_mean) for i in range(n))
        denominator = sum((x[i] - x_mean) ** 2 for i in range(n))
        
        if denominator == 0:
            return 'stable'
        
        slope = numerator / denominator
        
        if slope > 0.1:
            return 'increasing'
        elif slope < -0.1:
            return 'decreasing'
        else:
            return 'stable'
    
    def export_health_report(self, file_path: Optional[str] = None) -> str:
        """Export comprehensive health report."""
        report_data = {
            'timestamp': time.time(),
            'monitoring_interval': self.monitoring_interval,
            'current_status': self.get_current_status(),
            'active_alerts': [asdict(alert) for alert in self.active_alerts.values()],
            'recent_snapshots': [],
            'metric_summaries': {}
        }
        
        # Add recent snapshots
        with self._lock:
            for snapshot in list(self.health_history)[-10:]:  # Last 10 snapshots
                report_data['recent_snapshots'].append({
                    'timestamp': snapshot.timestamp,
                    'overall_status': snapshot.overall_status.value,
                    'metric_count': len(snapshot.metrics),
                    'alert_count': len(snapshot.alerts)
                })
            
            # Add metric summaries
            for metric_name, history in self.metric_history.items():
                if history:
                    values = [value for _, value, _ in history]
                    report_data['metric_summaries'][metric_name] = {
                        'current': values[-1],
                        'average': statistics.mean(values),
                        'min': min(values),
                        'max': max(values),
                        'data_points': len(values)
                    }
        
        # Save to file if path provided
        if file_path:
            with open(file_path, 'w') as f:
                json.dump(report_data, f, indent=2)
            self.logger.info(f"Health report exported to {file_path}")
        
        return json.dumps(report_data, indent=2)


# Utility functions for health monitoring

def create_health_dashboard(monitor: HealthMonitor) -> Dict[str, Any]:
    """Create a health dashboard summary."""
    status = monitor.get_current_status()
    
    dashboard = {
        'system_status': status,
        'alerts': {
            'active': len(monitor.active_alerts),
            'recent': len([a for a in monitor.alert_history 
                          if time.time() - a.timestamp < 3600])
        },
        'monitoring': {
            'is_active': monitor._monitoring,
            'interval_seconds': monitor.monitoring_interval,
            'history_size': len(monitor.health_history)
        }
    }
    
    return dashboard


def alert_callback_logger(alert: HealthAlert) -> None:
    """Example alert callback that logs alerts."""
    logger = logging.getLogger('health_alerts')
    logger.warning(f"HEALTH ALERT [{alert.severity.value.upper()}]: {alert.message}")


def alert_callback_email(alert: HealthAlert) -> None:
    """Example alert callback for email notifications (placeholder)."""
    # In a real implementation, this would send an email
    print(f"EMAIL ALERT: {alert.message}")


def setup_basic_monitoring(
    monitoring_interval: float = 30.0,
    enable_email_alerts: bool = False
) -> HealthMonitor:
    """Set up basic health monitoring with default configuration."""
    
    # Create alert callbacks
    callbacks = [alert_callback_logger]
    if enable_email_alerts:
        callbacks.append(alert_callback_email)
    
    # Create and configure monitor
    monitor = HealthMonitor(
        monitoring_interval=monitoring_interval,
        enable_alerts=True,
        alert_callbacks=callbacks
    )
    
    # Set up custom thresholds (example)
    monitor.set_threshold('cpu_usage', warning=75.0, critical=90.0)
    monitor.set_threshold('memory_usage', warning=80.0, critical=95.0)
    
    # Register example custom metrics
    def custom_error_rate():
        # This would typically calculate actual error rate
        return 0.02  # 2% error rate
    
    monitor.register_custom_metric('error_rate', custom_error_rate)
    
    return monitor