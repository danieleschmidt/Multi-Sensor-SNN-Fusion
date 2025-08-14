"""
Comprehensive Health Monitoring for Neuromorphic Systems

Real-time health monitoring, performance tracking, and predictive maintenance
for neuromorphic multi-modal fusion systems. Provides comprehensive metrics,
alerting, and automated recovery capabilities.
"""

import time
import threading
import queue
import statistics
from typing import Dict, List, Any, Optional, Callable, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from collections import deque, defaultdict
import json
import logging
import psutil
import numpy as np
from pathlib import Path

# For hardware monitoring (when available)
try:
    import GPUtil
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

from ..utils.robust_error_handling import robust_function, SecurityEvent
from ..algorithms.fusion import FusionResult, ModalityData


class HealthStatus(Enum):
    """System health status levels."""
    HEALTHY = "healthy"
    WARNING = "warning"
    DEGRADED = "degraded"
    CRITICAL = "critical"
    FAILED = "failed"


class MetricType(Enum):
    """Types of metrics to monitor."""
    PERFORMANCE = "performance"
    RESOURCE = "resource"
    QUALITY = "quality"
    SECURITY = "security"
    HARDWARE = "hardware"
    CUSTOM = "custom"


@dataclass
class HealthMetric:
    """Individual health metric."""
    name: str
    value: float
    unit: str
    metric_type: MetricType
    timestamp: float
    threshold_warning: Optional[float] = None
    threshold_critical: Optional[float] = None
    description: str = ""
    
    def get_status(self) -> HealthStatus:
        """Get health status based on thresholds."""
        if self.threshold_critical is not None and self.value >= self.threshold_critical:
            return HealthStatus.CRITICAL
        elif self.threshold_warning is not None and self.value >= self.threshold_warning:
            return HealthStatus.WARNING
        else:
            return HealthStatus.HEALTHY


@dataclass
class HealthReport:
    """Comprehensive health report."""
    timestamp: float
    overall_status: HealthStatus
    metrics: Dict[str, HealthMetric]
    alerts: List[str]
    recommendations: List[str]
    system_info: Dict[str, Any]
    performance_summary: Dict[str, float]


class MetricCollector:
    """Base class for metric collectors."""
    
    def __init__(self, name: str, collection_interval: float = 1.0):
        """
        Initialize metric collector.
        
        Args:
            name: Name of the collector
            collection_interval: Collection interval in seconds
        """
        self.name = name
        self.collection_interval = collection_interval
        self.enabled = True
        self.last_collection = 0.0
        
    def should_collect(self) -> bool:
        """Check if it's time to collect metrics."""
        current_time = time.time()
        return (current_time - self.last_collection) >= self.collection_interval
    
    def collect(self) -> Dict[str, HealthMetric]:
        """Collect metrics. Override in subclasses."""
        self.last_collection = time.time()
        return {}


class SystemResourceCollector(MetricCollector):
    """Collects system resource metrics."""
    
    def __init__(self, collection_interval: float = 5.0):
        super().__init__("system_resources", collection_interval)
        
    def collect(self) -> Dict[str, HealthMetric]:
        """Collect system resource metrics."""
        if not self.should_collect():
            return {}
        
        metrics = {}
        current_time = time.time()
        
        # CPU metrics
        cpu_percent = psutil.cpu_percent(interval=0.1)
        metrics['cpu_usage'] = HealthMetric(
            name='cpu_usage',
            value=cpu_percent,
            unit='%',
            metric_type=MetricType.RESOURCE,
            timestamp=current_time,
            threshold_warning=70.0,
            threshold_critical=90.0,
            description='CPU utilization percentage'
        )
        
        # Memory metrics
        memory = psutil.virtual_memory()
        metrics['memory_usage'] = HealthMetric(
            name='memory_usage',
            value=memory.percent,
            unit='%',
            metric_type=MetricType.RESOURCE,
            timestamp=current_time,
            threshold_warning=80.0,
            threshold_critical=95.0,
            description='Memory utilization percentage'
        )
        
        metrics['memory_available'] = HealthMetric(
            name='memory_available',
            value=memory.available / (1024**3),  # GB
            unit='GB',
            metric_type=MetricType.RESOURCE,
            timestamp=current_time,
            threshold_critical=0.5,  # Less than 500MB available
            description='Available memory in GB'
        )
        
        # Disk metrics
        disk = psutil.disk_usage('/')
        metrics['disk_usage'] = HealthMetric(
            name='disk_usage',
            value=disk.percent,
            unit='%',
            metric_type=MetricType.RESOURCE,
            timestamp=current_time,
            threshold_warning=80.0,
            threshold_critical=95.0,
            description='Disk usage percentage'
        )
        
        # Network metrics (if available)
        try:
            network = psutil.net_io_counters()
            if hasattr(network, 'bytes_sent'):
                metrics['network_bytes_sent'] = HealthMetric(
                    name='network_bytes_sent',
                    value=network.bytes_sent / (1024**2),  # MB
                    unit='MB',
                    metric_type=MetricType.RESOURCE,
                    timestamp=current_time,
                    description='Total network bytes sent'
                )
        except:
            pass  # Network monitoring not available
        
        # GPU metrics (if available)
        if GPU_AVAILABLE:
            try:
                gpus = GPUtil.getGPUs()
                for i, gpu in enumerate(gpus):
                    metrics[f'gpu_{i}_usage'] = HealthMetric(
                        name=f'gpu_{i}_usage',
                        value=gpu.load * 100,
                        unit='%',
                        metric_type=MetricType.HARDWARE,
                        timestamp=current_time,
                        threshold_warning=80.0,
                        threshold_critical=95.0,
                        description=f'GPU {i} utilization percentage'
                    )
                    
                    metrics[f'gpu_{i}_memory'] = HealthMetric(
                        name=f'gpu_{i}_memory',
                        value=(gpu.memoryUsed / gpu.memoryTotal) * 100,
                        unit='%',
                        metric_type=MetricType.HARDWARE,
                        timestamp=current_time,
                        threshold_warning=80.0,
                        threshold_critical=95.0,
                        description=f'GPU {i} memory usage percentage'
                    )
                    
                    metrics[f'gpu_{i}_temperature'] = HealthMetric(
                        name=f'gpu_{i}_temperature',
                        value=gpu.temperature,
                        unit='°C',
                        metric_type=MetricType.HARDWARE,
                        timestamp=current_time,
                        threshold_warning=75.0,
                        threshold_critical=85.0,
                        description=f'GPU {i} temperature'
                    )
            except:
                pass  # GPU monitoring not available
        
        self.last_collection = current_time
        return metrics


class PerformanceCollector(MetricCollector):
    """Collects performance metrics."""
    
    def __init__(self, collection_interval: float = 1.0):
        super().__init__("performance", collection_interval)
        self.latency_history = deque(maxlen=100)
        self.throughput_history = deque(maxlen=100)
        self.quality_history = deque(maxlen=100)
        
    def record_fusion_performance(
        self,
        latency_ms: float,
        throughput_ops_per_sec: float,
        quality_score: float,
    ) -> None:
        """Record performance metrics from fusion operations."""
        self.latency_history.append(latency_ms)
        self.throughput_history.append(throughput_ops_per_sec)
        self.quality_history.append(quality_score)
        
    def collect(self) -> Dict[str, HealthMetric]:
        """Collect performance metrics."""
        if not self.should_collect():
            return {}
        
        metrics = {}
        current_time = time.time()
        
        # Latency metrics
        if self.latency_history:
            mean_latency = statistics.mean(self.latency_history)
            p95_latency = np.percentile(list(self.latency_history), 95)
            
            metrics['mean_latency'] = HealthMetric(
                name='mean_latency',
                value=mean_latency,
                unit='ms',
                metric_type=MetricType.PERFORMANCE,
                timestamp=current_time,
                threshold_warning=10.0,  # 10ms
                threshold_critical=50.0,  # 50ms
                description='Mean processing latency'
            )
            
            metrics['p95_latency'] = HealthMetric(
                name='p95_latency',
                value=p95_latency,
                unit='ms',
                metric_type=MetricType.PERFORMANCE,
                timestamp=current_time,
                threshold_warning=20.0,  # 20ms
                threshold_critical=100.0,  # 100ms
                description='95th percentile latency'
            )
        
        # Throughput metrics
        if self.throughput_history:
            mean_throughput = statistics.mean(self.throughput_history)
            
            metrics['throughput'] = HealthMetric(
                name='throughput',
                value=mean_throughput,
                unit='ops/sec',
                metric_type=MetricType.PERFORMANCE,
                timestamp=current_time,
                threshold_critical=10.0,  # Less than 10 ops/sec is critical
                description='Processing throughput'
            )
        
        # Quality metrics
        if self.quality_history:
            mean_quality = statistics.mean(self.quality_history)
            quality_trend = self._calculate_trend(list(self.quality_history))
            
            metrics['fusion_quality'] = HealthMetric(
                name='fusion_quality',
                value=mean_quality,
                unit='score',
                metric_type=MetricType.QUALITY,
                timestamp=current_time,
                threshold_critical=0.3,  # Quality below 0.3 is critical
                threshold_warning=0.6,   # Quality below 0.6 is warning
                description='Fusion quality score'
            )
            
            metrics['quality_trend'] = HealthMetric(
                name='quality_trend',
                value=quality_trend,
                unit='slope',
                metric_type=MetricType.QUALITY,
                timestamp=current_time,
                threshold_critical=-0.01,  # Declining quality
                description='Quality trend slope'
            )
        
        self.last_collection = current_time
        return metrics
    
    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate trend slope for values."""
        if len(values) < 2:
            return 0.0
        
        x = np.arange(len(values))
        slope, _ = np.polyfit(x, values, 1)
        return float(slope)


class NeuromorphicHealthCollector(MetricCollector):
    """Collects neuromorphic-specific health metrics."""
    
    def __init__(self, collection_interval: float = 2.0):
        super().__init__("neuromorphic", collection_interval)
        self.spike_rate_history = defaultdict(lambda: deque(maxlen=50))
        self.attention_weight_history = defaultdict(lambda: deque(maxlen=50))
        self.fusion_weight_history = defaultdict(lambda: deque(maxlen=50))
        
    def record_spike_activity(
        self,
        modality_data: Dict[str, ModalityData],
        temporal_window: float = 100.0,  # ms
    ) -> None:
        """Record spike activity from modality data."""
        for modality, data in modality_data.items():
            if len(data.spike_times) > 0:
                # Calculate spike rate (spikes per second)
                duration = temporal_window / 1000.0  # Convert to seconds
                spike_rate = len(data.spike_times) / duration
                self.spike_rate_history[modality].append(spike_rate)
    
    def record_fusion_weights(self, fusion_result: FusionResult) -> None:
        """Record fusion weights from fusion result."""
        if hasattr(fusion_result, 'fusion_weights') and fusion_result.fusion_weights:
            for modality, weight in fusion_result.fusion_weights.items():
                self.fusion_weight_history[modality].append(weight)
    
    def collect(self) -> Dict[str, HealthMetric]:
        """Collect neuromorphic-specific metrics."""
        if not self.should_collect():
            return {}
        
        metrics = {}
        current_time = time.time()
        
        # Spike rate metrics
        for modality, rates in self.spike_rate_history.items():
            if rates:
                mean_rate = statistics.mean(rates)
                rate_variance = statistics.variance(rates) if len(rates) > 1 else 0.0
                
                metrics[f'{modality}_spike_rate'] = HealthMetric(
                    name=f'{modality}_spike_rate',
                    value=mean_rate,
                    unit='spikes/sec',
                    metric_type=MetricType.CUSTOM,
                    timestamp=current_time,
                    threshold_warning=200.0,  # High spike rate warning
                    threshold_critical=500.0,  # Very high spike rate critical
                    description=f'{modality} modality spike rate'
                )
                
                metrics[f'{modality}_spike_variance'] = HealthMetric(
                    name=f'{modality}_spike_variance',
                    value=rate_variance,
                    unit='variance',
                    metric_type=MetricType.CUSTOM,
                    timestamp=current_time,
                    threshold_warning=1000.0,  # High variance warning
                    description=f'{modality} modality spike rate variance'
                )
        
        # Fusion weight balance
        if self.fusion_weight_history:
            all_weights = []
            for weights in self.fusion_weight_history.values():
                if weights:
                    all_weights.extend(weights)
            
            if all_weights:
                weight_balance = statistics.stdev(all_weights) if len(all_weights) > 1 else 0.0
                
                metrics['fusion_weight_balance'] = HealthMetric(
                    name='fusion_weight_balance',
                    value=weight_balance,
                    unit='std_dev',
                    metric_type=MetricType.QUALITY,
                    timestamp=current_time,
                    threshold_warning=0.3,  # Imbalanced fusion warning
                    threshold_critical=0.5,  # Very imbalanced fusion critical
                    description='Fusion weight balance (lower is better)'
                )
        
        # Cross-modal coherence (simplified measure)
        coherence_score = self._calculate_cross_modal_coherence()
        if coherence_score is not None:
            metrics['cross_modal_coherence'] = HealthMetric(
                name='cross_modal_coherence',
                value=coherence_score,
                unit='score',
                metric_type=MetricType.QUALITY,
                timestamp=current_time,
                threshold_critical=0.3,  # Low coherence is critical
                threshold_warning=0.6,   # Moderate coherence is warning
                description='Cross-modal coherence score'
            )
        
        self.last_collection = current_time
        return metrics
    
    def _calculate_cross_modal_coherence(self) -> Optional[float]:
        """Calculate cross-modal coherence score."""
        if len(self.spike_rate_history) < 2:
            return None
        
        # Simple coherence based on correlation of spike rates
        modalities = list(self.spike_rate_history.keys())
        correlations = []
        
        for i in range(len(modalities)):
            for j in range(i + 1, len(modalities)):
                rates_i = list(self.spike_rate_history[modalities[i]])
                rates_j = list(self.spike_rate_history[modalities[j]])
                
                if len(rates_i) > 1 and len(rates_j) > 1:
                    # Ensure same length for correlation
                    min_len = min(len(rates_i), len(rates_j))
                    rates_i = rates_i[-min_len:]
                    rates_j = rates_j[-min_len:]
                    
                    if min_len > 1:
                        correlation = np.corrcoef(rates_i, rates_j)[0, 1]
                        if not np.isnan(correlation):
                            correlations.append(abs(correlation))  # Absolute correlation
        
        if correlations:
            return statistics.mean(correlations)
        
        return None


class HealthMonitor:
    """
    Comprehensive health monitoring system for neuromorphic systems.
    
    Features:
    - Real-time metric collection
    - Automated alerting and notifications
    - Predictive health analysis
    - Performance trend analysis
    - Automated recovery triggers
    """
    
    def __init__(
        self,
        monitoring_config: Optional[Dict[str, Any]] = None,
        enable_alerts: bool = True,
        enable_recovery: bool = True,
    ):
        """
        Initialize health monitoring system.
        
        Args:
            monitoring_config: Monitoring configuration
            enable_alerts: Enable alerting system
            enable_recovery: Enable automated recovery
        """
        self.config = monitoring_config or self._default_config()
        self.enable_alerts = enable_alerts
        self.enable_recovery = enable_recovery
        
        # Metric collectors
        self.collectors = {
            'system': SystemResourceCollector(),
            'performance': PerformanceCollector(),
            'neuromorphic': NeuromorphicHealthCollector(),
        }
        
        # Health tracking
        self.current_metrics: Dict[str, HealthMetric] = {}
        self.health_history: List[HealthReport] = []
        self.alerts_sent = set()
        
        # Monitoring state
        self.monitoring_active = False
        self.monitor_thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()
        
        # Alert callbacks
        self.alert_callbacks: List[Callable[[str, HealthMetric], None]] = []
        self.recovery_callbacks: List[Callable[[HealthStatus], None]] = []
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        self.logger.info("HealthMonitor initialized")
    
    def _default_config(self) -> Dict[str, Any]:
        """Default monitoring configuration."""
        return {
            'collection_interval': 5.0,  # seconds
            'max_health_history': 1000,
            'alert_cooldown': 300.0,  # 5 minutes
            'recovery_threshold': 3,  # Trigger recovery after 3 critical metrics
        }
    
    def add_alert_callback(self, callback: Callable[[str, HealthMetric], None]) -> None:
        """Add alert callback function."""
        self.alert_callbacks.append(callback)
    
    def add_recovery_callback(self, callback: Callable[[HealthStatus], None]) -> None:
        """Add recovery callback function."""
        self.recovery_callbacks.append(callback)
    
    def start_monitoring(self) -> None:
        """Start continuous health monitoring."""
        if self.monitoring_active:
            self.logger.warning("Monitoring already active")
            return
        
        self.monitoring_active = True
        self.stop_event.clear()
        
        self.monitor_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True
        )
        self.monitor_thread.start()
        
        self.logger.info("Health monitoring started")
    
    def stop_monitoring(self) -> None:
        """Stop health monitoring."""
        if not self.monitoring_active:
            return
        
        self.monitoring_active = False
        self.stop_event.set()
        
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=5.0)
        
        self.logger.info("Health monitoring stopped")
    
    def _monitoring_loop(self) -> None:
        """Main monitoring loop."""
        interval = self.config.get('collection_interval', 5.0)
        
        while not self.stop_event.wait(interval):
            try:
                self._collect_all_metrics()
                self._generate_health_report()
                self._check_alerts()
                self._check_recovery_triggers()
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                continue
    
    @robust_function(attempt_recovery=True)
    def _collect_all_metrics(self) -> None:
        """Collect metrics from all collectors."""
        for name, collector in self.collectors.items():
            if collector.enabled:
                try:
                    metrics = collector.collect()
                    for metric_name, metric in metrics.items():
                        self.current_metrics[metric_name] = metric
                except Exception as e:
                    self.logger.warning(f"Failed to collect metrics from {name}: {e}")
    
    def _generate_health_report(self) -> HealthReport:
        """Generate comprehensive health report."""
        current_time = time.time()
        
        # Determine overall status
        overall_status = HealthStatus.HEALTHY
        alerts = []
        recommendations = []
        
        critical_count = 0
        warning_count = 0
        
        for metric in self.current_metrics.values():
            status = metric.get_status()
            
            if status == HealthStatus.CRITICAL:
                critical_count += 1
                overall_status = HealthStatus.CRITICAL
                alerts.append(f"CRITICAL: {metric.name} = {metric.value} {metric.unit}")
                
            elif status == HealthStatus.WARNING and overall_status != HealthStatus.CRITICAL:
                warning_count += 1
                overall_status = HealthStatus.WARNING
                alerts.append(f"WARNING: {metric.name} = {metric.value} {metric.unit}")
        
        # Generate recommendations
        recommendations = self._generate_recommendations()
        
        # Create performance summary
        performance_summary = self._create_performance_summary()
        
        # System information
        system_info = {
            'timestamp': current_time,
            'monitoring_duration': current_time - (self.health_history[0].timestamp if self.health_history else current_time),
            'total_metrics': len(self.current_metrics),
            'critical_metrics': critical_count,
            'warning_metrics': warning_count,
        }
        
        report = HealthReport(
            timestamp=current_time,
            overall_status=overall_status,
            metrics=self.current_metrics.copy(),
            alerts=alerts,
            recommendations=recommendations,
            system_info=system_info,
            performance_summary=performance_summary,
        )
        
        # Store in history
        self.health_history.append(report)
        max_history = self.config.get('max_health_history', 1000)
        if len(self.health_history) > max_history:
            self.health_history = self.health_history[-max_history:]
        
        return report
    
    def _generate_recommendations(self) -> List[str]:
        """Generate health recommendations."""
        recommendations = []
        
        # Check for specific metric conditions
        for metric in self.current_metrics.values():
            status = metric.get_status()
            
            if status in [HealthStatus.WARNING, HealthStatus.CRITICAL]:
                if metric.name == 'cpu_usage':
                    recommendations.append("Consider reducing processing load or scaling horizontally")
                elif metric.name == 'memory_usage':
                    recommendations.append("Monitor for memory leaks and consider increasing available memory")
                elif metric.name == 'mean_latency':
                    recommendations.append("Investigate processing bottlenecks and optimize algorithms")
                elif metric.name == 'fusion_quality':
                    recommendations.append("Review fusion parameters and data quality")
                elif 'gpu' in metric.name and 'temperature' in metric.name:
                    recommendations.append("Check GPU cooling and reduce workload if necessary")
                elif metric.name == 'cross_modal_coherence':
                    recommendations.append("Investigate cross-modal synchronization issues")
        
        # General recommendations based on trends
        if len(self.health_history) > 10:
            recent_reports = self.health_history[-10:]
            
            # Check for degrading trends
            quality_metrics = []
            for report in recent_reports:
                if 'fusion_quality' in report.metrics:
                    quality_metrics.append(report.metrics['fusion_quality'].value)
            
            if len(quality_metrics) > 5:
                trend_slope = self._calculate_trend(quality_metrics)
                if trend_slope < -0.01:  # Declining quality
                    recommendations.append("Quality is declining - consider system maintenance")
        
        return list(set(recommendations))  # Remove duplicates
    
    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate trend slope."""
        if len(values) < 2:
            return 0.0
        
        x = np.arange(len(values))
        slope, _ = np.polyfit(x, values, 1)
        return float(slope)
    
    def _create_performance_summary(self) -> Dict[str, float]:
        """Create performance summary."""
        summary = {}
        
        # Performance metrics
        perf_metrics = ['mean_latency', 'p95_latency', 'throughput', 'fusion_quality']
        
        for metric_name in perf_metrics:
            if metric_name in self.current_metrics:
                summary[metric_name] = self.current_metrics[metric_name].value
        
        # Resource utilization
        resource_metrics = ['cpu_usage', 'memory_usage', 'disk_usage']
        
        for metric_name in resource_metrics:
            if metric_name in self.current_metrics:
                summary[metric_name] = self.current_metrics[metric_name].value
        
        return summary
    
    def _check_alerts(self) -> None:
        """Check for alert conditions and trigger notifications."""
        if not self.enable_alerts:
            return
        
        current_time = time.time()
        cooldown = self.config.get('alert_cooldown', 300.0)
        
        for metric in self.current_metrics.values():
            status = metric.get_status()
            
            if status in [HealthStatus.WARNING, HealthStatus.CRITICAL]:
                alert_key = f"{metric.name}_{status.value}"
                
                # Check cooldown
                if alert_key in self.alerts_sent:
                    last_alert_time = self.alerts_sent[alert_key]
                    if current_time - last_alert_time < cooldown:
                        continue  # Still in cooldown
                
                # Send alert
                for callback in self.alert_callbacks:
                    try:
                        callback(f"{status.value.upper()}: {metric.name}", metric)
                    except Exception as e:
                        self.logger.error(f"Alert callback failed: {e}")
                
                # Record alert
                self.alerts_sent[alert_key] = current_time
                
                self.logger.warning(
                    f"Health alert: {metric.name} = {metric.value} {metric.unit} [{status.value}]"
                )
    
    def _check_recovery_triggers(self) -> None:
        """Check for conditions that trigger automated recovery."""
        if not self.enable_recovery:
            return
        
        # Count critical metrics
        critical_count = sum(
            1 for metric in self.current_metrics.values()
            if metric.get_status() == HealthStatus.CRITICAL
        )
        
        recovery_threshold = self.config.get('recovery_threshold', 3)
        
        if critical_count >= recovery_threshold:
            self.logger.critical(f"Triggering recovery: {critical_count} critical metrics")
            
            # Trigger recovery callbacks
            for callback in self.recovery_callbacks:
                try:
                    callback(HealthStatus.CRITICAL)
                except Exception as e:
                    self.logger.error(f"Recovery callback failed: {e}")
    
    def get_current_status(self) -> HealthStatus:
        """Get current overall health status."""
        if self.health_history:
            return self.health_history[-1].overall_status
        return HealthStatus.HEALTHY
    
    def get_latest_report(self) -> Optional[HealthReport]:
        """Get latest health report."""
        return self.health_history[-1] if self.health_history else None
    
    def record_fusion_performance(
        self,
        latency_ms: float,
        throughput_ops_per_sec: float,
        quality_score: float,
    ) -> None:
        """Record performance metrics from fusion operations."""
        if 'performance' in self.collectors:
            self.collectors['performance'].record_fusion_performance(
                latency_ms, throughput_ops_per_sec, quality_score
            )
    
    def record_spike_activity(self, modality_data: Dict[str, ModalityData]) -> None:
        """Record spike activity from modality data."""
        if 'neuromorphic' in self.collectors:
            self.collectors['neuromorphic'].record_spike_activity(modality_data)
    
    def record_fusion_weights(self, fusion_result: FusionResult) -> None:
        """Record fusion weights from fusion result."""
        if 'neuromorphic' in self.collectors:
            self.collectors['neuromorphic'].record_fusion_weights(fusion_result)
    
    def export_health_report(self, filepath: str, include_history: bool = True) -> None:
        """Export comprehensive health report to file."""
        export_data = {
            'timestamp': time.time(),
            'current_status': self.get_current_status().value,
            'latest_report': self.get_latest_report().__dict__ if self.get_latest_report() else None,
            'configuration': self.config,
        }
        
        if include_history:
            export_data['health_history'] = [
                {
                    'timestamp': report.timestamp,
                    'overall_status': report.overall_status.value,
                    'metrics': {k: v.__dict__ for k, v in report.metrics.items()},
                    'alerts': report.alerts,
                    'recommendations': report.recommendations,
                    'system_info': report.system_info,
                    'performance_summary': report.performance_summary,
                }
                for report in self.health_history[-100:]  # Last 100 reports
            ]
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        self.logger.info(f"Health report exported to {filepath}")


# Utility functions for health monitoring
def create_default_health_monitor() -> HealthMonitor:
    """Create health monitor with default configuration."""
    return HealthMonitor()


def create_production_health_monitor() -> HealthMonitor:
    """Create health monitor optimized for production."""
    config = {
        'collection_interval': 10.0,  # Less frequent collection
        'max_health_history': 2000,   # More history
        'alert_cooldown': 600.0,      # 10-minute cooldown
        'recovery_threshold': 2,      # More sensitive recovery
    }
    
    monitor = HealthMonitor(config, enable_alerts=True, enable_recovery=True)
    
    # Add default alert callback (logging)
    def log_alert(alert_message: str, metric: HealthMetric) -> None:
        logging.getLogger('health_alerts').critical(f"{alert_message}: {metric.description}")
    
    monitor.add_alert_callback(log_alert)
    
    return monitor


# Export key components
__all__ = [
    'HealthStatus',
    'MetricType',
    'HealthMetric',
    'HealthReport',
    'MetricCollector',
    'SystemResourceCollector',
    'PerformanceCollector',
    'NeuromorphicHealthCollector',
    'HealthMonitor',
    'create_default_health_monitor',
    'create_production_health_monitor',
]