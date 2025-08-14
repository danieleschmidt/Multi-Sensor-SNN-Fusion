"""
Production Operational Dashboard

Comprehensive real-time dashboard system for monitoring, alerting,
and managing neuromorphic computing production deployments.
"""

import json
import logging
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import statistics
from collections import defaultdict, deque


class DashboardType(Enum):
    SYSTEM_OVERVIEW = "system_overview"
    PERFORMANCE_METRICS = "performance_metrics"
    ALERT_MANAGEMENT = "alert_management"
    DEPLOYMENT_STATUS = "deployment_status"
    NEUROMORPHIC_METRICS = "neuromorphic_metrics"


@dataclass
class MetricSnapshot:
    """Real-time metric snapshot."""
    timestamp: float
    metric_name: str
    value: float
    unit: str
    source: str
    tags: Dict[str, str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PerformanceSummary:
    """Performance metrics summary."""
    timestamp: float
    avg_response_time_ms: float
    max_response_time_ms: float
    requests_per_second: float
    error_rate_percent: float
    cpu_utilization_percent: float
    memory_utilization_percent: float
    active_connections: int
    queue_length: int
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class RealTimeMetrics:
    """
    Real-time metrics collection and aggregation system.
    
    Features:
    - High-frequency metrics collection
    - Statistical aggregation and windowing
    - Threshold-based alerting
    - Historical trend analysis
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._get_default_config()
        self.logger = logging.getLogger(__name__)
        
        # Metrics storage
        self.metrics_buffer: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=self.config['max_metrics_buffer'])
        )
        self.aggregated_metrics: Dict[str, List[MetricSnapshot]] = defaultdict(list)
        
        # Threading
        self.collection_thread: Optional[threading.Thread] = None
        self.aggregation_thread: Optional[threading.Thread] = None
        self.is_running = False
        
        # Performance tracking
        self.performance_history: List[PerformanceSummary] = []
        
        self.logger.info("Real-time metrics system initialized")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default metrics configuration."""
        return {
            'collection_interval_seconds': 5,
            'aggregation_interval_seconds': 60,
            'max_metrics_buffer': 1000,
            'retention_hours': 24,
            'alert_thresholds': {
                'cpu_percent': 85,
                'memory_percent': 85,
                'response_time_ms': 5000,
                'error_rate_percent': 5,
            },
        }
    
    def start_collection(self) -> None:
        """Start metrics collection."""
        if self.is_running:
            return
        
        self.is_running = True
        self.logger.info("Starting real-time metrics collection...")
        
        # Start collection thread
        self.collection_thread = threading.Thread(
            target=self._metrics_collection_loop,
            name="metrics-collector",
            daemon=True
        )
        self.collection_thread.start()
        
        # Start aggregation thread
        self.aggregation_thread = threading.Thread(
            target=self._metrics_aggregation_loop,
            name="metrics-aggregator",
            daemon=True
        )
        self.aggregation_thread.start()
        
        self.logger.info("Real-time metrics collection started")
    
    def stop_collection(self) -> None:
        """Stop metrics collection."""
        self.logger.info("Stopping metrics collection...")
        self.is_running = False
        
        if self.collection_thread and self.collection_thread.is_alive():
            self.collection_thread.join(timeout=5)
        
        if self.aggregation_thread and self.aggregation_thread.is_alive():
            self.aggregation_thread.join(timeout=5)
        
        self.logger.info("Metrics collection stopped")
    
    def _metrics_collection_loop(self) -> None:
        """Main metrics collection loop."""
        while self.is_running:
            try:
                # Collect system metrics
                self._collect_system_metrics()
                
                # Collect application metrics
                self._collect_application_metrics()
                
                # Collect neuromorphic metrics
                self._collect_neuromorphic_metrics()
                
                time.sleep(self.config['collection_interval_seconds'])
                
            except Exception as e:
                self.logger.error(f"Metrics collection error: {e}")
                time.sleep(1)
    
    def _collect_system_metrics(self) -> None:
        """Collect system-level metrics."""
        import psutil
        
        current_time = time.time()
        
        # CPU metrics
        cpu_percent = psutil.cpu_percent()
        self._add_metric("cpu_percent", cpu_percent, "%", "system", current_time)
        
        # Memory metrics
        memory = psutil.virtual_memory()
        self._add_metric("memory_percent", memory.percent, "%", "system", current_time)
        self._add_metric("memory_available_gb", memory.available / (1024**3), "GB", "system", current_time)
        
        # Disk metrics
        disk = psutil.disk_usage('/')
        disk_percent = (disk.used / disk.total) * 100
        self._add_metric("disk_percent", disk_percent, "%", "system", current_time)
        
        # Network metrics
        net_io = psutil.net_io_counters()
        self._add_metric("network_bytes_sent", net_io.bytes_sent, "bytes", "system", current_time)
        self._add_metric("network_bytes_recv", net_io.bytes_recv, "bytes", "system", current_time)
        
        # Process metrics
        self._add_metric("process_count", len(psutil.pids()), "count", "system", current_time)
    
    def _collect_application_metrics(self) -> None:
        """Collect application-level metrics."""
        import random
        
        current_time = time.time()
        
        # Simulate application metrics
        # In production, these would come from actual application instrumentation
        
        # Response time metrics
        response_time = random.normalvariate(150, 50)  # 150ms average
        response_time = max(10, response_time)  # Minimum 10ms
        self._add_metric("response_time_ms", response_time, "ms", "application", current_time)
        
        # Request rate
        request_rate = random.normalvariate(100, 20)  # 100 RPS average
        request_rate = max(0, request_rate)
        self._add_metric("requests_per_second", request_rate, "rps", "application", current_time)
        
        # Error rate
        error_rate = random.exponential(0.5)  # Low error rate
        error_rate = min(10, error_rate)  # Cap at 10%
        self._add_metric("error_rate_percent", error_rate, "%", "application", current_time)
        
        # Active connections
        active_connections = random.randint(50, 200)
        self._add_metric("active_connections", active_connections, "count", "application", current_time)
        
        # Queue metrics
        queue_length = random.poisson(5)
        self._add_metric("queue_length", queue_length, "count", "application", current_time)
    
    def _collect_neuromorphic_metrics(self) -> None:
        """Collect neuromorphic hardware metrics."""
        import random
        
        current_time = time.time()
        
        # Simulate neuromorphic hardware metrics
        # In production, these would interface with actual neuromorphic hardware
        
        # Spike rates
        spike_rate = random.normalvariate(1000, 200)  # 1000 spikes/sec average
        spike_rate = max(0, spike_rate)
        self._add_metric("spike_rate_hz", spike_rate, "Hz", "neuromorphic", current_time)
        
        # Neuron utilization
        neuron_utilization = random.uniform(40, 80)
        self._add_metric("neuron_utilization_percent", neuron_utilization, "%", "neuromorphic", current_time)
        
        # Power consumption
        power_consumption = random.normalvariate(150, 30)  # 150W average
        power_consumption = max(50, power_consumption)
        self._add_metric("power_consumption_watts", power_consumption, "W", "neuromorphic", current_time)
        
        # Temperature
        temperature = random.normalvariate(65, 10)  # 65°C average
        temperature = max(30, temperature)
        self._add_metric("temperature_celsius", temperature, "°C", "neuromorphic", current_time)
        
        # Inference latency
        inference_latency = random.exponential(2.0)  # 2ms average
        self._add_metric("inference_latency_ms", inference_latency, "ms", "neuromorphic", current_time)
    
    def _add_metric(self, metric_name: str, value: float, unit: str, source: str, timestamp: float) -> None:
        """Add metric to buffer."""
        metric = MetricSnapshot(
            timestamp=timestamp,
            metric_name=metric_name,
            value=value,
            unit=unit,
            source=source,
        )
        
        self.metrics_buffer[metric_name].append(metric)
    
    def _metrics_aggregation_loop(self) -> None:
        """Aggregate metrics periodically."""
        while self.is_running:
            try:
                self._aggregate_metrics()
                self._generate_performance_summary()
                self._cleanup_old_metrics()
                
                time.sleep(self.config['aggregation_interval_seconds'])
                
            except Exception as e:
                self.logger.error(f"Metrics aggregation error: {e}")
                time.sleep(5)
    
    def _aggregate_metrics(self) -> None:
        """Aggregate buffered metrics."""
        current_time = time.time()
        
        for metric_name, buffer in self.metrics_buffer.items():
            if not buffer:
                continue
            
            # Calculate statistics
            values = [m.value for m in buffer]
            
            aggregated = MetricSnapshot(
                timestamp=current_time,
                metric_name=metric_name,
                value=statistics.mean(values),
                unit=buffer[0].unit,
                source=buffer[0].source,
                tags={
                    'min': min(values),
                    'max': max(values),
                    'count': len(values),
                    'std': statistics.stdev(values) if len(values) > 1 else 0,
                }
            )
            
            self.aggregated_metrics[metric_name].append(aggregated)
    
    def _generate_performance_summary(self) -> None:
        """Generate performance summary."""
        current_time = time.time()
        
        # Get recent metrics
        def get_recent_metric_value(name: str, default: float = 0.0) -> float:
            if name in self.metrics_buffer and self.metrics_buffer[name]:
                return self.metrics_buffer[name][-1].value
            return default
        
        summary = PerformanceSummary(
            timestamp=current_time,
            avg_response_time_ms=get_recent_metric_value("response_time_ms", 0.0),
            max_response_time_ms=get_recent_metric_value("response_time_ms", 0.0) * 1.5,  # Estimate
            requests_per_second=get_recent_metric_value("requests_per_second", 0.0),
            error_rate_percent=get_recent_metric_value("error_rate_percent", 0.0),
            cpu_utilization_percent=get_recent_metric_value("cpu_percent", 0.0),
            memory_utilization_percent=get_recent_metric_value("memory_percent", 0.0),
            active_connections=int(get_recent_metric_value("active_connections", 0)),
            queue_length=int(get_recent_metric_value("queue_length", 0)),
        )
        
        self.performance_history.append(summary)
        
        # Keep only recent history
        cutoff_time = current_time - (self.config['retention_hours'] * 3600)
        self.performance_history = [
            s for s in self.performance_history 
            if s.timestamp > cutoff_time
        ]
    
    def _cleanup_old_metrics(self) -> None:
        """Clean up old aggregated metrics."""
        current_time = time.time()
        cutoff_time = current_time - (self.config['retention_hours'] * 3600)
        
        for metric_name in self.aggregated_metrics:
            self.aggregated_metrics[metric_name] = [
                m for m in self.aggregated_metrics[metric_name] 
                if m.timestamp > cutoff_time
            ]
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current metrics snapshot."""
        current_metrics = {}
        
        for metric_name, buffer in self.metrics_buffer.items():
            if buffer:
                latest = buffer[-1]
                current_metrics[metric_name] = {
                    'value': latest.value,
                    'unit': latest.unit,
                    'source': latest.source,
                    'timestamp': latest.timestamp,
                }
        
        return current_metrics
    
    def get_performance_trends(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get performance trends over specified time period."""
        cutoff_time = time.time() - (hours * 3600)
        
        return [
            summary.to_dict() 
            for summary in self.performance_history 
            if summary.timestamp > cutoff_time
        ]


class AlertDashboard:
    """
    Real-time alert management dashboard.
    
    Features:
    - Alert visualization and management
    - Alert routing and escalation
    - Historical alert analysis
    - Custom alert rules
    """
    
    def __init__(self, health_monitor):
        self.health_monitor = health_monitor
        self.logger = logging.getLogger(__name__)
        
        # Alert management
        self.alert_rules: List[Dict[str, Any]] = []
        self.alert_handlers: Dict[str, Callable] = {}
        
        # Initialize default alert rules
        self._initialize_default_alert_rules()
    
    def _initialize_default_alert_rules(self) -> None:
        """Initialize default alert rules."""
        self.alert_rules = [
            {
                'name': 'high_cpu_usage',
                'condition': lambda metrics: metrics.get('cpu_percent', {}).get('value', 0) > 85,
                'severity': 'warning',
                'message': 'CPU usage is high',
            },
            {
                'name': 'high_memory_usage',
                'condition': lambda metrics: metrics.get('memory_percent', {}).get('value', 0) > 85,
                'severity': 'warning',
                'message': 'Memory usage is high',
            },
            {
                'name': 'high_response_time',
                'condition': lambda metrics: metrics.get('response_time_ms', {}).get('value', 0) > 5000,
                'severity': 'critical',
                'message': 'Response time is too high',
            },
            {
                'name': 'high_error_rate',
                'condition': lambda metrics: metrics.get('error_rate_percent', {}).get('value', 0) > 5,
                'severity': 'critical',
                'message': 'Error rate is too high',
            },
        ]
    
    def evaluate_alert_rules(self, metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Evaluate alert rules against current metrics."""
        triggered_alerts = []
        
        for rule in self.alert_rules:
            try:
                if rule['condition'](metrics):
                    alert = {
                        'rule_name': rule['name'],
                        'severity': rule['severity'],
                        'message': rule['message'],
                        'timestamp': time.time(),
                        'metrics_snapshot': metrics,
                    }
                    triggered_alerts.append(alert)
                    
            except Exception as e:
                self.logger.error(f"Error evaluating alert rule {rule['name']}: {e}")
        
        return triggered_alerts
    
    def get_alert_summary(self) -> Dict[str, Any]:
        """Get alert summary from health monitor."""
        if not self.health_monitor:
            return {'active_alerts': {}, 'resolved_alerts': []}
        
        # Get alerts from health monitor
        active_alerts = getattr(self.health_monitor, 'active_alerts', {})
        resolved_alerts = getattr(self.health_monitor, 'resolved_alerts', [])
        
        # Categorize by severity
        alerts_by_severity = defaultdict(list)
        for alert in active_alerts.values():
            severity = getattr(alert, 'severity', 'unknown')
            if hasattr(severity, 'value'):
                severity = severity.value
            alerts_by_severity[severity].append(alert.to_dict() if hasattr(alert, 'to_dict') else alert)
        
        return {
            'active_alerts': dict(alerts_by_severity),
            'resolved_alerts': [
                alert.to_dict() if hasattr(alert, 'to_dict') else alert 
                for alert in resolved_alerts[-20:]  # Last 20 resolved alerts
            ],
            'total_active': len(active_alerts),
            'total_resolved_today': len([
                a for a in resolved_alerts 
                if getattr(a, 'resolved_timestamp', 0) > time.time() - 86400
            ]),
        }


class PerformanceAnalytics:
    """
    Advanced performance analytics and insights.
    
    Features:
    - Statistical analysis of performance trends
    - Anomaly detection
    - Performance forecasting
    - Optimization recommendations
    """
    
    def __init__(self, metrics_system: RealTimeMetrics):
        self.metrics_system = metrics_system
        self.logger = logging.getLogger(__name__)
    
    def analyze_performance_trends(self, hours: int = 24) -> Dict[str, Any]:
        """Analyze performance trends over specified period."""
        trends = self.metrics_system.get_performance_trends(hours)
        
        if not trends:
            return {'error': 'No performance data available'}
        
        # Calculate trend analysis
        response_times = [t['avg_response_time_ms'] for t in trends]
        cpu_usage = [t['cpu_utilization_percent'] for t in trends]
        memory_usage = [t['memory_utilization_percent'] for t in trends]
        error_rates = [t['error_rate_percent'] for t in trends]
        
        analysis = {
            'timespan_hours': hours,
            'data_points': len(trends),
            'response_time_analysis': {
                'current': response_times[-1] if response_times else 0,
                'average': statistics.mean(response_times) if response_times else 0,
                'min': min(response_times) if response_times else 0,
                'max': max(response_times) if response_times else 0,
                'trend': self._calculate_trend(response_times),
            },
            'cpu_analysis': {
                'current': cpu_usage[-1] if cpu_usage else 0,
                'average': statistics.mean(cpu_usage) if cpu_usage else 0,
                'peak': max(cpu_usage) if cpu_usage else 0,
                'trend': self._calculate_trend(cpu_usage),
            },
            'memory_analysis': {
                'current': memory_usage[-1] if memory_usage else 0,
                'average': statistics.mean(memory_usage) if memory_usage else 0,
                'peak': max(memory_usage) if memory_usage else 0,
                'trend': self._calculate_trend(memory_usage),
            },
            'error_analysis': {
                'current': error_rates[-1] if error_rates else 0,
                'average': statistics.mean(error_rates) if error_rates else 0,
                'peak': max(error_rates) if error_rates else 0,
                'trend': self._calculate_trend(error_rates),
            },
        }
        
        # Add recommendations
        analysis['recommendations'] = self._generate_recommendations(analysis)
        
        return analysis
    
    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction from time series data."""
        if len(values) < 2:
            return 'insufficient_data'
        
        # Simple linear trend calculation
        x = list(range(len(values)))
        n = len(values)
        
        sum_x = sum(x)
        sum_y = sum(values)
        sum_xy = sum(x[i] * values[i] for i in range(n))
        sum_x2 = sum(x[i] * x[i] for i in range(n))
        
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
        
        if abs(slope) < 0.01:
            return 'stable'
        elif slope > 0:
            return 'increasing'
        else:
            return 'decreasing'
    
    def _generate_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate performance optimization recommendations."""
        recommendations = []
        
        # Response time recommendations
        rt_current = analysis['response_time_analysis']['current']
        rt_avg = analysis['response_time_analysis']['average']
        
        if rt_current > 1000:
            recommendations.append("Response time is high - consider optimizing database queries")
        
        if analysis['response_time_analysis']['trend'] == 'increasing':
            recommendations.append("Response time trending upward - investigate performance bottlenecks")
        
        # CPU recommendations
        cpu_current = analysis['cpu_analysis']['current']
        cpu_peak = analysis['cpu_analysis']['peak']
        
        if cpu_current > 80:
            recommendations.append("High CPU usage - consider horizontal scaling")
        
        if cpu_peak > 90:
            recommendations.append("CPU usage spikes detected - optimize compute-intensive operations")
        
        # Memory recommendations
        mem_current = analysis['memory_analysis']['current']
        mem_trend = analysis['memory_analysis']['trend']
        
        if mem_current > 80:
            recommendations.append("High memory usage - check for memory leaks or increase memory allocation")
        
        if mem_trend == 'increasing':
            recommendations.append("Memory usage trending upward - investigate potential memory leaks")
        
        # Error rate recommendations
        error_current = analysis['error_analysis']['current']
        error_trend = analysis['error_analysis']['trend']
        
        if error_current > 2:
            recommendations.append("Elevated error rate - investigate application logs")
        
        if error_trend == 'increasing':
            recommendations.append("Error rate increasing - implement additional error handling")
        
        if not recommendations:
            recommendations.append("System performance is within normal parameters")
        
        return recommendations
    
    def detect_anomalies(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Detect performance anomalies."""
        trends = self.metrics_system.get_performance_trends(hours)
        
        if len(trends) < 10:
            return []
        
        anomalies = []
        
        # Simple anomaly detection using standard deviation
        for metric in ['avg_response_time_ms', 'cpu_utilization_percent', 'error_rate_percent']:
            values = [t[metric] for t in trends]
            
            if len(values) < 5:
                continue
            
            mean_val = statistics.mean(values)
            std_val = statistics.stdev(values)
            
            # Detect values that are more than 2 standard deviations from mean
            for i, value in enumerate(values):
                if abs(value - mean_val) > 2 * std_val:
                    anomalies.append({
                        'metric': metric,
                        'value': value,
                        'expected_range': [mean_val - 2*std_val, mean_val + 2*std_val],
                        'timestamp': trends[i]['timestamp'],
                        'severity': 'high' if abs(value - mean_val) > 3 * std_val else 'medium',
                    })
        
        return anomalies


class ProductionDashboard:
    """
    Main production dashboard orchestrator.
    
    Combines all dashboard components into a unified interface
    for production monitoring and management.
    """
    
    def __init__(self, health_monitor=None):
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.metrics_system = RealTimeMetrics()
        self.alert_dashboard = AlertDashboard(health_monitor)
        self.performance_analytics = PerformanceAnalytics(self.metrics_system)
        
        # Dashboard state
        self.is_active = False
        self.dashboard_config = self._get_default_dashboard_config()
        
        self.logger.info("Production dashboard initialized")
    
    def _get_default_dashboard_config(self) -> Dict[str, Any]:
        """Get default dashboard configuration."""
        return {
            'refresh_interval_seconds': 30,
            'alert_sound_enabled': True,
            'auto_refresh_enabled': True,
            'theme': 'dark',
            'layout': 'grid',
            'visible_panels': [
                'system_overview',
                'performance_metrics',
                'alert_summary',
                'neuromorphic_metrics',
            ],
        }
    
    def start_dashboard(self) -> None:
        """Start the production dashboard."""
        if self.is_active:
            return
        
        self.logger.info("Starting production dashboard...")
        
        # Start metrics collection
        self.metrics_system.start_collection()
        
        self.is_active = True
        self.logger.info("Production dashboard started successfully")
    
    def stop_dashboard(self) -> None:
        """Stop the production dashboard."""
        self.logger.info("Stopping production dashboard...")
        
        # Stop metrics collection
        self.metrics_system.stop_collection()
        
        self.is_active = False
        self.logger.info("Production dashboard stopped")
    
    def get_dashboard_data(self, dashboard_type: DashboardType = DashboardType.SYSTEM_OVERVIEW) -> Dict[str, Any]:
        """Get dashboard data based on type."""
        current_time = time.time()
        
        base_data = {
            'timestamp': current_time,
            'dashboard_type': dashboard_type.value,
            'is_active': self.is_active,
        }
        
        if dashboard_type == DashboardType.SYSTEM_OVERVIEW:
            return {
                **base_data,
                'current_metrics': self.metrics_system.get_current_metrics(),
                'alert_summary': self.alert_dashboard.get_alert_summary(),
                'performance_summary': self.performance_analytics.analyze_performance_trends(1),  # Last hour
            }
        
        elif dashboard_type == DashboardType.PERFORMANCE_METRICS:
            return {
                **base_data,
                'performance_trends': self.metrics_system.get_performance_trends(24),
                'performance_analysis': self.performance_analytics.analyze_performance_trends(24),
                'anomalies': self.performance_analytics.detect_anomalies(24),
            }
        
        elif dashboard_type == DashboardType.ALERT_MANAGEMENT:
            return {
                **base_data,
                'alert_summary': self.alert_dashboard.get_alert_summary(),
                'alert_rules': self.alert_dashboard.alert_rules,
                'recent_metrics': self.metrics_system.get_current_metrics(),
            }
        
        elif dashboard_type == DashboardType.NEUROMORPHIC_METRICS:
            current_metrics = self.metrics_system.get_current_metrics()
            neuromorphic_metrics = {
                k: v for k, v in current_metrics.items() 
                if v.get('source') == 'neuromorphic'
            }
            
            return {
                **base_data,
                'neuromorphic_metrics': neuromorphic_metrics,
                'neuromorphic_trends': [
                    summary for summary in self.metrics_system.get_performance_trends(6)
                ],
            }
        
        else:
            return base_data
    
    def export_dashboard_report(self, filepath: str, hours: int = 24) -> None:
        """Export comprehensive dashboard report."""
        report = {
            'report_metadata': {
                'generated_at': time.time(),
                'generated_date': datetime.now().isoformat(),
                'timespan_hours': hours,
                'dashboard_config': self.dashboard_config,
            },
            'system_overview': self.get_dashboard_data(DashboardType.SYSTEM_OVERVIEW),
            'performance_metrics': self.get_dashboard_data(DashboardType.PERFORMANCE_METRICS),
            'alert_management': self.get_dashboard_data(DashboardType.ALERT_MANAGEMENT),
            'neuromorphic_metrics': self.get_dashboard_data(DashboardType.NEUROMORPHIC_METRICS),
        }
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)
        
        self.logger.info(f"Dashboard report exported to {filepath}")
    
    def get_system_health_score(self) -> Dict[str, Any]:
        """Calculate overall system health score."""
        current_metrics = self.metrics_system.get_current_metrics()
        alert_summary = self.alert_dashboard.get_alert_summary()
        
        # Calculate health score based on various factors
        health_factors = {
            'cpu_health': 100 - min(100, current_metrics.get('cpu_percent', {}).get('value', 0)),
            'memory_health': 100 - min(100, current_metrics.get('memory_percent', {}).get('value', 0)),
            'response_time_health': max(0, 100 - (current_metrics.get('response_time_ms', {}).get('value', 0) / 50)),
            'error_rate_health': max(0, 100 - current_metrics.get('error_rate_percent', {}).get('value', 0) * 10),
            'alert_health': max(0, 100 - alert_summary['total_active'] * 10),
        }
        
        # Calculate weighted overall score
        weights = {
            'cpu_health': 0.2,
            'memory_health': 0.2,
            'response_time_health': 0.3,
            'error_rate_health': 0.2,
            'alert_health': 0.1,
        }
        
        overall_score = sum(
            health_factors[factor] * weights[factor] 
            for factor in health_factors
        )
        
        # Determine health status
        if overall_score >= 90:
            status = 'excellent'
        elif overall_score >= 75:
            status = 'good'
        elif overall_score >= 60:
            status = 'fair'
        else:
            status = 'poor'
        
        return {
            'overall_score': round(overall_score, 1),
            'status': status,
            'factor_scores': health_factors,
            'timestamp': time.time(),
        }