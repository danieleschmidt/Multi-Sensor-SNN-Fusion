"""
Comprehensive Monitoring and Health System

Advanced monitoring, metrics collection, alerting, and health checks
for production SNN-Fusion deployments.
"""

import time
import psutil
import logging
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, asdict
from enum import Enum
from collections import defaultdict, deque
import json
import os
from pathlib import Path
import socket
import requests
from concurrent.futures import ThreadPoolExecutor
import sqlite3


class HealthStatus(Enum):
    """Health check status levels."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    CRITICAL = "critical"


class MetricType(Enum):
    """Types of metrics."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class HealthCheck:
    """Health check definition."""
    name: str
    check_function: Callable[[], Dict[str, Any]]
    interval_seconds: int = 60
    timeout_seconds: int = 10
    enabled: bool = True


@dataclass
class Metric:
    """Metric data structure."""
    name: str
    value: Union[int, float]
    metric_type: MetricType
    timestamp: datetime
    labels: Dict[str, str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        data['metric_type'] = self.metric_type.value
        return data


@dataclass
class Alert:
    """Alert data structure."""
    id: str
    name: str
    message: str
    severity: AlertSeverity
    timestamp: datetime
    resolved: bool = False
    resolved_at: Optional[datetime] = None
    metadata: Dict[str, Any] = None
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        data['severity'] = self.severity.value
        if self.resolved_at:
            data['resolved_at'] = self.resolved_at.isoformat()
        return data


class MetricsCollector:
    """Advanced metrics collection and aggregation."""
    
    def __init__(self, retention_hours: int = 24):
        """
        Initialize metrics collector.
        
        Args:
            retention_hours: How long to retain metrics in memory
        """
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10000))
        self.retention_hours = retention_hours
        self.lock = threading.RLock()
        self.logger = logging.getLogger(__name__)
        
        # System metrics collection
        self._start_system_metrics_collection()
    
    def record_metric(
        self,
        name: str,
        value: Union[int, float],
        metric_type: MetricType = MetricType.GAUGE,
        labels: Optional[Dict[str, str]] = None
    ):
        """Record a metric value."""
        metric = Metric(
            name=name,
            value=value,
            metric_type=metric_type,
            timestamp=datetime.now(),
            labels=labels or {}
        )
        
        with self.lock:
            self.metrics[name].append(metric)
    
    def increment_counter(self, name: str, value: int = 1, labels: Optional[Dict[str, str]] = None):
        """Increment a counter metric."""
        self.record_metric(name, value, MetricType.COUNTER, labels)
    
    def set_gauge(self, name: str, value: Union[int, float], labels: Optional[Dict[str, str]] = None):
        """Set a gauge metric value."""
        self.record_metric(name, value, MetricType.GAUGE, labels)
    
    def get_metrics(self, name: str, since: Optional[datetime] = None) -> List[Metric]:
        """Get metrics for a specific name."""
        with self.lock:
            metrics = list(self.metrics.get(name, []))
            
            if since:
                metrics = [m for m in metrics if m.timestamp >= since]
            
            return metrics
    
    def get_latest_value(self, name: str) -> Optional[float]:
        """Get the latest value for a metric."""
        with self.lock:
            metric_list = self.metrics.get(name, [])
            return metric_list[-1].value if metric_list else None
    
    def get_aggregated_metrics(self, timeframe_minutes: int = 60) -> Dict[str, Dict[str, float]]:
        """Get aggregated metrics for the specified timeframe."""
        cutoff_time = datetime.now() - timedelta(minutes=timeframe_minutes)
        aggregated = {}
        
        with self.lock:
            for name, metric_list in self.metrics.items():
                recent_metrics = [m for m in metric_list if m.timestamp >= cutoff_time]
                
                if not recent_metrics:
                    continue
                
                values = [m.value for m in recent_metrics]
                
                aggregated[name] = {
                    'count': len(values),
                    'sum': sum(values),
                    'avg': sum(values) / len(values),
                    'min': min(values),
                    'max': max(values),
                    'latest': values[-1]
                }
        
        return aggregated
    
    def _start_system_metrics_collection(self):
        """Start collecting system-level metrics."""
        def collect_system_metrics():
            while True:
                try:
                    # CPU metrics
                    cpu_percent = psutil.cpu_percent(interval=1)
                    self.set_gauge('system.cpu.usage_percent', cpu_percent)
                    
                    # Memory metrics
                    memory = psutil.virtual_memory()
                    self.set_gauge('system.memory.usage_percent', memory.percent)
                    self.set_gauge('system.memory.available_gb', memory.available / (1024**3))
                    
                    # Disk metrics
                    disk = psutil.disk_usage('/')
                    self.set_gauge('system.disk.usage_percent', (disk.used / disk.total) * 100)
                    self.set_gauge('system.disk.free_gb', disk.free / (1024**3))
                    
                    # Network metrics
                    net_io = psutil.net_io_counters()
                    self.set_gauge('system.network.bytes_sent', net_io.bytes_sent)
                    self.set_gauge('system.network.bytes_recv', net_io.bytes_recv)
                    
                    # Process count
                    self.set_gauge('system.process.count', len(psutil.pids()))
                    
                except Exception as e:
                    self.logger.warning(f"Failed to collect system metrics: {e}")
                
                time.sleep(30)  # Collect every 30 seconds
        
        thread = threading.Thread(target=collect_system_metrics, daemon=True)
        thread.start()
    
    def cleanup_old_metrics(self):
        """Clean up old metrics beyond retention period."""
        cutoff_time = datetime.now() - timedelta(hours=self.retention_hours)
        
        with self.lock:
            for name, metric_list in self.metrics.items():
                # Remove old metrics
                while metric_list and metric_list[0].timestamp < cutoff_time:
                    metric_list.popleft()


class HealthMonitor:
    """Comprehensive health monitoring system."""
    
    def __init__(self, check_interval: int = 60):
        """
        Initialize health monitor.
        
        Args:
            check_interval: Default interval between health checks in seconds
        """
        self.health_checks: Dict[str, HealthCheck] = {}
        self.health_status: Dict[str, Dict[str, Any]] = {}
        self.check_interval = check_interval
        self.running = False
        self.lock = threading.RLock()
        self.logger = logging.getLogger(__name__)
        self.executor = ThreadPoolExecutor(max_workers=5, thread_name_prefix="health_check")
        
        # Register default health checks
        self._register_default_health_checks()
    
    def register_health_check(self, health_check: HealthCheck):
        """Register a health check."""
        with self.lock:
            self.health_checks[health_check.name] = health_check
            
        self.logger.info(f"Registered health check: {health_check.name}")
    
    def _register_default_health_checks(self):
        """Register default health checks."""
        # Database health check
        db_check = HealthCheck(
            name="database",
            check_function=self._check_database_health,
            interval_seconds=30
        )
        self.register_health_check(db_check)
        
        # API health check
        api_check = HealthCheck(
            name="api",
            check_function=self._check_api_health,
            interval_seconds=60
        )
        self.register_health_check(api_check)
        
        # Disk space check
        disk_check = HealthCheck(
            name="disk_space",
            check_function=self._check_disk_space,
            interval_seconds=120
        )
        self.register_health_check(disk_check)
        
        # Memory check
        memory_check = HealthCheck(
            name="memory",
            check_function=self._check_memory,
            interval_seconds=60
        )
        self.register_health_check(memory_check)
    
    def start_monitoring(self):
        """Start the health monitoring loop."""
        if self.running:
            return
        
        self.running = True
        
        def monitor_loop():
            while self.running:
                try:
                    self._run_health_checks()
                    time.sleep(self.check_interval)
                except Exception as e:
                    self.logger.error(f"Error in health monitoring loop: {e}")
                    time.sleep(5)  # Brief pause before retrying
        
        thread = threading.Thread(target=monitor_loop, daemon=True)
        thread.start()
        
        self.logger.info("Health monitoring started")
    
    def stop_monitoring(self):
        """Stop the health monitoring loop."""
        self.running = False
        self.executor.shutdown(wait=False)
        self.logger.info("Health monitoring stopped")
    
    def _run_health_checks(self):
        """Run all enabled health checks."""
        futures = {}
        
        with self.lock:
            for name, check in self.health_checks.items():
                if not check.enabled:
                    continue
                
                # Check if it's time to run this check
                last_run = self.health_status.get(name, {}).get('last_checked')
                if last_run:
                    time_since_last = (datetime.now() - last_run).total_seconds()
                    if time_since_last < check.interval_seconds:
                        continue
                
                # Submit health check to thread pool
                future = self.executor.submit(self._execute_health_check, check)
                futures[name] = future
        
        # Collect results
        for name, future in futures.items():
            try:
                result = future.result(timeout=self.health_checks[name].timeout_seconds)
                self._update_health_status(name, result)
            except Exception as e:
                self._update_health_status(name, {
                    'status': HealthStatus.UNHEALTHY.value,
                    'error': str(e)
                })
    
    def _execute_health_check(self, check: HealthCheck) -> Dict[str, Any]:
        """Execute a single health check."""
        try:
            result = check.check_function()
            result['last_checked'] = datetime.now()
            return result
        except Exception as e:
            return {
                'status': HealthStatus.UNHEALTHY.value,
                'error': str(e),
                'last_checked': datetime.now()
            }
    
    def _update_health_status(self, name: str, result: Dict[str, Any]):
        """Update health status for a check."""
        with self.lock:
            self.health_status[name] = result
    
    def get_overall_health(self) -> Dict[str, Any]:
        """Get overall system health status."""
        with self.lock:
            if not self.health_status:
                return {
                    'status': HealthStatus.UNHEALTHY.value,
                    'message': 'No health checks configured'
                }
            
            # Determine overall status
            statuses = [status.get('status') for status in self.health_status.values()]
            
            if all(s == HealthStatus.HEALTHY.value for s in statuses):
                overall_status = HealthStatus.HEALTHY.value
            elif any(s == HealthStatus.CRITICAL.value for s in statuses):
                overall_status = HealthStatus.CRITICAL.value
            elif any(s == HealthStatus.UNHEALTHY.value for s in statuses):
                overall_status = HealthStatus.UNHEALTHY.value
            else:
                overall_status = HealthStatus.DEGRADED.value
            
            # Count checks by status
            status_counts = defaultdict(int)
            for status in statuses:
                status_counts[status] += 1
            
            return {
                'status': overall_status,
                'checks': dict(self.health_status),
                'summary': {
                    'total_checks': len(statuses),
                    'status_counts': dict(status_counts)
                },
                'timestamp': datetime.now().isoformat()
            }
    
    def get_health_check_status(self, name: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific health check."""
        with self.lock:
            return self.health_status.get(name)
    
    # Default health check implementations
    
    def _check_database_health(self) -> Dict[str, Any]:
        """Check database connectivity and performance."""
        try:
            # Mock database check (replace with actual database check)
            start_time = time.time()
            
            # Simulate database query
            time.sleep(0.01)  # Simulate small delay
            
            response_time = (time.time() - start_time) * 1000  # Convert to ms
            
            if response_time > 1000:  # > 1 second
                status = HealthStatus.UNHEALTHY.value
            elif response_time > 500:  # > 500ms
                status = HealthStatus.DEGRADED.value
            else:
                status = HealthStatus.HEALTHY.value
            
            return {
                'status': status,
                'response_time_ms': response_time,
                'message': f'Database responding in {response_time:.2f}ms'
            }
        except Exception as e:
            return {
                'status': HealthStatus.CRITICAL.value,
                'error': str(e),
                'message': 'Database connection failed'
            }
    
    def _check_api_health(self) -> Dict[str, Any]:
        """Check API endpoint health."""
        try:
            # Check if the service is listening on its port
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5)
            result = sock.connect_ex(('localhost', 8080))  # Assuming API runs on port 8080
            sock.close()
            
            if result == 0:
                return {
                    'status': HealthStatus.HEALTHY.value,
                    'message': 'API endpoint reachable'
                }
            else:
                return {
                    'status': HealthStatus.UNHEALTHY.value,
                    'message': 'API endpoint not reachable'
                }
        except Exception as e:
            return {
                'status': HealthStatus.UNHEALTHY.value,
                'error': str(e),
                'message': 'API health check failed'
            }
    
    def _check_disk_space(self) -> Dict[str, Any]:
        """Check available disk space."""
        try:
            usage = psutil.disk_usage('/')
            free_percent = (usage.free / usage.total) * 100
            
            if free_percent < 5:  # Less than 5% free
                status = HealthStatus.CRITICAL.value
                message = f'Disk space critically low: {free_percent:.1f}% free'
            elif free_percent < 15:  # Less than 15% free
                status = HealthStatus.UNHEALTHY.value
                message = f'Disk space low: {free_percent:.1f}% free'
            elif free_percent < 25:  # Less than 25% free
                status = HealthStatus.DEGRADED.value
                message = f'Disk space getting low: {free_percent:.1f}% free'
            else:
                status = HealthStatus.HEALTHY.value
                message = f'Disk space healthy: {free_percent:.1f}% free'
            
            return {
                'status': status,
                'free_percent': free_percent,
                'free_gb': usage.free / (1024**3),
                'total_gb': usage.total / (1024**3),
                'message': message
            }
        except Exception as e:
            return {
                'status': HealthStatus.UNHEALTHY.value,
                'error': str(e),
                'message': 'Disk space check failed'
            }
    
    def _check_memory(self) -> Dict[str, Any]:
        """Check memory usage."""
        try:
            memory = psutil.virtual_memory()
            used_percent = memory.percent
            
            if used_percent > 95:  # Over 95% used
                status = HealthStatus.CRITICAL.value
                message = f'Memory usage critical: {used_percent:.1f}%'
            elif used_percent > 85:  # Over 85% used
                status = HealthStatus.UNHEALTHY.value
                message = f'Memory usage high: {used_percent:.1f}%'
            elif used_percent > 75:  # Over 75% used
                status = HealthStatus.DEGRADED.value
                message = f'Memory usage elevated: {used_percent:.1f}%'
            else:
                status = HealthStatus.HEALTHY.value
                message = f'Memory usage normal: {used_percent:.1f}%'
            
            return {
                'status': status,
                'used_percent': used_percent,
                'available_gb': memory.available / (1024**3),
                'total_gb': memory.total / (1024**3),
                'message': message
            }
        except Exception as e:
            return {
                'status': HealthStatus.UNHEALTHY.value,
                'error': str(e),
                'message': 'Memory check failed'
            }


class AlertManager:
    """Advanced alerting system with multiple notification channels."""
    
    def __init__(self, storage_file: str = "alerts.db"):
        """
        Initialize alert manager.
        
        Args:
            storage_file: SQLite file for alert persistence
        """
        self.storage_file = storage_file
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: List[Alert] = []
        self.notification_channels: List[Callable] = []
        self.lock = threading.RLock()
        self.logger = logging.getLogger(__name__)
        
        # Initialize storage
        self._init_storage()
    
    def _init_storage(self):
        """Initialize SQLite storage for alerts."""
        try:
            with sqlite3.connect(self.storage_file) as conn:
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS alerts (
                        id TEXT PRIMARY KEY,
                        name TEXT,
                        message TEXT,
                        severity TEXT,
                        timestamp TEXT,
                        resolved INTEGER DEFAULT 0,
                        resolved_at TEXT,
                        metadata TEXT
                    )
                ''')
                conn.commit()
        except Exception as e:
            self.logger.error(f"Failed to initialize alert storage: {e}")
    
    def add_notification_channel(self, channel: Callable[[Alert], None]):
        """Add a notification channel."""
        self.notification_channels.append(channel)
        self.logger.info("Added notification channel")
    
    def trigger_alert(
        self,
        name: str,
        message: str,
        severity: AlertSeverity = AlertSeverity.WARNING,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Trigger a new alert."""
        alert_id = f"alert_{int(time.time())}_{hash(name + message) % 10000:04d}"
        
        alert = Alert(
            id=alert_id,
            name=name,
            message=message,
            severity=severity,
            timestamp=datetime.now(),
            metadata=metadata or {}
        )
        
        with self.lock:
            self.active_alerts[alert_id] = alert
            self.alert_history.append(alert)
        
        # Store in database
        self._store_alert(alert)
        
        # Send notifications
        self._send_notifications(alert)
        
        self.logger.warning(f"Alert triggered: {name} - {message}")
        return alert_id
    
    def resolve_alert(self, alert_id: str) -> bool:
        """Resolve an active alert."""
        with self.lock:
            if alert_id in self.active_alerts:
                alert = self.active_alerts[alert_id]
                alert.resolved = True
                alert.resolved_at = datetime.now()
                
                # Update in database
                self._update_alert(alert)
                
                # Remove from active alerts
                del self.active_alerts[alert_id]
                
                self.logger.info(f"Alert resolved: {alert.name}")
                return True
        
        return False
    
    def get_active_alerts(self, severity: Optional[AlertSeverity] = None) -> List[Alert]:
        """Get list of active alerts."""
        with self.lock:
            alerts = list(self.active_alerts.values())
            
            if severity:
                alerts = [a for a in alerts if a.severity == severity]
            
            return sorted(alerts, key=lambda x: x.timestamp, reverse=True)
    
    def get_alert_summary(self) -> Dict[str, Any]:
        """Get alert summary statistics."""
        with self.lock:
            active_count = len(self.active_alerts)
            
            # Count by severity
            severity_counts = defaultdict(int)
            for alert in self.active_alerts.values():
                severity_counts[alert.severity.value] += 1
            
            # Recent alerts (last 24 hours)
            recent_cutoff = datetime.now() - timedelta(hours=24)
            recent_alerts = [a for a in self.alert_history if a.timestamp > recent_cutoff]
            
            return {
                'active_alerts': active_count,
                'recent_alerts_24h': len(recent_alerts),
                'severity_breakdown': dict(severity_counts),
                'total_alerts_all_time': len(self.alert_history)
            }
    
    def _store_alert(self, alert: Alert):
        """Store alert in database."""
        try:
            with sqlite3.connect(self.storage_file) as conn:
                conn.execute('''
                    INSERT INTO alerts 
                    (id, name, message, severity, timestamp, resolved, resolved_at, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    alert.id,
                    alert.name,
                    alert.message,
                    alert.severity.value,
                    alert.timestamp.isoformat(),
                    int(alert.resolved),
                    alert.resolved_at.isoformat() if alert.resolved_at else None,
                    json.dumps(alert.metadata) if alert.metadata else None
                ))
                conn.commit()
        except Exception as e:
            self.logger.error(f"Failed to store alert: {e}")
    
    def _update_alert(self, alert: Alert):
        """Update alert in database."""
        try:
            with sqlite3.connect(self.storage_file) as conn:
                conn.execute('''
                    UPDATE alerts 
                    SET resolved = ?, resolved_at = ?
                    WHERE id = ?
                ''', (
                    int(alert.resolved),
                    alert.resolved_at.isoformat() if alert.resolved_at else None,
                    alert.id
                ))
                conn.commit()
        except Exception as e:
            self.logger.error(f"Failed to update alert: {e}")
    
    def _send_notifications(self, alert: Alert):
        """Send alert notifications through all channels."""
        for channel in self.notification_channels:
            try:
                channel(alert)
            except Exception as e:
                self.logger.error(f"Failed to send notification: {e}")


# Notification channel implementations
def console_notification_channel(alert: Alert):
    """Simple console notification channel."""
    print(f"🚨 ALERT: {alert.name} - {alert.message} (Severity: {alert.severity.value})")


def log_notification_channel(alert: Alert):
    """Log file notification channel."""
    logger = logging.getLogger("alerts")
    log_message = f"Alert: {alert.name} - {alert.message} (ID: {alert.id})"
    
    if alert.severity == AlertSeverity.CRITICAL:
        logger.critical(log_message)
    elif alert.severity == AlertSeverity.ERROR:
        logger.error(log_message)
    elif alert.severity == AlertSeverity.WARNING:
        logger.warning(log_message)
    else:
        logger.info(log_message)


class ComprehensiveMonitor:
    """Main monitoring system that coordinates all monitoring components."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize comprehensive monitoring system.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        
        # Initialize components
        self.metrics = MetricsCollector(
            retention_hours=self.config.get('metrics_retention_hours', 24)
        )
        self.health = HealthMonitor(
            check_interval=self.config.get('health_check_interval', 60)
        )
        self.alerts = AlertManager(
            storage_file=self.config.get('alert_storage_file', 'alerts.db')
        )
        
        self.logger = logging.getLogger(__name__)
        
        # Setup default notification channels
        self.alerts.add_notification_channel(console_notification_channel)
        self.alerts.add_notification_channel(log_notification_channel)
        
        # Start monitoring
        self.start()
    
    def start(self):
        """Start all monitoring components."""
        self.health.start_monitoring()
        
        # Start alert monitoring based on health checks
        self._start_alert_monitoring()
        
        self.logger.info("Comprehensive monitoring system started")
    
    def stop(self):
        """Stop all monitoring components."""
        self.health.stop_monitoring()
        self.logger.info("Comprehensive monitoring system stopped")
    
    def _start_alert_monitoring(self):
        """Start monitoring health checks and trigger alerts."""
        def alert_monitor():
            while True:
                try:
                    # Check overall health and trigger alerts if needed
                    overall_health = self.health.get_overall_health()
                    
                    # Critical system health
                    if overall_health['status'] == HealthStatus.CRITICAL.value:
                        self.alerts.trigger_alert(
                            name="system_critical",
                            message="System health is critical",
                            severity=AlertSeverity.CRITICAL,
                            metadata=overall_health
                        )
                    
                    # Check individual health checks
                    for name, status in overall_health.get('checks', {}).items():
                        if status.get('status') == HealthStatus.CRITICAL.value:
                            self.alerts.trigger_alert(
                                name=f"{name}_critical",
                                message=f"{name} health check is critical: {status.get('message', '')}",
                                severity=AlertSeverity.ERROR,
                                metadata=status
                            )
                    
                    # Check system metrics for anomalies
                    self._check_metric_anomalies()
                    
                except Exception as e:
                    self.logger.error(f"Error in alert monitoring: {e}")
                
                time.sleep(60)  # Check every minute
        
        thread = threading.Thread(target=alert_monitor, daemon=True)
        thread.start()
    
    def _check_metric_anomalies(self):
        """Check metrics for anomalies and trigger alerts."""
        try:
            # CPU usage anomaly
            cpu_usage = self.metrics.get_latest_value('system.cpu.usage_percent')
            if cpu_usage and cpu_usage > 90:
                self.alerts.trigger_alert(
                    name="high_cpu_usage",
                    message=f"CPU usage is {cpu_usage:.1f}%",
                    severity=AlertSeverity.WARNING
                )
            
            # Memory usage anomaly  
            memory_usage = self.metrics.get_latest_value('system.memory.usage_percent')
            if memory_usage and memory_usage > 90:
                self.alerts.trigger_alert(
                    name="high_memory_usage",
                    message=f"Memory usage is {memory_usage:.1f}%",
                    severity=AlertSeverity.WARNING
                )
            
            # Disk space anomaly
            disk_usage = self.metrics.get_latest_value('system.disk.usage_percent')
            if disk_usage and disk_usage > 85:
                self.alerts.trigger_alert(
                    name="low_disk_space",
                    message=f"Disk usage is {disk_usage:.1f}%",
                    severity=AlertSeverity.ERROR
                )
        
        except Exception as e:
            self.logger.warning(f"Failed to check metric anomalies: {e}")
    
    def get_monitoring_dashboard(self) -> Dict[str, Any]:
        """Get comprehensive monitoring dashboard data."""
        return {
            'timestamp': datetime.now().isoformat(),
            'system_health': self.health.get_overall_health(),
            'metrics_summary': self.metrics.get_aggregated_metrics(60),
            'alert_summary': self.alerts.get_alert_summary(),
            'active_alerts': [alert.to_dict() for alert in self.alerts.get_active_alerts()],
            'system_info': {
                'cpu_count': psutil.cpu_count(),
                'memory_total_gb': psutil.virtual_memory().total / (1024**3),
                'disk_total_gb': psutil.disk_usage('/').total / (1024**3)
            }
        }


# Example usage and testing
if __name__ == "__main__":
    print("Testing Comprehensive Monitoring System...")
    
    # Initialize monitoring
    monitor = ComprehensiveMonitor({
        'health_check_interval': 30,
        'metrics_retention_hours': 1  # Short retention for testing
    })
    
    # Record some test metrics
    monitor.metrics.set_gauge('test.value', 42.5)
    monitor.metrics.increment_counter('test.requests', 1)
    monitor.metrics.set_gauge('test.temperature', 68.2, {'location': 'server_room'})
    
    # Trigger test alert
    monitor.alerts.trigger_alert(
        name="test_alert",
        message="This is a test alert",
        severity=AlertSeverity.INFO
    )
    
    # Wait for health checks to run
    print("Running health checks...")
    time.sleep(35)
    
    # Get dashboard data
    dashboard = monitor.get_monitoring_dashboard()
    
    print("\n=== Monitoring Dashboard ===")
    print(f"System Status: {dashboard['system_health']['status']}")
    print(f"Total Health Checks: {dashboard['system_health']['summary']['total_checks']}")
    print(f"Active Alerts: {dashboard['alert_summary']['active_alerts']}")
    print(f"Metrics Collected: {len(dashboard['metrics_summary'])} types")
    
    print("\nHealth Check Details:")
    for name, status in dashboard['system_health']['checks'].items():
        print(f"  {name}: {status.get('status', 'unknown')} - {status.get('message', '')}")
    
    print("\nRecent Metrics:")
    for name, metrics in list(dashboard['metrics_summary'].items())[:5]:
        print(f"  {name}: {metrics['latest']:.2f} (avg: {metrics['avg']:.2f})")
    
    # Test alert resolution
    active_alerts = monitor.alerts.get_active_alerts()
    if active_alerts:
        alert_id = active_alerts[0].id
        print(f"\nResolving test alert: {alert_id}")
        monitor.alerts.resolve_alert(alert_id)
    
    # Cleanup
    monitor.stop()
    
    print("✓ Comprehensive monitoring test completed!")