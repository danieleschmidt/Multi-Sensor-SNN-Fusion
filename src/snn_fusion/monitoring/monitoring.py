"""
System and Experiment Monitoring

Provides comprehensive monitoring capabilities for neuromorphic computing
experiments, system health, and performance tracking.
"""

import time
import threading
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass, asdict
import json
from collections import defaultdict

from .metrics import MetricsCollector, NeuromorphicMetrics, SystemMetrics, get_global_collector


@dataclass
class MonitoringAlert:
    """Monitoring alert data structure."""
    alert_id: str
    timestamp: float
    severity: str  # 'info', 'warning', 'error', 'critical'
    title: str
    message: str
    source: str
    tags: Dict[str, str]
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class ExperimentStatus:
    """Experiment monitoring status."""
    experiment_id: str
    status: str  # 'running', 'completed', 'failed', 'paused'
    start_time: float
    last_update: float
    progress_percent: float
    current_epoch: int
    total_epochs: int
    best_accuracy: float
    current_loss: float
    estimated_completion: Optional[float] = None
    resource_usage: Optional[Dict[str, float]] = None


class SystemMonitor:
    """
    System-wide monitoring for neuromorphic computing infrastructure.
    
    Monitors system health, resource utilization, and provides
    alerting for critical conditions.
    """
    
    def __init__(
        self,
        metrics_collector: Optional[MetricsCollector] = None,
        alert_thresholds: Optional[Dict[str, float]] = None,
    ):
        """
        Initialize system monitor.
        
        Args:
            metrics_collector: Metrics collector instance
            alert_thresholds: Thresholds for system alerts
        """
        self.metrics_collector = metrics_collector or get_global_collector()
        self.logger = logging.getLogger(__name__)
        
        # Default alert thresholds
        self.alert_thresholds = {
            'cpu_percent': 90.0,
            'memory_percent': 85.0,
            'disk_usage_percent': 90.0,
            'gpu_memory_percent': 95.0,
            'load_average_1m': 4.0,
            **( alert_thresholds or {})
        }
        
        # Monitoring state
        self.monitoring_active = False
        self.monitoring_thread: Optional[threading.Thread] = None
        self.monitoring_interval = 10.0  # seconds
        
        # Alert callbacks
        self.alert_callbacks: List[Callable[[MonitoringAlert], None]] = []
        
        # System health status
        self.system_health_status = 'unknown'
        self.last_health_check = 0.0
        
        self.logger.info("Initialized system monitor")
    
    def start_monitoring(self, interval: float = 10.0) -> None:
        """
        Start system monitoring.
        
        Args:
            interval: Monitoring interval in seconds
        """
        if self.monitoring_active:
            self.logger.warning("System monitoring already active")
            return
        
        self.monitoring_interval = interval
        self.monitoring_active = True
        
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True
        )
        self.monitoring_thread.start()
        
        self.logger.info(f"Started system monitoring with {interval}s interval")
    
    def stop_monitoring(self) -> None:
        """Stop system monitoring."""
        if not self.monitoring_active:
            return
        
        self.monitoring_active = False
        
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=5)
        
        self.logger.info("Stopped system monitoring")
    
    def get_system_health(self) -> Dict[str, Any]:
        """
        Get current system health status.
        
        Returns:
            System health information
        """
        try:
            # Get latest system metrics
            health_data = {
                'status': self.system_health_status,
                'last_check': self.last_health_check,
                'timestamp': time.time(),
                'metrics': {},
                'alerts': [],
            }
            
            # Collect current metric values
            metric_names = [
                'system.cpu_percent',
                'system.memory_percent',
                'system.disk_usage_percent',
                'system.gpu_memory_percent',
                'system.load_average_1m',
            ]
            
            for metric_name in metric_names:
                value = self.metrics_collector.get_latest_value(metric_name)
                if value is not None:
                    health_data['metrics'][metric_name] = value
            
            # Check for active alerts
            for metric_name, threshold in self.alert_thresholds.items():
                full_metric_name = f'system.{metric_name}'
                value = health_data['metrics'].get(full_metric_name)
                
                if value is not None and value > threshold:
                    health_data['alerts'].append({
                        'metric': metric_name,
                        'value': value,
                        'threshold': threshold,
                        'severity': self._get_alert_severity(metric_name, value, threshold)
                    })
            
            # Determine overall health status
            if health_data['alerts']:
                max_severity = max(alert['severity'] for alert in health_data['alerts'])
                if max_severity == 'critical':
                    health_data['status'] = 'critical'
                elif max_severity == 'error':
                    health_data['status'] = 'degraded'
                else:
                    health_data['status'] = 'warning'
            else:
                health_data['status'] = 'healthy'
            
            self.system_health_status = health_data['status']
            self.last_health_check = health_data['timestamp']
            
            return health_data
            
        except Exception as e:
            self.logger.error(f"Failed to get system health: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'timestamp': time.time(),
            }
    
    def add_alert_callback(self, callback: Callable[[MonitoringAlert], None]) -> None:
        """
        Add callback for alert notifications.
        
        Args:
            callback: Function to call with alert data
        """
        self.alert_callbacks.append(callback)
        self.logger.info("Added alert callback")
    
    def remove_alert_callback(self, callback: Callable[[MonitoringAlert], None]) -> None:
        """Remove alert callback."""
        if callback in self.alert_callbacks:
            self.alert_callbacks.remove(callback)
            self.logger.info("Removed alert callback")
    
    def _monitoring_loop(self) -> None:
        """Main monitoring loop."""
        self.logger.info("Started system monitoring loop")
        
        while self.monitoring_active:
            try:
                # Perform health check
                health_data = self.get_system_health()
                
                # Process alerts
                for alert_info in health_data.get('alerts', []):
                    self._process_alert(alert_info)
                
                # Sleep until next check
                time.sleep(self.monitoring_interval)
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                time.sleep(self.monitoring_interval)
        
        self.logger.info("Stopped system monitoring loop")
    
    def _process_alert(self, alert_info: Dict[str, Any]) -> None:
        """Process and dispatch alert."""
        try:
            alert = MonitoringAlert(
                alert_id=f"system_{alert_info['metric']}_{int(time.time())}",
                timestamp=time.time(),
                severity=alert_info['severity'],
                title=f"System Alert: {alert_info['metric']}",
                message=f"{alert_info['metric']} is {alert_info['value']:.2f}, exceeding threshold of {alert_info['threshold']:.2f}",
                source='system_monitor',
                tags={'metric': alert_info['metric'], 'type': 'system'},
                metadata=alert_info
            )
            
            # Call alert callbacks
            for callback in self.alert_callbacks:
                try:
                    callback(alert)
                except Exception as e:
                    self.logger.error(f"Alert callback error: {e}")
                    
        except Exception as e:
            self.logger.error(f"Failed to process alert: {e}")
    
    def _get_alert_severity(self, metric_name: str, value: float, threshold: float) -> str:
        """Determine alert severity based on metric and value."""
        excess_ratio = (value - threshold) / threshold
        
        # Critical thresholds
        if metric_name in ['memory_percent', 'disk_usage_percent'] and excess_ratio > 0.1:
            return 'critical'
        elif metric_name == 'cpu_percent' and excess_ratio > 0.2:
            return 'critical'
        elif excess_ratio > 0.5:
            return 'critical'
        
        # Error thresholds
        elif excess_ratio > 0.2:
            return 'error'
        
        # Warning thresholds
        else:
            return 'warning'


class ExperimentMonitor:
    """
    Experiment-specific monitoring for neuromorphic computing workflows.
    
    Tracks experiment progress, resource usage, and performance metrics
    with real-time updates and alerting.
    """
    
    def __init__(
        self,
        metrics_collector: Optional[MetricsCollector] = None,
    ):
        """
        Initialize experiment monitor.
        
        Args:
            metrics_collector: Metrics collector instance
        """
        self.metrics_collector = metrics_collector or get_global_collector()
        self.logger = logging.getLogger(__name__)
        
        # Experiment tracking
        self.active_experiments: Dict[str, ExperimentStatus] = {}
        self.experiment_history: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        
        # Monitoring callbacks
        self.status_callbacks: List[Callable[[str, ExperimentStatus], None]] = []
        
        self.logger.info("Initialized experiment monitor")
    
    def start_experiment_tracking(
        self,
        experiment_id: str,
        total_epochs: int = 100,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Start tracking a new experiment.
        
        Args:
            experiment_id: Unique experiment identifier
            total_epochs: Total number of training epochs
            metadata: Additional experiment metadata
        """
        try:
            status = ExperimentStatus(
                experiment_id=experiment_id,
                status='running',
                start_time=time.time(),
                last_update=time.time(),
                progress_percent=0.0,
                current_epoch=0,
                total_epochs=total_epochs,
                best_accuracy=0.0,
                current_loss=float('inf'),
            )
            
            self.active_experiments[experiment_id] = status
            
            # Log experiment start
            self.experiment_history[experiment_id].append({
                'timestamp': time.time(),
                'event': 'started',
                'metadata': metadata or {},
            })
            
            self.logger.info(f"Started tracking experiment: {experiment_id}")
            
            # Notify callbacks
            self._notify_status_callbacks(experiment_id, status)
            
        except Exception as e:
            self.logger.error(f"Failed to start experiment tracking: {e}")
    
    def update_experiment_progress(
        self,
        experiment_id: str,
        current_epoch: int,
        current_loss: float,
        current_accuracy: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Update experiment progress.
        
        Args:
            experiment_id: Experiment identifier
            current_epoch: Current training epoch
            current_loss: Current loss value
            current_accuracy: Current accuracy (optional)
            metadata: Additional metadata
        """
        try:
            if experiment_id not in self.active_experiments:
                self.logger.warning(f"Unknown experiment: {experiment_id}")
                return
            
            status = self.active_experiments[experiment_id]
            
            # Update status
            status.current_epoch = current_epoch
            status.current_loss = current_loss
            status.last_update = time.time()
            status.progress_percent = (current_epoch / status.total_epochs) * 100
            
            if current_accuracy is not None:
                status.best_accuracy = max(status.best_accuracy, current_accuracy)
            
            # Estimate completion time
            if current_epoch > 0:
                elapsed_time = time.time() - status.start_time
                time_per_epoch = elapsed_time / current_epoch
                remaining_epochs = status.total_epochs - current_epoch
                status.estimated_completion = time.time() + (remaining_epochs * time_per_epoch)
            
            # Record metrics
            if current_accuracy is not None:
                self.metrics_collector.record_metric(
                    name=f"experiment.{experiment_id}.accuracy",
                    value=current_accuracy,
                    tags={'experiment_id': experiment_id, 'type': 'training'},
                )
            
            self.metrics_collector.record_metric(
                name=f"experiment.{experiment_id}.loss",
                value=current_loss,
                tags={'experiment_id': experiment_id, 'type': 'training'},
            )
            
            # Log progress update
            self.experiment_history[experiment_id].append({
                'timestamp': time.time(),
                'event': 'progress_update',
                'epoch': current_epoch,
                'loss': current_loss,
                'accuracy': current_accuracy,
                'metadata': metadata or {},
            })
            
            # Notify callbacks
            self._notify_status_callbacks(experiment_id, status)
            
        except Exception as e:
            self.logger.error(f"Failed to update experiment progress: {e}")
    
    def complete_experiment(
        self,
        experiment_id: str,
        final_accuracy: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Mark experiment as completed.
        
        Args:
            experiment_id: Experiment identifier
            final_accuracy: Final accuracy achieved
            metadata: Additional completion metadata
        """
        try:
            if experiment_id not in self.active_experiments:
                self.logger.warning(f"Unknown experiment: {experiment_id}")
                return
            
            status = self.active_experiments[experiment_id]
            status.status = 'completed'
            status.last_update = time.time()
            status.progress_percent = 100.0
            
            if final_accuracy is not None:
                status.best_accuracy = final_accuracy
            
            # Log completion
            completion_data = {
                'timestamp': time.time(),
                'event': 'completed',
                'final_accuracy': final_accuracy,
                'duration_seconds': time.time() - status.start_time,
                'total_epochs': status.current_epoch,
                'metadata': metadata or {},
            }
            
            self.experiment_history[experiment_id].append(completion_data)
            
            self.logger.info(f"Experiment completed: {experiment_id}")
            
            # Notify callbacks
            self._notify_status_callbacks(experiment_id, status)
            
        except Exception as e:
            self.logger.error(f"Failed to complete experiment: {e}")
    
    def fail_experiment(
        self,
        experiment_id: str,
        error_message: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Mark experiment as failed.
        
        Args:
            experiment_id: Experiment identifier
            error_message: Error description
            metadata: Additional error metadata
        """
        try:
            if experiment_id not in self.active_experiments:
                self.logger.warning(f"Unknown experiment: {experiment_id}")
                return
            
            status = self.active_experiments[experiment_id]
            status.status = 'failed'
            status.last_update = time.time()
            
            # Log failure
            self.experiment_history[experiment_id].append({
                'timestamp': time.time(),
                'event': 'failed',
                'error': error_message,
                'duration_seconds': time.time() - status.start_time,
                'metadata': metadata or {},
            })
            
            self.logger.error(f"Experiment failed: {experiment_id} - {error_message}")
            
            # Notify callbacks
            self._notify_status_callbacks(experiment_id, status)
            
        except Exception as e:
            self.logger.error(f"Failed to mark experiment as failed: {e}")
    
    def get_experiment_status(self, experiment_id: str) -> Optional[ExperimentStatus]:
        """
        Get current experiment status.
        
        Args:
            experiment_id: Experiment identifier
            
        Returns:
            Experiment status or None if not found
        """
        return self.active_experiments.get(experiment_id)
    
    def get_experiment_history(self, experiment_id: str) -> List[Dict[str, Any]]:
        """
        Get experiment history.
        
        Args:
            experiment_id: Experiment identifier
            
        Returns:
            List of experiment events
        """
        return self.experiment_history.get(experiment_id, [])
    
    def get_active_experiments(self) -> Dict[str, ExperimentStatus]:
        """Get all active experiments."""
        return {
            exp_id: status for exp_id, status in self.active_experiments.items()
            if status.status in ['running', 'paused']
        }
    
    def add_status_callback(self, callback: Callable[[str, ExperimentStatus], None]) -> None:
        """
        Add callback for experiment status updates.
        
        Args:
            callback: Function to call with (experiment_id, status)
        """
        self.status_callbacks.append(callback)
        self.logger.info("Added experiment status callback")
    
    def remove_status_callback(self, callback: Callable[[str, ExperimentStatus], None]) -> None:
        """Remove experiment status callback."""
        if callback in self.status_callbacks:
            self.status_callbacks.remove(callback)
            self.logger.info("Removed experiment status callback")
    
    def _notify_status_callbacks(self, experiment_id: str, status: ExperimentStatus) -> None:
        """Notify all status callbacks of experiment update."""
        for callback in self.status_callbacks:
            try:
                callback(experiment_id, status)
            except Exception as e:
                self.logger.error(f"Status callback error: {e}")


class PerformanceTracker:
    """
    Performance tracking and optimization for neuromorphic computing.
    
    Tracks and analyzes performance metrics to identify bottlenecks
    and optimization opportunities in neuromorphic workflows.
    """
    
    def __init__(
        self,
        metrics_collector: Optional[MetricsCollector] = None,
    ):
        """
        Initialize performance tracker.
        
        Args:
            metrics_collector: Metrics collector instance
        """
        self.metrics_collector = metrics_collector or get_global_collector()
        self.logger = logging.getLogger(__name__)
        
        # Performance tracking
        self.operation_timings: Dict[str, List[float]] = defaultdict(list)
        self.resource_usage_history: List[Dict[str, Any]] = []
        
        # Performance baselines
        self.performance_baselines: Dict[str, float] = {}
        
        self.logger.info("Initialized performance tracker")
    
    def track_operation(self, operation_name: str):
        """
        Context manager for tracking operation performance.
        
        Args:
            operation_name: Name of the operation to track
            
        Usage:
            with tracker.track_operation('model_training'):
                # training code here
                pass
        """
        return OperationTimer(self, operation_name)
    
    def record_operation_time(self, operation_name: str, duration: float) -> None:
        """
        Record operation timing.
        
        Args:
            operation_name: Operation name
            duration: Duration in seconds
        """
        try:
            self.operation_timings[operation_name].append(duration)
            
            # Record as metric
            self.metrics_collector.record_metric(
                name=f"performance.{operation_name}.duration",
                value=duration,
                tags={'type': 'performance', 'operation': operation_name},
            )
            
            # Check against baseline
            if operation_name in self.performance_baselines:
                baseline = self.performance_baselines[operation_name]
                if duration > baseline * 1.5:  # 50% slower than baseline
                    self.logger.warning(
                        f"Performance degradation detected for {operation_name}: "
                        f"{duration:.3f}s vs baseline {baseline:.3f}s"
                    )
            
        except Exception as e:
            self.logger.error(f"Failed to record operation time: {e}")
    
    def set_performance_baseline(self, operation_name: str, baseline_duration: float) -> None:
        """
        Set performance baseline for operation.
        
        Args:
            operation_name: Operation name
            baseline_duration: Baseline duration in seconds
        """
        self.performance_baselines[operation_name] = baseline_duration
        self.logger.info(f"Set performance baseline for {operation_name}: {baseline_duration:.3f}s")
    
    def get_performance_summary(self, operation_name: str) -> Optional[Dict[str, Any]]:
        """
        Get performance summary for operation.
        
        Args:
            operation_name: Operation name
            
        Returns:
            Performance statistics or None
        """
        if operation_name not in self.operation_timings:
            return None
        
        timings = self.operation_timings[operation_name]
        if not timings:
            return None
        
        import numpy as np
        
        return {
            'operation': operation_name,
            'count': len(timings),
            'mean_duration': np.mean(timings),
            'median_duration': np.median(timings),
            'min_duration': np.min(timings),
            'max_duration': np.max(timings),
            'std_duration': np.std(timings),
            'baseline': self.performance_baselines.get(operation_name),
            'latest_duration': timings[-1],
        }
    
    def get_all_performance_summaries(self) -> Dict[str, Dict[str, Any]]:
        """Get performance summaries for all tracked operations."""
        return {
            operation: self.get_performance_summary(operation)
            for operation in self.operation_timings.keys()
        }


class OperationTimer:
    """Context manager for timing operations."""
    
    def __init__(self, tracker: PerformanceTracker, operation_name: str):
        self.tracker = tracker
        self.operation_name = operation_name
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time is not None:
            duration = time.time() - self.start_time
            self.tracker.record_operation_time(self.operation_name, duration)