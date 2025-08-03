"""
Alert Management and Anomaly Detection

Implements comprehensive alerting system with threshold-based alerts,
anomaly detection, and automated response for neuromorphic computing systems.
"""

import time
import threading
from typing import Dict, Any, Optional, List, Callable, Union, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import logging
import json
import numpy as np
from collections import defaultdict, deque

from .metrics import MetricsCollector, get_global_collector
from .monitoring import MonitoringAlert


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AlertStatus(Enum):
    """Alert status."""
    ACTIVE = "active"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"
    SUPPRESSED = "suppressed"


@dataclass
class AlertRule:
    """Alert rule configuration."""
    rule_id: str
    name: str
    description: str
    metric_name: str
    condition: str  # 'greater_than', 'less_than', 'equals', 'not_equals', 'anomaly'
    threshold: Optional[float] = None
    severity: AlertSeverity = AlertSeverity.WARNING
    evaluation_window: float = 60.0  # seconds
    minimum_samples: int = 3
    cooldown_period: float = 300.0  # seconds
    enabled: bool = True
    tags: Optional[Dict[str, str]] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class AlertInstance:
    """Active alert instance."""
    alert_id: str
    rule_id: str
    metric_name: str
    severity: AlertSeverity
    status: AlertStatus
    title: str
    message: str
    triggered_at: float
    last_update: float
    threshold: Optional[float]
    current_value: float
    tags: Dict[str, str]
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[float] = None
    resolved_at: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None


class ThresholdAlert:
    """
    Threshold-based alerting for metric values.
    
    Monitors metrics against configurable thresholds and triggers
    alerts when conditions are met.
    """
    
    def __init__(
        self,
        metrics_collector: Optional[MetricsCollector] = None,
        evaluation_interval: float = 30.0,
    ):
        """
        Initialize threshold alert system.
        
        Args:
            metrics_collector: Metrics collector instance
            evaluation_interval: How often to evaluate rules (seconds)
        """
        self.metrics_collector = metrics_collector or get_global_collector()
        self.evaluation_interval = evaluation_interval
        self.logger = logging.getLogger(__name__)
        
        # Alert rules and instances
        self.alert_rules: Dict[str, AlertRule] = {}
        self.active_alerts: Dict[str, AlertInstance] = {}
        self.alert_history: List[AlertInstance] = []
        
        # Evaluation state
        self.evaluation_active = False
        self.evaluation_thread: Optional[threading.Thread] = None
        self.last_evaluation: Dict[str, float] = {}
        
        # Callbacks for alert actions
        self.alert_callbacks: List[Callable[[AlertInstance], None]] = []
        
        # Thread safety
        self._lock = threading.Lock()
        
        self.logger.info("Initialized threshold alert system")
    
    def add_rule(self, rule: AlertRule) -> None:
        """
        Add alert rule.
        
        Args:
            rule: Alert rule configuration
        """
        with self._lock:
            self.alert_rules[rule.rule_id] = rule
        
        self.logger.info(f"Added alert rule: {rule.name} ({rule.rule_id})")
    
    def remove_rule(self, rule_id: str) -> bool:
        """
        Remove alert rule.
        
        Args:
            rule_id: Rule identifier
            
        Returns:
            True if rule was removed
        """
        with self._lock:
            if rule_id in self.alert_rules:
                del self.alert_rules[rule_id]
                self.logger.info(f"Removed alert rule: {rule_id}")
                return True
        
        return False
    
    def enable_rule(self, rule_id: str) -> bool:
        """Enable alert rule."""
        with self._lock:
            if rule_id in self.alert_rules:
                self.alert_rules[rule_id].enabled = True
                self.logger.info(f"Enabled alert rule: {rule_id}")
                return True
        return False
    
    def disable_rule(self, rule_id: str) -> bool:
        """Disable alert rule."""
        with self._lock:
            if rule_id in self.alert_rules:
                self.alert_rules[rule_id].enabled = False
                self.logger.info(f"Disabled alert rule: {rule_id}")
                return True
        return False
    
    def start_evaluation(self) -> None:
        """Start alert rule evaluation."""
        if self.evaluation_active:
            self.logger.warning("Alert evaluation already active")
            return
        
        self.evaluation_active = True
        self.evaluation_thread = threading.Thread(
            target=self._evaluation_loop,
            daemon=True
        )
        self.evaluation_thread.start()
        
        self.logger.info(f"Started alert evaluation with {self.evaluation_interval}s interval")
    
    def stop_evaluation(self) -> None:
        """Stop alert rule evaluation."""
        if not self.evaluation_active:
            return
        
        self.evaluation_active = False
        
        if self.evaluation_thread and self.evaluation_thread.is_alive():
            self.evaluation_thread.join(timeout=5)
        
        self.logger.info("Stopped alert evaluation")
    
    def acknowledge_alert(self, alert_id: str, acknowledged_by: str) -> bool:
        """
        Acknowledge an active alert.
        
        Args:
            alert_id: Alert identifier
            acknowledged_by: User/system acknowledging the alert
            
        Returns:
            True if alert was acknowledged
        """
        with self._lock:
            if alert_id in self.active_alerts:
                alert = self.active_alerts[alert_id]
                alert.status = AlertStatus.ACKNOWLEDGED
                alert.acknowledged_by = acknowledged_by
                alert.acknowledged_at = time.time()
                alert.last_update = time.time()
                
                self.logger.info(f"Alert acknowledged: {alert_id} by {acknowledged_by}")
                return True
        
        return False
    
    def resolve_alert(self, alert_id: str, resolved_by: Optional[str] = None) -> bool:
        """
        Resolve an active alert.
        
        Args:
            alert_id: Alert identifier
            resolved_by: User/system resolving the alert
            
        Returns:
            True if alert was resolved
        """
        with self._lock:
            if alert_id in self.active_alerts:
                alert = self.active_alerts[alert_id]
                alert.status = AlertStatus.RESOLVED
                alert.resolved_at = time.time()
                alert.last_update = time.time()
                
                # Move to history
                self.alert_history.append(alert)
                del self.active_alerts[alert_id]
                
                self.logger.info(f"Alert resolved: {alert_id}")
                return True
        
        return False
    
    def get_active_alerts(self, severity: Optional[AlertSeverity] = None) -> List[AlertInstance]:
        """
        Get active alerts.
        
        Args:
            severity: Filter by severity level
            
        Returns:
            List of active alerts
        """
        with self._lock:
            alerts = list(self.active_alerts.values())
        
        if severity:
            alerts = [alert for alert in alerts if alert.severity == severity]
        
        return alerts
    
    def get_alert_history(
        self,
        hours: int = 24,
        severity: Optional[AlertSeverity] = None
    ) -> List[AlertInstance]:
        """
        Get alert history.
        
        Args:
            hours: Number of hours to look back
            severity: Filter by severity level
            
        Returns:
            List of historical alerts
        """
        cutoff_time = time.time() - (hours * 3600)
        
        with self._lock:
            alerts = [
                alert for alert in self.alert_history
                if alert.triggered_at >= cutoff_time
            ]
        
        if severity:
            alerts = [alert for alert in alerts if alert.severity == severity]
        
        return alerts
    
    def add_callback(self, callback: Callable[[AlertInstance], None]) -> None:
        """
        Add callback for alert notifications.
        
        Args:
            callback: Function to call when alert is triggered
        """
        self.alert_callbacks.append(callback)
        self.logger.info("Added alert callback")
    
    def remove_callback(self, callback: Callable[[AlertInstance], None]) -> None:
        """Remove alert callback."""
        if callback in self.alert_callbacks:
            self.alert_callbacks.remove(callback)
            self.logger.info("Removed alert callback")
    
    def _evaluation_loop(self) -> None:
        """Main alert evaluation loop."""
        self.logger.info("Started alert evaluation loop")
        
        while self.evaluation_active:
            try:
                current_time = time.time()
                
                # Evaluate all enabled rules
                with self._lock:
                    rules_to_evaluate = [
                        rule for rule in self.alert_rules.values()
                        if rule.enabled
                    ]
                
                for rule in rules_to_evaluate:
                    self._evaluate_rule(rule, current_time)
                
                # Check for auto-resolution
                self._check_auto_resolution(current_time)
                
                time.sleep(self.evaluation_interval)
                
            except Exception as e:
                self.logger.error(f"Error in alert evaluation loop: {e}")
                time.sleep(self.evaluation_interval)
        
        self.logger.info("Stopped alert evaluation loop")
    
    def _evaluate_rule(self, rule: AlertRule, current_time: float) -> None:
        """Evaluate a single alert rule."""
        try:
            # Check cooldown period
            if rule.rule_id in self.last_evaluation:
                if current_time - self.last_evaluation[rule.rule_id] < rule.cooldown_period:
                    return
            
            # Get metric history for evaluation window
            start_time = current_time - rule.evaluation_window
            metric_points = self.metrics_collector.get_metric_history(
                rule.metric_name, start_time=start_time, end_time=current_time
            )
            
            if len(metric_points) < rule.minimum_samples:
                return  # Not enough data
            
            # Extract values
            values = [point.value for point in metric_points]
            latest_value = values[-1]
            
            # Evaluate condition
            condition_met = self._evaluate_condition(rule, latest_value, values)
            
            if condition_met:
                # Check if alert already exists
                existing_alert_id = f"{rule.rule_id}_{rule.metric_name}"
                
                if existing_alert_id not in self.active_alerts:
                    # Create new alert
                    alert = self._create_alert(rule, latest_value, current_time)
                    
                    with self._lock:
                        self.active_alerts[alert.alert_id] = alert
                    
                    # Notify callbacks
                    self._notify_callbacks(alert)
                    
                    self.logger.warning(f"Alert triggered: {alert.title}")
                
                # Update last evaluation time
                self.last_evaluation[rule.rule_id] = current_time
                
        except Exception as e:
            self.logger.error(f"Failed to evaluate rule {rule.rule_id}: {e}")
    
    def _evaluate_condition(self, rule: AlertRule, latest_value: float, values: List[float]) -> bool:
        """Evaluate alert condition."""
        try:
            if rule.condition == 'greater_than':
                return latest_value > rule.threshold
            elif rule.condition == 'less_than':
                return latest_value < rule.threshold
            elif rule.condition == 'equals':
                return abs(latest_value - rule.threshold) < 1e-6
            elif rule.condition == 'not_equals':
                return abs(latest_value - rule.threshold) >= 1e-6
            elif rule.condition == 'anomaly':
                # Simple anomaly detection based on standard deviation
                if len(values) < 5:
                    return False
                
                mean_val = np.mean(values[:-1])  # Exclude latest value
                std_val = np.std(values[:-1])
                
                if std_val == 0:
                    return False
                
                z_score = abs((latest_value - mean_val) / std_val)
                return z_score > 3.0  # 3-sigma rule
            
            return False
            
        except Exception as e:
            self.logger.error(f"Failed to evaluate condition: {e}")
            return False
    
    def _create_alert(self, rule: AlertRule, value: float, timestamp: float) -> AlertInstance:
        """Create new alert instance."""
        alert_id = f"{rule.rule_id}_{rule.metric_name}_{int(timestamp)}"
        
        # Generate alert message
        if rule.threshold is not None:
            message = f"{rule.metric_name} is {value:.3f}, threshold: {rule.threshold:.3f}"
        else:
            message = f"{rule.metric_name} triggered anomaly detection with value: {value:.3f}"
        
        return AlertInstance(
            alert_id=alert_id,
            rule_id=rule.rule_id,
            metric_name=rule.metric_name,
            severity=rule.severity,
            status=AlertStatus.ACTIVE,
            title=rule.name,
            message=message,
            triggered_at=timestamp,
            last_update=timestamp,
            threshold=rule.threshold,
            current_value=value,
            tags=rule.tags or {},
            metadata=rule.metadata,
        )
    
    def _check_auto_resolution(self, current_time: float) -> None:
        """Check for automatic alert resolution."""
        try:
            alerts_to_resolve = []
            
            with self._lock:
                for alert in self.active_alerts.values():
                    # Get current metric value
                    latest_value = self.metrics_collector.get_latest_value(alert.metric_name)
                    if latest_value is None:
                        continue
                    
                    # Get corresponding rule
                    rule = self.alert_rules.get(alert.rule_id)
                    if not rule:
                        continue
                    
                    # Check if condition is no longer met
                    condition_met = self._evaluate_condition(rule, latest_value, [latest_value])
                    
                    if not condition_met:
                        alerts_to_resolve.append(alert.alert_id)
            
            # Resolve alerts
            for alert_id in alerts_to_resolve:
                self.resolve_alert(alert_id, "auto-resolution")
                
        except Exception as e:
            self.logger.error(f"Failed to check auto-resolution: {e}")
    
    def _notify_callbacks(self, alert: AlertInstance) -> None:
        """Notify alert callbacks."""
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                self.logger.error(f"Alert callback error: {e}")


class AnomalyDetector:
    """
    Advanced anomaly detection for neuromorphic metrics.
    
    Implements statistical and machine learning-based anomaly detection
    for identifying unusual patterns in neuromorphic system behavior.
    """
    
    def __init__(
        self,
        metrics_collector: Optional[MetricsCollector] = None,
        detection_interval: float = 60.0,
        training_window: int = 1000,  # samples
    ):
        """
        Initialize anomaly detector.
        
        Args:
            metrics_collector: Metrics collector instance
            detection_interval: Detection interval in seconds
            training_window: Number of samples for training
        """
        self.metrics_collector = metrics_collector or get_global_collector()
        self.detection_interval = detection_interval
        self.training_window = training_window
        self.logger = logging.getLogger(__name__)
        
        # Detection models
        self.statistical_models: Dict[str, Dict[str, Any]] = {}
        self.anomaly_thresholds: Dict[str, float] = {}
        
        # Detection state
        self.detection_active = False
        self.detection_thread: Optional[threading.Thread] = None
        
        # Anomaly callbacks
        self.anomaly_callbacks: List[Callable[[str, float, float, Dict[str, Any]], None]] = []
        
        # Anomaly history
        self.anomaly_history: List[Dict[str, Any]] = []
        
        self.logger.info("Initialized anomaly detector")
    
    def add_metric_for_detection(
        self,
        metric_name: str,
        detection_method: str = 'statistical',
        sensitivity: float = 3.0,
    ) -> None:
        """
        Add metric for anomaly detection.
        
        Args:
            metric_name: Metric to monitor
            detection_method: Detection method ('statistical', 'isolation_forest')
            sensitivity: Detection sensitivity (lower = more sensitive)
        """
        self.anomaly_thresholds[metric_name] = sensitivity
        
        if detection_method == 'statistical':
            self.statistical_models[metric_name] = {
                'method': 'statistical',
                'mean': 0.0,
                'std': 1.0,
                'training_data': deque(maxlen=self.training_window),
                'trained': False,
            }
        
        self.logger.info(f"Added metric for anomaly detection: {metric_name}")
    
    def start_detection(self) -> None:
        """Start anomaly detection."""
        if self.detection_active:
            self.logger.warning("Anomaly detection already active")
            return
        
        self.detection_active = True
        self.detection_thread = threading.Thread(
            target=self._detection_loop,
            daemon=True
        )
        self.detection_thread.start()
        
        self.logger.info(f"Started anomaly detection with {self.detection_interval}s interval")
    
    def stop_detection(self) -> None:
        """Stop anomaly detection."""
        if not self.detection_active:
            return
        
        self.detection_active = False
        
        if self.detection_thread and self.detection_thread.is_alive():
            self.detection_thread.join(timeout=5)
        
        self.logger.info("Stopped anomaly detection")
    
    def add_callback(self, callback: Callable[[str, float, float, Dict[str, Any]], None]) -> None:
        """
        Add callback for anomaly notifications.
        
        Args:
            callback: Function called with (metric_name, value, anomaly_score, metadata)
        """
        self.anomaly_callbacks.append(callback)
        self.logger.info("Added anomaly callback")
    
    def remove_callback(self, callback: Callable) -> None:
        """Remove anomaly callback."""
        if callback in self.anomaly_callbacks:
            self.anomaly_callbacks.remove(callback)
            self.logger.info("Removed anomaly callback")
    
    def get_anomaly_history(self, metric_name: Optional[str] = None, hours: int = 24) -> List[Dict[str, Any]]:
        """
        Get anomaly detection history.
        
        Args:
            metric_name: Filter by metric name
            hours: Number of hours to look back
            
        Returns:
            List of anomaly events
        """
        cutoff_time = time.time() - (hours * 3600)
        
        anomalies = [
            anomaly for anomaly in self.anomaly_history
            if anomaly['timestamp'] >= cutoff_time
        ]
        
        if metric_name:
            anomalies = [
                anomaly for anomaly in anomalies
                if anomaly['metric_name'] == metric_name
            ]
        
        return anomalies
    
    def _detection_loop(self) -> None:
        """Main anomaly detection loop."""
        self.logger.info("Started anomaly detection loop")
        
        while self.detection_active:
            try:
                current_time = time.time()
                
                # Process each monitored metric
                for metric_name in self.statistical_models.keys():
                    self._detect_anomalies(metric_name, current_time)
                
                time.sleep(self.detection_interval)
                
            except Exception as e:
                self.logger.error(f"Error in anomaly detection loop: {e}")
                time.sleep(self.detection_interval)
        
        self.logger.info("Stopped anomaly detection loop")
    
    def _detect_anomalies(self, metric_name: str, current_time: float) -> None:
        """Detect anomalies for a specific metric."""
        try:
            # Get latest metric value
            latest_value = self.metrics_collector.get_latest_value(metric_name)
            if latest_value is None:
                return
            
            model = self.statistical_models[metric_name]
            
            # Add to training data
            model['training_data'].append(latest_value)
            
            # Train model if enough data
            if len(model['training_data']) >= 50 and not model['trained']:
                self._train_statistical_model(metric_name)
            
            # Detect anomaly if model is trained
            if model['trained']:
                anomaly_score = self._calculate_anomaly_score(metric_name, latest_value)
                threshold = self.anomaly_thresholds.get(metric_name, 3.0)
                
                if anomaly_score > threshold:
                    self._report_anomaly(metric_name, latest_value, anomaly_score, current_time)
                    
        except Exception as e:
            self.logger.error(f"Failed to detect anomalies for {metric_name}: {e}")
    
    def _train_statistical_model(self, metric_name: str) -> None:
        """Train statistical anomaly detection model."""
        try:
            model = self.statistical_models[metric_name]
            training_data = list(model['training_data'])
            
            if len(training_data) < 10:
                return
            
            # Calculate statistics
            model['mean'] = np.mean(training_data)
            model['std'] = np.std(training_data)
            
            # Ensure non-zero std
            if model['std'] == 0:
                model['std'] = 1e-6
            
            model['trained'] = True
            
            self.logger.info(f"Trained statistical model for {metric_name}")
            
        except Exception as e:
            self.logger.error(f"Failed to train model for {metric_name}: {e}")
    
    def _calculate_anomaly_score(self, metric_name: str, value: float) -> float:
        """Calculate anomaly score for value."""
        try:
            model = self.statistical_models[metric_name]
            
            if not model['trained']:
                return 0.0
            
            # Z-score based anomaly score
            z_score = abs((value - model['mean']) / model['std'])
            return z_score
            
        except Exception as e:
            self.logger.error(f"Failed to calculate anomaly score: {e}")
            return 0.0
    
    def _report_anomaly(
        self,
        metric_name: str,
        value: float,
        anomaly_score: float,
        timestamp: float,
    ) -> None:
        """Report detected anomaly."""
        try:
            # Create anomaly record
            anomaly = {
                'metric_name': metric_name,
                'value': value,
                'anomaly_score': anomaly_score,
                'timestamp': timestamp,
                'detection_method': 'statistical',
                'model_stats': {
                    'mean': self.statistical_models[metric_name]['mean'],
                    'std': self.statistical_models[metric_name]['std'],
                }
            }
            
            # Add to history
            self.anomaly_history.append(anomaly)
            
            # Keep history size manageable
            if len(self.anomaly_history) > 10000:
                self.anomaly_history = self.anomaly_history[-5000:]
            
            self.logger.warning(
                f"Anomaly detected in {metric_name}: value={value:.3f}, "
                f"score={anomaly_score:.2f}"
            )
            
            # Notify callbacks
            for callback in self.anomaly_callbacks:
                try:
                    callback(metric_name, value, anomaly_score, anomaly)
                except Exception as e:
                    self.logger.error(f"Anomaly callback error: {e}")
                    
        except Exception as e:
            self.logger.error(f"Failed to report anomaly: {e}")


class AlertManager:
    """
    Central alert management system combining threshold and anomaly detection.
    
    Coordinates multiple alert systems and provides unified alert management
    for neuromorphic computing infrastructure.
    """
    
    def __init__(
        self,
        metrics_collector: Optional[MetricsCollector] = None,
        threshold_alert: Optional[ThresholdAlert] = None,
        anomaly_detector: Optional[AnomalyDetector] = None,
    ):
        """
        Initialize alert manager.
        
        Args:
            metrics_collector: Metrics collector instance
            threshold_alert: Threshold alert system
            anomaly_detector: Anomaly detector
        """
        self.metrics_collector = metrics_collector or get_global_collector()
        self.threshold_alert = threshold_alert or ThresholdAlert(self.metrics_collector)
        self.anomaly_detector = anomaly_detector or AnomalyDetector(self.metrics_collector)
        self.logger = logging.getLogger(__name__)
        
        # Integration callbacks
        self.threshold_alert.add_callback(self._handle_threshold_alert)
        self.anomaly_detector.add_callback(self._handle_anomaly_detection)
        
        # Global alert callbacks
        self.global_callbacks: List[Callable[[Dict[str, Any]], None]] = []
        
        self.logger.info("Initialized alert manager")
    
    def start_all_monitoring(self) -> None:
        """Start all alert and anomaly monitoring."""
        self.threshold_alert.start_evaluation()
        self.anomaly_detector.start_detection()
        self.logger.info("Started all alert monitoring")
    
    def stop_all_monitoring(self) -> None:
        """Stop all alert and anomaly monitoring."""
        self.threshold_alert.stop_evaluation()
        self.anomaly_detector.stop_detection()
        self.logger.info("Stopped all alert monitoring")
    
    def add_global_callback(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """Add global alert callback."""
        self.global_callbacks.append(callback)
        self.logger.info("Added global alert callback")
    
    def get_alert_summary(self) -> Dict[str, Any]:
        """Get comprehensive alert summary."""
        try:
            active_alerts = self.threshold_alert.get_active_alerts()
            recent_anomalies = self.anomaly_detector.get_anomaly_history(hours=1)
            
            # Count by severity
            severity_counts = {severity.value: 0 for severity in AlertSeverity}
            for alert in active_alerts:
                severity_counts[alert.severity.value] += 1
            
            return {
                'active_alerts_count': len(active_alerts),
                'recent_anomalies_count': len(recent_anomalies),
                'severity_breakdown': severity_counts,
                'most_recent_alert': max(
                    active_alerts, key=lambda x: x.triggered_at, default=None
                ),
                'timestamp': time.time(),
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get alert summary: {e}")
            return {'error': str(e)}
    
    def _handle_threshold_alert(self, alert: AlertInstance) -> None:
        """Handle threshold alert notifications."""
        try:
            alert_data = {
                'type': 'threshold_alert',
                'alert': asdict(alert),
                'timestamp': time.time(),
            }
            
            # Notify global callbacks
            for callback in self.global_callbacks:
                try:
                    callback(alert_data)
                except Exception as e:
                    self.logger.error(f"Global callback error: {e}")
                    
        except Exception as e:
            self.logger.error(f"Failed to handle threshold alert: {e}")
    
    def _handle_anomaly_detection(
        self,
        metric_name: str,
        value: float,
        anomaly_score: float,
        metadata: Dict[str, Any],
    ) -> None:
        """Handle anomaly detection notifications."""
        try:
            anomaly_data = {
                'type': 'anomaly_detection',
                'metric_name': metric_name,
                'value': value,
                'anomaly_score': anomaly_score,
                'metadata': metadata,
                'timestamp': time.time(),
            }
            
            # Notify global callbacks
            for callback in self.global_callbacks:
                try:
                    callback(anomaly_data)
                except Exception as e:
                    self.logger.error(f"Global callback error: {e}")
                    
        except Exception as e:
            self.logger.error(f"Failed to handle anomaly detection: {e}")