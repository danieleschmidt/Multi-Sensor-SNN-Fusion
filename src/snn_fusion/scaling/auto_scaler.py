"""
Auto-Scaling System for Neuromorphic Computing

Implements intelligent auto-scaling based on load, performance metrics,
and resource utilization for dynamic scaling of SNN workloads.
"""

import time
import threading
import logging
from typing import Dict, List, Optional, Callable, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import statistics
from collections import deque, defaultdict
import json
import warnings


class ScalingDirection(Enum):
    """Scaling directions."""
    UP = "up"
    DOWN = "down"
    STABLE = "stable"


class ScalingStrategy(Enum):
    """Auto-scaling strategies."""
    REACTIVE = "reactive"
    PREDICTIVE = "predictive"
    HYBRID = "hybrid"


class ResourceMetricType(Enum):
    """Types of resource metrics."""
    CPU_UTILIZATION = "cpu_utilization"
    MEMORY_UTILIZATION = "memory_utilization"
    GPU_UTILIZATION = "gpu_utilization"
    QUEUE_LENGTH = "queue_length"
    THROUGHPUT = "throughput"
    LATENCY = "latency"
    ERROR_RATE = "error_rate"


@dataclass
class ScalingMetrics:
    """Metrics for scaling decisions."""
    timestamp: float
    cpu_utilization: float = 0.0
    memory_utilization: float = 0.0
    gpu_utilization: float = 0.0
    queue_length: int = 0
    throughput: float = 0.0
    average_latency: float = 0.0
    error_rate: float = 0.0
    active_workers: int = 0
    pending_tasks: int = 0
    custom_metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class ScalingThreshold:
    """Scaling threshold configuration."""
    metric_type: ResourceMetricType
    scale_up_threshold: float
    scale_down_threshold: float
    min_samples: int = 3
    evaluation_window_seconds: int = 60
    cooldown_seconds: int = 300


@dataclass
class ScalingRule:
    """Scaling rule configuration."""
    name: str
    thresholds: List[ScalingThreshold]
    min_instances: int = 1
    max_instances: int = 100
    scale_up_factor: float = 1.5
    scale_down_factor: float = 0.7
    enabled: bool = True
    priority: int = 1


@dataclass
class ScalingAction:
    """Represents a scaling action."""
    timestamp: float
    direction: ScalingDirection
    rule_name: str
    from_instances: int
    to_instances: int
    reason: str
    metrics: ScalingMetrics


class PredictiveModel:
    """
    Simple predictive model for load forecasting.
    """
    
    def __init__(self, window_size: int = 60):
        self.window_size = window_size
        self.historical_data = deque(maxlen=window_size)
        self.trend_weights = [0.1, 0.2, 0.3, 0.4]  # Recent data weighted more heavily
    
    def add_sample(self, metrics: ScalingMetrics):
        """Add a metrics sample."""
        self.historical_data.append(metrics)
    
    def predict_load(self, forecast_minutes: int = 5) -> float:
        """
        Predict future load based on historical data.
        
        Args:
            forecast_minutes: Minutes into the future to predict
            
        Returns:
            Predicted CPU utilization percentage
        """
        if len(self.historical_data) < 4:
            return 50.0  # Default moderate load
        
        # Get recent samples
        recent_samples = list(self.historical_data)[-4:]
        
        # Calculate weighted trend
        weighted_sum = sum(
            sample.cpu_utilization * weight 
            for sample, weight in zip(recent_samples, self.trend_weights)
        )
        
        # Simple linear extrapolation
        if len(recent_samples) >= 2:
            trend = (recent_samples[-1].cpu_utilization - recent_samples[0].cpu_utilization) / 3
            predicted_load = weighted_sum + (trend * forecast_minutes / 5)
        else:
            predicted_load = weighted_sum
        
        return max(0.0, min(100.0, predicted_load))
    
    def get_trend(self) -> float:
        """Get current load trend (positive = increasing)."""
        if len(self.historical_data) < 2:
            return 0.0
        
        recent = list(self.historical_data)[-5:]  # Last 5 samples
        if len(recent) < 2:
            return 0.0
        
        # Calculate average change per sample
        changes = [
            recent[i].cpu_utilization - recent[i-1].cpu_utilization
            for i in range(1, len(recent))
        ]
        
        return statistics.mean(changes) if changes else 0.0


class AutoScaler:
    """
    Intelligent auto-scaling system for neuromorphic workloads.
    """
    
    def __init__(self, strategy: ScalingStrategy = ScalingStrategy.HYBRID):
        self.strategy = strategy
        self.logger = logging.getLogger(__name__)
        
        # Scaling configuration
        self.scaling_rules: Dict[str, ScalingRule] = {}
        self.current_instances = 1
        self.target_instances = 1
        
        # Metrics collection
        self.metrics_history = deque(maxlen=1440)  # 24 hours at 1-minute intervals
        self.metrics_callbacks: List[Callable[[], ScalingMetrics]] = []
        
        # Predictive modeling
        self.predictive_model = PredictiveModel()
        
        # Scaling state
        self.last_scaling_action: Optional[ScalingAction] = None
        self.cooldown_until = 0.0
        self.scaling_actions_history = deque(maxlen=100)
        
        # Control
        self.running = False
        self.monitoring_thread: Optional[threading.Thread] = None
        self.decision_thread: Optional[threading.Thread] = None
        
        # Statistics
        self.stats = {
            'scaling_actions': 0,
            'scale_up_actions': 0,
            'scale_down_actions': 0,
            'cooldown_periods': 0,
            'prediction_accuracy': 0.0
        }
        
        self.stats_lock = threading.Lock()
        
        # Setup default scaling rules
        self._setup_default_rules()
    
    def _setup_default_rules(self):
        """Setup default scaling rules."""
        # CPU-based scaling
        cpu_rule = ScalingRule(
            name="cpu_scaling",
            thresholds=[
                ScalingThreshold(
                    metric_type=ResourceMetricType.CPU_UTILIZATION,
                    scale_up_threshold=70.0,
                    scale_down_threshold=30.0,
                    min_samples=3,
                    evaluation_window_seconds=60,
                    cooldown_seconds=300
                )
            ],
            min_instances=1,
            max_instances=20,
            scale_up_factor=1.5,
            scale_down_factor=0.7,
            priority=1
        )
        
        # Memory-based scaling
        memory_rule = ScalingRule(
            name="memory_scaling",
            thresholds=[
                ScalingThreshold(
                    metric_type=ResourceMetricType.MEMORY_UTILIZATION,
                    scale_up_threshold=80.0,
                    scale_down_threshold=40.0,
                    min_samples=2,
                    evaluation_window_seconds=120,
                    cooldown_seconds=600
                )
            ],
            min_instances=1,
            max_instances=15,
            scale_up_factor=1.3,
            scale_down_factor=0.8,
            priority=2
        )
        
        # Queue-based scaling
        queue_rule = ScalingRule(
            name="queue_scaling",
            thresholds=[
                ScalingThreshold(
                    metric_type=ResourceMetricType.QUEUE_LENGTH,
                    scale_up_threshold=100.0,
                    scale_down_threshold=10.0,
                    min_samples=2,
                    evaluation_window_seconds=30,
                    cooldown_seconds=180
                )
            ],
            min_instances=1,
            max_instances=50,
            scale_up_factor=2.0,
            scale_down_factor=0.5,
            priority=3
        )
        
        self.scaling_rules["cpu"] = cpu_rule
        self.scaling_rules["memory"] = memory_rule
        self.scaling_rules["queue"] = queue_rule
    
    def add_scaling_rule(self, rule: ScalingRule):
        """Add a custom scaling rule."""
        self.scaling_rules[rule.name] = rule
        self.logger.info(f"Added scaling rule: {rule.name}")
    
    def remove_scaling_rule(self, rule_name: str):
        """Remove a scaling rule."""
        if rule_name in self.scaling_rules:
            del self.scaling_rules[rule_name]
            self.logger.info(f"Removed scaling rule: {rule_name}")
    
    def add_metrics_callback(self, callback: Callable[[], ScalingMetrics]):
        """Add a callback to collect metrics."""
        self.metrics_callbacks.append(callback)
    
    def start(self):
        """Start the auto-scaler."""
        if self.running:
            return
        
        self.running = True
        
        # Start monitoring thread
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True
        )
        self.monitoring_thread.start()
        
        # Start decision thread
        self.decision_thread = threading.Thread(
            target=self._decision_loop,
            daemon=True
        )
        self.decision_thread.start()
        
        self.logger.info(f"AutoScaler started with {self.strategy.value} strategy")
    
    def stop(self):
        """Stop the auto-scaler."""
        self.running = False
        
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5.0)
        
        if self.decision_thread:
            self.decision_thread.join(timeout=5.0)
        
        self.logger.info("AutoScaler stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.running:
            try:
                # Collect metrics from all callbacks
                if self.metrics_callbacks:
                    combined_metrics = self._collect_metrics()
                    
                    # Store metrics
                    self.metrics_history.append(combined_metrics)
                    
                    # Update predictive model
                    if self.strategy in [ScalingStrategy.PREDICTIVE, ScalingStrategy.HYBRID]:
                        self.predictive_model.add_sample(combined_metrics)
                
                time.sleep(10.0)  # Collect metrics every 10 seconds
                
            except Exception as e:
                self.logger.error(f"Monitoring loop error: {e}")
                time.sleep(5.0)
    
    def _decision_loop(self):
        """Main scaling decision loop."""
        while self.running:
            try:
                if len(self.metrics_history) >= 3:  # Need minimum samples
                    decision = self._make_scaling_decision()
                    
                    if decision and decision.direction != ScalingDirection.STABLE:
                        self._execute_scaling_action(decision)
                
                time.sleep(30.0)  # Make decisions every 30 seconds
                
            except Exception as e:
                self.logger.error(f"Decision loop error: {e}")
                time.sleep(15.0)
    
    def _collect_metrics(self) -> ScalingMetrics:
        """Collect metrics from all callbacks."""
        timestamp = time.time()
        aggregated_metrics = ScalingMetrics(timestamp=timestamp)
        
        # Collect from all callbacks
        all_metrics = []
        for callback in self.metrics_callbacks:
            try:
                metrics = callback()
                all_metrics.append(metrics)
            except Exception as e:
                self.logger.warning(f"Metrics callback error: {e}")
        
        if not all_metrics:
            return aggregated_metrics
        
        # Aggregate metrics
        aggregated_metrics.cpu_utilization = statistics.mean(
            m.cpu_utilization for m in all_metrics
        )
        aggregated_metrics.memory_utilization = statistics.mean(
            m.memory_utilization for m in all_metrics
        )
        aggregated_metrics.gpu_utilization = statistics.mean(
            m.gpu_utilization for m in all_metrics
        )
        aggregated_metrics.queue_length = sum(m.queue_length for m in all_metrics)
        aggregated_metrics.throughput = sum(m.throughput for m in all_metrics)
        aggregated_metrics.average_latency = statistics.mean(
            m.average_latency for m in all_metrics if m.average_latency > 0
        ) if any(m.average_latency > 0 for m in all_metrics) else 0.0
        aggregated_metrics.error_rate = statistics.mean(
            m.error_rate for m in all_metrics
        )
        aggregated_metrics.active_workers = max(m.active_workers for m in all_metrics)
        aggregated_metrics.pending_tasks = sum(m.pending_tasks for m in all_metrics)
        
        return aggregated_metrics
    
    def _make_scaling_decision(self) -> Optional[ScalingAction]:
        """Make scaling decision based on current strategy."""
        current_time = time.time()
        
        # Check cooldown
        if current_time < self.cooldown_until:
            return None
        
        # Get recent metrics
        recent_metrics = list(self.metrics_history)[-10:]  # Last 10 samples
        current_metrics = recent_metrics[-1]
        
        # Evaluate all rules
        scaling_decisions = []
        
        for rule_name, rule in self.scaling_rules.items():
            if not rule.enabled:
                continue
            
            decision = self._evaluate_rule(rule, recent_metrics, current_metrics)
            if decision:
                scaling_decisions.append((rule.priority, decision))
        
        if not scaling_decisions:
            return None
        
        # Sort by priority (higher priority first)
        scaling_decisions.sort(key=lambda x: x[0], reverse=True)
        
        # Return highest priority decision
        return scaling_decisions[0][1]
    
    def _evaluate_rule(self, rule: ScalingRule, recent_metrics: List[ScalingMetrics],
                      current_metrics: ScalingMetrics) -> Optional[ScalingAction]:
        """Evaluate a single scaling rule."""
        for threshold in rule.thresholds:
            # Get metric values
            metric_values = self._get_metric_values(threshold.metric_type, recent_metrics)
            
            if len(metric_values) < threshold.min_samples:
                continue
            
            # Calculate average over evaluation window
            avg_value = statistics.mean(metric_values[-threshold.min_samples:])
            
            # Check for scaling decision
            if avg_value >= threshold.scale_up_threshold:
                # Scale up
                new_instances = min(
                    rule.max_instances,
                    max(1, int(self.current_instances * rule.scale_up_factor))
                )
                
                if new_instances > self.current_instances:
                    return ScalingAction(
                        timestamp=time.time(),
                        direction=ScalingDirection.UP,
                        rule_name=rule.name,
                        from_instances=self.current_instances,
                        to_instances=new_instances,
                        reason=f"{threshold.metric_type.value} ({avg_value:.2f}) > {threshold.scale_up_threshold}",
                        metrics=current_metrics
                    )
            
            elif avg_value <= threshold.scale_down_threshold:
                # Scale down
                new_instances = max(
                    rule.min_instances,
                    max(1, int(self.current_instances * rule.scale_down_factor))
                )
                
                if new_instances < self.current_instances:
                    return ScalingAction(
                        timestamp=time.time(),
                        direction=ScalingDirection.DOWN,
                        rule_name=rule.name,
                        from_instances=self.current_instances,
                        to_instances=new_instances,
                        reason=f"{threshold.metric_type.value} ({avg_value:.2f}) < {threshold.scale_down_threshold}",
                        metrics=current_metrics
                    )
        
        return None
    
    def _get_metric_values(self, metric_type: ResourceMetricType, 
                          metrics_list: List[ScalingMetrics]) -> List[float]:
        """Extract metric values of a specific type."""
        values = []
        
        for metrics in metrics_list:
            if metric_type == ResourceMetricType.CPU_UTILIZATION:
                values.append(metrics.cpu_utilization)
            elif metric_type == ResourceMetricType.MEMORY_UTILIZATION:
                values.append(metrics.memory_utilization)
            elif metric_type == ResourceMetricType.GPU_UTILIZATION:
                values.append(metrics.gpu_utilization)
            elif metric_type == ResourceMetricType.QUEUE_LENGTH:
                values.append(float(metrics.queue_length))
            elif metric_type == ResourceMetricType.THROUGHPUT:
                values.append(metrics.throughput)
            elif metric_type == ResourceMetricType.LATENCY:
                values.append(metrics.average_latency)
            elif metric_type == ResourceMetricType.ERROR_RATE:
                values.append(metrics.error_rate)
        
        return values
    
    def _execute_scaling_action(self, action: ScalingAction):
        """Execute a scaling action."""
        self.logger.info(
            f"Scaling {action.direction.value}: {action.from_instances} -> {action.to_instances} "
            f"(Rule: {action.rule_name}, Reason: {action.reason})"
        )
        
        # Update current instances
        self.current_instances = action.to_instances
        self.target_instances = action.to_instances
        
        # Set cooldown period
        cooldown_time = 300  # Default 5 minutes
        for rule in self.scaling_rules.values():
            for threshold in rule.thresholds:
                cooldown_time = max(cooldown_time, threshold.cooldown_seconds)
        
        self.cooldown_until = time.time() + cooldown_time
        
        # Store action
        self.last_scaling_action = action
        self.scaling_actions_history.append(action)
        
        # Update statistics
        with self.stats_lock:
            self.stats['scaling_actions'] += 1
            if action.direction == ScalingDirection.UP:
                self.stats['scale_up_actions'] += 1
            else:
                self.stats['scale_down_actions'] += 1
            
            if time.time() < self.cooldown_until:
                self.stats['cooldown_periods'] += 1
    
    def get_current_state(self) -> Dict[str, Any]:
        """Get current auto-scaler state."""
        return {
            'strategy': self.strategy.value,
            'current_instances': self.current_instances,
            'target_instances': self.target_instances,
            'is_running': self.running,
            'in_cooldown': time.time() < self.cooldown_until,
            'cooldown_remaining': max(0, self.cooldown_until - time.time()),
            'last_scaling_action': self.last_scaling_action.__dict__ if self.last_scaling_action else None,
            'active_rules': len([r for r in self.scaling_rules.values() if r.enabled]),
            'metrics_history_length': len(self.metrics_history)
        }
    
    def get_recommendations(self) -> List[str]:
        """Get scaling recommendations based on current state."""
        recommendations = []
        
        if not self.metrics_history:
            recommendations.append("Insufficient metrics data for recommendations")
            return recommendations
        
        recent_metrics = list(self.metrics_history)[-5:]
        current = recent_metrics[-1]
        
        # CPU recommendations
        if current.cpu_utilization > 80:
            recommendations.append("High CPU utilization detected - consider scaling up")
        elif current.cpu_utilization < 20 and self.current_instances > 1:
            recommendations.append("Low CPU utilization - consider scaling down")
        
        # Memory recommendations
        if current.memory_utilization > 85:
            recommendations.append("High memory utilization - scaling up recommended")
        
        # Queue recommendations
        if current.queue_length > 200:
            recommendations.append("Large queue detected - immediate scaling up recommended")
        
        # Error rate recommendations
        if current.error_rate > 5.0:
            recommendations.append("High error rate - check system health before scaling")
        
        # Trend analysis
        if self.strategy in [ScalingStrategy.PREDICTIVE, ScalingStrategy.HYBRID]:
            trend = self.predictive_model.get_trend()
            if trend > 2.0:
                recommendations.append("Increasing load trend - proactive scaling up suggested")
            elif trend < -2.0:
                recommendations.append("Decreasing load trend - scaling down opportunity")
        
        if not recommendations:
            recommendations.append("System appears to be operating within normal parameters")
        
        return recommendations
    
    def get_scaling_history(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get recent scaling history."""
        recent_actions = list(self.scaling_actions_history)[-limit:]
        return [
            {
                'timestamp': action.timestamp,
                'direction': action.direction.value,
                'rule_name': action.rule_name,
                'from_instances': action.from_instances,
                'to_instances': action.to_instances,
                'reason': action.reason
            }
            for action in recent_actions
        ]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get auto-scaling statistics."""
        with self.stats_lock:
            stats = self.stats.copy()
        
        # Add derived statistics
        if stats['scaling_actions'] > 0:
            stats['scale_up_percentage'] = (stats['scale_up_actions'] / stats['scaling_actions']) * 100
            stats['scale_down_percentage'] = (stats['scale_down_actions'] / stats['scaling_actions']) * 100
        else:
            stats['scale_up_percentage'] = 0.0
            stats['scale_down_percentage'] = 0.0
        
        return stats


# Example usage and testing
if __name__ == "__main__":
    print("Testing Auto-Scaling System...")
    
    # Create auto-scaler
    auto_scaler = AutoScaler(ScalingStrategy.HYBRID)
    
    # Create mock metrics callback
    def mock_metrics_callback():
        return ScalingMetrics(
            timestamp=time.time(),
            cpu_utilization=60.0 + (time.time() % 30),  # Varying load
            memory_utilization=45.0,
            queue_length=50,
            throughput=100.0,
            average_latency=0.05,
            error_rate=1.0,
            active_workers=5,
            pending_tasks=25
        )
    
    # Add metrics callback
    auto_scaler.add_metrics_callback(mock_metrics_callback)
    
    # Test rule management
    print("\n1. Testing Rule Management:")
    
    custom_rule = ScalingRule(
        name="custom_test",
        thresholds=[
            ScalingThreshold(
                metric_type=ResourceMetricType.THROUGHPUT,
                scale_up_threshold=150.0,
                scale_down_threshold=50.0
            )
        ],
        min_instances=2,
        max_instances=10
    )
    
    auto_scaler.add_scaling_rule(custom_rule)
    assert "custom_test" in auto_scaler.scaling_rules
    print("  ✓ Custom rule added")
    
    auto_scaler.remove_scaling_rule("custom_test")
    assert "custom_test" not in auto_scaler.scaling_rules
    print("  ✓ Custom rule removed")
    
    # Test state and recommendations
    print("\n2. Testing State and Recommendations:")
    
    # Start auto-scaler briefly to collect some data
    auto_scaler.start()
    time.sleep(2.0)  # Let it collect some metrics
    
    state = auto_scaler.get_current_state()
    assert state['is_running'] == True
    print(f"  ✓ Current state: {state['current_instances']} instances")
    
    recommendations = auto_scaler.get_recommendations()
    assert len(recommendations) > 0
    print(f"  ✓ Got {len(recommendations)} recommendations")
    
    # Test predictive model
    print("\n3. Testing Predictive Model:")
    
    model = PredictiveModel(window_size=10)
    
    # Add some test data
    for i in range(10):
        metrics = ScalingMetrics(
            timestamp=time.time() + i,
            cpu_utilization=50.0 + i * 2  # Increasing trend
        )
        model.add_sample(metrics)
    
    predicted_load = model.predict_load(5)
    trend = model.get_trend()
    
    assert predicted_load > 50.0  # Should predict higher load
    assert trend > 0.0  # Should detect increasing trend
    
    print(f"  ✓ Predicted load: {predicted_load:.1f}%")
    print(f"  ✓ Load trend: {trend:.1f}")
    
    # Get statistics
    print("\n4. Testing Statistics:")
    
    stats = auto_scaler.get_statistics()
    history = auto_scaler.get_scaling_history()
    
    print(f"  ✓ Statistics: {stats['scaling_actions']} total actions")
    print(f"  ✓ History: {len(history)} recorded actions")
    
    # Stop auto-scaler
    auto_scaler.stop()
    print("\n✓ Auto-scaling system test completed!")