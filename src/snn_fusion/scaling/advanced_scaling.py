"""
Advanced Auto-Scaling and Load Balancing for SNN-Fusion

Intelligent resource management, dynamic scaling, and load distribution
for production SNN-Fusion deployments with predictive scaling algorithms.
"""

import time
import asyncio
import threading
import statistics
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging
import json
import numpy as np
from collections import deque, defaultdict
import heapq
import psutil
import socket
from concurrent.futures import ThreadPoolExecutor
import uuid


class ScalingDirection(Enum):
    """Scaling directions."""
    UP = "up"
    DOWN = "down"
    STABLE = "stable"


class LoadBalancingStrategy(Enum):
    """Load balancing strategies."""
    ROUND_ROBIN = "round_robin"
    LEAST_CONNECTIONS = "least_connections"
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"
    LEAST_RESPONSE_TIME = "least_response_time"
    RESOURCE_BASED = "resource_based"
    PREDICTIVE = "predictive"


class NodeStatus(Enum):
    """Node status states."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    DRAINING = "draining"
    OFFLINE = "offline"


@dataclass
class WorkerNode:
    """Represents a worker node in the cluster."""
    node_id: str
    host: str
    port: int
    capacity: int = 100  # Max concurrent requests
    current_load: int = 0
    status: NodeStatus = NodeStatus.HEALTHY
    weight: float = 1.0
    last_health_check: datetime = field(default_factory=datetime.now)
    response_times: deque = field(default_factory=lambda: deque(maxlen=100))
    total_requests: int = 0
    failed_requests: int = 0
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    created_at: datetime = field(default_factory=datetime.now)
    
    @property
    def load_percentage(self) -> float:
        """Calculate current load as percentage of capacity."""
        return (self.current_load / max(self.capacity, 1)) * 100
    
    @property
    def average_response_time(self) -> float:
        """Calculate average response time."""
        return statistics.mean(self.response_times) if self.response_times else 0.0
    
    @property
    def error_rate(self) -> float:
        """Calculate error rate as percentage."""
        if self.total_requests == 0:
            return 0.0
        return (self.failed_requests / self.total_requests) * 100
    
    @property
    def health_score(self) -> float:
        """Calculate overall health score (0-100)."""
        if self.status == NodeStatus.OFFLINE:
            return 0.0
        
        # Factors contributing to health score
        load_score = max(0, 100 - self.load_percentage)
        response_time_score = max(0, 100 - min(self.average_response_time * 10, 100))
        error_score = max(0, 100 - self.error_rate)
        resource_score = max(0, 100 - max(self.cpu_usage, self.memory_usage))
        
        # Weighted average
        health_score = (
            load_score * 0.3 +
            response_time_score * 0.2 +
            error_score * 0.3 +
            resource_score * 0.2
        )
        
        return min(100, max(0, health_score))


@dataclass
class ScalingMetrics:
    """Metrics for scaling decisions."""
    timestamp: datetime
    total_requests_per_second: float
    average_response_time: float
    error_rate: float
    cpu_usage: float
    memory_usage: float
    active_connections: int
    queue_depth: int
    prediction_confidence: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'total_rps': self.total_requests_per_second,
            'avg_response_time': self.average_response_time,
            'error_rate': self.error_rate,
            'cpu_usage': self.cpu_usage,
            'memory_usage': self.memory_usage,
            'active_connections': self.active_connections,
            'queue_depth': self.queue_depth,
            'prediction_confidence': self.prediction_confidence
        }


@dataclass
class ScalingConfig:
    """Configuration for auto-scaling."""
    min_nodes: int = 2
    max_nodes: int = 20
    target_cpu_utilization: float = 70.0
    target_response_time: float = 1.0  # seconds
    scale_up_threshold: float = 80.0
    scale_down_threshold: float = 30.0
    cooldown_period: int = 300  # seconds
    prediction_window: int = 600  # seconds for predictive scaling
    health_check_interval: int = 30
    enable_predictive_scaling: bool = True
    aggressive_scaling: bool = False


class PredictiveScaler:
    """Predictive scaling using time series analysis."""
    
    def __init__(self, window_size: int = 100):
        """Initialize predictive scaler."""
        self.window_size = window_size
        self.metrics_history: deque = deque(maxlen=window_size)
        self.logger = logging.getLogger(__name__)
    
    def add_metrics(self, metrics: ScalingMetrics):
        """Add metrics to history."""
        self.metrics_history.append(metrics)
    
    def predict_load(self, horizon_minutes: int = 10) -> Tuple[float, float]:
        """
        Predict future load using simple trend analysis.
        
        Returns:
            Tuple of (predicted_rps, confidence_score)
        """
        if len(self.metrics_history) < 10:
            # Not enough data for prediction
            if self.metrics_history:
                current_rps = self.metrics_history[-1].total_requests_per_second
                return current_rps, 0.1
            return 0.0, 0.0
        
        # Extract RPS values
        rps_values = [m.total_requests_per_second for m in self.metrics_history]
        time_points = list(range(len(rps_values)))
        
        # Simple linear trend analysis
        try:
            # Calculate linear trend
            n = len(rps_values)
            sum_x = sum(time_points)
            sum_y = sum(rps_values)
            sum_xy = sum(x * y for x, y in zip(time_points, rps_values))
            sum_x2 = sum(x * x for x in time_points)
            
            # Linear regression coefficients
            slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
            intercept = (sum_y - slope * sum_x) / n
            
            # Predict future value
            future_point = len(rps_values) + horizon_minutes
            predicted_rps = slope * future_point + intercept
            
            # Calculate confidence based on recent trend consistency
            recent_values = rps_values[-min(20, len(rps_values)):]
            variance = statistics.variance(recent_values) if len(recent_values) > 1 else 0
            
            # Confidence decreases with variance and prediction distance
            confidence = max(0.1, min(0.9, 1.0 / (1.0 + variance * 0.1 + horizon_minutes * 0.02)))
            
            return max(0, predicted_rps), confidence
            
        except (ValueError, ZeroDivisionError):
            # Fallback to current value
            current_rps = rps_values[-1]
            return current_rps, 0.2
    
    def detect_pattern(self) -> Dict[str, Any]:
        """Detect patterns in the metrics history."""
        if len(self.metrics_history) < 20:
            return {'pattern': 'insufficient_data'}
        
        rps_values = [m.total_requests_per_second for m in self.metrics_history]
        
        # Check for periodic patterns (simplified)
        # Look for daily, hourly patterns
        patterns = {}
        
        # Trend detection
        if len(rps_values) >= 10:
            recent_avg = statistics.mean(rps_values[-10:])
            older_avg = statistics.mean(rps_values[-20:-10])
            
            if recent_avg > older_avg * 1.2:
                patterns['trend'] = 'increasing'
            elif recent_avg < older_avg * 0.8:
                patterns['trend'] = 'decreasing'
            else:
                patterns['trend'] = 'stable'
        
        # Volatility detection
        if len(rps_values) >= 10:
            variance = statistics.variance(rps_values[-10:])
            if variance > statistics.mean(rps_values[-10:]) * 0.5:
                patterns['volatility'] = 'high'
            elif variance > statistics.mean(rps_values[-10:]) * 0.2:
                patterns['volatility'] = 'medium'
            else:
                patterns['volatility'] = 'low'
        
        return patterns


class LoadBalancer:
    """Advanced load balancer with multiple strategies."""
    
    def __init__(self, strategy: LoadBalancingStrategy = LoadBalancingStrategy.LEAST_RESPONSE_TIME):
        """Initialize load balancer."""
        self.strategy = strategy
        self.nodes: Dict[str, WorkerNode] = {}
        self.round_robin_index = 0
        self.logger = logging.getLogger(__name__)
        self.request_history: deque = deque(maxlen=10000)
        self.lock = threading.RLock()
    
    def add_node(self, node: WorkerNode):
        """Add a worker node."""
        with self.lock:
            self.nodes[node.node_id] = node
            self.logger.info(f"Added node {node.node_id} ({node.host}:{node.port})")
    
    def remove_node(self, node_id: str):
        """Remove a worker node."""
        with self.lock:
            if node_id in self.nodes:
                del self.nodes[node_id]
                self.logger.info(f"Removed node {node_id}")
    
    def get_node(self, request_id: str = None) -> Optional[WorkerNode]:
        """Get next available node based on load balancing strategy."""
        with self.lock:
            available_nodes = [
                node for node in self.nodes.values()
                if node.status in [NodeStatus.HEALTHY, NodeStatus.DEGRADED] and 
                node.current_load < node.capacity
            ]
            
            if not available_nodes:
                self.logger.warning("No available nodes for load balancing")
                return None
            
            if self.strategy == LoadBalancingStrategy.ROUND_ROBIN:
                return self._round_robin_selection(available_nodes)
            elif self.strategy == LoadBalancingStrategy.LEAST_CONNECTIONS:
                return self._least_connections_selection(available_nodes)
            elif self.strategy == LoadBalancingStrategy.WEIGHTED_ROUND_ROBIN:
                return self._weighted_round_robin_selection(available_nodes)
            elif self.strategy == LoadBalancingStrategy.LEAST_RESPONSE_TIME:
                return self._least_response_time_selection(available_nodes)
            elif self.strategy == LoadBalancingStrategy.RESOURCE_BASED:
                return self._resource_based_selection(available_nodes)
            elif self.strategy == LoadBalancingStrategy.PREDICTIVE:
                return self._predictive_selection(available_nodes)
            else:
                return available_nodes[0]  # Fallback
    
    def _round_robin_selection(self, nodes: List[WorkerNode]) -> WorkerNode:
        """Round-robin node selection."""
        node = nodes[self.round_robin_index % len(nodes)]
        self.round_robin_index += 1
        return node
    
    def _least_connections_selection(self, nodes: List[WorkerNode]) -> WorkerNode:
        """Select node with least connections."""
        return min(nodes, key=lambda n: n.current_load)
    
    def _weighted_round_robin_selection(self, nodes: List[WorkerNode]) -> WorkerNode:
        """Weighted round-robin selection based on node weight."""
        # Build weighted list
        weighted_nodes = []
        for node in nodes:
            weight_count = max(1, int(node.weight * 10))
            weighted_nodes.extend([node] * weight_count)
        
        if not weighted_nodes:
            return nodes[0]
        
        selected = weighted_nodes[self.round_robin_index % len(weighted_nodes)]
        self.round_robin_index += 1
        return selected
    
    def _least_response_time_selection(self, nodes: List[WorkerNode]) -> WorkerNode:
        """Select node with best response time."""
        return min(nodes, key=lambda n: n.average_response_time or float('inf'))
    
    def _resource_based_selection(self, nodes: List[WorkerNode]) -> WorkerNode:
        """Select node based on resource utilization."""
        return min(nodes, key=lambda n: max(n.cpu_usage, n.memory_usage))
    
    def _predictive_selection(self, nodes: List[WorkerNode]) -> WorkerNode:
        """Predictive node selection based on health score."""
        return max(nodes, key=lambda n: n.health_score)
    
    def update_node_metrics(self, node_id: str, response_time: float, success: bool, 
                          cpu_usage: float = None, memory_usage: float = None):
        """Update node metrics after request completion."""
        with self.lock:
            if node_id in self.nodes:
                node = self.nodes[node_id]
                
                # Update response times
                node.response_times.append(response_time)
                
                # Update request counts
                node.total_requests += 1
                if not success:
                    node.failed_requests += 1
                
                # Update resource usage if provided
                if cpu_usage is not None:
                    node.cpu_usage = cpu_usage
                if memory_usage is not None:
                    node.memory_usage = memory_usage
                
                # Update load
                node.current_load = max(0, node.current_load - 1)
    
    def increment_node_load(self, node_id: str):
        """Increment node load when request starts."""
        with self.lock:
            if node_id in self.nodes:
                self.nodes[node_id].current_load += 1
    
    def get_cluster_stats(self) -> Dict[str, Any]:
        """Get cluster-wide statistics."""
        with self.lock:
            if not self.nodes:
                return {
                    'total_nodes': 0,
                    'healthy_nodes': 0,
                    'total_capacity': 0,
                    'total_load': 0,
                    'average_response_time': 0.0,
                    'error_rate': 0.0
                }
            
            healthy_nodes = [n for n in self.nodes.values() if n.status == NodeStatus.HEALTHY]
            total_capacity = sum(node.capacity for node in self.nodes.values())
            total_load = sum(node.current_load for node in self.nodes.values())
            
            # Calculate averages
            response_times = []
            total_requests = 0
            failed_requests = 0
            
            for node in self.nodes.values():
                if node.response_times:
                    response_times.extend(list(node.response_times))
                total_requests += node.total_requests
                failed_requests += node.failed_requests
            
            avg_response_time = statistics.mean(response_times) if response_times else 0.0
            error_rate = (failed_requests / max(total_requests, 1)) * 100
            
            return {
                'total_nodes': len(self.nodes),
                'healthy_nodes': len(healthy_nodes),
                'total_capacity': total_capacity,
                'total_load': total_load,
                'load_percentage': (total_load / max(total_capacity, 1)) * 100,
                'average_response_time': avg_response_time,
                'error_rate': error_rate,
                'node_details': {
                    node_id: {
                        'status': node.status.value,
                        'load_percentage': node.load_percentage,
                        'health_score': node.health_score,
                        'avg_response_time': node.average_response_time
                    }
                    for node_id, node in self.nodes.items()
                }
            }


class AutoScaler:
    """Intelligent auto-scaling system."""
    
    def __init__(self, config: ScalingConfig, load_balancer: LoadBalancer):
        """Initialize auto-scaler."""
        self.config = config
        self.load_balancer = load_balancer
        self.predictive_scaler = PredictiveScaler()
        self.logger = logging.getLogger(__name__)
        
        # Scaling state
        self.last_scaling_action = datetime.now()
        self.scaling_history: List[Dict[str, Any]] = []
        self.metrics_history: deque = deque(maxlen=1000)
        
        # Node management
        self.node_counter = 0
        self.pending_nodes: Dict[str, datetime] = {}
        
        # Background threads
        self.monitoring_active = False
        self.monitor_thread: Optional[threading.Thread] = None
        
    def start_monitoring(self):
        """Start background monitoring and scaling."""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True,
            name="AutoScaler-Monitor"
        )
        self.monitor_thread.start()
        self.logger.info("Auto-scaler monitoring started")
    
    def stop_monitoring(self):
        """Stop background monitoring."""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        self.logger.info("Auto-scaler monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.monitoring_active:
            try:
                # Collect current metrics
                metrics = self._collect_metrics()
                self.metrics_history.append(metrics)
                self.predictive_scaler.add_metrics(metrics)
                
                # Make scaling decision
                scaling_decision = self._make_scaling_decision(metrics)
                
                if scaling_decision['action'] != ScalingDirection.STABLE:
                    self._execute_scaling_action(scaling_decision)
                
                # Cleanup unhealthy nodes
                self._cleanup_unhealthy_nodes()
                
                # Health checks
                self._perform_health_checks()
                
                time.sleep(self.config.health_check_interval)
                
            except Exception as e:
                self.logger.error(f"Error in auto-scaler monitoring loop: {e}")
                time.sleep(5)  # Brief pause before retrying
    
    def _collect_metrics(self) -> ScalingMetrics:
        """Collect current cluster metrics."""
        cluster_stats = self.load_balancer.get_cluster_stats()
        
        # Get system metrics
        cpu_usage = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        memory_usage = memory.percent
        
        # Calculate RPS (simplified)
        current_time = time.time()
        recent_requests = len([
            req for req in self.load_balancer.request_history
            if current_time - req.get('timestamp', 0) < 60
        ]) if hasattr(self.load_balancer, 'request_history') else 0
        
        rps = recent_requests / 60.0
        
        return ScalingMetrics(
            timestamp=datetime.now(),
            total_requests_per_second=rps,
            average_response_time=cluster_stats['average_response_time'],
            error_rate=cluster_stats['error_rate'],
            cpu_usage=cpu_usage,
            memory_usage=memory_usage,
            active_connections=cluster_stats['total_load'],
            queue_depth=0  # Would be populated from actual queue
        )
    
    def _make_scaling_decision(self, metrics: ScalingMetrics) -> Dict[str, Any]:
        """Make scaling decision based on metrics."""
        current_nodes = len(self.load_balancer.nodes)
        
        # Check cooldown period
        time_since_last_scale = (datetime.now() - self.last_scaling_action).total_seconds()
        if time_since_last_scale < self.config.cooldown_period:
            return {
                'action': ScalingDirection.STABLE,
                'reason': f'Cooldown period active ({time_since_last_scale:.0f}s remaining)',
                'target_nodes': current_nodes
            }
        
        # Base scaling logic
        scale_up_signals = 0
        scale_down_signals = 0
        reasons = []
        
        # CPU-based scaling
        if metrics.cpu_usage > self.config.scale_up_threshold:
            scale_up_signals += 1
            reasons.append(f"CPU usage high ({metrics.cpu_usage:.1f}%)")
        elif metrics.cpu_usage < self.config.scale_down_threshold:
            scale_down_signals += 1
            reasons.append(f"CPU usage low ({metrics.cpu_usage:.1f}%)")
        
        # Memory-based scaling
        if metrics.memory_usage > self.config.scale_up_threshold:
            scale_up_signals += 1
            reasons.append(f"Memory usage high ({metrics.memory_usage:.1f}%)")
        elif metrics.memory_usage < self.config.scale_down_threshold:
            scale_down_signals += 1
            reasons.append(f"Memory usage low ({metrics.memory_usage:.1f}%)")
        
        # Response time-based scaling
        if metrics.average_response_time > self.config.target_response_time:
            scale_up_signals += 1
            reasons.append(f"Response time high ({metrics.average_response_time:.2f}s)")
        
        # Error rate-based scaling
        if metrics.error_rate > 5.0:  # 5% error rate threshold
            scale_up_signals += 1
            reasons.append(f"Error rate high ({metrics.error_rate:.1f}%)")
        
        # Predictive scaling (if enabled)
        if self.config.enable_predictive_scaling:
            predicted_rps, confidence = self.predictive_scaler.predict_load(10)
            current_capacity = sum(node.capacity for node in self.load_balancer.nodes.values())
            
            if confidence > 0.7 and predicted_rps > current_capacity * 0.8:
                scale_up_signals += 1
                reasons.append(f"Predicted load increase ({predicted_rps:.1f} RPS)")
                metrics.prediction_confidence = confidence
        
        # Make decision
        if scale_up_signals >= 2 and current_nodes < self.config.max_nodes:
            target_nodes = min(
                current_nodes + (2 if self.config.aggressive_scaling else 1),
                self.config.max_nodes
            )
            return {
                'action': ScalingDirection.UP,
                'reason': '; '.join(reasons),
                'target_nodes': target_nodes,
                'current_nodes': current_nodes
            }
        elif scale_down_signals >= 2 and current_nodes > self.config.min_nodes:
            target_nodes = max(
                current_nodes - (2 if self.config.aggressive_scaling else 1),
                self.config.min_nodes
            )
            return {
                'action': ScalingDirection.DOWN,
                'reason': '; '.join(reasons),
                'target_nodes': target_nodes,
                'current_nodes': current_nodes
            }
        else:
            return {
                'action': ScalingDirection.STABLE,
                'reason': 'No scaling signals or at limits',
                'target_nodes': current_nodes
            }
    
    def _execute_scaling_action(self, decision: Dict[str, Any]):
        """Execute scaling action."""
        action = decision['action']
        target_nodes = decision['target_nodes']
        current_nodes = decision['current_nodes']
        
        try:
            if action == ScalingDirection.UP:
                # Scale up
                nodes_to_add = target_nodes - current_nodes
                for i in range(nodes_to_add):
                    self._create_node()
                
                self.logger.info(f"Scaling up: added {nodes_to_add} nodes. Reason: {decision['reason']}")
                
            elif action == ScalingDirection.DOWN:
                # Scale down
                nodes_to_remove = current_nodes - target_nodes
                self._remove_nodes(nodes_to_remove)
                
                self.logger.info(f"Scaling down: removed {nodes_to_remove} nodes. Reason: {decision['reason']}")
            
            # Record scaling action
            self.last_scaling_action = datetime.now()
            self.scaling_history.append({
                'timestamp': datetime.now().isoformat(),
                'action': action.value,
                'from_nodes': current_nodes,
                'to_nodes': target_nodes,
                'reason': decision['reason']
            })
            
            # Keep history bounded
            if len(self.scaling_history) > 100:
                self.scaling_history = self.scaling_history[-50:]
                
        except Exception as e:
            self.logger.error(f"Failed to execute scaling action: {e}")
    
    def _create_node(self):
        """Create a new worker node."""
        self.node_counter += 1
        node_id = f"worker-{self.node_counter}-{int(time.time())}"
        
        # In a real implementation, this would provision actual infrastructure
        # For now, create a mock node
        node = WorkerNode(
            node_id=node_id,
            host="localhost",
            port=8000 + self.node_counter,
            capacity=100,
            weight=1.0
        )
        
        self.load_balancer.add_node(node)
        self.pending_nodes[node_id] = datetime.now()
        
        # In production, trigger actual node creation (e.g., Kubernetes pod, EC2 instance)
        self.logger.debug(f"Created mock node {node_id}")
    
    def _remove_nodes(self, count: int):
        """Remove worker nodes gracefully."""
        # Select nodes to remove (prefer unhealthy or low-loaded nodes)
        nodes_by_priority = sorted(
            self.load_balancer.nodes.values(),
            key=lambda n: (n.status != NodeStatus.HEALTHY, n.health_score, -n.current_load)
        )
        
        nodes_to_remove = nodes_by_priority[:count]
        
        for node in nodes_to_remove:
            # Mark node as draining
            node.status = NodeStatus.DRAINING
            
            # In production, gracefully drain connections and terminate
            self.load_balancer.remove_node(node.node_id)
            self.logger.debug(f"Removed node {node.node_id}")
    
    def _cleanup_unhealthy_nodes(self):
        """Remove persistently unhealthy nodes."""
        unhealthy_threshold = timedelta(minutes=5)
        current_time = datetime.now()
        
        nodes_to_remove = []
        for node in self.load_balancer.nodes.values():
            if (node.status == NodeStatus.UNHEALTHY and 
                current_time - node.last_health_check > unhealthy_threshold):
                nodes_to_remove.append(node.node_id)
        
        for node_id in nodes_to_remove:
            self.load_balancer.remove_node(node_id)
            self.logger.info(f"Removed unhealthy node {node_id}")
    
    def _perform_health_checks(self):
        """Perform health checks on all nodes."""
        for node in self.load_balancer.nodes.values():
            # In production, this would be actual HTTP health checks
            # For now, simulate based on load and error rate
            
            health_score = node.health_score
            
            if health_score > 80:
                node.status = NodeStatus.HEALTHY
            elif health_score > 50:
                node.status = NodeStatus.DEGRADED
            else:
                node.status = NodeStatus.UNHEALTHY
            
            node.last_health_check = datetime.now()
    
    def get_scaling_report(self) -> Dict[str, Any]:
        """Get comprehensive scaling report."""
        cluster_stats = self.load_balancer.get_cluster_stats()
        current_metrics = self.metrics_history[-1] if self.metrics_history else None
        
        # Predictive analysis
        predicted_rps, confidence = self.predictive_scaler.predict_load(10)
        patterns = self.predictive_scaler.detect_pattern()
        
        return {
            'timestamp': datetime.now().isoformat(),
            'cluster_status': {
                'total_nodes': cluster_stats['total_nodes'],
                'healthy_nodes': cluster_stats['healthy_nodes'],
                'load_percentage': cluster_stats['load_percentage'],
                'average_response_time': cluster_stats['average_response_time'],
                'error_rate': cluster_stats['error_rate']
            },
            'scaling_config': {
                'min_nodes': self.config.min_nodes,
                'max_nodes': self.config.max_nodes,
                'target_cpu': self.config.target_cpu_utilization,
                'scale_up_threshold': self.config.scale_up_threshold,
                'scale_down_threshold': self.config.scale_down_threshold,
                'cooldown_period': self.config.cooldown_period
            },
            'current_metrics': current_metrics.to_dict() if current_metrics else {},
            'predictions': {
                'predicted_rps_10min': predicted_rps,
                'confidence': confidence,
                'detected_patterns': patterns
            },
            'scaling_history': self.scaling_history[-10:],  # Last 10 actions
            'time_since_last_scaling': (datetime.now() - self.last_scaling_action).total_seconds()
        }


# Example usage and testing
if __name__ == "__main__":
    print("Testing Advanced Auto-Scaling System...")
    
    # Create configuration
    config = ScalingConfig(
        min_nodes=2,
        max_nodes=10,
        target_cpu_utilization=70.0,
        scale_up_threshold=80.0,
        scale_down_threshold=30.0,
        cooldown_period=60,  # Shorter for testing
        enable_predictive_scaling=True
    )
    
    # Initialize load balancer
    load_balancer = LoadBalancer(LoadBalancingStrategy.LEAST_RESPONSE_TIME)
    
    # Add initial nodes
    for i in range(config.min_nodes):
        node = WorkerNode(
            node_id=f"initial-node-{i}",
            host="localhost",
            port=8000 + i,
            capacity=100,
            weight=1.0
        )
        load_balancer.add_node(node)
    
    # Initialize auto-scaler
    auto_scaler = AutoScaler(config, load_balancer)
    auto_scaler.start_monitoring()
    
    print(f"Started with {len(load_balancer.nodes)} nodes")
    
    # Simulate load testing
    print("\n1. Simulating normal load...")
    for i in range(10):
        # Get node for request
        node = load_balancer.get_node(f"request-{i}")
        if node:
            load_balancer.increment_node_load(node.node_id)
            
            # Simulate request processing
            time.sleep(0.1)
            response_time = 0.5 + np.random.normal(0, 0.1)
            success = np.random.random() > 0.05  # 5% error rate
            
            # Update metrics
            load_balancer.update_node_metrics(
                node.node_id,
                response_time,
                success,
                cpu_usage=60.0 + np.random.normal(0, 10),
                memory_usage=50.0 + np.random.normal(0, 10)
            )
    
    # Wait for monitoring cycle
    time.sleep(5)
    
    print("\n2. Simulating high load (should trigger scale-up)...")
    # Simulate high CPU usage that should trigger scaling
    for node in load_balancer.nodes.values():
        node.cpu_usage = 85.0  # High CPU
        node.memory_usage = 80.0  # High memory
        node.current_load = int(node.capacity * 0.9)  # High load
    
    # Wait for auto-scaler to react
    time.sleep(config.health_check_interval + 5)
    
    print(f"After high load: {len(load_balancer.nodes)} nodes")
    
    print("\n3. Simulating low load (should trigger scale-down after cooldown)...")
    # Simulate low resource usage
    for node in load_balancer.nodes.values():
        node.cpu_usage = 20.0  # Low CPU
        node.memory_usage = 25.0  # Low memory
        node.current_load = 1  # Low load
    
    # Wait for cooldown and scale-down
    time.sleep(config.cooldown_period + config.health_check_interval + 5)
    
    print(f"After low load: {len(load_balancer.nodes)} nodes")
    
    # Get comprehensive report
    report = auto_scaler.get_scaling_report()
    
    print("\n=== Auto-Scaling Report ===")
    print(f"Cluster Status:")
    print(f"  Total nodes: {report['cluster_status']['total_nodes']}")
    print(f"  Healthy nodes: {report['cluster_status']['healthy_nodes']}")
    print(f"  Load percentage: {report['cluster_status']['load_percentage']:.1f}%")
    print(f"  Average response time: {report['cluster_status']['average_response_time']:.3f}s")
    
    print(f"\nPredictions:")
    print(f"  Predicted RPS (10min): {report['predictions']['predicted_rps_10min']:.2f}")
    print(f"  Confidence: {report['predictions']['confidence']:.2%}")
    print(f"  Patterns: {report['predictions']['detected_patterns']}")
    
    print(f"\nRecent Scaling Actions:")
    for action in report['scaling_history']:
        print(f"  {action['timestamp']}: {action['action']} ({action['from_nodes']} → {action['to_nodes']}) - {action['reason']}")
    
    # Cleanup
    auto_scaler.stop_monitoring()
    
    print("\n✓ Advanced auto-scaling test completed!")