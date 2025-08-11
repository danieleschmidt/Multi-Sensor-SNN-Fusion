"""
Distributed Deployment System for Large-Scale SNN Fusion

Implements distributed deployment, load balancing, auto-scaling, and
multi-region support for production neuromorphic systems at scale.
"""

import os
import json
import time
import threading
import asyncio
from typing import Dict, List, Optional, Any, Callable, Tuple, Union
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
import logging
import hashlib
from collections import defaultdict, deque
import uuid
import socket


class DeploymentStrategy(Enum):
    """Deployment strategies for distributed systems."""
    SINGLE_NODE = "single_node"
    MULTI_NODE = "multi_node"
    KUBERNETES = "kubernetes"
    DOCKER_SWARM = "docker_swarm"
    CLOUD_NATIVE = "cloud_native"


class LoadBalancingStrategy(Enum):
    """Load balancing strategies."""
    ROUND_ROBIN = "round_robin"
    LEAST_CONNECTIONS = "least_connections"
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"
    PERFORMANCE_BASED = "performance_based"
    HASH_BASED = "hash_based"


class ScalingPolicy(Enum):
    """Auto-scaling policies."""
    REACTIVE = "reactive"
    PREDICTIVE = "predictive"
    SCHEDULED = "scheduled"
    HYBRID = "hybrid"


@dataclass
class NodeInfo:
    """Information about a deployment node."""
    node_id: str
    hostname: str
    ip_address: str
    port: int
    status: str
    capabilities: Dict[str, Any]
    resource_usage: Dict[str, float]
    last_heartbeat: float
    deployment_region: str = "default"
    node_weight: float = 1.0
    

@dataclass
class DeploymentConfig:
    """Configuration for distributed deployment."""
    deployment_id: str
    strategy: DeploymentStrategy
    load_balancing: LoadBalancingStrategy
    scaling_policy: ScalingPolicy
    min_nodes: int = 1
    max_nodes: int = 10
    target_cpu_utilization: float = 70.0
    target_memory_utilization: float = 80.0
    health_check_interval: float = 30.0
    scaling_cooldown: float = 300.0
    regions: List[str] = None
    
    def __post_init__(self):
        if self.regions is None:
            self.regions = ["default"]


@dataclass
class ServiceMetrics:
    """Service performance metrics."""
    timestamp: float
    request_count: int
    response_time_ms: float
    error_rate: float
    throughput_rps: float
    cpu_usage: float
    memory_usage: float
    active_connections: int


class DistributedDeploymentManager:
    """
    Comprehensive distributed deployment manager for SNN Fusion systems.
    
    Provides:
    - Multi-node deployment coordination
    - Load balancing and traffic distribution
    - Auto-scaling based on demand
    - Health monitoring and failure recovery
    - Multi-region deployment support
    """
    
    def __init__(
        self,
        deployment_config: DeploymentConfig,
        node_discovery_service: Optional[str] = None
    ):
        """
        Initialize distributed deployment manager.
        
        Args:
            deployment_config: Deployment configuration
            node_discovery_service: Node discovery service endpoint
        """
        self.config = deployment_config
        self.node_discovery_service = node_discovery_service
        
        self.logger = logging.getLogger(__name__)
        
        # Node management
        self.active_nodes: Dict[str, NodeInfo] = {}
        self.node_metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self.failed_nodes: Dict[str, float] = {}  # node_id -> failure_timestamp
        
        # Load balancing
        self.load_balancer = LoadBalancer(deployment_config.load_balancing)
        self.request_router = RequestRouter(self)
        
        # Auto-scaling
        self.auto_scaler = AutoScaler(self, deployment_config)
        
        # Service discovery
        self.service_registry = ServiceRegistry()
        
        # Monitoring
        self.metrics_collector = MetricsCollector()
        self.health_monitor = HealthMonitor(self)
        
        # Thread safety
        self._lock = threading.RLock()
        self._shutdown_event = threading.Event()
        
        # Background tasks
        self._background_tasks: List[threading.Thread] = []
        
        self.logger.info(f"Distributed deployment manager initialized for {deployment_config.deployment_id}")
    
    def start_deployment(self) -> bool:
        """Start the distributed deployment."""
        try:
            # Initialize service registry
            self.service_registry.initialize()
            
            # Start health monitoring
            self.health_monitor.start()
            
            # Start auto-scaling
            self.auto_scaler.start()
            
            # Start metrics collection
            self.metrics_collector.start()
            
            # Start background tasks
            self._start_background_tasks()
            
            self.logger.info("Distributed deployment started successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start deployment: {e}")
            return False
    
    def stop_deployment(self) -> None:
        """Stop the distributed deployment gracefully."""
        self.logger.info("Stopping distributed deployment...")
        
        # Signal shutdown
        self._shutdown_event.set()
        
        # Stop components
        self.auto_scaler.stop()
        self.health_monitor.stop()
        self.metrics_collector.stop()
        
        # Wait for background tasks
        for task in self._background_tasks:
            task.join(timeout=10.0)
        
        self.logger.info("Distributed deployment stopped")
    
    def register_node(self, node_info: NodeInfo) -> bool:
        """Register a new node in the deployment."""
        try:
            with self._lock:
                # Validate node info
                if not self._validate_node_info(node_info):
                    return False
                
                # Add to active nodes
                self.active_nodes[node_info.node_id] = node_info
                
                # Initialize metrics
                self.node_metrics[node_info.node_id] = deque(maxlen=100)
                
                # Remove from failed nodes if present
                if node_info.node_id in self.failed_nodes:
                    del self.failed_nodes[node_info.node_id]
                
                # Update load balancer
                self.load_balancer.add_node(node_info)
                
                # Register in service discovery
                self.service_registry.register_node(node_info)
            
            self.logger.info(f"Node {node_info.node_id} registered successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to register node {node_info.node_id}: {e}")
            return False
    
    def unregister_node(self, node_id: str) -> bool:
        """Unregister a node from the deployment."""
        try:
            with self._lock:
                if node_id not in self.active_nodes:
                    return False
                
                # Remove from active nodes
                node_info = self.active_nodes[node_id]
                del self.active_nodes[node_id]
                
                # Update load balancer
                self.load_balancer.remove_node(node_id)
                
                # Unregister from service discovery
                self.service_registry.unregister_node(node_id)
                
                # Clean up metrics
                if node_id in self.node_metrics:
                    del self.node_metrics[node_id]
            
            self.logger.info(f"Node {node_id} unregistered successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to unregister node {node_id}: {e}")
            return False
    
    def route_request(self, request_data: Dict[str, Any]) -> Optional[NodeInfo]:
        """Route a request to an appropriate node."""
        return self.request_router.route_request(request_data)
    
    def update_node_metrics(self, node_id: str, metrics: Dict[str, float]) -> None:
        """Update metrics for a specific node."""
        with self._lock:
            if node_id in self.active_nodes:
                # Update resource usage
                self.active_nodes[node_id].resource_usage.update(metrics)
                self.active_nodes[node_id].last_heartbeat = time.time()
                
                # Store metrics history
                self.node_metrics[node_id].append((time.time(), metrics.copy()))
    
    def get_deployment_status(self) -> Dict[str, Any]:
        """Get comprehensive deployment status."""
        with self._lock:
            status = {
                'deployment_id': self.config.deployment_id,
                'strategy': self.config.strategy.value,
                'active_nodes': len(self.active_nodes),
                'failed_nodes': len(self.failed_nodes),
                'node_details': {},
                'load_balancer_status': self.load_balancer.get_status(),
                'auto_scaler_status': self.auto_scaler.get_status(),
                'service_registry_status': self.service_registry.get_status()
            }
            
            # Add node details
            for node_id, node_info in self.active_nodes.items():
                status['node_details'][node_id] = {
                    'hostname': node_info.hostname,
                    'status': node_info.status,
                    'region': node_info.deployment_region,
                    'resource_usage': node_info.resource_usage,
                    'last_heartbeat': node_info.last_heartbeat
                }
        
        return status
    
    def _validate_node_info(self, node_info: NodeInfo) -> bool:
        """Validate node information."""
        required_fields = ['node_id', 'hostname', 'ip_address', 'port']
        
        for field in required_fields:
            if not hasattr(node_info, field) or getattr(node_info, field) is None:
                self.logger.error(f"Node info missing required field: {field}")
                return False
        
        # Validate IP address format
        try:
            socket.inet_aton(node_info.ip_address)
        except socket.error:
            self.logger.error(f"Invalid IP address: {node_info.ip_address}")
            return False
        
        return True
    
    def _start_background_tasks(self) -> None:
        """Start background maintenance tasks."""
        # Node heartbeat checker
        heartbeat_task = threading.Thread(
            target=self._heartbeat_checker,
            daemon=True,
            name="HeartbeatChecker"
        )
        heartbeat_task.start()
        self._background_tasks.append(heartbeat_task)
        
        # Metrics aggregator
        metrics_task = threading.Thread(
            target=self._metrics_aggregator,
            daemon=True,
            name="MetricsAggregator"
        )
        metrics_task.start()
        self._background_tasks.append(metrics_task)
    
    def _heartbeat_checker(self) -> None:
        """Check node heartbeats and handle failures."""
        while not self._shutdown_event.is_set():
            try:
                current_time = time.time()
                failed_nodes = []
                
                with self._lock:
                    for node_id, node_info in self.active_nodes.items():
                        # Check if node missed heartbeat
                        time_since_heartbeat = current_time - node_info.last_heartbeat
                        
                        if time_since_heartbeat > self.config.health_check_interval * 2:
                            failed_nodes.append(node_id)
                
                # Handle failed nodes
                for node_id in failed_nodes:
                    self._handle_node_failure(node_id)
                
                time.sleep(self.config.health_check_interval)
                
            except Exception as e:
                self.logger.error(f"Error in heartbeat checker: {e}")
                time.sleep(10)
    
    def _metrics_aggregator(self) -> None:
        """Aggregate metrics from all nodes."""
        while not self._shutdown_event.is_set():
            try:
                self._aggregate_cluster_metrics()
                time.sleep(60)  # Aggregate every minute
            except Exception as e:
                self.logger.error(f"Error in metrics aggregator: {e}")
                time.sleep(10)
    
    def _handle_node_failure(self, node_id: str) -> None:
        """Handle node failure."""
        self.logger.warning(f"Node {node_id} failed - removing from active nodes")
        
        with self._lock:
            if node_id in self.active_nodes:
                # Mark node as failed
                self.failed_nodes[node_id] = time.time()
                
                # Remove from active nodes
                self.unregister_node(node_id)
        
        # Trigger auto-scaling if needed
        self.auto_scaler.handle_node_failure(node_id)
    
    def _aggregate_cluster_metrics(self) -> None:
        """Aggregate metrics across the cluster."""
        with self._lock:
            cluster_metrics = {
                'timestamp': time.time(),
                'active_nodes': len(self.active_nodes),
                'total_cpu_usage': 0.0,
                'total_memory_usage': 0.0,
                'total_requests': 0,
                'average_response_time': 0.0
            }
            
            if self.active_nodes:
                cpu_sum = sum(node.resource_usage.get('cpu_usage', 0) 
                            for node in self.active_nodes.values())
                memory_sum = sum(node.resource_usage.get('memory_usage', 0) 
                               for node in self.active_nodes.values())
                
                cluster_metrics['total_cpu_usage'] = cpu_sum / len(self.active_nodes)
                cluster_metrics['total_memory_usage'] = memory_sum / len(self.active_nodes)
        
        # Store cluster metrics
        self.metrics_collector.record_cluster_metrics(cluster_metrics)


class LoadBalancer:
    """Load balancer for distributing requests across nodes."""
    
    def __init__(self, strategy: LoadBalancingStrategy):
        """Initialize load balancer with specified strategy."""
        self.strategy = strategy
        self.nodes: List[NodeInfo] = []
        self.current_index = 0
        self.node_connections: Dict[str, int] = defaultdict(int)
        self.node_weights: Dict[str, float] = defaultdict(lambda: 1.0)
        self._lock = threading.RLock()
        
        self.logger = logging.getLogger(__name__)
    
    def add_node(self, node_info: NodeInfo) -> None:
        """Add a node to the load balancer."""
        with self._lock:
            self.nodes.append(node_info)
            self.node_weights[node_info.node_id] = node_info.node_weight
        
        self.logger.info(f"Added node {node_info.node_id} to load balancer")
    
    def remove_node(self, node_id: str) -> None:
        """Remove a node from the load balancer."""
        with self._lock:
            self.nodes = [node for node in self.nodes if node.node_id != node_id]
            
            if node_id in self.node_connections:
                del self.node_connections[node_id]
            if node_id in self.node_weights:
                del self.node_weights[node_id]
        
        self.logger.info(f"Removed node {node_id} from load balancer")
    
    def select_node(self, request_data: Dict[str, Any] = None) -> Optional[NodeInfo]:
        """Select a node based on the load balancing strategy."""
        with self._lock:
            if not self.nodes:
                return None
            
            if self.strategy == LoadBalancingStrategy.ROUND_ROBIN:
                return self._round_robin_selection()
            elif self.strategy == LoadBalancingStrategy.LEAST_CONNECTIONS:
                return self._least_connections_selection()
            elif self.strategy == LoadBalancingStrategy.WEIGHTED_ROUND_ROBIN:
                return self._weighted_round_robin_selection()
            elif self.strategy == LoadBalancingStrategy.PERFORMANCE_BASED:
                return self._performance_based_selection()
            elif self.strategy == LoadBalancingStrategy.HASH_BASED:
                return self._hash_based_selection(request_data)
            else:
                return self._round_robin_selection()
    
    def _round_robin_selection(self) -> NodeInfo:
        """Round-robin node selection."""
        node = self.nodes[self.current_index]
        self.current_index = (self.current_index + 1) % len(self.nodes)
        return node
    
    def _least_connections_selection(self) -> NodeInfo:
        """Select node with least connections."""
        return min(self.nodes, key=lambda n: self.node_connections[n.node_id])
    
    def _weighted_round_robin_selection(self) -> NodeInfo:
        """Weighted round-robin selection."""
        # Simplified weighted selection
        total_weight = sum(self.node_weights[node.node_id] for node in self.nodes)
        
        if total_weight <= 0:
            return self._round_robin_selection()
        
        # Select based on weights (simplified implementation)
        weights = [self.node_weights[node.node_id] / total_weight for node in self.nodes]
        
        import random
        return random.choices(self.nodes, weights=weights)[0]
    
    def _performance_based_selection(self) -> NodeInfo:
        """Select node based on performance metrics."""
        # Select node with lowest CPU usage
        return min(self.nodes, 
                  key=lambda n: n.resource_usage.get('cpu_usage', 100))
    
    def _hash_based_selection(self, request_data: Dict[str, Any]) -> NodeInfo:
        """Hash-based consistent node selection."""
        if not request_data:
            return self._round_robin_selection()
        
        # Create hash from request data
        request_hash = hashlib.md5(
            json.dumps(request_data, sort_keys=True).encode()
        ).hexdigest()
        
        # Select node based on hash
        node_index = int(request_hash, 16) % len(self.nodes)
        return self.nodes[node_index]
    
    def update_node_connections(self, node_id: str, delta: int) -> None:
        """Update connection count for a node."""
        with self._lock:
            self.node_connections[node_id] += delta
            if self.node_connections[node_id] < 0:
                self.node_connections[node_id] = 0
    
    def get_status(self) -> Dict[str, Any]:
        """Get load balancer status."""
        with self._lock:
            return {
                'strategy': self.strategy.value,
                'total_nodes': len(self.nodes),
                'node_connections': dict(self.node_connections),
                'node_weights': dict(self.node_weights)
            }


class AutoScaler:
    """Auto-scaling manager for dynamic resource allocation."""
    
    def __init__(self, deployment_manager: DistributedDeploymentManager, config: DeploymentConfig):
        """Initialize auto-scaler."""
        self.deployment_manager = deployment_manager
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Scaling state
        self.last_scaling_action = 0.0
        self.scaling_in_progress = False
        self._stop_event = threading.Event()
        self._scaling_thread: Optional[threading.Thread] = None
        
        # Metrics for scaling decisions
        self.metrics_window = deque(maxlen=20)  # 20 samples for trend analysis
    
    def start(self) -> None:
        """Start auto-scaling monitoring."""
        if self._scaling_thread and self._scaling_thread.is_alive():
            return
        
        self._stop_event.clear()
        self._scaling_thread = threading.Thread(
            target=self._scaling_loop,
            daemon=True,
            name="AutoScaler"
        )
        self._scaling_thread.start()
        
        self.logger.info("Auto-scaler started")
    
    def stop(self) -> None:
        """Stop auto-scaling."""
        self._stop_event.set()
        if self._scaling_thread:
            self._scaling_thread.join(timeout=10.0)
        
        self.logger.info("Auto-scaler stopped")
    
    def _scaling_loop(self) -> None:
        """Main scaling loop."""
        while not self._stop_event.is_set():
            try:
                self._evaluate_scaling_need()
                time.sleep(30)  # Check every 30 seconds
            except Exception as e:
                self.logger.error(f"Error in scaling loop: {e}")
                time.sleep(10)
    
    def _evaluate_scaling_need(self) -> None:
        """Evaluate if scaling action is needed."""
        current_time = time.time()
        
        # Check cooldown period
        if current_time - self.last_scaling_action < self.config.scaling_cooldown:
            return
        
        if self.scaling_in_progress:
            return
        
        # Get current cluster metrics
        cluster_metrics = self._get_cluster_metrics()
        if not cluster_metrics:
            return
        
        # Store metrics for trend analysis
        self.metrics_window.append(cluster_metrics)
        
        # Make scaling decision
        scaling_decision = self._make_scaling_decision(cluster_metrics)
        
        if scaling_decision != 'none':
            self._execute_scaling_action(scaling_decision)
    
    def _get_cluster_metrics(self) -> Optional[Dict[str, float]]:
        """Get current cluster metrics."""
        with self.deployment_manager._lock:
            nodes = self.deployment_manager.active_nodes
            
            if not nodes:
                return None
            
            total_cpu = sum(node.resource_usage.get('cpu_usage', 0) for node in nodes.values())
            total_memory = sum(node.resource_usage.get('memory_usage', 0) for node in nodes.values())
            
            return {
                'node_count': len(nodes),
                'avg_cpu_usage': total_cpu / len(nodes),
                'avg_memory_usage': total_memory / len(nodes),
                'timestamp': time.time()
            }
    
    def _make_scaling_decision(self, metrics: Dict[str, float]) -> str:
        """Make scaling decision based on metrics."""
        node_count = metrics['node_count']
        cpu_usage = metrics['avg_cpu_usage']
        memory_usage = metrics['avg_memory_usage']
        
        # Scale up conditions
        if (cpu_usage > self.config.target_cpu_utilization or 
            memory_usage > self.config.target_memory_utilization):
            
            if node_count < self.config.max_nodes:
                # Check if this is a sustained trend
                if self._is_sustained_high_usage():
                    return 'scale_up'
        
        # Scale down conditions
        elif (cpu_usage < self.config.target_cpu_utilization * 0.5 and 
              memory_usage < self.config.target_memory_utilization * 0.5):
            
            if node_count > self.config.min_nodes:
                # Check if this is a sustained trend
                if self._is_sustained_low_usage():
                    return 'scale_down'
        
        return 'none'
    
    def _is_sustained_high_usage(self) -> bool:
        """Check if high usage is sustained over multiple samples."""
        if len(self.metrics_window) < 5:
            return False
        
        recent_samples = list(self.metrics_window)[-5:]
        high_usage_count = 0
        
        for sample in recent_samples:
            if (sample['avg_cpu_usage'] > self.config.target_cpu_utilization or
                sample['avg_memory_usage'] > self.config.target_memory_utilization):
                high_usage_count += 1
        
        return high_usage_count >= 4  # 4 out of 5 samples
    
    def _is_sustained_low_usage(self) -> bool:
        """Check if low usage is sustained over multiple samples."""
        if len(self.metrics_window) < 10:
            return False
        
        recent_samples = list(self.metrics_window)[-10:]
        low_usage_count = 0
        
        for sample in recent_samples:
            if (sample['avg_cpu_usage'] < self.config.target_cpu_utilization * 0.5 and
                sample['avg_memory_usage'] < self.config.target_memory_utilization * 0.5):
                low_usage_count += 1
        
        return low_usage_count >= 8  # 8 out of 10 samples
    
    def _execute_scaling_action(self, action: str) -> None:
        """Execute scaling action."""
        self.scaling_in_progress = True
        self.last_scaling_action = time.time()
        
        try:
            if action == 'scale_up':
                success = self._scale_up()
                action_desc = "Scale up"
            elif action == 'scale_down':
                success = self._scale_down()
                action_desc = "Scale down"
            else:
                success = False
                action_desc = "Unknown action"
            
            if success:
                self.logger.info(f"{action_desc} completed successfully")
            else:
                self.logger.warning(f"{action_desc} failed")
                
        except Exception as e:
            self.logger.error(f"Scaling action failed: {e}")
        finally:
            self.scaling_in_progress = False
    
    def _scale_up(self) -> bool:
        """Scale up by adding nodes."""
        # In a real implementation, this would:
        # 1. Request new instances from cloud provider
        # 2. Deploy application to new instances
        # 3. Register new nodes with deployment manager
        
        self.logger.info("Scaling up - requesting new nodes")
        # Placeholder implementation
        return True
    
    def _scale_down(self) -> bool:
        """Scale down by removing nodes."""
        # In a real implementation, this would:
        # 1. Select nodes to remove (least loaded)
        # 2. Drain traffic from selected nodes
        # 3. Gracefully shutdown nodes
        # 4. Terminate instances
        
        self.logger.info("Scaling down - removing excess nodes")
        # Placeholder implementation
        return True
    
    def handle_node_failure(self, node_id: str) -> None:
        """Handle node failure by potentially scaling up."""
        current_time = time.time()
        
        # Check if we need to scale up due to node failure
        with self.deployment_manager._lock:
            active_nodes = len(self.deployment_manager.active_nodes)
            
            if active_nodes < self.config.min_nodes:
                # Force scale up due to minimum nodes requirement
                self.logger.info(f"Node failure caused cluster to fall below minimum nodes, scaling up")
                self.last_scaling_action = current_time
                self._execute_scaling_action('scale_up')
    
    def get_status(self) -> Dict[str, Any]:
        """Get auto-scaler status."""
        return {
            'scaling_policy': self.config.scaling_policy.value,
            'min_nodes': self.config.min_nodes,
            'max_nodes': self.config.max_nodes,
            'target_cpu_utilization': self.config.target_cpu_utilization,
            'target_memory_utilization': self.config.target_memory_utilization,
            'last_scaling_action': self.last_scaling_action,
            'scaling_in_progress': self.scaling_in_progress,
            'metrics_samples': len(self.metrics_window)
        }


class ServiceRegistry:
    """Service registry for node discovery and management."""
    
    def __init__(self):
        """Initialize service registry."""
        self.services: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.RLock()
        self.logger = logging.getLogger(__name__)
    
    def initialize(self) -> None:
        """Initialize service registry."""
        self.logger.info("Service registry initialized")
    
    def register_node(self, node_info: NodeInfo) -> None:
        """Register a node in the service registry."""
        with self._lock:
            self.services[node_info.node_id] = {
                'hostname': node_info.hostname,
                'ip_address': node_info.ip_address,
                'port': node_info.port,
                'status': node_info.status,
                'region': node_info.deployment_region,
                'registered_at': time.time()
            }
        
        self.logger.info(f"Registered service for node {node_info.node_id}")
    
    def unregister_node(self, node_id: str) -> None:
        """Unregister a node from the service registry."""
        with self._lock:
            if node_id in self.services:
                del self.services[node_id]
        
        self.logger.info(f"Unregistered service for node {node_id}")
    
    def discover_nodes(self, region: str = None) -> List[Dict[str, Any]]:
        """Discover available nodes, optionally filtered by region."""
        with self._lock:
            if region:
                return [service for service in self.services.values() 
                       if service.get('region') == region]
            else:
                return list(self.services.values())
    
    def get_status(self) -> Dict[str, Any]:
        """Get service registry status."""
        with self._lock:
            return {
                'total_services': len(self.services),
                'services': list(self.services.keys())
            }


class RequestRouter:
    """Request router for directing traffic to appropriate nodes."""
    
    def __init__(self, deployment_manager: DistributedDeploymentManager):
        """Initialize request router."""
        self.deployment_manager = deployment_manager
        self.logger = logging.getLogger(__name__)
    
    def route_request(self, request_data: Dict[str, Any]) -> Optional[NodeInfo]:
        """Route request to appropriate node."""
        # Select node using load balancer
        node = self.deployment_manager.load_balancer.select_node(request_data)
        
        if node:
            # Update connection count
            self.deployment_manager.load_balancer.update_node_connections(
                node.node_id, 1
            )
            
            self.logger.debug(f"Routed request to node {node.node_id}")
        else:
            self.logger.warning("No available nodes for request routing")
        
        return node


class MetricsCollector:
    """Metrics collector for gathering system metrics."""
    
    def __init__(self):
        """Initialize metrics collector."""
        self.metrics_history: deque = deque(maxlen=1000)
        self._stop_event = threading.Event()
        self._metrics_thread: Optional[threading.Thread] = None
        self.logger = logging.getLogger(__name__)
    
    def start(self) -> None:
        """Start metrics collection."""
        if self._metrics_thread and self._metrics_thread.is_alive():
            return
        
        self._stop_event.clear()
        self._metrics_thread = threading.Thread(
            target=self._collection_loop,
            daemon=True,
            name="MetricsCollector"
        )
        self._metrics_thread.start()
        
        self.logger.info("Metrics collector started")
    
    def stop(self) -> None:
        """Stop metrics collection."""
        self._stop_event.set()
        if self._metrics_thread:
            self._metrics_thread.join(timeout=5.0)
        
        self.logger.info("Metrics collector stopped")
    
    def _collection_loop(self) -> None:
        """Main metrics collection loop."""
        while not self._stop_event.is_set():
            try:
                # Collect system metrics (placeholder)
                metrics = self._collect_system_metrics()
                self.metrics_history.append(metrics)
                
                time.sleep(10)  # Collect every 10 seconds
            except Exception as e:
                self.logger.error(f"Error in metrics collection: {e}")
                time.sleep(10)
    
    def _collect_system_metrics(self) -> Dict[str, Any]:
        """Collect system metrics."""
        return {
            'timestamp': time.time(),
            'cpu_usage': 0.0,  # Would be implemented with actual monitoring
            'memory_usage': 0.0,
            'disk_usage': 0.0,
            'network_io': 0.0
        }
    
    def record_cluster_metrics(self, metrics: Dict[str, Any]) -> None:
        """Record cluster-level metrics."""
        self.metrics_history.append({
            'type': 'cluster',
            'timestamp': time.time(),
            'metrics': metrics
        })


class HealthMonitor:
    """Health monitor for checking node health."""
    
    def __init__(self, deployment_manager: DistributedDeploymentManager):
        """Initialize health monitor."""
        self.deployment_manager = deployment_manager
        self._stop_event = threading.Event()
        self._monitor_thread: Optional[threading.Thread] = None
        self.logger = logging.getLogger(__name__)
    
    def start(self) -> None:
        """Start health monitoring."""
        if self._monitor_thread and self._monitor_thread.is_alive():
            return
        
        self._stop_event.clear()
        self._monitor_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True,
            name="HealthMonitor"
        )
        self._monitor_thread.start()
        
        self.logger.info("Health monitor started")
    
    def stop(self) -> None:
        """Stop health monitoring."""
        self._stop_event.set()
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5.0)
        
        self.logger.info("Health monitor stopped")
    
    def _monitoring_loop(self) -> None:
        """Main health monitoring loop."""
        while not self._stop_event.is_set():
            try:
                self._check_node_health()
                time.sleep(self.deployment_manager.config.health_check_interval)
            except Exception as e:
                self.logger.error(f"Error in health monitoring: {e}")
                time.sleep(10)
    
    def _check_node_health(self) -> None:
        """Check health of all nodes."""
        # This would perform actual health checks
        # For now, it's handled by the heartbeat checker
        pass


# Utility functions

def create_deployment_config(
    deployment_id: str,
    strategy: DeploymentStrategy = DeploymentStrategy.MULTI_NODE,
    min_nodes: int = 1,
    max_nodes: int = 5
) -> DeploymentConfig:
    """Create a basic deployment configuration."""
    return DeploymentConfig(
        deployment_id=deployment_id,
        strategy=strategy,
        load_balancing=LoadBalancingStrategy.PERFORMANCE_BASED,
        scaling_policy=ScalingPolicy.REACTIVE,
        min_nodes=min_nodes,
        max_nodes=max_nodes
    )


def create_node_info(
    hostname: str,
    ip_address: str,
    port: int,
    capabilities: Dict[str, Any] = None,
    region: str = "default"
) -> NodeInfo:
    """Create node information structure."""
    return NodeInfo(
        node_id=str(uuid.uuid4()),
        hostname=hostname,
        ip_address=ip_address,
        port=port,
        status="active",
        capabilities=capabilities or {},
        resource_usage={},
        last_heartbeat=time.time(),
        deployment_region=region
    )