"""
Load Balancing and Scaling for SNN-Fusion

This module provides comprehensive load balancing and scaling capabilities
for distributed spiking neural network processing, including dynamic
scaling, resource management, and workload distribution.
"""

import time
import threading
import multiprocessing as mp
import queue
import hashlib
import random
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple, Callable, Union
import logging
from dataclasses import dataclass, field
from enum import Enum
import psutil
import numpy as np
from pathlib import Path
import pickle
import json


class LoadBalancingStrategy(Enum):
    """Load balancing strategies."""
    ROUND_ROBIN = "round_robin"
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin" 
    LEAST_CONNECTIONS = "least_connections"
    LEAST_RESPONSE_TIME = "least_response_time"
    RESOURCE_AWARE = "resource_aware"
    CONSISTENT_HASH = "consistent_hash"


class NodeStatus(Enum):
    """Node status states."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    OVERLOADED = "overloaded"
    FAILED = "failed"
    MAINTENANCE = "maintenance"


@dataclass
class WorkerNode:
    """Represents a worker node in the cluster."""
    node_id: str
    host: str
    port: int
    weight: float = 1.0
    max_concurrent_tasks: int = 10
    
    # Runtime state
    status: NodeStatus = NodeStatus.HEALTHY
    current_load: int = 0
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    avg_response_time: float = 0.0
    last_health_check: float = field(default_factory=time.time)
    
    # Resource metrics
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    network_latency: float = 0.0
    
    def __post_init__(self):
        """Initialize node after creation."""
        self.connection_pool = queue.Queue(maxsize=self.max_concurrent_tasks)
        self.metrics_lock = threading.RLock()
        self.task_queue = queue.Queue()
        
    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.total_requests == 0:
            return 1.0
        return self.successful_requests / self.total_requests
    
    @property
    def is_available(self) -> bool:
        """Check if node is available for new tasks."""
        return (self.status in [NodeStatus.HEALTHY, NodeStatus.DEGRADED] and
                self.current_load < self.max_concurrent_tasks)
    
    @property
    def load_factor(self) -> float:
        """Calculate normalized load factor."""
        if self.max_concurrent_tasks == 0:
            return 1.0
        return self.current_load / self.max_concurrent_tasks
    
    def update_metrics(self, response_time: float, success: bool):
        """Update node metrics after task completion."""
        with self.metrics_lock:
            self.total_requests += 1
            if success:
                self.successful_requests += 1
            else:
                self.failed_requests += 1
            
            # Update average response time using exponential moving average
            alpha = 0.1  # Smoothing factor
            self.avg_response_time = (alpha * response_time + 
                                    (1 - alpha) * self.avg_response_time)
    
    def update_resource_metrics(self, cpu: float, memory: float, latency: float):
        """Update resource utilization metrics."""
        with self.metrics_lock:
            self.cpu_usage = cpu
            self.memory_usage = memory
            self.network_latency = latency
            self.last_health_check = time.time()
            
            # Update status based on resource usage
            if cpu > 90 or memory > 90:
                self.status = NodeStatus.OVERLOADED
            elif cpu > 70 or memory > 70:
                self.status = NodeStatus.DEGRADED
            else:
                self.status = NodeStatus.HEALTHY
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert node to dictionary."""
        return {
            'node_id': self.node_id,
            'host': self.host,
            'port': self.port,
            'status': self.status.value,
            'current_load': self.current_load,
            'success_rate': self.success_rate,
            'avg_response_time': self.avg_response_time,
            'cpu_usage': self.cpu_usage,
            'memory_usage': self.memory_usage,
            'network_latency': self.network_latency
        }


@dataclass
class Task:
    """Represents a task to be distributed."""
    task_id: str
    payload: Any
    priority: int = 0
    timeout: float = 30.0
    retry_count: int = 0
    max_retries: int = 3
    created_at: float = field(default_factory=time.time)
    
    # Task requirements
    required_memory_mb: Optional[float] = None
    required_cpu_cores: Optional[int] = None
    gpu_required: bool = False
    
    def __post_init__(self):
        """Initialize task after creation."""
        if not hasattr(self, 'task_id') or not self.task_id:
            # Generate unique task ID
            self.task_id = hashlib.md5(
                f"{time.time()}_{random.random()}".encode()
            ).hexdigest()[:12]
    
    @property
    def age(self) -> float:
        """Get task age in seconds."""
        return time.time() - self.created_at
    
    @property
    def can_retry(self) -> bool:
        """Check if task can be retried."""
        return self.retry_count < self.max_retries


class LoadBalancer:
    """
    Advanced load balancer for SNN-Fusion distributed processing.
    
    Supports multiple load balancing strategies, health monitoring,
    dynamic scaling, and fault tolerance.
    """
    
    def __init__(
        self,
        strategy: LoadBalancingStrategy = LoadBalancingStrategy.RESOURCE_AWARE,
        health_check_interval: float = 30.0,
        task_timeout: float = 300.0
    ):
        """Initialize load balancer."""
        self.strategy = strategy
        self.health_check_interval = health_check_interval
        self.task_timeout = task_timeout
        self.logger = logging.getLogger(__name__)
        
        # Node management
        self.nodes: Dict[str, WorkerNode] = {}
        self.nodes_lock = threading.RLock()
        
        # Load balancing state
        self.round_robin_counter = 0
        self.consistent_hash_ring: List[Tuple[int, str]] = []
        
        # Task management
        self.pending_tasks: queue.PriorityQueue = queue.PriorityQueue()
        self.active_tasks: Dict[str, Tuple[Task, WorkerNode, float]] = {}
        self.completed_tasks: Dict[str, Any] = {}
        
        # Statistics
        self.total_tasks = 0
        self.successful_tasks = 0
        self.failed_tasks = 0
        self.start_time = time.time()
        
        # Background threads
        self.health_monitor_thread = None
        self.task_dispatcher_thread = None
        self.cleanup_thread = None
        self.running = False
        
        self.logger.info(f"LoadBalancer initialized with {strategy.value} strategy")
    
    def add_node(self, node: WorkerNode):
        """Add a worker node to the cluster."""
        with self.nodes_lock:
            self.nodes[node.node_id] = node
            self._rebuild_consistent_hash()
            self.logger.info(f"Added node {node.node_id} ({node.host}:{node.port})")
    
    def remove_node(self, node_id: str):
        """Remove a worker node from the cluster."""
        with self.nodes_lock:
            if node_id in self.nodes:
                node = self.nodes[node_id]
                # Gracefully handle active tasks on this node
                self._handle_node_failure(node)
                del self.nodes[node_id]
                self._rebuild_consistent_hash()
                self.logger.info(f"Removed node {node_id}")
    
    def submit_task(self, task: Task) -> str:
        """Submit a task for distributed processing."""
        # Add to pending queue (priority queue: higher priority = lower number)
        priority = -task.priority  # Negate for correct ordering
        self.pending_tasks.put((priority, task.created_at, task))
        
        self.total_tasks += 1
        self.logger.debug(f"Submitted task {task.task_id} with priority {task.priority}")
        
        return task.task_id
    
    def get_task_result(self, task_id: str, timeout: float = None) -> Any:
        """Get result for a completed task."""
        start_time = time.time()
        timeout = timeout or self.task_timeout
        
        while time.time() - start_time < timeout:
            if task_id in self.completed_tasks:
                result = self.completed_tasks.pop(task_id)
                if isinstance(result, Exception):
                    raise result
                return result
            
            time.sleep(0.1)
        
        raise TimeoutError(f"Task {task_id} did not complete within {timeout}s")
    
    def start(self):
        """Start the load balancer."""
        if self.running:
            return
        
        self.running = True
        
        # Start background threads
        self.health_monitor_thread = threading.Thread(
            target=self._health_monitor_loop,
            daemon=True
        )
        self.task_dispatcher_thread = threading.Thread(
            target=self._task_dispatcher_loop,
            daemon=True
        )
        self.cleanup_thread = threading.Thread(
            target=self._cleanup_loop,
            daemon=True
        )
        
        self.health_monitor_thread.start()
        self.task_dispatcher_thread.start()
        self.cleanup_thread.start()
        
        self.logger.info("LoadBalancer started")
    
    def stop(self):
        """Stop the load balancer."""
        self.running = False
        
        # Wait for threads to finish
        for thread in [self.health_monitor_thread, 
                      self.task_dispatcher_thread, 
                      self.cleanup_thread]:
            if thread and thread.is_alive():
                thread.join(timeout=5.0)
        
        self.logger.info("LoadBalancer stopped")
    
    def get_cluster_status(self) -> Dict[str, Any]:
        """Get comprehensive cluster status."""
        with self.nodes_lock:
            nodes_status = {
                node_id: node.to_dict()
                for node_id, node in self.nodes.items()
            }
        
        # Calculate cluster metrics
        healthy_nodes = sum(1 for node in self.nodes.values() 
                          if node.status == NodeStatus.HEALTHY)
        total_capacity = sum(node.max_concurrent_tasks for node in self.nodes.values())
        current_load = sum(node.current_load for node in self.nodes.values())
        
        uptime = time.time() - self.start_time
        success_rate = (self.successful_tasks / max(self.total_tasks, 1)) * 100
        
        return {
            'strategy': self.strategy.value,
            'uptime_seconds': uptime,
            'nodes': nodes_status,
            'cluster_metrics': {
                'total_nodes': len(self.nodes),
                'healthy_nodes': healthy_nodes,
                'total_capacity': total_capacity,
                'current_load': current_load,
                'utilization': (current_load / max(total_capacity, 1)) * 100
            },
            'task_statistics': {
                'total_tasks': self.total_tasks,
                'successful_tasks': self.successful_tasks,
                'failed_tasks': self.failed_tasks,
                'success_rate': success_rate,
                'pending_tasks': self.pending_tasks.qsize(),
                'active_tasks': len(self.active_tasks)
            }
        }
    
    def _select_node(self, task: Task) -> Optional[WorkerNode]:
        """Select the best node for a task based on load balancing strategy."""
        available_nodes = [node for node in self.nodes.values() if node.is_available]
        
        if not available_nodes:
            return None
        
        if self.strategy == LoadBalancingStrategy.ROUND_ROBIN:
            return self._round_robin_select(available_nodes)
        elif self.strategy == LoadBalancingStrategy.WEIGHTED_ROUND_ROBIN:
            return self._weighted_round_robin_select(available_nodes)
        elif self.strategy == LoadBalancingStrategy.LEAST_CONNECTIONS:
            return self._least_connections_select(available_nodes)
        elif self.strategy == LoadBalancingStrategy.LEAST_RESPONSE_TIME:
            return self._least_response_time_select(available_nodes)
        elif self.strategy == LoadBalancingStrategy.RESOURCE_AWARE:
            return self._resource_aware_select(available_nodes, task)
        elif self.strategy == LoadBalancingStrategy.CONSISTENT_HASH:
            return self._consistent_hash_select(available_nodes, task)
        else:
            return random.choice(available_nodes)
    
    def _round_robin_select(self, nodes: List[WorkerNode]) -> WorkerNode:
        """Round-robin node selection."""
        node = nodes[self.round_robin_counter % len(nodes)]
        self.round_robin_counter += 1
        return node
    
    def _weighted_round_robin_select(self, nodes: List[WorkerNode]) -> WorkerNode:
        """Weighted round-robin node selection."""
        weights = [node.weight for node in nodes]
        total_weight = sum(weights)
        
        if total_weight == 0:
            return random.choice(nodes)
        
        # Select based on cumulative weights
        random_value = random.uniform(0, total_weight)
        cumulative_weight = 0
        
        for node, weight in zip(nodes, weights):
            cumulative_weight += weight
            if random_value <= cumulative_weight:
                return node
        
        return nodes[-1]  # Fallback
    
    def _least_connections_select(self, nodes: List[WorkerNode]) -> WorkerNode:
        """Select node with least connections."""
        return min(nodes, key=lambda n: n.current_load)
    
    def _least_response_time_select(self, nodes: List[WorkerNode]) -> WorkerNode:
        """Select node with least average response time."""
        return min(nodes, key=lambda n: n.avg_response_time or float('inf'))
    
    def _resource_aware_select(self, nodes: List[WorkerNode], task: Task) -> WorkerNode:
        """Select node based on resource requirements and availability."""
        def score_node(node: WorkerNode) -> float:
            # Base score from resource utilization
            cpu_score = 1.0 - (node.cpu_usage / 100.0)
            memory_score = 1.0 - (node.memory_usage / 100.0)
            load_score = 1.0 - node.load_factor
            
            # Response time penalty
            response_penalty = 1.0 / (1.0 + node.avg_response_time)
            
            # Success rate bonus
            success_bonus = node.success_rate
            
            # Network latency penalty
            latency_penalty = 1.0 / (1.0 + node.network_latency)
            
            # Combine scores with weights
            total_score = (
                0.3 * cpu_score +
                0.3 * memory_score +
                0.2 * load_score +
                0.1 * response_penalty +
                0.05 * success_bonus +
                0.05 * latency_penalty
            )
            
            # Apply task-specific requirements
            if task.required_memory_mb and node.memory_usage > 80:
                total_score *= 0.5  # Penalize high memory nodes for memory-intensive tasks
            
            if task.required_cpu_cores and node.cpu_usage > 80:
                total_score *= 0.5  # Penalize high CPU nodes for CPU-intensive tasks
            
            return total_score
        
        return max(nodes, key=score_node)
    
    def _consistent_hash_select(self, nodes: List[WorkerNode], task: Task) -> WorkerNode:
        """Select node using consistent hashing."""
        if not self.consistent_hash_ring:
            return random.choice(nodes)
        
        # Hash the task ID
        task_hash = hash(task.task_id) % (2**32)
        
        # Find the first node with hash >= task_hash
        for node_hash, node_id in self.consistent_hash_ring:
            if node_hash >= task_hash and node_id in self.nodes:
                node = self.nodes[node_id]
                if node in nodes:  # Check if node is available
                    return node
        
        # Wrap around to first node
        for node_hash, node_id in self.consistent_hash_ring:
            if node_id in self.nodes:
                node = self.nodes[node_id]
                if node in nodes:
                    return node
        
        return random.choice(nodes)
    
    def _rebuild_consistent_hash(self):
        """Rebuild consistent hash ring."""
        self.consistent_hash_ring.clear()
        
        # Add virtual nodes for better distribution
        virtual_nodes_per_node = 150
        
        for node_id in self.nodes:
            for i in range(virtual_nodes_per_node):
                virtual_node_id = f"{node_id}:{i}"
                node_hash = hash(virtual_node_id) % (2**32)
                self.consistent_hash_ring.append((node_hash, node_id))
        
        # Sort by hash value
        self.consistent_hash_ring.sort(key=lambda x: x[0])
    
    def _health_monitor_loop(self):
        """Background thread for monitoring node health."""
        while self.running:
            try:
                with self.nodes_lock:
                    for node in self.nodes.values():
                        self._check_node_health(node)
                
                time.sleep(self.health_check_interval)
                
            except Exception as e:
                self.logger.error(f"Health monitor error: {e}")
                time.sleep(5.0)
    
    def _check_node_health(self, node: WorkerNode):
        """Check health of a specific node."""
        try:
            # In a real implementation, this would ping the actual node
            # For now, simulate health check based on metrics
            
            current_time = time.time()
            time_since_check = current_time - node.last_health_check
            
            if time_since_check > self.health_check_interval * 3:
                # Node hasn't been updated recently
                if node.status != NodeStatus.FAILED:
                    self.logger.warning(f"Node {node.node_id} appears unhealthy")
                    node.status = NodeStatus.FAILED
                    self._handle_node_failure(node)
            
            # Check resource thresholds
            if node.cpu_usage > 95 or node.memory_usage > 95:
                node.status = NodeStatus.OVERLOADED
            elif node.success_rate < 0.8:
                node.status = NodeStatus.DEGRADED
                
        except Exception as e:
            self.logger.error(f"Health check failed for node {node.node_id}: {e}")
            node.status = NodeStatus.FAILED
    
    def _task_dispatcher_loop(self):
        """Background thread for dispatching tasks to nodes."""
        while self.running:
            try:
                # Get next task from queue
                try:
                    priority, created_at, task = self.pending_tasks.get(timeout=1.0)
                except queue.Empty:
                    continue
                
                # Select node for task
                selected_node = self._select_node(task)
                
                if selected_node is None:
                    # No available nodes, put task back in queue
                    self.pending_tasks.put((priority, created_at, task))
                    time.sleep(1.0)
                    continue
                
                # Dispatch task to selected node
                self._dispatch_task(task, selected_node)
                
            except Exception as e:
                self.logger.error(f"Task dispatcher error: {e}")
                time.sleep(1.0)
    
    def _dispatch_task(self, task: Task, node: WorkerNode):
        """Dispatch a task to a specific node."""
        try:
            # Update node load
            node.current_load += 1
            
            # Track active task
            self.active_tasks[task.task_id] = (task, node, time.time())
            
            # In a real implementation, this would send the task to the actual node
            # For simulation, we'll process it in a thread pool
            threading.Thread(
                target=self._simulate_task_execution,
                args=(task, node),
                daemon=True
            ).start()
            
            self.logger.debug(f"Dispatched task {task.task_id} to node {node.node_id}")
            
        except Exception as e:
            # Task dispatch failed
            node.current_load -= 1
            if task.task_id in self.active_tasks:
                del self.active_tasks[task.task_id]
            
            # Retry if possible
            if task.can_retry:
                task.retry_count += 1
                priority = -task.priority
                self.pending_tasks.put((priority, task.created_at, task))
                self.logger.warning(f"Task {task.task_id} dispatch failed, retrying")
            else:
                self.completed_tasks[task.task_id] = e
                self.failed_tasks += 1
                self.logger.error(f"Task {task.task_id} dispatch failed permanently: {e}")
    
    def _simulate_task_execution(self, task: Task, node: WorkerNode):
        """Simulate task execution (for testing/development)."""
        try:
            # Simulate processing time
            processing_time = random.uniform(0.1, 2.0)
            time.sleep(processing_time)
            
            # Simulate success/failure
            success = random.random() < 0.95  # 95% success rate
            
            if success:
                # Task completed successfully
                result = f"Task {task.task_id} completed successfully"
                self.completed_tasks[task.task_id] = result
                self.successful_tasks += 1
                node.update_metrics(processing_time, True)
            else:
                # Task failed
                error = Exception(f"Task {task.task_id} execution failed")
                if task.can_retry:
                    task.retry_count += 1
                    priority = -task.priority
                    self.pending_tasks.put((priority, task.created_at, task))
                else:
                    self.completed_tasks[task.task_id] = error
                    self.failed_tasks += 1
                node.update_metrics(processing_time, False)
            
        except Exception as e:
            self.completed_tasks[task.task_id] = e
            self.failed_tasks += 1
            node.update_metrics(0.0, False)
            
        finally:
            # Clean up
            node.current_load -= 1
            if task.task_id in self.active_tasks:
                del self.active_tasks[task.task_id]
    
    def _cleanup_loop(self):
        """Background thread for cleanup operations."""
        while self.running:
            try:
                current_time = time.time()
                
                # Clean up old completed tasks
                expired_tasks = [
                    task_id for task_id, result in self.completed_tasks.items()
                    if current_time - getattr(result, '_completed_at', current_time) > 3600
                ]
                for task_id in expired_tasks:
                    self.completed_tasks.pop(task_id, None)
                
                # Check for timed out active tasks
                timed_out_tasks = [
                    task_id for task_id, (task, node, start_time) in self.active_tasks.items()
                    if current_time - start_time > task.timeout
                ]
                
                for task_id in timed_out_tasks:
                    task, node, start_time = self.active_tasks[task_id]
                    self.completed_tasks[task_id] = TimeoutError(f"Task {task_id} timed out")
                    self.failed_tasks += 1
                    node.current_load -= 1
                    del self.active_tasks[task_id]
                    self.logger.warning(f"Task {task_id} timed out after {task.timeout}s")
                
                time.sleep(60)  # Run cleanup every minute
                
            except Exception as e:
                self.logger.error(f"Cleanup error: {e}")
                time.sleep(60)
    
    def _handle_node_failure(self, failed_node: WorkerNode):
        """Handle node failure by redistributing its tasks."""
        # Find tasks assigned to failed node
        failed_tasks = [
            (task_id, task) for task_id, (task, node, start_time) in self.active_tasks.items()
            if node.node_id == failed_node.node_id
        ]
        
        # Redistribute tasks
        for task_id, task in failed_tasks:
            del self.active_tasks[task_id]
            failed_node.current_load -= 1
            
            if task.can_retry:
                task.retry_count += 1
                priority = -task.priority
                self.pending_tasks.put((priority, task.created_at, task))
                self.logger.info(f"Redistributing task {task_id} due to node failure")
            else:
                error = Exception(f"Task {task_id} failed due to node failure")
                self.completed_tasks[task_id] = error
                self.failed_tasks += 1


# Example usage and testing
if __name__ == "__main__":
    print("Testing Load Balancer...")
    
    # Create load balancer
    lb = LoadBalancer(
        strategy=LoadBalancingStrategy.RESOURCE_AWARE,
        health_check_interval=10.0
    )
    
    # Add worker nodes
    nodes = [
        WorkerNode("node1", "localhost", 8001, weight=1.0, max_concurrent_tasks=5),
        WorkerNode("node2", "localhost", 8002, weight=1.5, max_concurrent_tasks=8),
        WorkerNode("node3", "localhost", 8003, weight=0.8, max_concurrent_tasks=3),
    ]
    
    for node in nodes:
        # Simulate different resource states
        if node.node_id == "node1":
            node.update_resource_metrics(30, 40, 0.05)
        elif node.node_id == "node2":
            node.update_resource_metrics(60, 55, 0.08)
        else:
            node.update_resource_metrics(80, 70, 0.12)
        
        lb.add_node(node)
    
    # Start load balancer
    lb.start()
    
    print(f"Added {len(nodes)} nodes to cluster")
    
    # Submit test tasks
    print("\nSubmitting test tasks...")
    task_ids = []
    for i in range(20):
        task = Task(
            task_id=f"test_task_{i}",
            payload={"data": f"processing_job_{i}", "size": random.randint(100, 1000)},
            priority=random.randint(1, 5)
        )
        task_id = lb.submit_task(task)
        task_ids.append(task_id)
    
    print(f"Submitted {len(task_ids)} tasks")
    
    # Wait for tasks to complete
    print("\nWaiting for task completion...")
    completed = 0
    for task_id in task_ids:
        try:
            result = lb.get_task_result(task_id, timeout=10.0)
            completed += 1
        except Exception as e:
            print(f"Task {task_id} failed: {e}")
    
    print(f"Completed {completed}/{len(task_ids)} tasks")
    
    # Get cluster status
    status = lb.get_cluster_status()
    print(f"\nCluster Status:")
    print(f"  Strategy: {status['strategy']}")
    print(f"  Total nodes: {status['cluster_metrics']['total_nodes']}")
    print(f"  Healthy nodes: {status['cluster_metrics']['healthy_nodes']}")
    print(f"  Cluster utilization: {status['cluster_metrics']['utilization']:.1f}%")
    print(f"  Success rate: {status['task_statistics']['success_rate']:.1f}%")
    
    # Node details
    print(f"\nNode Details:")
    for node_id, node_info in status['nodes'].items():
        print(f"  {node_id}: {node_info['status']} "
              f"(load: {node_info['current_load']}, "
              f"success: {node_info['success_rate']:.1%}, "
              f"response: {node_info['avg_response_time']:.3f}s)")
    
    # Stop load balancer
    lb.stop()
    print("✓ Load balancer test completed!")