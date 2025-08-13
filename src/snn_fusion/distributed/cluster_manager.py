"""
Distributed Cluster Management for Neuromorphic Computing

Implements distributed cluster management, node discovery, load balancing,
and fault tolerance for large-scale SNN deployments.
"""

import time
import json
import socket
import threading
import hashlib
import uuid
from typing import Dict, List, Optional, Any, Callable, Set, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
import logging
from collections import defaultdict, deque
import queue
import weakref
import warnings


class NodeStatus(Enum):
    """Node status states."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    OFFLINE = "offline"
    JOINING = "joining"
    LEAVING = "leaving"


class NodeRole(Enum):
    """Node roles in the cluster."""
    COORDINATOR = "coordinator"
    WORKER = "worker"
    HYBRID = "hybrid"


class TaskDistributionStrategy(Enum):
    """Task distribution strategies."""
    ROUND_ROBIN = "round_robin"
    LEAST_LOADED = "least_loaded"
    HASH_BASED = "hash_based"
    CUSTOM = "custom"


@dataclass
class NodeInfo:
    """Information about a cluster node."""
    node_id: str
    hostname: str
    ip_address: str
    port: int
    role: NodeRole
    status: NodeStatus = NodeStatus.OFFLINE
    capabilities: Dict[str, Any] = field(default_factory=dict)
    load_metrics: Dict[str, float] = field(default_factory=dict)
    last_heartbeat: float = field(default_factory=time.time)
    joined_at: float = field(default_factory=time.time)
    version: str = "1.0.0"
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'NodeInfo':
        """Create from dictionary."""
        # Convert enum values
        data['role'] = NodeRole(data['role'])
        data['status'] = NodeStatus(data['status'])
        return cls(**data)
    
    def is_healthy(self, heartbeat_timeout: float = 30.0) -> bool:
        """Check if node is healthy based on heartbeat."""
        return (
            self.status == NodeStatus.HEALTHY and
            (time.time() - self.last_heartbeat) < heartbeat_timeout
        )


@dataclass
class DistributedTask:
    """Represents a distributed task."""
    task_id: str
    task_type: str
    data: Any
    target_node: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    priority: int = 1
    timeout: float = 300.0
    retries: int = 0
    max_retries: int = 3
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskResult:
    """Result of a distributed task."""
    task_id: str
    success: bool
    result: Any = None
    error: Optional[str] = None
    node_id: Optional[str] = None
    execution_time: float = 0.0
    completed_at: float = field(default_factory=time.time)


class HeartbeatManager:
    """
    Manages heartbeat communication between cluster nodes.
    """
    
    def __init__(self, node_id: str, port: int = 0):
        self.node_id = node_id
        self.port = port or self._find_free_port()
        self.running = False
        self.heartbeat_interval = 10.0  # seconds
        
        # Heartbeat tracking
        self.last_heartbeats: Dict[str, float] = {}
        self.heartbeat_callbacks: List[Callable[[str, Dict], None]] = []
        
        # Network components
        self.socket = None
        self.receiver_thread = None
        self.sender_thread = None
        
        self.lock = threading.Lock()
    
    def _find_free_port(self) -> int:
        """Find a free port for heartbeat communication."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('', 0))
            return s.getsockname()[1]
    
    def start(self):
        """Start heartbeat manager."""
        if self.running:
            return
        
        self.running = True
        
        # Setup UDP socket for heartbeat
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.socket.bind(('', self.port))
        self.socket.settimeout(1.0)
        
        # Start receiver thread
        self.receiver_thread = threading.Thread(
            target=self._receiver_loop,
            daemon=True
        )
        self.receiver_thread.start()
        
        # Start sender thread
        self.sender_thread = threading.Thread(
            target=self._sender_loop,
            daemon=True
        )
        self.sender_thread.start()
        
        logging.info(f"HeartbeatManager started on port {self.port}")
    
    def stop(self):
        """Stop heartbeat manager."""
        self.running = False
        
        if self.socket:
            self.socket.close()
        
        if self.receiver_thread:
            self.receiver_thread.join(timeout=2.0)
        
        if self.sender_thread:
            self.sender_thread.join(timeout=2.0)
    
    def add_heartbeat_callback(self, callback: Callable[[str, Dict], None]):
        """Add callback for heartbeat events."""
        self.heartbeat_callbacks.append(callback)
    
    def send_heartbeat_to(self, target_ip: str, target_port: int, data: Dict[str, Any]):
        """Send heartbeat to specific node."""
        try:
            heartbeat_data = {
                'node_id': self.node_id,
                'timestamp': time.time(),
                'type': 'heartbeat',
                **data
            }
            
            message = json.dumps(heartbeat_data).encode('utf-8')
            self.socket.sendto(message, (target_ip, target_port))
            
        except Exception as e:
            logging.warning(f"Failed to send heartbeat to {target_ip}:{target_port}: {e}")
    
    def _receiver_loop(self):
        """Receive heartbeats from other nodes."""
        while self.running:
            try:
                data, addr = self.socket.recvfrom(1024)
                heartbeat = json.loads(data.decode('utf-8'))
                
                if heartbeat.get('type') == 'heartbeat':
                    sender_id = heartbeat.get('node_id')
                    
                    if sender_id and sender_id != self.node_id:
                        with self.lock:
                            self.last_heartbeats[sender_id] = time.time()
                        
                        # Notify callbacks
                        for callback in self.heartbeat_callbacks:
                            try:
                                callback(sender_id, heartbeat)
                            except Exception as e:
                                logging.error(f"Heartbeat callback error: {e}")
                
            except socket.timeout:
                continue
            except Exception as e:
                if self.running:
                    logging.error(f"Heartbeat receiver error: {e}")
                break
    
    def _sender_loop(self):
        """Send periodic heartbeats."""
        # This would typically broadcast to known nodes
        # For now, just a placeholder
        while self.running:
            try:
                time.sleep(self.heartbeat_interval)
                # Heartbeat sending logic would go here
                
            except Exception as e:
                logging.error(f"Heartbeat sender error: {e}")
    
    def get_node_status(self, node_id: str, timeout: float = 30.0) -> bool:
        """Check if a node is alive based on heartbeat."""
        with self.lock:
            last_heartbeat = self.last_heartbeats.get(node_id, 0)
        
        return (time.time() - last_heartbeat) < timeout


class ClusterManager:
    """
    Main cluster management system for distributed neuromorphic computing.
    """
    
    def __init__(self, node_id: Optional[str] = None, 
                 role: NodeRole = NodeRole.HYBRID,
                 port: int = 0):
        self.node_id = node_id or str(uuid.uuid4())
        self.role = role
        self.port = port or self._find_free_port()
        
        # Cluster state
        self.nodes: Dict[str, NodeInfo] = {}
        self.coordinator_id: Optional[str] = None
        self.is_coordinator = False
        
        # Task management
        self.pending_tasks: Dict[str, DistributedTask] = {}
        self.completed_tasks: Dict[str, TaskResult] = {}
        self.task_handlers: Dict[str, Callable] = {}
        
        # Distribution strategy
        self.distribution_strategy = TaskDistributionStrategy.LEAST_LOADED
        self.custom_distributor: Optional[Callable] = None
        
        # Heartbeat management
        self.heartbeat_manager = HeartbeatManager(self.node_id, port + 1000)
        self.heartbeat_manager.add_heartbeat_callback(self._handle_heartbeat)
        
        # Control
        self.running = False
        self.management_thread = None
        
        # Statistics
        self.stats = {
            'tasks_distributed': 0,
            'tasks_completed': 0,
            'tasks_failed': 0,
            'node_join_events': 0,
            'node_leave_events': 0,
            'coordinator_changes': 0
        }
        
        self.stats_lock = threading.Lock()
        
        # Local node info
        self.local_node = NodeInfo(
            node_id=self.node_id,
            hostname=socket.gethostname(),
            ip_address=self._get_local_ip(),
            port=self.port,
            role=role,
            status=NodeStatus.JOINING
        )
        
        self.logger = logging.getLogger(__name__)
    
    def _find_free_port(self) -> int:
        """Find a free port."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('', 0))
            return s.getsockname()[1]
    
    def _get_local_ip(self) -> str:
        """Get local IP address."""
        try:
            # Connect to a remote address to determine local IP
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                s.connect(("8.8.8.8", 80))
                return s.getsockname()[0]
        except Exception:
            return "127.0.0.1"
    
    def start(self, bootstrap_nodes: Optional[List[Tuple[str, int]]] = None):
        """
        Start cluster manager.
        
        Args:
            bootstrap_nodes: List of (ip, port) tuples for bootstrap nodes
        """
        if self.running:
            return
        
        self.running = True
        
        # Start heartbeat manager
        self.heartbeat_manager.start()
        
        # Start management thread
        self.management_thread = threading.Thread(
            target=self._management_loop,
            daemon=True
        )
        self.management_thread.start()
        
        # Add local node to cluster
        self.nodes[self.node_id] = self.local_node
        
        # Bootstrap cluster connection
        if bootstrap_nodes:
            self._bootstrap_cluster(bootstrap_nodes)
        else:
            # Become coordinator if no bootstrap nodes
            self._become_coordinator()
        
        self.logger.info(f"ClusterManager started (Node: {self.node_id}, Role: {self.role.value})")
    
    def stop(self):
        """Stop cluster manager."""
        self.running = False
        
        # Update local node status
        if self.node_id in self.nodes:
            self.nodes[self.node_id].status = NodeStatus.LEAVING
        
        # Stop heartbeat manager
        self.heartbeat_manager.stop()
        
        # Wait for management thread
        if self.management_thread:
            self.management_thread.join(timeout=5.0)
        
        self.logger.info("ClusterManager stopped")
    
    def _bootstrap_cluster(self, bootstrap_nodes: List[Tuple[str, int]]):
        """Bootstrap connection to existing cluster."""
        for ip, port in bootstrap_nodes:
            try:
                # Send join request (simplified)
                self.heartbeat_manager.send_heartbeat_to(
                    ip, port + 1000,  # Heartbeat port offset
                    {
                        'action': 'join_request',
                        'node_info': self.local_node.to_dict()
                    }
                )
            except Exception as e:
                self.logger.warning(f"Failed to bootstrap with {ip}:{port}: {e}")
    
    def _become_coordinator(self):
        """Become the cluster coordinator."""
        self.is_coordinator = True
        self.coordinator_id = self.node_id
        self.local_node.status = NodeStatus.HEALTHY
        
        with self.stats_lock:
            self.stats['coordinator_changes'] += 1
        
        self.logger.info(f"Node {self.node_id} became coordinator")
    
    def _management_loop(self):
        """Main cluster management loop."""
        while self.running:
            try:
                # Check node health
                self._check_node_health()
                
                # Handle coordinator election if needed
                if not self.coordinator_id or self.coordinator_id not in self.nodes:
                    self._elect_coordinator()
                
                # Clean up old tasks
                self._cleanup_old_tasks()
                
                time.sleep(5.0)  # Management cycle every 5 seconds
                
            except Exception as e:
                self.logger.error(f"Management loop error: {e}")
                time.sleep(1.0)
    
    def _check_node_health(self):
        """Check health of all nodes."""
        current_time = time.time()
        unhealthy_nodes = []
        
        for node_id, node in self.nodes.items():
            if node_id == self.node_id:
                continue  # Skip self
            
            # Check heartbeat timeout
            if (current_time - node.last_heartbeat) > 30.0:
                if node.status == NodeStatus.HEALTHY:
                    node.status = NodeStatus.DEGRADED
                    self.logger.warning(f"Node {node_id} marked as degraded")
                elif (current_time - node.last_heartbeat) > 60.0:
                    node.status = NodeStatus.OFFLINE
                    unhealthy_nodes.append(node_id)
                    self.logger.warning(f"Node {node_id} marked as offline")
        
        # Remove offline nodes
        for node_id in unhealthy_nodes:
            if node_id in self.nodes:
                del self.nodes[node_id]
                with self.stats_lock:
                    self.stats['node_leave_events'] += 1
    
    def _elect_coordinator(self):
        """Elect a new coordinator using simple algorithm."""
        healthy_nodes = [
            (node_id, node) for node_id, node in self.nodes.items()
            if node.status == NodeStatus.HEALTHY
        ]
        
        if not healthy_nodes:
            # No healthy nodes, become coordinator
            self._become_coordinator()
            return
        
        # Simple election: node with lowest ID becomes coordinator
        coordinator_candidate = min(healthy_nodes, key=lambda x: x[0])
        new_coordinator_id = coordinator_candidate[0]
        
        if new_coordinator_id == self.node_id:
            self._become_coordinator()
        else:
            self.coordinator_id = new_coordinator_id
            self.is_coordinator = False
            
            with self.stats_lock:
                self.stats['coordinator_changes'] += 1
            
            self.logger.info(f"Node {new_coordinator_id} elected as coordinator")
    
    def _handle_heartbeat(self, node_id: str, data: Dict[str, Any]):
        """Handle heartbeat from another node."""
        current_time = time.time()
        
        # Handle join requests
        if data.get('action') == 'join_request' and 'node_info' in data:
            node_info = NodeInfo.from_dict(data['node_info'])
            node_info.last_heartbeat = current_time
            node_info.status = NodeStatus.HEALTHY
            
            self.nodes[node_id] = node_info
            
            with self.stats_lock:
                self.stats['node_join_events'] += 1
            
            self.logger.info(f"Node {node_id} joined cluster")
        
        # Update existing node
        elif node_id in self.nodes:
            self.nodes[node_id].last_heartbeat = current_time
            if self.nodes[node_id].status == NodeStatus.DEGRADED:
                self.nodes[node_id].status = NodeStatus.HEALTHY
                self.logger.info(f"Node {node_id} recovered")
    
    def register_task_handler(self, task_type: str, handler: Callable):
        """Register a handler for a specific task type."""
        self.task_handlers[task_type] = handler
        self.logger.info(f"Registered handler for task type: {task_type}")
    
    def distribute_task(self, task: DistributedTask) -> bool:
        """
        Distribute a task to an appropriate node.
        
        Args:
            task: Task to distribute
            
        Returns:
            True if task was successfully distributed
        """
        if not self.is_coordinator:
            self.logger.warning("Only coordinator can distribute tasks")
            return False
        
        # Find target node
        target_node_id = self._select_target_node(task)
        
        if not target_node_id:
            self.logger.error("No available nodes for task distribution")
            return False
        
        # Store task
        task.target_node = target_node_id
        self.pending_tasks[task.task_id] = task
        
        # Send task to target node (simplified)
        try:
            # In real implementation, would send over network
            self._execute_local_task(task)
            
            with self.stats_lock:
                self.stats['tasks_distributed'] += 1
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to distribute task {task.task_id}: {e}")
            return False
    
    def _select_target_node(self, task: DistributedTask) -> Optional[str]:
        """Select target node for task based on distribution strategy."""
        healthy_nodes = [
            (node_id, node) for node_id, node in self.nodes.items()
            if node.status == NodeStatus.HEALTHY and node.role in [NodeRole.WORKER, NodeRole.HYBRID]
        ]
        
        if not healthy_nodes:
            return None
        
        if self.distribution_strategy == TaskDistributionStrategy.ROUND_ROBIN:
            # Simple round-robin based on task count
            task_counts = defaultdict(int)
            for t in self.pending_tasks.values():
                if t.target_node:
                    task_counts[t.target_node] += 1
            
            return min(healthy_nodes, key=lambda x: task_counts[x[0]])[0]
        
        elif self.distribution_strategy == TaskDistributionStrategy.LEAST_LOADED:
            # Select node with lowest load
            return min(healthy_nodes, key=lambda x: x[1].load_metrics.get('cpu', 0))[0]
        
        elif self.distribution_strategy == TaskDistributionStrategy.HASH_BASED:
            # Hash-based selection for consistency
            task_hash = int(hashlib.md5(task.task_id.encode()).hexdigest(), 16)
            return healthy_nodes[task_hash % len(healthy_nodes)][0]
        
        elif self.distribution_strategy == TaskDistributionStrategy.CUSTOM and self.custom_distributor:
            # Use custom distribution function
            return self.custom_distributor(task, healthy_nodes)
        
        else:
            # Default: first available node
            return healthy_nodes[0][0]
    
    def _execute_local_task(self, task: DistributedTask):
        """Execute task locally (placeholder for network execution)."""
        if task.task_type in self.task_handlers:
            handler = self.task_handlers[task.task_type]
            
            start_time = time.time()
            try:
                result = handler(task.data)
                execution_time = time.time() - start_time
                
                task_result = TaskResult(
                    task_id=task.task_id,
                    success=True,
                    result=result,
                    node_id=self.node_id,
                    execution_time=execution_time
                )
                
                self.completed_tasks[task.task_id] = task_result
                
                if task.task_id in self.pending_tasks:
                    del self.pending_tasks[task.task_id]
                
                with self.stats_lock:
                    self.stats['tasks_completed'] += 1
                
            except Exception as e:
                execution_time = time.time() - start_time
                
                task_result = TaskResult(
                    task_id=task.task_id,
                    success=False,
                    error=str(e),
                    node_id=self.node_id,
                    execution_time=execution_time
                )
                
                self.completed_tasks[task.task_id] = task_result
                
                with self.stats_lock:
                    self.stats['tasks_failed'] += 1
    
    def _cleanup_old_tasks(self):
        """Clean up old completed tasks."""
        current_time = time.time()
        cleanup_age = 3600  # 1 hour
        
        old_tasks = [
            task_id for task_id, result in self.completed_tasks.items()
            if (current_time - result.completed_at) > cleanup_age
        ]
        
        for task_id in old_tasks:
            del self.completed_tasks[task_id]
    
    def get_cluster_state(self) -> Dict[str, Any]:
        """Get current cluster state."""
        healthy_nodes = sum(1 for node in self.nodes.values() if node.status == NodeStatus.HEALTHY)
        
        return {
            'node_id': self.node_id,
            'is_coordinator': self.is_coordinator,
            'coordinator_id': self.coordinator_id,
            'total_nodes': len(self.nodes),
            'healthy_nodes': healthy_nodes,
            'pending_tasks': len(self.pending_tasks),
            'completed_tasks': len(self.completed_tasks),
            'distribution_strategy': self.distribution_strategy.value,
            'uptime': time.time() - self.local_node.joined_at
        }
    
    def get_node_list(self) -> List[Dict[str, Any]]:
        """Get list of all nodes with their information."""
        return [node.to_dict() for node in self.nodes.values()]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get cluster statistics."""
        with self.stats_lock:
            return self.stats.copy()


# Example usage and testing
if __name__ == "__main__":
    print("Testing Distributed Cluster Manager...")
    
    # Test node info
    print("\n1. Testing Node Info:")
    node = NodeInfo(
        node_id="test_node",
        hostname="localhost",
        ip_address="127.0.0.1",
        port=8080,
        role=NodeRole.WORKER
    )
    
    # Test serialization
    node_dict = node.to_dict()
    restored_node = NodeInfo.from_dict(node_dict)
    
    assert restored_node.node_id == node.node_id
    assert restored_node.role == node.role
    print("  ✓ Node info serialization working")
    
    # Test heartbeat manager
    print("\n2. Testing Heartbeat Manager:")
    heartbeat_mgr = HeartbeatManager("test_node")
    
    # Mock callback
    received_heartbeats = []
    def heartbeat_callback(node_id, data):
        received_heartbeats.append((node_id, data))
    
    heartbeat_mgr.add_heartbeat_callback(heartbeat_callback)
    
    # Start briefly
    heartbeat_mgr.start()
    time.sleep(0.5)
    heartbeat_mgr.stop()
    
    print(f"  ✓ Heartbeat manager started and stopped on port {heartbeat_mgr.port}")
    
    # Test cluster manager
    print("\n3. Testing Cluster Manager:")
    cluster_mgr = ClusterManager("test_cluster_node", NodeRole.HYBRID)
    
    # Register a test task handler
    def test_task_handler(data):
        return f"processed: {data}"
    
    cluster_mgr.register_task_handler("test_task", test_task_handler)
    
    # Start cluster manager
    cluster_mgr.start()
    time.sleep(1.0)  # Let it initialize
    
    # Test task distribution
    test_task = DistributedTask(
        task_id="test_task_1",
        task_type="test_task",
        data="hello world"
    )
    
    distributed = cluster_mgr.distribute_task(test_task)
    print(f"  ✓ Task distribution: {distributed}")
    
    # Get cluster state
    state = cluster_mgr.get_cluster_state()
    print(f"  ✓ Cluster state: {state['total_nodes']} nodes, coordinator: {state['is_coordinator']}")
    
    # Get statistics
    stats = cluster_mgr.get_statistics()
    print(f"  ✓ Statistics: {stats['tasks_distributed']} distributed, {stats['tasks_completed']} completed")
    
    # Stop cluster manager
    cluster_mgr.stop()
    
    print("\n✓ Distributed cluster manager test completed!")