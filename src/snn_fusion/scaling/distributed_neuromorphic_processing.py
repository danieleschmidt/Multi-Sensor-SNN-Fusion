"""
Distributed Neuromorphic Processing System

Advanced distributed processing system for neuromorphic multi-modal fusion
with support for cluster computing, edge deployment, and hybrid cloud-edge
architectures optimized for ultra-low latency processing.
"""

import asyncio
import json
import time
import threading
import queue
import multiprocessing as mp
from typing import Dict, List, Any, Optional, Tuple, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
import socket
import pickle
import numpy as np
from pathlib import Path
import uuid

# Async networking
try:
    import aioredis
    import aiokafka
    ASYNC_AVAILABLE = True
except ImportError:
    ASYNC_AVAILABLE = False

# Distributed computing
try:
    import ray
    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False

# Container orchestration
try:
    from kubernetes import client, config as k8s_config
    K8S_AVAILABLE = True
except ImportError:
    K8S_AVAILABLE = False

from ..algorithms.fusion import CrossModalFusion, ModalityData, FusionResult
from ..algorithms.temporal_spike_attention import TemporalSpikeAttention
from ..optimization.advanced_performance_optimizer import OptimizationConfig
from ..utils.robust_error_handling import robust_function, RobustErrorHandler
from ..security.neuromorphic_security import NeuromorphicSecurityManager


class NodeType(Enum):
    """Types of processing nodes in the distributed system."""
    EDGE = "edge"                    # Edge computing nodes
    COMPUTE = "compute"              # Main compute nodes
    COORDINATOR = "coordinator"      # Coordination nodes
    STORAGE = "storage"             # Storage nodes
    NEUROMORPHIC = "neuromorphic"   # Neuromorphic hardware nodes


class ProcessingMode(Enum):
    """Processing modes for distributed system."""
    PIPELINE = "pipeline"           # Sequential pipeline processing
    PARALLEL = "parallel"          # Parallel processing
    HYBRID = "hybrid"              # Hybrid pipeline-parallel
    FEDERATED = "federated"        # Federated learning mode


@dataclass
class NodeConfig:
    """Configuration for a processing node."""
    node_id: str
    node_type: NodeType
    address: str
    port: int
    capabilities: Dict[str, Any]
    resource_limits: Dict[str, float]
    security_config: Optional[Dict[str, Any]] = None
    priority: int = 1  # Higher priority = preferred node


@dataclass
class ProcessingJob:
    """Job for distributed processing."""
    job_id: str
    modality_data: Dict[str, ModalityData]
    model_config: Dict[str, Any]
    priority: int = 1
    max_latency_ms: float = 100.0
    target_nodes: Optional[List[str]] = None
    callback: Optional[Callable] = None
    created_at: float = field(default_factory=time.time)


@dataclass
class ProcessingResult:
    """Result from distributed processing."""
    job_id: str
    fusion_result: FusionResult
    processing_node: str
    processing_time_ms: float
    queue_time_ms: float
    network_latency_ms: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class DistributedLoadBalancer:
    """
    Intelligent load balancer for neuromorphic processing nodes.
    
    Features:
    - Latency-aware routing
    - Node capability matching
    - Adaptive load balancing
    - Health-based routing
    """
    
    def __init__(self, health_check_interval: float = 5.0):
        """
        Initialize load balancer.
        
        Args:
            health_check_interval: Interval for health checks in seconds
        """
        self.nodes: Dict[str, NodeConfig] = {}
        self.node_health: Dict[str, float] = {}  # 0.0 to 1.0
        self.node_load: Dict[str, float] = {}    # Current load factor
        self.node_metrics: Dict[str, Dict[str, float]] = {}
        
        self.health_check_interval = health_check_interval
        self.routing_history = []
        
        self.logger = logging.getLogger(__name__)
        
        # Start health monitoring
        self._start_health_monitoring()
    
    def register_node(self, node_config: NodeConfig):
        """Register a new processing node."""
        self.nodes[node_config.node_id] = node_config
        self.node_health[node_config.node_id] = 1.0  # Assume healthy initially
        self.node_load[node_config.node_id] = 0.0
        self.node_metrics[node_config.node_id] = {
            'avg_latency_ms': 0.0,
            'throughput_ops_per_sec': 0.0,
            'error_rate': 0.0,
            'queue_depth': 0.0,
        }
        
        self.logger.info(f"Registered node {node_config.node_id} ({node_config.node_type.value})")
    
    def unregister_node(self, node_id: str):
        """Unregister a processing node."""
        if node_id in self.nodes:
            del self.nodes[node_id]
            del self.node_health[node_id]
            del self.node_load[node_id]
            del self.node_metrics[node_id]
            
            self.logger.info(f"Unregistered node {node_id}")
    
    def select_node(
        self,
        job: ProcessingJob,
        exclude_nodes: Optional[List[str]] = None,
    ) -> Optional[str]:
        """
        Select optimal node for processing job.
        
        Args:
            job: Processing job to route
            exclude_nodes: Nodes to exclude from selection
            
        Returns:
            Selected node ID or None if no suitable node found
        """
        if not self.nodes:
            return None
        
        exclude_nodes = exclude_nodes or []
        
        # Filter available nodes
        available_nodes = []
        for node_id, node_config in self.nodes.items():
            if (node_id not in exclude_nodes and 
                self.node_health[node_id] > 0.5 and  # Healthy nodes only
                self._node_can_handle_job(node_config, job)):
                available_nodes.append(node_id)
        
        if not available_nodes:
            return None
        
        # Score nodes based on multiple criteria
        node_scores = {}
        for node_id in available_nodes:
            score = self._calculate_node_score(node_id, job)
            node_scores[node_id] = score
        
        # Select node with highest score
        selected_node = max(node_scores.keys(), key=lambda n: node_scores[n])
        
        # Update load balancing history
        self.routing_history.append({
            'timestamp': time.time(),
            'job_id': job.job_id,
            'selected_node': selected_node,
            'available_nodes': available_nodes,
            'scores': node_scores,
        })
        
        # Keep limited history
        if len(self.routing_history) > 1000:
            self.routing_history = self.routing_history[-1000:]
        
        return selected_node
    
    def _node_can_handle_job(self, node_config: NodeConfig, job: ProcessingJob) -> bool:
        """Check if node can handle the job."""
        # Check target nodes constraint
        if job.target_nodes and node_config.node_id not in job.target_nodes:
            return False
        
        # Check node type compatibility
        if node_config.node_type == NodeType.STORAGE:
            return False  # Storage nodes don't process
        
        # Check resource limits
        current_load = self.node_load[node_config.node_id]
        if current_load > 0.9:  # Node overloaded
            return False
        
        # Check latency requirements
        node_metrics = self.node_metrics[node_config.node_id]
        if node_metrics['avg_latency_ms'] > job.max_latency_ms * 0.8:
            return False  # Node likely to exceed latency requirement
        
        # Check modality capabilities
        required_modalities = set(job.modality_data.keys())
        supported_modalities = set(node_config.capabilities.get('modalities', []))
        
        if required_modalities and not required_modalities.issubset(supported_modalities):
            return False  # Node doesn't support required modalities
        
        return True
    
    def _calculate_node_score(self, node_id: str, job: ProcessingJob) -> float:
        """Calculate score for node selection."""
        node_config = self.nodes[node_id]
        node_metrics = self.node_metrics[node_id]
        
        score = 0.0
        
        # Health factor (0-30 points)
        health_score = self.node_health[node_id] * 30
        score += health_score
        
        # Load factor (0-25 points) - lower load is better
        load_score = (1.0 - self.node_load[node_id]) * 25
        score += load_score
        
        # Latency factor (0-20 points) - lower latency is better
        if node_metrics['avg_latency_ms'] > 0:
            latency_score = max(0, 20 - (node_metrics['avg_latency_ms'] / job.max_latency_ms) * 20)
        else:
            latency_score = 20  # No history, assume good latency
        score += latency_score
        
        # Throughput factor (0-15 points)
        throughput_score = min(15, node_metrics['throughput_ops_per_sec'] / 10.0)
        score += throughput_score
        
        # Priority factor (0-10 points)
        priority_score = min(10, node_config.priority * 2)
        score += priority_score
        
        # Error rate penalty
        error_penalty = node_metrics['error_rate'] * 20
        score -= error_penalty
        
        # Node type bonus
        if node_config.node_type == NodeType.NEUROMORPHIC:
            score += 10  # Prefer neuromorphic hardware
        elif node_config.node_type == NodeType.EDGE:
            score += 5   # Prefer edge nodes for low latency
        
        return max(0, score)  # Ensure non-negative score
    
    def update_node_metrics(
        self,
        node_id: str,
        latency_ms: float,
        throughput_ops_per_sec: float,
        error_occurred: bool = False,
        queue_depth: int = 0,
    ):
        """Update node performance metrics."""
        if node_id not in self.node_metrics:
            return
        
        metrics = self.node_metrics[node_id]
        
        # Update with exponential moving average
        alpha = 0.1
        
        metrics['avg_latency_ms'] = (
            (1 - alpha) * metrics['avg_latency_ms'] + alpha * latency_ms
        )
        
        metrics['throughput_ops_per_sec'] = (
            (1 - alpha) * metrics['throughput_ops_per_sec'] + alpha * throughput_ops_per_sec
        )
        
        # Update error rate
        error_value = 1.0 if error_occurred else 0.0
        metrics['error_rate'] = (
            (1 - alpha) * metrics['error_rate'] + alpha * error_value
        )
        
        metrics['queue_depth'] = queue_depth
        
        # Update node load based on queue depth and throughput
        max_queue = self.nodes[node_id].resource_limits.get('max_queue_depth', 100)
        self.node_load[node_id] = min(1.0, queue_depth / max_queue)
    
    def _start_health_monitoring(self):
        """Start health monitoring thread."""
        def health_monitor():
            while True:
                try:
                    for node_id in list(self.nodes.keys()):
                        health = self._check_node_health(node_id)
                        self.node_health[node_id] = health
                    
                    time.sleep(self.health_check_interval)
                    
                except Exception as e:
                    self.logger.error(f"Health monitoring error: {e}")
                    time.sleep(self.health_check_interval)
        
        monitor_thread = threading.Thread(target=health_monitor, daemon=True)
        monitor_thread.start()
    
    def _check_node_health(self, node_id: str) -> float:
        """Check health of a specific node."""
        if node_id not in self.nodes:
            return 0.0
        
        node_config = self.nodes[node_id]
        
        try:
            # Simple TCP connection test
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1.0)
            result = sock.connect_ex((node_config.address, node_config.port))
            sock.close()
            
            if result == 0:
                # Additional health factors
                metrics = self.node_metrics[node_id]
                
                # Penalize high error rate
                health = 1.0 - min(0.5, metrics['error_rate'])
                
                # Penalize high latency
                if metrics['avg_latency_ms'] > 100:
                    health *= 0.8
                
                return health
            else:
                return 0.0  # Connection failed
                
        except Exception:
            return 0.0
    
    def get_load_balancing_stats(self) -> Dict[str, Any]:
        """Get load balancing statistics."""
        return {
            'total_nodes': len(self.nodes),
            'healthy_nodes': sum(1 for h in self.node_health.values() if h > 0.5),
            'average_load': np.mean(list(self.node_load.values())) if self.node_load else 0,
            'node_health': self.node_health.copy(),
            'node_load': self.node_load.copy(),
            'node_metrics': {k: v.copy() for k, v in self.node_metrics.items()},
        }


class NeuromorphicProcessingNode:
    """
    Individual processing node in the distributed system.
    
    Features:
    - Asynchronous job processing
    - Local caching and optimization
    - Health monitoring and reporting
    - Security and authentication
    """
    
    def __init__(
        self,
        node_config: NodeConfig,
        fusion_models: Dict[str, CrossModalFusion],
        max_concurrent_jobs: int = 10,
    ):
        """
        Initialize processing node.
        
        Args:
            node_config: Node configuration
            fusion_models: Dictionary of fusion models by type
            max_concurrent_jobs: Maximum concurrent jobs
        """
        self.config = node_config
        self.fusion_models = fusion_models
        self.max_concurrent_jobs = max_concurrent_jobs
        
        # Job processing
        self.job_queue = queue.PriorityQueue(maxsize=100)
        self.active_jobs = {}
        self.job_results = {}
        
        # Performance tracking
        self.processing_stats = {
            'jobs_processed': 0,
            'jobs_failed': 0,
            'total_processing_time': 0.0,
            'average_latency_ms': 0.0,
            'queue_depth': 0,
        }
        
        # Security
        if node_config.security_config:
            self.security_manager = NeuromorphicSecurityManager(
                node_config.security_config
            )
        else:
            self.security_manager = None
        
        # Error handling
        self.error_handler = RobustErrorHandler()
        
        self.logger = logging.getLogger(f"{__name__}.{node_config.node_id}")
        
        # Start processing threads
        self.processing_active = True
        self._start_job_processing()
    
    def _start_job_processing(self):
        """Start job processing threads."""
        for i in range(min(self.max_concurrent_jobs, mp.cpu_count())):
            worker_thread = threading.Thread(
                target=self._job_processing_worker,
                args=(f"worker_{i}",),
                daemon=True
            )
            worker_thread.start()
    
    def submit_job(self, job: ProcessingJob) -> bool:
        """
        Submit job for processing.
        
        Args:
            job: Processing job to submit
            
        Returns:
            True if job was accepted, False otherwise
        """
        try:
            # Security validation if enabled
            if self.security_manager:
                is_valid, validation_report = self.security_manager.validate_modality_data(
                    job.modality_data
                )
                if not is_valid:
                    self.logger.warning(f"Job {job.job_id} failed security validation")
                    return False
            
            # Check queue capacity
            if self.job_queue.qsize() >= self.job_queue.maxsize:
                self.logger.warning(f"Job queue full, rejecting job {job.job_id}")
                return False
            
            # Add to queue with priority
            priority_score = (-job.priority, job.created_at)  # Higher priority first, then FIFO
            self.job_queue.put((priority_score, job))
            
            self.processing_stats['queue_depth'] = self.job_queue.qsize()
            
            self.logger.debug(f"Accepted job {job.job_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to submit job {job.job_id}: {e}")
            return False
    
    def get_job_result(self, job_id: str) -> Optional[ProcessingResult]:
        """Get result for completed job."""
        return self.job_results.get(job_id)
    
    def _job_processing_worker(self, worker_id: str):
        """Job processing worker thread."""
        while self.processing_active:
            try:
                # Get job from queue (with timeout)
                try:
                    priority_score, job = self.job_queue.get(timeout=1.0)
                except queue.Empty:
                    continue
                
                self.processing_stats['queue_depth'] = self.job_queue.qsize()
                
                # Process job
                result = self._process_job(job, worker_id)
                
                # Store result
                self.job_results[job.job_id] = result
                
                # Call callback if provided
                if job.callback:
                    try:
                        job.callback(result)
                    except Exception as e:
                        self.logger.error(f"Job callback failed for {job.job_id}: {e}")
                
                # Update statistics
                self.processing_stats['jobs_processed'] += 1
                if result.fusion_result is None:
                    self.processing_stats['jobs_failed'] += 1
                
                self.processing_stats['total_processing_time'] += result.processing_time_ms
                self.processing_stats['average_latency_ms'] = (
                    self.processing_stats['total_processing_time'] / 
                    max(1, self.processing_stats['jobs_processed'])
                )
                
                # Cleanup old results
                self._cleanup_old_results()
                
                self.job_queue.task_done()
                
            except Exception as e:
                self.logger.error(f"Worker {worker_id} error: {e}")
                self.processing_stats['jobs_failed'] += 1
    
    @robust_function(critical_path=True)
    def _process_job(self, job: ProcessingJob, worker_id: str) -> ProcessingResult:
        """Process individual job."""
        start_time = time.time()
        queue_time_ms = (start_time - job.created_at) * 1000
        
        self.active_jobs[job.job_id] = {
            'job': job,
            'worker': worker_id,
            'start_time': start_time,
        }
        
        try:
            # Select appropriate model
            model_type = job.model_config.get('type', 'tsa')
            if model_type not in self.fusion_models:
                raise ValueError(f"Model type {model_type} not available on this node")
            
            model = self.fusion_models[model_type]
            
            # Process fusion
            fusion_result = model.fuse_modalities(job.modality_data)
            
            processing_time_ms = (time.time() - start_time) * 1000
            
            result = ProcessingResult(
                job_id=job.job_id,
                fusion_result=fusion_result,
                processing_node=self.config.node_id,
                processing_time_ms=processing_time_ms,
                queue_time_ms=queue_time_ms,
                network_latency_ms=0.0,  # Set by coordinator
                metadata={
                    'worker_id': worker_id,
                    'model_type': model_type,
                    'node_type': self.config.node_type.value,
                }
            )
            
            self.logger.debug(f"Processed job {job.job_id} in {processing_time_ms:.2f}ms")
            
        except Exception as e:
            processing_time_ms = (time.time() - start_time) * 1000
            
            self.error_handler.handle_error(e, f"_process_job_{job.job_id}")
            
            result = ProcessingResult(
                job_id=job.job_id,
                fusion_result=None,  # Indicate failure
                processing_node=self.config.node_id,
                processing_time_ms=processing_time_ms,
                queue_time_ms=queue_time_ms,
                network_latency_ms=0.0,
                metadata={
                    'error': str(e),
                    'worker_id': worker_id,
                }
            )
            
            self.logger.error(f"Failed to process job {job.job_id}: {e}")
        
        finally:
            if job.job_id in self.active_jobs:
                del self.active_jobs[job.job_id]
        
        return result
    
    def _cleanup_old_results(self):
        """Clean up old job results."""
        current_time = time.time()
        cutoff_time = current_time - 3600  # Keep results for 1 hour
        
        old_jobs = [
            job_id for job_id, result in self.job_results.items()
            if hasattr(result, 'created_at') and result.created_at < cutoff_time
        ]
        
        for job_id in old_jobs:
            del self.job_results[job_id]
    
    def get_node_status(self) -> Dict[str, Any]:
        """Get current node status."""
        return {
            'node_id': self.config.node_id,
            'node_type': self.config.node_type.value,
            'processing_active': self.processing_active,
            'queue_depth': self.job_queue.qsize(),
            'active_jobs': len(self.active_jobs),
            'available_models': list(self.fusion_models.keys()),
            'processing_stats': self.processing_stats.copy(),
            'resource_usage': self._get_resource_usage(),
        }
    
    def _get_resource_usage(self) -> Dict[str, float]:
        """Get current resource usage."""
        try:
            import psutil
            process = psutil.Process()
            
            return {
                'cpu_percent': process.cpu_percent(),
                'memory_percent': process.memory_percent(),
                'memory_mb': process.memory_info().rss / (1024**2),
                'threads': process.num_threads(),
            }
        except ImportError:
            return {'cpu_percent': 0, 'memory_percent': 0}
    
    def shutdown(self):
        """Shutdown processing node."""
        self.processing_active = False
        
        # Wait for active jobs to complete
        timeout = 30  # seconds
        start_time = time.time()
        
        while self.active_jobs and (time.time() - start_time) < timeout:
            time.sleep(0.1)
        
        self.logger.info(f"Node {self.config.node_id} shutdown complete")


class DistributedNeuromorphicCoordinator:
    """
    Central coordinator for distributed neuromorphic processing.
    
    Features:
    - Job scheduling and distribution
    - Load balancing and optimization
    - Health monitoring and recovery
    - Performance analytics
    """
    
    def __init__(self, processing_mode: ProcessingMode = ProcessingMode.HYBRID):
        """
        Initialize distributed coordinator.
        
        Args:
            processing_mode: Processing mode for the system
        """
        self.processing_mode = processing_mode
        self.load_balancer = DistributedLoadBalancer()
        
        # Job management
        self.pending_jobs = {}
        self.completed_jobs = {}
        self.job_callbacks = {}
        
        # System state
        self.system_stats = {
            'total_jobs': 0,
            'completed_jobs': 0,
            'failed_jobs': 0,
            'average_latency_ms': 0.0,
            'throughput_ops_per_sec': 0.0,
        }
        
        # Networking
        self.nodes: Dict[str, NeuromorphicProcessingNode] = {}
        self.node_connections = {}
        
        self.logger = logging.getLogger(__name__)
        
        # Start monitoring
        self._start_system_monitoring()
    
    def register_node(self, node: NeuromorphicProcessingNode):
        """Register a processing node."""
        self.nodes[node.config.node_id] = node
        self.load_balancer.register_node(node.config)
        
        self.logger.info(f"Registered processing node {node.config.node_id}")
    
    def unregister_node(self, node_id: str):
        """Unregister a processing node."""
        if node_id in self.nodes:
            del self.nodes[node_id]
            self.load_balancer.unregister_node(node_id)
            
            self.logger.info(f"Unregistered processing node {node_id}")
    
    async def submit_job_async(
        self,
        modality_data: Dict[str, ModalityData],
        model_config: Dict[str, Any],
        priority: int = 1,
        max_latency_ms: float = 100.0,
        callback: Optional[Callable] = None,
    ) -> str:
        """
        Submit job for asynchronous processing.
        
        Args:
            modality_data: Multi-modal spike data
            model_config: Model configuration
            priority: Job priority (higher = more important)
            max_latency_ms: Maximum acceptable latency
            callback: Optional completion callback
            
        Returns:
            Job ID
        """
        job_id = str(uuid.uuid4())
        
        job = ProcessingJob(
            job_id=job_id,
            modality_data=modality_data,
            model_config=model_config,
            priority=priority,
            max_latency_ms=max_latency_ms,
            callback=callback,
        )
        
        # Select processing node
        selected_node = self.load_balancer.select_node(job)
        
        if not selected_node:
            raise RuntimeError("No suitable processing node available")
        
        # Submit to node
        node = self.nodes[selected_node]
        success = node.submit_job(job)
        
        if not success:
            # Try alternative nodes
            exclude_nodes = [selected_node]
            
            for _ in range(3):  # Try up to 3 alternative nodes
                alternative_node = self.load_balancer.select_node(job, exclude_nodes)
                if not alternative_node:
                    break
                
                node = self.nodes[alternative_node]
                success = node.submit_job(job)
                
                if success:
                    selected_node = alternative_node
                    break
                else:
                    exclude_nodes.append(alternative_node)
        
        if not success:
            raise RuntimeError("Failed to submit job to any available node")
        
        # Track job
        self.pending_jobs[job_id] = {
            'job': job,
            'assigned_node': selected_node,
            'submit_time': time.time(),
        }
        
        if callback:
            self.job_callbacks[job_id] = callback
        
        self.system_stats['total_jobs'] += 1
        
        self.logger.debug(f"Submitted job {job_id} to node {selected_node}")
        
        return job_id
    
    def submit_job(self, *args, **kwargs) -> str:
        """Synchronous wrapper for job submission."""
        import asyncio
        
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        return loop.run_until_complete(self.submit_job_async(*args, **kwargs))
    
    def get_job_result(self, job_id: str) -> Optional[ProcessingResult]:
        """Get result for completed job."""
        if job_id in self.completed_jobs:
            return self.completed_jobs[job_id]
        
        # Check if job is still pending
        if job_id in self.pending_jobs:
            job_info = self.pending_jobs[job_id]
            node = self.nodes[job_info['assigned_node']]
            
            result = node.get_job_result(job_id)
            if result:
                # Job completed, move to completed jobs
                self.completed_jobs[job_id] = result
                del self.pending_jobs[job_id]
                
                # Update statistics
                self._update_system_stats(result)
                
                # Update load balancer metrics
                self.load_balancer.update_node_metrics(
                    result.processing_node,
                    result.processing_time_ms,
                    1000.0 / max(0.1, result.processing_time_ms),  # Rough throughput estimate
                    result.fusion_result is None,  # Error occurred
                    node.job_queue.qsize(),
                )
                
                return result
        
        return None
    
    async def wait_for_job(self, job_id: str, timeout: float = 30.0) -> Optional[ProcessingResult]:
        """
        Wait for job completion with timeout.
        
        Args:
            job_id: Job ID to wait for
            timeout: Maximum wait time in seconds
            
        Returns:
            Processing result or None if timeout
        """
        start_time = time.time()
        
        while (time.time() - start_time) < timeout:
            result = self.get_job_result(job_id)
            if result:
                return result
            
            await asyncio.sleep(0.1)  # Check every 100ms
        
        return None  # Timeout
    
    def _update_system_stats(self, result: ProcessingResult):
        """Update system-wide statistics."""
        if result.fusion_result is not None:
            self.system_stats['completed_jobs'] += 1
        else:
            self.system_stats['failed_jobs'] += 1
        
        # Update running averages
        total_jobs = self.system_stats['completed_jobs'] + self.system_stats['failed_jobs']
        
        if total_jobs > 0:
            # Exponential moving average for latency
            alpha = 0.1
            self.system_stats['average_latency_ms'] = (
                (1 - alpha) * self.system_stats['average_latency_ms'] +
                alpha * result.processing_time_ms
            )
            
            # Update throughput
            completed_jobs = self.system_stats['completed_jobs']
            if completed_jobs > 0:
                # Rough throughput estimate
                self.system_stats['throughput_ops_per_sec'] = 1000.0 / max(
                    0.1, self.system_stats['average_latency_ms']
                )
    
    def _start_system_monitoring(self):
        """Start system monitoring thread."""
        def system_monitor():
            while True:
                try:
                    self._monitor_system_health()
                    time.sleep(10.0)  # Monitor every 10 seconds
                except Exception as e:
                    self.logger.error(f"System monitoring error: {e}")
        
        monitor_thread = threading.Thread(target=system_monitor, daemon=True)
        monitor_thread.start()
    
    def _monitor_system_health(self):
        """Monitor overall system health."""
        # Check for stalled jobs
        current_time = time.time()
        stalled_threshold = 300.0  # 5 minutes
        
        stalled_jobs = []
        for job_id, job_info in self.pending_jobs.items():
            if (current_time - job_info['submit_time']) > stalled_threshold:
                stalled_jobs.append(job_id)
        
        if stalled_jobs:
            self.logger.warning(f"Found {len(stalled_jobs)} stalled jobs")
            
            # Try to recover stalled jobs
            for job_id in stalled_jobs:
                self._recover_stalled_job(job_id)
    
    def _recover_stalled_job(self, job_id: str):
        """Attempt to recover a stalled job."""
        if job_id not in self.pending_jobs:
            return
        
        job_info = self.pending_jobs[job_id]
        current_node = job_info['assigned_node']
        
        self.logger.info(f"Attempting to recover stalled job {job_id} from node {current_node}")
        
        # Try to resubmit to a different node
        job = job_info['job']
        alternative_node = self.load_balancer.select_node(job, exclude_nodes=[current_node])
        
        if alternative_node:
            node = self.nodes[alternative_node]
            success = node.submit_job(job)
            
            if success:
                job_info['assigned_node'] = alternative_node
                job_info['submit_time'] = time.time()  # Reset submit time
                self.logger.info(f"Recovered job {job_id} to node {alternative_node}")
            else:
                self.logger.error(f"Failed to recover job {job_id}")
        else:
            self.logger.error(f"No alternative node available for job {job_id}")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        node_statuses = {}
        for node_id, node in self.nodes.items():
            node_statuses[node_id] = node.get_node_status()
        
        return {
            'coordinator_id': id(self),
            'processing_mode': self.processing_mode.value,
            'system_stats': self.system_stats.copy(),
            'pending_jobs': len(self.pending_jobs),
            'completed_jobs': len(self.completed_jobs),
            'registered_nodes': len(self.nodes),
            'load_balancer_stats': self.load_balancer.get_load_balancing_stats(),
            'node_statuses': node_statuses,
        }
    
    def shutdown(self):
        """Shutdown distributed system."""
        self.logger.info("Shutting down distributed coordinator...")
        
        # Shutdown all nodes
        for node in self.nodes.values():
            node.shutdown()
        
        self.logger.info("Distributed coordinator shutdown complete")


# Factory functions for easy deployment
def create_edge_node(
    node_id: str,
    address: str,
    port: int,
    modalities: List[str],
) -> NeuromorphicProcessingNode:
    """Create an edge processing node."""
    from ..algorithms.temporal_spike_attention import create_temporal_spike_attention
    
    node_config = NodeConfig(
        node_id=node_id,
        node_type=NodeType.EDGE,
        address=address,
        port=port,
        capabilities={'modalities': modalities},
        resource_limits={'max_queue_depth': 50},  # Lower queue for edge
        priority=2,  # Higher priority for edge nodes
    )
    
    # Create lightweight model for edge
    fusion_models = {
        'tsa': create_temporal_spike_attention(
            modalities,
            config={'temporal_window': 50.0, 'enable_predictive': False}  # Simplified for edge
        )
    }
    
    return NeuromorphicProcessingNode(
        node_config=node_config,
        fusion_models=fusion_models,
        max_concurrent_jobs=5,  # Lower concurrency for edge
    )


def create_compute_node(
    node_id: str,
    address: str,
    port: int,
    modalities: List[str],
    optimization_config: Optional[OptimizationConfig] = None,
) -> NeuromorphicProcessingNode:
    """Create a high-performance compute node."""
    from ..algorithms.temporal_spike_attention import create_temporal_spike_attention
    from ..optimization.advanced_performance_optimizer import create_optimized_fusion_model
    
    node_config = NodeConfig(
        node_id=node_id,
        node_type=NodeType.COMPUTE,
        address=address,
        port=port,
        capabilities={'modalities': modalities, 'optimization': True},
        resource_limits={'max_queue_depth': 200},
        priority=1,  # Standard priority
    )
    
    # Create optimized models
    fusion_models = {}
    
    if optimization_config:
        tsa_model, _ = create_optimized_fusion_model(
            'tsa', modalities, optimization_config
        )
        fusion_models['tsa'] = tsa_model
    else:
        fusion_models['tsa'] = create_temporal_spike_attention(modalities)
    
    return NeuromorphicProcessingNode(
        node_config=node_config,
        fusion_models=fusion_models,
        max_concurrent_jobs=20,  # High concurrency for compute nodes
    )


# Export key components
__all__ = [
    'NodeType',
    'ProcessingMode', 
    'NodeConfig',
    'ProcessingJob',
    'ProcessingResult',
    'DistributedLoadBalancer',
    'NeuromorphicProcessingNode',
    'DistributedNeuromorphicCoordinator',
    'create_edge_node',
    'create_compute_node',
]