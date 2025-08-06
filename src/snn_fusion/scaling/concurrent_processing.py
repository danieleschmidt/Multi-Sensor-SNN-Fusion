"""
Concurrent Processing for SNN-Fusion

This module provides advanced concurrent processing capabilities
for spiking neural networks, including parallel spike processing,
distributed training, and asynchronous inference.
"""

import asyncio
import time
import threading
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple, Callable, Union, AsyncIterator
import logging
from dataclasses import dataclass
from enum import Enum
import queue
import numpy as np
from pathlib import Path
import pickle
import contextlib


class ProcessingMode(Enum):
    """Processing execution modes."""
    SEQUENTIAL = "sequential"
    THREADED = "threaded"
    MULTIPROCESS = "multiprocess"
    ASYNC = "async"
    HYBRID = "hybrid"


class TaskPriority(Enum):
    """Task priority levels."""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class ProcessingConfig:
    """Configuration for concurrent processing."""
    mode: ProcessingMode = ProcessingMode.HYBRID
    max_threads: Optional[int] = None
    max_processes: Optional[int] = None
    batch_size: int = 32
    queue_size: int = 1000
    enable_profiling: bool = True
    memory_limit_mb: float = 2000.0
    
    def __post_init__(self):
        """Initialize default values."""
        if self.max_threads is None:
            self.max_threads = min(mp.cpu_count() * 2, 16)
        if self.max_processes is None:
            self.max_processes = min(mp.cpu_count(), 8)


@dataclass
class ProcessingTask:
    """A task for concurrent processing."""
    task_id: str
    function: Callable
    args: Tuple = ()
    kwargs: Dict[str, Any] = None
    priority: TaskPriority = TaskPriority.NORMAL
    timeout: float = 30.0
    dependencies: List[str] = None
    
    def __post_init__(self):
        """Initialize defaults."""
        if self.kwargs is None:
            self.kwargs = {}
        if self.dependencies is None:
            self.dependencies = []
    
    @property
    def priority_value(self) -> int:
        """Get numeric priority value (higher = more important)."""
        return self.priority.value


class ConcurrentProcessor:
    """
    Advanced concurrent processor for SNN operations.
    
    Supports multiple execution modes, task dependencies,
    priority queuing, and resource-aware scheduling.
    """
    
    def __init__(self, config: ProcessingConfig):
        """Initialize concurrent processor."""
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Execution contexts
        self.thread_pool: Optional[ThreadPoolExecutor] = None
        self.process_pool: Optional[ProcessPoolExecutor] = None
        
        # Task management
        self.task_queue: queue.PriorityQueue = queue.PriorityQueue(maxsize=config.queue_size)
        self.dependency_graph: Dict[str, List[str]] = {}
        self.completed_tasks: Dict[str, Any] = {}
        self.failed_tasks: Dict[str, Exception] = {}
        self.active_tasks: Dict[str, asyncio.Task] = {}
        
        # Statistics
        self.total_tasks = 0
        self.completed_count = 0
        self.failed_count = 0
        self.start_time = time.time()
        
        # Synchronization
        self.lock = threading.RLock()
        self.shutdown_event = threading.Event()
        
        # Initialize execution contexts
        self._initialize_executors()
        
        self.logger.info(f"ConcurrentProcessor initialized with {config.mode.value} mode")
    
    def _initialize_executors(self):
        """Initialize thread and process pools."""
        if self.config.mode in [ProcessingMode.THREADED, ProcessingMode.HYBRID]:
            self.thread_pool = ThreadPoolExecutor(
                max_workers=self.config.max_threads,
                thread_name_prefix="SNN-Thread"
            )
        
        if self.config.mode in [ProcessingMode.MULTIPROCESS, ProcessingMode.HYBRID]:
            self.process_pool = ProcessPoolExecutor(
                max_workers=self.config.max_processes
            )
    
    def submit_task(self, task: ProcessingTask) -> str:
        """Submit a task for concurrent processing."""
        with self.lock:
            # Add task to priority queue
            priority = -task.priority_value  # Negate for correct ordering (higher priority first)
            self.task_queue.put((priority, time.time(), task))
            
            # Track dependencies
            if task.dependencies:
                self.dependency_graph[task.task_id] = task.dependencies.copy()
            
            self.total_tasks += 1
            self.logger.debug(f"Submitted task {task.task_id} with priority {task.priority.name}")
            
            return task.task_id
    
    def get_result(self, task_id: str, timeout: float = None) -> Any:
        """Get result of a completed task."""
        start_time = time.time()
        timeout = timeout or 30.0
        
        while time.time() - start_time < timeout:
            with self.lock:
                if task_id in self.completed_tasks:
                    return self.completed_tasks.pop(task_id)
                if task_id in self.failed_tasks:
                    raise self.failed_tasks.pop(task_id)
            
            time.sleep(0.1)
        
        raise TimeoutError(f"Task {task_id} did not complete within {timeout}s")
    
    async def process_batch_async(
        self, 
        spike_data: List[np.ndarray],
        processing_func: Callable[[np.ndarray], np.ndarray],
        batch_size: Optional[int] = None
    ) -> List[np.ndarray]:
        """Process spike data batches asynchronously."""
        batch_size = batch_size or self.config.batch_size
        results = []
        
        # Create batches
        batches = [
            spike_data[i:i + batch_size]
            for i in range(0, len(spike_data), batch_size)
        ]
        
        # Process batches concurrently
        tasks = []
        for i, batch in enumerate(batches):
            task = asyncio.create_task(
                self._process_batch_worker(batch, processing_func, f"batch_{i}")
            )
            tasks.append(task)
        
        # Wait for all batches to complete
        batch_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Flatten results
        for batch_result in batch_results:
            if isinstance(batch_result, Exception):
                self.logger.error(f"Batch processing failed: {batch_result}")
                continue
            results.extend(batch_result)
        
        return results
    
    async def _process_batch_worker(
        self,
        batch: List[np.ndarray],
        processing_func: Callable,
        batch_id: str
    ) -> List[np.ndarray]:
        """Worker function for processing a batch."""
        try:
            if self.config.mode == ProcessingMode.ASYNC:
                # Pure async processing
                results = []
                for item in batch:
                    if asyncio.iscoroutinefunction(processing_func):
                        result = await processing_func(item)
                    else:
                        result = processing_func(item)
                    results.append(result)
                return results
            
            elif self.config.mode in [ProcessingMode.THREADED, ProcessingMode.HYBRID]:
                # Use thread pool for CPU-bound operations
                loop = asyncio.get_event_loop()
                with ThreadPoolExecutor(max_workers=4) as executor:
                    futures = [
                        loop.run_in_executor(executor, processing_func, item)
                        for item in batch
                    ]
                    results = await asyncio.gather(*futures)
                return results
            
            else:
                # Sequential fallback
                return [processing_func(item) for item in batch]
                
        except Exception as e:
            self.logger.error(f"Batch {batch_id} processing failed: {e}")
            raise
    
    def process_spike_trains_parallel(
        self,
        spike_trains: List[np.ndarray],
        processing_func: Callable[[np.ndarray], np.ndarray],
        chunk_size: Optional[int] = None
    ) -> List[np.ndarray]:
        """Process spike trains in parallel using multiple processes."""
        if not self.process_pool:
            # Fallback to sequential processing
            return [processing_func(train) for train in spike_trains]
        
        chunk_size = chunk_size or max(1, len(spike_trains) // self.config.max_processes)
        
        # Split spike trains into chunks
        chunks = [
            spike_trains[i:i + chunk_size]
            for i in range(0, len(spike_trains), chunk_size)
        ]
        
        # Submit chunks to process pool
        futures = []
        for chunk in chunks:
            future = self.process_pool.submit(
                self._process_spike_chunk,
                chunk,
                processing_func
            )
            futures.append(future)
        
        # Collect results
        results = []
        for future in as_completed(futures):
            try:
                chunk_results = future.result(timeout=60.0)
                results.extend(chunk_results)
            except Exception as e:
                self.logger.error(f"Spike train processing failed: {e}")
                raise
        
        return results
    
    @staticmethod
    def _process_spike_chunk(
        spike_chunk: List[np.ndarray],
        processing_func: Callable
    ) -> List[np.ndarray]:
        """Process a chunk of spike trains (static method for multiprocessing)."""
        return [processing_func(train) for train in spike_chunk]
    
    def execute_pipeline(self, tasks: List[ProcessingTask]) -> Dict[str, Any]:
        """Execute a pipeline of dependent tasks."""
        # Build dependency graph
        dependency_map = {}
        for task in tasks:
            dependency_map[task.task_id] = task.dependencies.copy()
        
        # Topological sort to determine execution order
        execution_order = self._topological_sort(dependency_map)
        
        results = {}
        
        for task_id in execution_order:
            # Find the task object
            task = next((t for t in tasks if t.task_id == task_id), None)
            if not task:
                continue
            
            try:
                # Check if dependencies are satisfied
                missing_deps = [
                    dep for dep in task.dependencies
                    if dep not in results
                ]
                
                if missing_deps:
                    raise RuntimeError(f"Missing dependencies for {task_id}: {missing_deps}")
                
                # Execute task
                start_time = time.time()
                
                # Inject dependency results into kwargs
                if task.dependencies:
                    dependency_results = {
                        f"dep_{dep}": results[dep]
                        for dep in task.dependencies
                    }
                    task.kwargs.update(dependency_results)
                
                # Execute based on processing mode
                if self.config.mode == ProcessingMode.THREADED and self.thread_pool:
                    future = self.thread_pool.submit(task.function, *task.args, **task.kwargs)
                    result = future.result(timeout=task.timeout)
                elif self.config.mode == ProcessingMode.MULTIPROCESS and self.process_pool:
                    future = self.process_pool.submit(task.function, *task.args, **task.kwargs)
                    result = future.result(timeout=task.timeout)
                else:
                    result = task.function(*task.args, **task.kwargs)
                
                execution_time = time.time() - start_time
                results[task_id] = result
                self.completed_count += 1
                
                self.logger.debug(f"Task {task_id} completed in {execution_time:.3f}s")
                
            except Exception as e:
                self.logger.error(f"Task {task_id} failed: {e}")
                results[task_id] = e
                self.failed_count += 1
                
                # Check if this failure should stop the pipeline
                if task.priority == TaskPriority.CRITICAL:
                    raise RuntimeError(f"Critical task {task_id} failed: {e}")
        
        return results
    
    def _topological_sort(self, dependency_map: Dict[str, List[str]]) -> List[str]:
        """Perform topological sort on task dependencies."""
        # Build adjacency list and in-degree count
        graph = {task_id: [] for task_id in dependency_map}
        in_degree = {task_id: 0 for task_id in dependency_map}
        
        for task_id, deps in dependency_map.items():
            for dep in deps:
                if dep in graph:
                    graph[dep].append(task_id)
                    in_degree[task_id] += 1
        
        # Kahn's algorithm
        queue = [task_id for task_id, degree in in_degree.items() if degree == 0]
        result = []
        
        while queue:
            current = queue.pop(0)
            result.append(current)
            
            for neighbor in graph[current]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
        
        # Check for cycles
        if len(result) != len(dependency_map):
            raise RuntimeError("Circular dependency detected in task pipeline")
        
        return result
    
    def process_multi_modal_batch(
        self,
        modal_data: Dict[str, List[np.ndarray]],
        modal_processors: Dict[str, Callable]
    ) -> Dict[str, List[Any]]:
        """Process multi-modal data batches concurrently."""
        results = {}
        futures = {}
        
        # Submit processing for each modality
        for modality, data_list in modal_data.items():
            if modality not in modal_processors:
                self.logger.warning(f"No processor for modality {modality}")
                continue
            
            processor = modal_processors[modality]
            
            if self.config.mode in [ProcessingMode.THREADED, ProcessingMode.HYBRID] and self.thread_pool:
                future = self.thread_pool.submit(
                    self._process_modality_batch,
                    data_list,
                    processor,
                    modality
                )
                futures[modality] = future
            else:
                # Sequential processing
                results[modality] = self._process_modality_batch(data_list, processor, modality)
        
        # Collect results from futures
        for modality, future in futures.items():
            try:
                results[modality] = future.result(timeout=60.0)
            except Exception as e:
                self.logger.error(f"Modality {modality} processing failed: {e}")
                results[modality] = e
        
        return results
    
    def _process_modality_batch(
        self,
        data_list: List[np.ndarray],
        processor: Callable,
        modality: str
    ) -> List[Any]:
        """Process a batch of data for a specific modality."""
        try:
            return [processor(data) for data in data_list]
        except Exception as e:
            self.logger.error(f"Processing failed for modality {modality}: {e}")
            raise
    
    async def stream_process(
        self,
        data_stream: AsyncIterator[np.ndarray],
        processing_func: Callable,
        buffer_size: int = 100
    ) -> AsyncIterator[Any]:
        """Process streaming data with concurrent processing."""
        buffer = []
        
        async for data_item in data_stream:
            buffer.append(data_item)
            
            # Process buffer when full
            if len(buffer) >= buffer_size:
                # Process buffer concurrently
                results = await self.process_batch_async(
                    buffer,
                    processing_func,
                    batch_size=self.config.batch_size
                )
                
                # Yield results
                for result in results:
                    yield result
                
                # Clear buffer
                buffer.clear()
        
        # Process remaining items in buffer
        if buffer:
            results = await self.process_batch_async(
                buffer,
                processing_func,
                batch_size=self.config.batch_size
            )
            for result in results:
                yield result
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get processing performance statistics."""
        uptime = time.time() - self.start_time
        
        return {
            'mode': self.config.mode.value,
            'uptime_seconds': uptime,
            'total_tasks': self.total_tasks,
            'completed_tasks': self.completed_count,
            'failed_tasks': self.failed_count,
            'success_rate': self.completed_count / max(self.total_tasks, 1) * 100,
            'tasks_per_second': self.completed_count / max(uptime, 1),
            'active_tasks': len(self.active_tasks),
            'queue_size': self.task_queue.qsize(),
            'thread_pool_active': self.thread_pool is not None,
            'process_pool_active': self.process_pool is not None
        }
    
    def shutdown(self, wait: bool = True):
        """Shutdown the concurrent processor."""
        self.shutdown_event.set()
        
        if self.thread_pool:
            self.thread_pool.shutdown(wait=wait)
        
        if self.process_pool:
            self.process_pool.shutdown(wait=wait)
        
        self.logger.info("ConcurrentProcessor shutdown completed")


class SpikeProcessingPipeline:
    """
    High-level pipeline for concurrent spike processing.
    
    Provides specialized methods for common SNN operations
    with optimized concurrent execution.
    """
    
    def __init__(self, processor: ConcurrentProcessor):
        """Initialize spike processing pipeline."""
        self.processor = processor
        self.logger = logging.getLogger(__name__)
    
    async def encode_spike_trains(
        self,
        input_data: List[np.ndarray],
        encoder_configs: Dict[str, Any]
    ) -> List[np.ndarray]:
        """Encode input data into spike trains concurrently."""
        def spike_encoder(data: np.ndarray) -> np.ndarray:
            # Simple Poisson encoding example
            rate_factor = encoder_configs.get('rate_factor', 100.0)
            time_steps = encoder_configs.get('time_steps', 100)
            
            # Normalize data to [0, 1]
            normalized = (data - np.min(data)) / (np.max(data) - np.min(data) + 1e-8)
            
            # Generate Poisson spike trains
            rates = normalized * rate_factor
            spikes = np.random.poisson(rates[..., np.newaxis] * 0.01, 
                                     size=(*rates.shape, time_steps))
            
            return (spikes > 0).astype(np.float32)
        
        return await self.processor.process_batch_async(
            input_data,
            spike_encoder
        )
    
    def process_liquid_state_machine(
        self,
        spike_inputs: List[np.ndarray],
        reservoir_config: Dict[str, Any]
    ) -> List[np.ndarray]:
        """Process spike inputs through LSM reservoirs concurrently."""
        def lsm_processor(spikes: np.ndarray) -> np.ndarray:
            # Simplified LSM processing
            n_neurons = reservoir_config.get('n_neurons', 1000)
            leak_rate = reservoir_config.get('leak_rate', 0.95)
            
            # Initialize reservoir state
            state = np.zeros(n_neurons)
            outputs = []
            
            for t in range(spikes.shape[-1]):  # Time dimension
                # Update reservoir state
                input_current = np.dot(spikes[..., t], 
                                     np.random.randn(spikes.shape[0], n_neurons) * 0.1)
                state = state * leak_rate + input_current
                
                # Apply activation (simplified)
                activated = np.tanh(state)
                outputs.append(activated.copy())
            
            return np.stack(outputs, axis=-1)
        
        return self.processor.process_spike_trains_parallel(
            spike_inputs,
            lsm_processor
        )
    
    def apply_stdp_learning(
        self,
        pre_spikes: List[np.ndarray],
        post_spikes: List[np.ndarray],
        stdp_config: Dict[str, Any]
    ) -> List[np.ndarray]:
        """Apply STDP learning rule concurrently."""
        def stdp_updater(spike_pair: Tuple[np.ndarray, np.ndarray]) -> np.ndarray:
            pre, post = spike_pair
            
            # STDP parameters
            tau_plus = stdp_config.get('tau_plus', 20.0)
            tau_minus = stdp_config.get('tau_minus', 20.0)
            a_plus = stdp_config.get('a_plus', 0.1)
            a_minus = stdp_config.get('a_minus', 0.12)
            
            # Calculate weight changes
            weight_changes = np.zeros((pre.shape[0], post.shape[0]))
            
            for t in range(min(pre.shape[-1], post.shape[-1])):
                pre_active = pre[..., t] > 0
                post_active = post[..., t] > 0
                
                if np.any(pre_active) and np.any(post_active):
                    # Simplified STDP calculation
                    weight_changes += np.outer(pre_active.astype(float), 
                                             post_active.astype(float)) * a_plus
            
            return weight_changes
        
        # Pair pre and post spikes
        spike_pairs = list(zip(pre_spikes, post_spikes))
        
        return self.processor.process_spike_trains_parallel(
            spike_pairs,
            stdp_updater
        )


# Example usage and testing
if __name__ == "__main__":
    print("Testing Concurrent Processing...")
    
    # Create processing configuration
    config = ProcessingConfig(
        mode=ProcessingMode.HYBRID,
        max_threads=8,
        max_processes=4,
        batch_size=16
    )
    
    # Initialize processor
    processor = ConcurrentProcessor(config)
    
    # Test 1: Batch processing
    print("\n1. Testing batch processing...")
    test_data = [np.random.randn(100, 64) for _ in range(50)]
    
    def dummy_processor(data: np.ndarray) -> np.ndarray:
        # Simulate processing time
        time.sleep(0.01)
        return data * 2 + np.random.randn(*data.shape) * 0.1
    
    start_time = time.time()
    results = asyncio.run(processor.process_batch_async(
        test_data[:10],  # Process subset for testing
        dummy_processor,
        batch_size=4
    ))
    processing_time = time.time() - start_time
    
    print(f"Processed {len(results)} batches in {processing_time:.3f}s")
    
    # Test 2: Pipeline execution
    print("\n2. Testing pipeline execution...")
    
    def task_a(x: int) -> int:
        time.sleep(0.1)
        return x * 2
    
    def task_b(y: int, dep_a: int) -> int:
        time.sleep(0.1)
        return y + dep_a
    
    def task_c(dep_a: int, dep_b: int) -> int:
        time.sleep(0.1)
        return dep_a * dep_b
    
    pipeline_tasks = [
        ProcessingTask("task_a", task_a, args=(5,), priority=TaskPriority.HIGH),
        ProcessingTask("task_b", task_b, args=(3,), dependencies=["task_a"], priority=TaskPriority.NORMAL),
        ProcessingTask("task_c", task_c, dependencies=["task_a", "task_b"], priority=TaskPriority.LOW)
    ]
    
    start_time = time.time()
    pipeline_results = processor.execute_pipeline(pipeline_tasks)
    pipeline_time = time.time() - start_time
    
    print(f"Pipeline completed in {pipeline_time:.3f}s")
    print(f"Results: {pipeline_results}")
    
    # Test 3: Multi-modal processing
    print("\n3. Testing multi-modal processing...")
    
    modal_data = {
        'audio': [np.random.randn(100, 64) for _ in range(10)],
        'events': [np.random.randn(100, 128) for _ in range(10)],
        'imu': [np.random.randn(100, 6) for _ in range(10)]
    }
    
    modal_processors = {
        'audio': lambda x: np.mean(x, axis=0),
        'events': lambda x: np.max(x, axis=0),
        'imu': lambda x: np.std(x, axis=0)
    }
    
    start_time = time.time()
    modal_results = processor.process_multi_modal_batch(modal_data, modal_processors)
    modal_time = time.time() - start_time
    
    print(f"Multi-modal processing completed in {modal_time:.3f}s")
    print(f"Processed modalities: {list(modal_results.keys())}")
    
    # Test 4: Spike processing pipeline
    print("\n4. Testing spike processing pipeline...")
    
    spike_pipeline = SpikeProcessingPipeline(processor)
    
    # Generate test input data
    input_data = [np.random.randn(50, 32) for _ in range(20)]
    encoder_config = {'rate_factor': 50.0, 'time_steps': 50}
    
    start_time = time.time()
    encoded_spikes = asyncio.run(spike_pipeline.encode_spike_trains(
        input_data[:5],  # Process subset for testing
        encoder_config
    ))
    encoding_time = time.time() - start_time
    
    print(f"Encoded {len(encoded_spikes)} spike trains in {encoding_time:.3f}s")
    print(f"Spike train shape: {encoded_spikes[0].shape}")
    
    # Get performance statistics
    stats = processor.get_performance_stats()
    print(f"\nPerformance Statistics:")
    print(f"  Processing mode: {stats['mode']}")
    print(f"  Total tasks: {stats['total_tasks']}")
    print(f"  Success rate: {stats['success_rate']:.1f}%")
    print(f"  Tasks per second: {stats['tasks_per_second']:.2f}")
    
    # Shutdown processor
    processor.shutdown()
    
    print("✓ Concurrent processing test completed!")