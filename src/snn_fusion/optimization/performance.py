"""
Performance Optimization for SNN-Fusion

This module provides comprehensive performance optimization strategies
for spiking neural networks, including memory management, computation
optimization, and resource utilization improvements.
"""

import time
import threading
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Callable, Union
import logging
from dataclasses import dataclass
from enum import Enum
import psutil
import gc
from pathlib import Path
import pickle
import functools


class OptimizationLevel(Enum):
    """Performance optimization levels."""
    NONE = "none"
    BASIC = "basic"
    AGGRESSIVE = "aggressive"
    MAXIMUM = "maximum"


@dataclass
class PerformanceMetrics:
    """Performance metrics tracking."""
    execution_time: float
    memory_usage_mb: float
    cpu_utilization: float
    cache_hit_rate: float
    throughput: float
    latency_p95: float
    operations_per_second: float
    
    def to_dict(self) -> Dict[str, float]:
        return {
            'execution_time': self.execution_time,
            'memory_usage_mb': self.memory_usage_mb,
            'cpu_utilization': self.cpu_utilization,
            'cache_hit_rate': self.cache_hit_rate,
            'throughput': self.throughput,
            'latency_p95': self.latency_p95,
            'operations_per_second': self.operations_per_second
        }


class PerformanceProfiler:
    """
    Comprehensive performance profiler for SNN operations.
    
    Tracks execution time, memory usage, CPU utilization,
    and other performance metrics across the system.
    """
    
    def __init__(self, enable_detailed_profiling: bool = True):
        """Initialize performance profiler."""
        self.enable_detailed_profiling = enable_detailed_profiling
        self.logger = logging.getLogger(__name__)
        
        # Metrics storage
        self.metrics_history: List[Tuple[str, PerformanceMetrics]] = []
        self.operation_counts: Dict[str, int] = {}
        self.timing_stats: Dict[str, List[float]] = {}
        
        # Profiling state
        self.active_profiles: Dict[str, Dict[str, Any]] = {}
        
    def start_profiling(self, operation_name: str) -> str:
        """Start profiling an operation."""
        profile_id = f"{operation_name}_{int(time.time()*1000)}"
        
        self.active_profiles[profile_id] = {
            'operation': operation_name,
            'start_time': time.time(),
            'start_memory': self._get_memory_usage(),
            'start_cpu': self._get_cpu_usage()
        }
        
        return profile_id
    
    def end_profiling(self, profile_id: str) -> PerformanceMetrics:
        """End profiling and calculate metrics."""
        if profile_id not in self.active_profiles:
            raise ValueError(f"Profile {profile_id} not found")
        
        profile = self.active_profiles[profile_id]
        
        # Calculate metrics
        execution_time = time.time() - profile['start_time']
        memory_usage = self._get_memory_usage() - profile['start_memory']
        cpu_utilization = self._get_cpu_usage() - profile['start_cpu']
        
        metrics = PerformanceMetrics(
            execution_time=execution_time,
            memory_usage_mb=max(0, memory_usage),
            cpu_utilization=max(0, cpu_utilization),
            cache_hit_rate=0.0,  # To be updated by cache systems
            throughput=0.0,      # To be calculated based on operations
            latency_p95=0.0,     # To be calculated from timing history
            operations_per_second=1.0 / execution_time if execution_time > 0 else 0
        )
        
        # Store metrics
        operation_name = profile['operation']
        self.metrics_history.append((operation_name, metrics))
        self.operation_counts[operation_name] = self.operation_counts.get(operation_name, 0) + 1
        
        # Update timing statistics
        if operation_name not in self.timing_stats:
            self.timing_stats[operation_name] = []
        self.timing_stats[operation_name].append(execution_time)
        
        # Cleanup
        del self.active_profiles[profile_id]
        
        return metrics
    
    def get_operation_stats(self, operation_name: str) -> Dict[str, Any]:
        """Get statistics for a specific operation."""
        if operation_name not in self.timing_stats:
            return {}
        
        timings = self.timing_stats[operation_name]
        
        return {
            'count': len(timings),
            'avg_time': np.mean(timings),
            'min_time': np.min(timings),
            'max_time': np.max(timings),
            'p95_time': np.percentile(timings, 95),
            'p99_time': np.percentile(timings, 99),
            'total_time': np.sum(timings)
        }
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        return psutil.Process().memory_info().rss / (1024 * 1024)
    
    def _get_cpu_usage(self) -> float:
        """Get current CPU usage percentage."""
        return psutil.cpu_percent(interval=0.1)


def profile_performance(operation_name: str, profiler: Optional[PerformanceProfiler] = None):
    """Decorator for automatic performance profiling."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            _profiler = profiler or PerformanceProfiler()
            
            profile_id = _profiler.start_profiling(operation_name)
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                metrics = _profiler.end_profiling(profile_id)
                if hasattr(func, '_last_metrics'):
                    func._last_metrics = metrics
        
        return wrapper
    return decorator


class MemoryOptimizer:
    """
    Memory optimization strategies for SNN processing.
    
    Implements memory pooling, garbage collection optimization,
    and memory-efficient data structures.
    """
    
    def __init__(self, optimization_level: OptimizationLevel = OptimizationLevel.BASIC):
        """Initialize memory optimizer."""
        self.optimization_level = optimization_level
        self.logger = logging.getLogger(__name__)
        
        # Memory pools
        self.memory_pools: Dict[str, List[np.ndarray]] = {}
        self.pool_locks: Dict[str, threading.Lock] = {}
        
        # Memory tracking
        self.allocated_memory = 0
        self.max_memory_limit = self._get_memory_limit()
        
        # Optimization settings based on level
        self._configure_optimization()
        
    def _configure_optimization(self):
        """Configure optimization based on level."""
        if self.optimization_level == OptimizationLevel.BASIC:
            self.enable_pooling = True
            self.gc_threshold = 1000
            self.pool_size_limit = 100
        elif self.optimization_level == OptimizationLevel.AGGRESSIVE:
            self.enable_pooling = True
            self.gc_threshold = 500
            self.pool_size_limit = 200
        elif self.optimization_level == OptimizationLevel.MAXIMUM:
            self.enable_pooling = True
            self.gc_threshold = 100
            self.pool_size_limit = 500
        else:
            self.enable_pooling = False
            self.gc_threshold = 10000
            self.pool_size_limit = 10
    
    def allocate_array(self, shape: Tuple[int, ...], dtype: np.dtype = np.float32) -> np.ndarray:
        """Allocate array from memory pool if available."""
        if not self.enable_pooling:
            return np.empty(shape, dtype=dtype)
        
        pool_key = f"{shape}_{dtype}"
        
        # Create pool if doesn't exist
        if pool_key not in self.memory_pools:
            self.memory_pools[pool_key] = []
            self.pool_locks[pool_key] = threading.Lock()
        
        # Try to get from pool
        with self.pool_locks[pool_key]:
            pool = self.memory_pools[pool_key]
            if pool:
                array = pool.pop()
                array.fill(0)  # Reset array
                return array
        
        # Create new array if pool is empty
        return np.empty(shape, dtype=dtype)
    
    def deallocate_array(self, array: np.ndarray):
        """Return array to memory pool."""
        if not self.enable_pooling:
            return
        
        pool_key = f"{array.shape}_{array.dtype}"
        
        if pool_key in self.memory_pools:
            with self.pool_locks[pool_key]:
                pool = self.memory_pools[pool_key]
                if len(pool) < self.pool_size_limit:
                    pool.append(array)
                    return
        
        # Let garbage collector handle it
        del array
    
    def optimize_garbage_collection(self):
        """Optimize garbage collection settings."""
        if self.optimization_level in [OptimizationLevel.AGGRESSIVE, OptimizationLevel.MAXIMUM]:
            gc.set_threshold(self.gc_threshold, self.gc_threshold // 10, self.gc_threshold // 100)
            gc.collect()
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get current memory statistics."""
        process = psutil.Process()
        memory_info = process.memory_info()
        
        return {
            'rss_mb': memory_info.rss / (1024 * 1024),
            'vms_mb': memory_info.vms / (1024 * 1024),
            'memory_percent': process.memory_percent(),
            'pool_sizes': {k: len(v) for k, v in self.memory_pools.items()},
            'allocated_pools': len(self.memory_pools)
        }
    
    def _get_memory_limit(self) -> int:
        """Get system memory limit."""
        return psutil.virtual_memory().total


class ComputeOptimizer:
    """
    Computation optimization for SNN operations.
    
    Implements vectorization, parallelization, and
    algorithm-specific optimizations.
    """
    
    def __init__(self, max_workers: Optional[int] = None):
        """Initialize compute optimizer."""
        self.max_workers = max_workers or min(mp.cpu_count(), 8)
        self.logger = logging.getLogger(__name__)
        
        # Thread pools for different types of operations
        self.cpu_pool = ThreadPoolExecutor(max_workers=self.max_workers)
        self.io_pool = ThreadPoolExecutor(max_workers=self.max_workers * 2)
        
    def parallelize_computation(
        self, 
        func: Callable, 
        data_chunks: List[Any],
        use_processes: bool = False
    ) -> List[Any]:
        """Parallelize computation across multiple workers."""
        if len(data_chunks) == 1:
            return [func(data_chunks[0])]
        
        if use_processes:
            with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                results = list(executor.map(func, data_chunks))
        else:
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                results = list(executor.map(func, data_chunks))
        
        return results
    
    def optimize_spike_processing(self, spike_data: np.ndarray) -> np.ndarray:
        """Optimize spike processing using vectorized operations."""
        # Use vectorized operations instead of loops
        if len(spike_data.shape) == 2:
            # Batch processing
            return self._vectorized_spike_ops(spike_data)
        else:
            return spike_data
    
    def _vectorized_spike_ops(self, spikes: np.ndarray) -> np.ndarray:
        """Apply vectorized operations to spike data."""
        # Example optimizations
        # 1. Use np.where instead of loops
        # 2. Use broadcasting for batch operations
        # 3. Minimize array copies
        
        # Spike rate calculation (vectorized)
        spike_rates = np.mean(spikes, axis=1, keepdims=True)
        
        # Normalized spikes
        normalized_spikes = spikes / (spike_rates + 1e-8)
        
        return normalized_spikes
    
    def batch_process(
        self, 
        data: List[Any], 
        process_func: Callable,
        batch_size: int = 32
    ) -> List[Any]:
        """Process data in optimized batches."""
        results = []
        
        for i in range(0, len(data), batch_size):
            batch = data[i:i + batch_size]
            batch_result = process_func(batch)
            results.extend(batch_result if isinstance(batch_result, list) else [batch_result])
        
        return results


class CacheManager:
    """
    Intelligent caching system for SNN-Fusion operations.
    
    Implements LRU caching, memory-aware caching, and
    operation-specific cache strategies.
    """
    
    def __init__(self, max_memory_mb: float = 1000.0):
        """Initialize cache manager."""
        self.max_memory_bytes = int(max_memory_mb * 1024 * 1024)
        self.logger = logging.getLogger(__name__)
        
        # Cache storage
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.cache_stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'total_size_bytes': 0
        }
        
        # LRU tracking
        self.access_order: List[str] = []
        self.cache_lock = threading.RLock()
    
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache."""
        with self.cache_lock:
            if key in self.cache:
                # Update access order (LRU)
                self.access_order.remove(key)
                self.access_order.append(key)
                
                self.cache_stats['hits'] += 1
                return self.cache[key]['data']
            else:
                self.cache_stats['misses'] += 1
                return None
    
    def put(self, key: str, data: Any, size_bytes: Optional[int] = None):
        """Put item in cache."""
        if size_bytes is None:
            size_bytes = self._estimate_size(data)
        
        with self.cache_lock:
            # Remove existing entry if present
            if key in self.cache:
                self._remove_from_cache(key)
            
            # Ensure we have space
            while (self.cache_stats['total_size_bytes'] + size_bytes > self.max_memory_bytes 
                   and self.access_order):
                self._evict_lru()
            
            # Add to cache
            self.cache[key] = {
                'data': data,
                'size': size_bytes,
                'timestamp': time.time()
            }
            self.access_order.append(key)
            self.cache_stats['total_size_bytes'] += size_bytes
    
    def clear(self):
        """Clear entire cache."""
        with self.cache_lock:
            self.cache.clear()
            self.access_order.clear()
            self.cache_stats['total_size_bytes'] = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self.cache_lock:
            total_requests = self.cache_stats['hits'] + self.cache_stats['misses']
            hit_rate = self.cache_stats['hits'] / max(total_requests, 1)
            
            return {
                'hit_rate': hit_rate,
                'total_requests': total_requests,
                'cache_size_mb': self.cache_stats['total_size_bytes'] / (1024 * 1024),
                'cached_items': len(self.cache),
                'evictions': self.cache_stats['evictions']
            }
    
    def _evict_lru(self):
        """Evict least recently used item."""
        if self.access_order:
            lru_key = self.access_order[0]
            self._remove_from_cache(lru_key)
            self.cache_stats['evictions'] += 1
    
    def _remove_from_cache(self, key: str):
        """Remove item from cache."""
        if key in self.cache:
            self.cache_stats['total_size_bytes'] -= self.cache[key]['size']
            del self.cache[key]
            if key in self.access_order:
                self.access_order.remove(key)
    
    def _estimate_size(self, data: Any) -> int:
        """Estimate memory size of data."""
        if isinstance(data, np.ndarray):
            return data.nbytes
        elif isinstance(data, (list, tuple)):
            return sum(self._estimate_size(item) for item in data)
        elif isinstance(data, dict):
            return sum(self._estimate_size(v) + len(str(k)) for k, v in data.items())
        else:
            try:
                return len(pickle.dumps(data))
            except:
                return 1024  # Default estimate


def cached_operation(cache_manager: CacheManager, expire_after: Optional[int] = None):
    """Decorator for caching operation results."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Create cache key from function name and arguments
            cache_key = f"{func.__name__}_{hash(str(args) + str(sorted(kwargs.items())))}"
            
            # Try to get from cache
            result = cache_manager.get(cache_key)
            if result is not None:
                return result
            
            # Execute function and cache result
            result = func(*args, **kwargs)
            cache_manager.put(cache_key, result)
            
            return result
        
        return wrapper
    return decorator


class PerformanceOptimizer:
    """
    Main performance optimizer that coordinates all optimization strategies.
    
    Combines memory optimization, compute optimization, and caching
    for maximum SNN-Fusion performance.
    """
    
    def __init__(
        self,
        optimization_level: OptimizationLevel = OptimizationLevel.BASIC,
        max_memory_mb: float = 2000.0,
        max_workers: Optional[int] = None
    ):
        """Initialize performance optimizer."""
        self.optimization_level = optimization_level
        self.logger = logging.getLogger(__name__)
        
        # Initialize sub-optimizers
        self.memory_optimizer = MemoryOptimizer(optimization_level)
        self.compute_optimizer = ComputeOptimizer(max_workers)
        self.cache_manager = CacheManager(max_memory_mb)
        self.profiler = PerformanceProfiler()
        
        # Performance tracking
        self.optimization_history: List[Dict[str, Any]] = []
        
        self.logger.info(f"PerformanceOptimizer initialized with {optimization_level.value} level")
    
    def optimize_spike_encoding(self, data: np.ndarray, encoder_func: Callable) -> np.ndarray:
        """Optimize spike encoding operation."""
        profile_id = self.profiler.start_profiling("spike_encoding")
        
        try:
            # Check cache first
            cache_key = f"encoding_{hash(data.tobytes())}"
            cached_result = self.cache_manager.get(cache_key)
            if cached_result is not None:
                return cached_result
            
            # Optimize memory allocation
            result_shape = data.shape  # Assume same shape for encoding
            result_array = self.memory_optimizer.allocate_array(result_shape, np.float32)
            
            # Apply compute optimizations
            if len(data) > 1000:  # Large dataset
                # Process in parallel chunks
                chunk_size = len(data) // self.compute_optimizer.max_workers
                chunks = [data[i:i+chunk_size] for i in range(0, len(data), chunk_size)]
                results = self.compute_optimizer.parallelize_computation(encoder_func, chunks)
                result = np.concatenate(results)
            else:
                result = encoder_func(data)
            
            # Cache result
            self.cache_manager.put(cache_key, result)
            
            return result
            
        finally:
            metrics = self.profiler.end_profiling(profile_id)
            self._track_optimization("spike_encoding", metrics)
    
    def optimize_training_batch(self, batch_data: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize training batch processing."""
        profile_id = self.profiler.start_profiling("training_batch")
        
        try:
            optimized_batch = {}
            
            for modality, data in batch_data.items():
                if isinstance(data, np.ndarray):
                    # Apply compute optimizations
                    optimized_data = self.compute_optimizer.optimize_spike_processing(data)
                    optimized_batch[modality] = optimized_data
                else:
                    optimized_batch[modality] = data
            
            return optimized_batch
            
        finally:
            metrics = self.profiler.end_profiling(profile_id)
            self._track_optimization("training_batch", metrics)
    
    def get_optimization_report(self) -> Dict[str, Any]:
        """Get comprehensive optimization report."""
        memory_stats = self.memory_optimizer.get_memory_stats()
        cache_stats = self.cache_manager.get_stats()
        
        # Calculate overall performance improvements
        if self.optimization_history:
            recent_ops = self.optimization_history[-100:]  # Last 100 operations
            avg_execution_time = np.mean([op['metrics']['execution_time'] for op in recent_ops])
            avg_memory_usage = np.mean([op['metrics']['memory_usage_mb'] for op in recent_ops])
            avg_throughput = np.mean([op['metrics']['operations_per_second'] for op in recent_ops])
        else:
            avg_execution_time = 0
            avg_memory_usage = 0
            avg_throughput = 0
        
        return {
            'optimization_level': self.optimization_level.value,
            'memory_stats': memory_stats,
            'cache_stats': cache_stats,
            'performance_metrics': {
                'avg_execution_time': avg_execution_time,
                'avg_memory_usage_mb': avg_memory_usage,
                'avg_throughput_ops': avg_throughput
            },
            'total_optimizations': len(self.optimization_history),
            'profiler_stats': {
                name: self.profiler.get_operation_stats(name)
                for name in self.profiler.operation_counts.keys()
            }
        }
    
    def cleanup_resources(self):
        """Cleanup optimization resources."""
        self.cache_manager.clear()
        self.memory_optimizer.optimize_garbage_collection()
        self.compute_optimizer.cpu_pool.shutdown(wait=False)
        self.compute_optimizer.io_pool.shutdown(wait=False)
    
    def _track_optimization(self, operation: str, metrics: PerformanceMetrics):
        """Track optimization metrics."""
        self.optimization_history.append({
            'operation': operation,
            'timestamp': time.time(),
            'metrics': metrics.to_dict()
        })
        
        # Keep history bounded
        if len(self.optimization_history) > 10000:
            self.optimization_history = self.optimization_history[-5000:]


# Example usage and testing
if __name__ == "__main__":
    print("Testing Performance Optimization...")
    
    # Create performance optimizer
    optimizer = PerformanceOptimizer(
        optimization_level=OptimizationLevel.AGGRESSIVE,
        max_memory_mb=500
    )
    
    # Test spike encoding optimization
    print("\nTesting spike encoding optimization...")
    test_data = np.random.randn(1000, 64)
    
    def dummy_encoder(data):
        return data * 2 + np.random.randn(*data.shape) * 0.1
    
    # Test with optimization
    start_time = time.time()
    optimized_result = optimizer.optimize_spike_encoding(test_data, dummy_encoder)
    optimized_time = time.time() - start_time
    
    print(f"Optimized encoding took {optimized_time:.4f} seconds")
    print(f"Result shape: {optimized_result.shape}")
    
    # Test caching (second call should be faster)
    start_time = time.time()
    cached_result = optimizer.optimize_spike_encoding(test_data, dummy_encoder)
    cached_time = time.time() - start_time
    
    print(f"Cached encoding took {cached_time:.4f} seconds")
    print(f"Speedup: {optimized_time/cached_time:.2f}x")
    
    # Test batch optimization
    print("\nTesting batch optimization...")
    batch_data = {
        'audio': np.random.randn(32, 100, 64),
        'events': np.random.randn(32, 100, 128),
        'labels': np.random.randint(0, 10, 32)
    }
    
    optimized_batch = optimizer.optimize_training_batch(batch_data)
    print(f"Batch optimization completed for {len(optimized_batch)} modalities")
    
    # Get performance report
    report = optimizer.get_optimization_report()
    print(f"\nOptimization Report:")
    print(f"  Cache hit rate: {report['cache_stats']['hit_rate']:.2%}")
    print(f"  Memory usage: {report['memory_stats']['rss_mb']:.1f} MB")
    print(f"  Total optimizations: {report['total_optimizations']}")
    
    # Cleanup
    optimizer.cleanup_resources()
    
    print("✓ Performance optimization test completed!")