"""
Advanced Performance Optimization for Neuromorphic Systems

Implements comprehensive performance optimization including memory management,
computational acceleration, and resource pooling for scalable SNN deployment.
"""

import time
import threading
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, Future
from typing import Any, Dict, List, Optional, Union, Callable, Tuple
import queue
import gc
import weakref
from dataclasses import dataclass
from enum import Enum
import hashlib
import pickle
import sys
import warnings


class OptimizationLevel(Enum):
    """Performance optimization levels."""
    CONSERVATIVE = "conservative"
    BALANCED = "balanced"
    AGGRESSIVE = "aggressive"
    MAXIMUM = "maximum"


class ResourceType(Enum):
    """Types of computational resources."""
    CPU = "cpu"
    MEMORY = "memory"
    GPU = "gpu"
    STORAGE = "storage"
    NETWORK = "network"


@dataclass
class PerformanceMetrics:
    """Performance measurement data."""
    operation_name: str
    duration: float
    memory_usage: int
    cpu_usage: float
    throughput: float
    timestamp: float
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class OptimizationResult:
    """Result of optimization operation."""
    success: bool
    performance_gain: float
    memory_saved: int
    optimization_applied: List[str]
    metrics_before: PerformanceMetrics
    metrics_after: PerformanceMetrics
    recommendations: List[str]


class MemoryOptimizer:
    """
    Advanced memory management and optimization.
    """
    
    def __init__(self, max_cache_size: int = 1024 * 1024 * 1024):  # 1GB default
        self.max_cache_size = max_cache_size
        self.cache = {}
        self.cache_access_times = {}
        self.cache_sizes = {}
        self.total_cache_size = 0
        self.lock = threading.Lock()
        
        # Memory pool for efficient allocation
        self.memory_pools = {}
        self.pool_lock = threading.Lock()
        
        # Weak reference tracking for automatic cleanup
        self.tracked_objects = weakref.WeakSet()
        
    def cache_result(self, key: str, value: Any, size_hint: Optional[int] = None) -> bool:
        """
        Cache a computation result with intelligent eviction.
        
        Args:
            key: Cache key
            value: Value to cache
            size_hint: Estimated size in bytes
            
        Returns:
            True if successfully cached
        """
        with self.lock:
            # Estimate size if not provided
            if size_hint is None:
                try:
                    size_hint = sys.getsizeof(value)
                    if hasattr(value, '__dict__'):
                        size_hint += sys.getsizeof(value.__dict__)
                except Exception:
                    size_hint = 1024  # Default estimate
            
            # Check if we need to evict
            if self.total_cache_size + size_hint > self.max_cache_size:
                self._evict_lru_items(size_hint)
            
            # Store in cache
            self.cache[key] = value
            self.cache_access_times[key] = time.time()
            self.cache_sizes[key] = size_hint
            self.total_cache_size += size_hint
            
            return True
    
    def get_cached(self, key: str) -> Optional[Any]:
        """Get cached result if available."""
        with self.lock:
            if key in self.cache:
                self.cache_access_times[key] = time.time()
                return self.cache[key]
            return None
    
    def _evict_lru_items(self, space_needed: int):
        """Evict least recently used items to free space."""
        # Sort by access time (oldest first)
        items_by_age = sorted(
            self.cache_access_times.items(),
            key=lambda x: x[1]
        )
        
        space_freed = 0
        for key, _ in items_by_age:
            if space_freed >= space_needed:
                break
            
            if key in self.cache:
                space_freed += self.cache_sizes[key]
                self.total_cache_size -= self.cache_sizes[key]
                
                del self.cache[key]
                del self.cache_access_times[key]
                del self.cache_sizes[key]
    
    def clear_cache(self):
        """Clear all cached items."""
        with self.lock:
            self.cache.clear()
            self.cache_access_times.clear()
            self.cache_sizes.clear()
            self.total_cache_size = 0
    
    def get_memory_pool(self, pool_name: str, item_size: int, 
                       initial_size: int = 100) -> 'MemoryPool':
        """Get or create a memory pool for efficient allocation."""
        with self.pool_lock:
            if pool_name not in self.memory_pools:
                self.memory_pools[pool_name] = MemoryPool(
                    item_size, initial_size
                )
            return self.memory_pools[pool_name]
    
    def track_object(self, obj: Any):
        """Track object for automatic cleanup."""
        self.tracked_objects.add(obj)
    
    def force_cleanup(self):
        """Force garbage collection and cleanup."""
        # Clear weak references
        self.tracked_objects.clear()
        
        # Run garbage collection
        collected = gc.collect()
        
        # Clear caches if memory pressure
        try:
            import psutil
            memory_percent = psutil.virtual_memory().percent
            if memory_percent > 80:  # High memory usage
                self.clear_cache()
                
        except ImportError:
            # Fallback: clear cache if too large
            if self.total_cache_size > self.max_cache_size * 0.8:
                self.clear_cache()
        
        return collected
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get comprehensive memory statistics."""
        with self.lock:
            return {
                'cache_items': len(self.cache),
                'cache_size_bytes': self.total_cache_size,
                'cache_size_mb': self.total_cache_size / (1024 * 1024),
                'max_cache_size_mb': self.max_cache_size / (1024 * 1024),
                'cache_utilization': self.total_cache_size / self.max_cache_size,
                'memory_pools': len(self.memory_pools),
                'tracked_objects': len(self.tracked_objects)
            }


class MemoryPool:
    """Efficient memory pool for frequent allocations."""
    
    def __init__(self, item_size: int, initial_size: int = 100):
        self.item_size = item_size
        self.available_items = queue.Queue()
        self.lock = threading.Lock()
        
        # Pre-allocate items
        for _ in range(initial_size):
            self.available_items.put(bytearray(item_size))
    
    def get_item(self) -> bytearray:
        """Get an item from the pool."""
        try:
            return self.available_items.get_nowait()
        except queue.Empty:
            # Create new item if pool is empty
            return bytearray(self.item_size)
    
    def return_item(self, item: bytearray):
        """Return an item to the pool."""
        if len(item) == self.item_size:
            # Reset the item
            item[:] = b'\x00' * self.item_size
            self.available_items.put(item)


class ComputationOptimizer:
    """
    Optimizes computational operations through various techniques.
    """
    
    def __init__(self, optimization_level: OptimizationLevel = OptimizationLevel.BALANCED):
        self.optimization_level = optimization_level
        self.compiled_functions = {}
        self.operation_cache = {}
        
        # Threading pools for different types of operations
        self.cpu_pool = ThreadPoolExecutor(
            max_workers=mp.cpu_count(),
            thread_name_prefix="cpu_worker"
        )
        
        self.io_pool = ThreadPoolExecutor(
            max_workers=min(32, mp.cpu_count() + 4),
            thread_name_prefix="io_worker"
        )
        
        # Process pool for CPU-intensive tasks
        try:
            self.process_pool = ProcessPoolExecutor(
                max_workers=max(1, mp.cpu_count() - 1)
            )
        except Exception:
            self.process_pool = None
            warnings.warn("Process pool unavailable, using thread pool only")
    
    def optimize_function(self, func: Callable, 
                         cache_results: bool = True,
                         use_multiprocessing: bool = False) -> Callable:
        """
        Optimize a function with caching and compilation.
        
        Args:
            func: Function to optimize
            cache_results: Whether to cache function results
            use_multiprocessing: Whether to use multiprocessing for parallelization
            
        Returns:
            Optimized function
        """
        func_name = f"{func.__module__}.{func.__name__}"
        
        if func_name in self.compiled_functions:
            return self.compiled_functions[func_name]
        
        def optimized_func(*args, **kwargs):
            # Generate cache key if caching enabled
            if cache_results:
                try:
                    cache_key = self._generate_cache_key(func_name, args, kwargs)
                    
                    # Check cache
                    if cache_key in self.operation_cache:
                        return self.operation_cache[cache_key]
                except Exception:
                    cache_key = None
            else:
                cache_key = None
            
            # Execute function
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                
                # Cache result if enabled
                if cache_key is not None:
                    self.operation_cache[cache_key] = result
                
                # Record performance
                duration = time.time() - start_time
                self._record_performance(func_name, duration, True)
                
                return result
                
            except Exception as e:
                duration = time.time() - start_time
                self._record_performance(func_name, duration, False)
                raise
        
        self.compiled_functions[func_name] = optimized_func
        return optimized_func
    
    def _generate_cache_key(self, func_name: str, args: tuple, kwargs: dict) -> str:
        """Generate deterministic cache key for function call."""
        # Create a simplified representation for hashing
        key_data = {
            'function': func_name,
            'args': str(args)[:200],  # Truncate to avoid huge keys
            'kwargs': str(sorted(kwargs.items()))[:200]
        }
        
        key_string = str(key_data)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def _record_performance(self, func_name: str, duration: float, success: bool):
        """Record function performance metrics."""
        # Store in a simple performance log
        if not hasattr(self, 'performance_log'):
            self.performance_log = []
        
        self.performance_log.append({
            'function': func_name,
            'duration': duration,
            'success': success,
            'timestamp': time.time()
        })
        
        # Keep only recent entries
        if len(self.performance_log) > 1000:
            self.performance_log = self.performance_log[-1000:]
    
    def parallel_map(self, func: Callable, items: List[Any], 
                    chunk_size: Optional[int] = None,
                    use_processes: bool = False) -> List[Any]:
        """
        Parallel map operation with automatic chunking.
        
        Args:
            func: Function to apply
            items: Items to process
            chunk_size: Size of chunks for processing
            use_processes: Whether to use process pool
            
        Returns:
            List of results
        """
        if not items:
            return []
        
        # Auto-determine chunk size
        if chunk_size is None:
            num_workers = mp.cpu_count()
            chunk_size = max(1, len(items) // (num_workers * 4))
        
        # Choose execution pool
        if use_processes and self.process_pool is not None:
            executor = self.process_pool
        else:
            executor = self.cpu_pool
        
        # Submit chunks
        futures = []
        for i in range(0, len(items), chunk_size):
            chunk = items[i:i + chunk_size]
            future = executor.submit(self._process_chunk, func, chunk)
            futures.append(future)
        
        # Collect results
        results = []
        for future in futures:
            chunk_results = future.result()
            results.extend(chunk_results)
        
        return results
    
    def _process_chunk(self, func: Callable, chunk: List[Any]) -> List[Any]:
        """Process a chunk of items."""
        return [func(item) for item in chunk]
    
    def async_execute(self, func: Callable, *args, 
                     io_bound: bool = False, **kwargs) -> Future:
        """
        Execute function asynchronously.
        
        Args:
            func: Function to execute
            *args: Function arguments
            io_bound: Whether operation is I/O bound
            **kwargs: Function keyword arguments
            
        Returns:
            Future object
        """
        if io_bound:
            return self.io_pool.submit(func, *args, **kwargs)
        else:
            return self.cpu_pool.submit(func, *args, **kwargs)
    
    def batch_execute(self, operations: List[Tuple[Callable, tuple, dict]],
                     max_concurrent: int = None) -> List[Any]:
        """
        Execute multiple operations in batches.
        
        Args:
            operations: List of (function, args, kwargs) tuples
            max_concurrent: Maximum concurrent operations
            
        Returns:
            List of results
        """
        if max_concurrent is None:
            max_concurrent = mp.cpu_count() * 2
        
        results = [None] * len(operations)
        futures = {}
        
        # Submit initial batch
        for i, (func, args, kwargs) in enumerate(operations[:max_concurrent]):
            future = self.cpu_pool.submit(func, *args, **kwargs)
            futures[future] = i
        
        # Process completions and submit remaining
        remaining_ops = operations[max_concurrent:]
        next_op_index = max_concurrent
        
        while futures:
            # Wait for first completion
            completed = None
            for future in list(futures.keys()):
                if future.done():
                    completed = future
                    break
            
            if completed is None:
                time.sleep(0.001)  # Short sleep to avoid busy waiting
                continue
            
            # Store result
            result_index = futures[completed]
            try:
                results[result_index] = completed.result()
            except Exception as e:
                results[result_index] = e
            
            del futures[completed]
            
            # Submit next operation if available
            if remaining_ops:
                func, args, kwargs = remaining_ops.pop(0)
                future = self.cpu_pool.submit(func, *args, **kwargs)
                futures[future] = next_op_index
                next_op_index += 1
        
        return results
    
    def get_optimization_stats(self) -> Dict[str, Any]:
        """Get optimization statistics."""
        stats = {
            'compiled_functions': len(self.compiled_functions),
            'cached_operations': len(self.operation_cache),
            'cpu_pool_workers': self.cpu_pool._max_workers,
            'io_pool_workers': self.io_pool._max_workers,
            'process_pool_available': self.process_pool is not None
        }
        
        if hasattr(self, 'performance_log'):
            total_operations = len(self.performance_log)
            successful_operations = sum(1 for op in self.performance_log if op['success'])
            
            stats.update({
                'total_operations': total_operations,
                'successful_operations': successful_operations,
                'success_rate': successful_operations / max(total_operations, 1),
                'avg_duration': sum(op['duration'] for op in self.performance_log) / max(total_operations, 1)
            })
        
        return stats
    
    def cleanup(self):
        """Clean up resources."""
        self.cpu_pool.shutdown(wait=True)
        self.io_pool.shutdown(wait=True)
        if self.process_pool:
            self.process_pool.shutdown(wait=True)


class ResourceManager:
    """
    Manages computational resources and load balancing.
    """
    
    def __init__(self):
        self.resource_usage = {resource: 0.0 for resource in ResourceType}
        self.resource_limits = {resource: 100.0 for resource in ResourceType}
        self.resource_monitors = {}
        self.lock = threading.Lock()
        
        # Start resource monitoring
        self._start_monitoring()
    
    def _start_monitoring(self):
        """Start background resource monitoring."""
        def monitor_resources():
            while True:
                try:
                    self._update_resource_usage()
                    time.sleep(1.0)  # Update every second
                except Exception:
                    pass  # Continue monitoring on errors
        
        monitor_thread = threading.Thread(target=monitor_resources, daemon=True)
        monitor_thread.start()
    
    def _update_resource_usage(self):
        """Update current resource usage."""
        try:
            import psutil
            
            with self.lock:
                # CPU usage
                self.resource_usage[ResourceType.CPU] = psutil.cpu_percent(interval=None)
                
                # Memory usage
                memory = psutil.virtual_memory()
                self.resource_usage[ResourceType.MEMORY] = memory.percent
                
                # Storage usage (for primary disk)
                disk = psutil.disk_usage('/')
                self.resource_usage[ResourceType.STORAGE] = (disk.used / disk.total) * 100
                
        except ImportError:
            # Fallback: use simple estimation
            pass
    
    def check_resource_availability(self, resource_type: ResourceType,
                                  required_amount: float) -> bool:
        """
        Check if sufficient resources are available.
        
        Args:
            resource_type: Type of resource to check
            required_amount: Required amount (percentage)
            
        Returns:
            True if resources are available
        """
        with self.lock:
            current_usage = self.resource_usage[resource_type]
            limit = self.resource_limits[resource_type]
            
            return (current_usage + required_amount) <= limit
    
    def reserve_resources(self, resource_requirements: Dict[ResourceType, float]) -> bool:
        """
        Reserve resources for an operation.
        
        Args:
            resource_requirements: Dictionary of resource requirements
            
        Returns:
            True if all resources successfully reserved
        """
        with self.lock:
            # Check if all resources are available
            for resource_type, amount in resource_requirements.items():
                if not self.check_resource_availability(resource_type, amount):
                    return False
            
            # Reserve all resources
            for resource_type, amount in resource_requirements.items():
                self.resource_usage[resource_type] += amount
            
            return True
    
    def release_resources(self, resource_requirements: Dict[ResourceType, float]):
        """Release previously reserved resources."""
        with self.lock:
            for resource_type, amount in resource_requirements.items():
                self.resource_usage[resource_type] = max(
                    0.0, self.resource_usage[resource_type] - amount
                )
    
    def get_resource_status(self) -> Dict[str, Any]:
        """Get current resource status."""
        with self.lock:
            return {
                'usage': {rt.value: usage for rt, usage in self.resource_usage.items()},
                'limits': {rt.value: limit for rt, limit in self.resource_limits.items()},
                'available': {
                    rt.value: max(0, limit - usage)
                    for rt, (usage, limit) in zip(
                        self.resource_usage.keys(),
                        zip(self.resource_usage.values(), self.resource_limits.values())
                    )
                }
            }


class PerformanceProfiler:
    """
    Comprehensive performance profiling and analysis.
    """
    
    def __init__(self):
        self.profiles = {}
        self.current_profiles = {}
        self.lock = threading.Lock()
    
    def start_profiling(self, operation_name: str) -> str:
        """Start profiling an operation."""
        profile_id = f"{operation_name}_{int(time.time() * 1000000)}"
        
        with self.lock:
            self.current_profiles[profile_id] = {
                'operation': operation_name,
                'start_time': time.time(),
                'start_memory': self._get_memory_usage()
            }
        
        return profile_id
    
    def end_profiling(self, profile_id: str) -> PerformanceMetrics:
        """End profiling and return metrics."""
        end_time = time.time()
        end_memory = self._get_memory_usage()
        
        with self.lock:
            if profile_id not in self.current_profiles:
                raise ValueError(f"Profile {profile_id} not found")
            
            profile_data = self.current_profiles[profile_id]
            del self.current_profiles[profile_id]
        
        # Calculate metrics
        duration = end_time - profile_data['start_time']
        memory_delta = end_memory - profile_data['start_memory']
        
        metrics = PerformanceMetrics(
            operation_name=profile_data['operation'],
            duration=duration,
            memory_usage=memory_delta,
            cpu_usage=self._get_cpu_usage(),
            throughput=1.0 / duration if duration > 0 else 0.0,
            timestamp=end_time
        )
        
        # Store metrics
        operation_name = profile_data['operation']
        if operation_name not in self.profiles:
            self.profiles[operation_name] = []
        
        self.profiles[operation_name].append(metrics)
        
        # Keep only recent profiles
        if len(self.profiles[operation_name]) > 1000:
            self.profiles[operation_name] = self.profiles[operation_name][-1000:]
        
        return metrics
    
    def _get_memory_usage(self) -> int:
        """Get current memory usage in bytes."""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss
        except ImportError:
            # Fallback: use simple estimation
            return 0
    
    def _get_cpu_usage(self) -> float:
        """Get current CPU usage percentage."""
        try:
            import psutil
            return psutil.cpu_percent()
        except ImportError:
            return 0.0
    
    def analyze_performance(self, operation_name: str) -> Dict[str, Any]:
        """Analyze performance for an operation."""
        with self.lock:
            if operation_name not in self.profiles:
                return {'error': 'No profiles found for operation'}
            
            metrics_list = self.profiles[operation_name]
        
        if not metrics_list:
            return {'error': 'No metrics available'}
        
        durations = [m.duration for m in metrics_list]
        memory_usage = [m.memory_usage for m in metrics_list]
        throughputs = [m.throughput for m in metrics_list]
        
        return {
            'operation': operation_name,
            'total_executions': len(metrics_list),
            'duration_stats': {
                'mean': sum(durations) / len(durations),
                'min': min(durations),
                'max': max(durations),
                'median': sorted(durations)[len(durations) // 2]
            },
            'memory_stats': {
                'mean': sum(memory_usage) / len(memory_usage),
                'min': min(memory_usage),
                'max': max(memory_usage)
            },
            'throughput_stats': {
                'mean': sum(throughputs) / len(throughputs),
                'min': min(throughputs),
                'max': max(throughputs)
            }
        }
    
    def profile_context(self, operation_name: str):
        """Context manager for profiling operations."""
        return ProfilingContext(self, operation_name)


class ProfilingContext:
    """Context manager for performance profiling."""
    
    def __init__(self, profiler: PerformanceProfiler, operation_name: str):
        self.profiler = profiler
        self.operation_name = operation_name
        self.profile_id = None
    
    def __enter__(self):
        self.profile_id = self.profiler.start_profiling(self.operation_name)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.profile_id:
            return self.profiler.end_profiling(self.profile_id)


# Global instances
default_memory_optimizer = MemoryOptimizer()
default_computation_optimizer = ComputationOptimizer()
default_resource_manager = ResourceManager()
default_profiler = PerformanceProfiler()


def optimize_function(func: Callable, **kwargs) -> Callable:
    """Convenience function for function optimization."""
    return default_computation_optimizer.optimize_function(func, **kwargs)


def profile_operation(operation_name: str):
    """Convenience function for operation profiling."""
    return default_profiler.profile_context(operation_name)


def cache_result(key: str, value: Any, **kwargs) -> bool:
    """Convenience function for result caching."""
    return default_memory_optimizer.cache_result(key, value, **kwargs)


def get_cached(key: str) -> Optional[Any]:
    """Convenience function for cache retrieval."""
    return default_memory_optimizer.get_cached(key)


# Example usage and testing
if __name__ == "__main__":
    print("Testing Performance Optimization System...")
    
    # Test memory optimization
    print("\n1. Testing Memory Optimization:")
    memory_opt = MemoryOptimizer(max_cache_size=1024 * 1024)  # 1MB cache
    
    # Cache some results
    memory_opt.cache_result("test_key_1", "test_value_1")
    memory_opt.cache_result("test_key_2", [1, 2, 3, 4, 5])
    
    # Retrieve cached results
    result1 = memory_opt.get_cached("test_key_1")
    result2 = memory_opt.get_cached("test_key_2")
    
    assert result1 == "test_value_1"
    assert result2 == [1, 2, 3, 4, 5]
    print("  ✓ Caching and retrieval working")
    
    # Test memory stats
    stats = memory_opt.get_memory_stats()
    print(f"  ✓ Memory stats: {stats['cache_items']} items, {stats['cache_size_mb']:.2f}MB")
    
    # Test computation optimization
    print("\n2. Testing Computation Optimization:")
    comp_opt = ComputationOptimizer()
    
    # Define a test function
    def expensive_computation(n):
        return sum(i * i for i in range(n))
    
    # Optimize the function
    optimized_func = comp_opt.optimize_function(expensive_computation, cache_results=True)
    
    # Test optimized function
    start_time = time.time()
    result1 = optimized_func(1000)
    first_duration = time.time() - start_time
    
    start_time = time.time()
    result2 = optimized_func(1000)  # Should be cached
    second_duration = time.time() - start_time
    
    assert result1 == result2
    assert second_duration < first_duration  # Cached version should be faster
    print(f"  ✓ Function optimization working (speedup: {first_duration/second_duration:.2f}x)")
    
    # Test parallel processing
    print("\n3. Testing Parallel Processing:")
    
    def square_number(x):
        return x * x
    
    numbers = list(range(100))
    
    # Sequential processing
    start_time = time.time()
    sequential_results = [square_number(x) for x in numbers]
    sequential_duration = time.time() - start_time
    
    # Parallel processing
    start_time = time.time()
    parallel_results = comp_opt.parallel_map(square_number, numbers)
    parallel_duration = time.time() - start_time
    
    assert sequential_results == parallel_results
    print(f"  ✓ Parallel processing working (speedup: {sequential_duration/parallel_duration:.2f}x)")
    
    # Test resource management
    print("\n4. Testing Resource Management:")
    resource_mgr = ResourceManager()
    
    # Check resource status
    status = resource_mgr.get_resource_status()
    print(f"  ✓ CPU usage: {status['usage']['cpu']:.1f}%")
    print(f"  ✓ Memory usage: {status['usage']['memory']:.1f}%")
    
    # Test resource reservation
    requirements = {ResourceType.CPU: 10.0, ResourceType.MEMORY: 5.0}
    reserved = resource_mgr.reserve_resources(requirements)
    print(f"  ✓ Resource reservation: {reserved}")
    
    if reserved:
        resource_mgr.release_resources(requirements)
        print("  ✓ Resource release completed")
    
    # Test performance profiling
    print("\n5. Testing Performance Profiling:")
    profiler = PerformanceProfiler()
    
    # Profile an operation
    with profiler.profile_context("test_operation") as profile:
        time.sleep(0.01)  # Simulate work
        result = sum(range(1000))
    
    # Analyze performance
    analysis = profiler.analyze_performance("test_operation")
    print(f"  ✓ Profiling working: {analysis['total_executions']} executions")
    print(f"  ✓ Average duration: {analysis['duration_stats']['mean']:.4f}s")
    
    # Test convenience functions
    print("\n6. Testing Convenience Functions:")
    
    @optimize_function
    def test_function(x):
        return x ** 2
    
    with profile_operation("convenience_test"):
        result = test_function(10)
    
    assert result == 100
    print("  ✓ Convenience functions working")
    
    # Get optimization statistics
    opt_stats = comp_opt.get_optimization_stats()
    print(f"  ✓ Optimization stats: {opt_stats['compiled_functions']} optimized functions")
    
    print("\n✓ Performance optimization test completed!")
    
    # Cleanup
    comp_opt.cleanup()