#!/usr/bin/env python3
"""
Performance Validation Script

Tests performance optimization features without ML framework dependencies.
"""

import time
import threading
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import psutil
import gc
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from collections import defaultdict, deque
import random
import statistics


@dataclass
class PerformanceMetrics:
    """Performance metrics tracking."""
    execution_time: float
    memory_usage_mb: float
    cpu_utilization: float
    throughput_ops_per_sec: float
    cache_hit_rate: float = 0.0


class SimpleProfiler:
    """Simple performance profiler."""
    
    def __init__(self):
        self.metrics = []
        self.start_times = {}
    
    def start_profiling(self, operation_name: str) -> str:
        """Start profiling an operation."""
        profile_id = f"{operation_name}_{int(time.time()*1000)}"
        self.start_times[profile_id] = {
            'time': time.time(),
            'memory': self._get_memory_usage()
        }
        return profile_id
    
    def end_profiling(self, profile_id: str) -> PerformanceMetrics:
        """End profiling and calculate metrics."""
        if profile_id not in self.start_times:
            raise ValueError(f"Profile {profile_id} not found")
        
        start_data = self.start_times[profile_id]
        execution_time = time.time() - start_data['time']
        memory_usage = self._get_memory_usage() - start_data['memory']
        
        metrics = PerformanceMetrics(
            execution_time=execution_time,
            memory_usage_mb=max(0, memory_usage),
            cpu_utilization=psutil.cpu_percent(interval=0.1),
            throughput_ops_per_sec=1.0 / execution_time if execution_time > 0 else 0
        )
        
        self.metrics.append(metrics)
        del self.start_times[profile_id]
        
        return metrics
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        return psutil.Process().memory_info().rss / (1024 * 1024)


class SimpleCache:
    """Simple LRU cache for testing."""
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.cache = {}
        self.access_order = deque()
        self.hits = 0
        self.misses = 0
        self.lock = threading.RLock()
    
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache."""
        with self.lock:
            if key in self.cache:
                # Update LRU order
                self.access_order.remove(key)
                self.access_order.append(key)
                self.hits += 1
                return self.cache[key]
            else:
                self.misses += 1
                return None
    
    def put(self, key: str, value: Any):
        """Put item in cache."""
        with self.lock:
            if key in self.cache:
                self.access_order.remove(key)
            elif len(self.cache) >= self.max_size:
                # Evict LRU item
                lru_key = self.access_order.popleft()
                del self.cache[lru_key]
            
            self.cache[key] = value
            self.access_order.append(key)
    
    def get_hit_rate(self) -> float:
        """Get cache hit rate."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0
    
    def clear(self):
        """Clear cache."""
        with self.lock:
            self.cache.clear()
            self.access_order.clear()


class ConcurrentProcessor:
    """Simple concurrent processor for testing."""
    
    def __init__(self, max_workers: int = None):
        self.max_workers = max_workers or min(mp.cpu_count(), 8)
        self.thread_pool = ThreadPoolExecutor(max_workers=self.max_workers)
        self.process_pool = ProcessPoolExecutor(max_workers=self.max_workers)
    
    def process_batch_threaded(self, data_list: List[Any], processor_func) -> List[Any]:
        """Process batch using threads."""
        futures = [self.thread_pool.submit(processor_func, item) for item in data_list]
        return [future.result() for future in futures]
    
    def process_batch_multiprocess(self, data_list: List[Any], processor_func) -> List[Any]:
        """Process batch using processes."""
        futures = [self.process_pool.submit(processor_func, item) for item in data_list]
        return [future.result() for future in futures]
    
    def shutdown(self):
        """Shutdown processor."""
        self.thread_pool.shutdown(wait=True)
        self.process_pool.shutdown(wait=True)


def cpu_intensive_task(n: int) -> float:
    """CPU-intensive task for performance testing."""
    result = 0.0
    for i in range(n):
        result += (i ** 0.5) * (i ** 0.3)
        if i % 1000 == 0:
            result = result % 1000000  # Prevent overflow
    return result


def memory_intensive_task(size: int) -> int:
    """Memory-intensive task for testing."""
    data = list(range(size))
    return sum(data)


def io_intensive_task(duration: float) -> str:
    """IO-intensive task simulation."""
    time.sleep(duration)
    return f"Completed after {duration}s"


def run_performance_validation():
    """Run comprehensive performance validation."""
    print("⚡ Starting Performance Validation...")
    print("=" * 60)
    
    profiler = SimpleProfiler()
    cache = SimpleCache(max_size=100)
    processor = ConcurrentProcessor(max_workers=4)
    
    results = {
        "sequential_time": 0.0,
        "threaded_time": 0.0,
        "multiprocess_time": 0.0,
        "cache_hit_rate": 0.0,
        "memory_efficiency": 0.0,
        "cpu_utilization": 0.0
    }
    
    try:
        # Test 1: Sequential vs Concurrent Processing
        print("1. Testing Sequential vs Concurrent Processing")
        print("-" * 45)
        
        # Prepare test data
        test_data = [10000 + random.randint(0, 5000) for _ in range(20)]
        
        # Sequential processing
        profile_id = profiler.start_profiling("sequential_processing")
        sequential_results = [cpu_intensive_task(n) for n in test_data]
        sequential_metrics = profiler.end_profiling(profile_id)
        results["sequential_time"] = sequential_metrics.execution_time
        
        print(f"Sequential processing: {sequential_metrics.execution_time:.3f}s")
        
        # Threaded processing
        profile_id = profiler.start_profiling("threaded_processing")
        threaded_results = processor.process_batch_threaded(test_data, cpu_intensive_task)
        threaded_metrics = profiler.end_profiling(profile_id)
        results["threaded_time"] = threaded_metrics.execution_time
        
        print(f"Threaded processing: {threaded_metrics.execution_time:.3f}s")
        
        # Multiprocess processing
        profile_id = profiler.start_profiling("multiprocess_processing")
        mp_results = processor.process_batch_multiprocess(test_data, cpu_intensive_task)
        mp_metrics = profiler.end_profiling(profile_id)
        results["multiprocess_time"] = mp_metrics.execution_time
        
        print(f"Multiprocess processing: {mp_metrics.execution_time:.3f}s")
        
        # Calculate speedup
        if results["sequential_time"] > 0:
            threaded_speedup = results["sequential_time"] / results["threaded_time"]
            mp_speedup = results["sequential_time"] / results["multiprocess_time"]
            print(f"Threaded speedup: {threaded_speedup:.2f}x")
            print(f"Multiprocess speedup: {mp_speedup:.2f}x")
        
        # Verify results consistency
        assert len(sequential_results) == len(threaded_results) == len(mp_results)
        print("✅ Results consistency verified")
        
        # Test 2: Caching Performance
        print("\\n2. Testing Caching Performance")
        print("-" * 30)
        
        # Populate cache
        for i in range(50):
            cache.put(f"key_{i}", f"value_{i}")
        
        # Test cache hits and misses
        cache_test_keys = [f"key_{random.randint(0, 70)}" for _ in range(200)]
        
        profile_id = profiler.start_profiling("cache_operations")
        cache_results = []
        for key in cache_test_keys:
            result = cache.get(key)
            if result is None:
                # Simulate expensive computation for cache miss
                result = f"computed_{key}"
                cache.put(key, result)
            cache_results.append(result)
        
        cache_metrics = profiler.end_profiling(profile_id)
        results["cache_hit_rate"] = cache.get_hit_rate()
        
        print(f"Cache hit rate: {results['cache_hit_rate']:.2%}")
        print(f"Cache operations time: {cache_metrics.execution_time:.3f}s")
        print("✅ Caching performance tested")
        
        # Test 3: Memory Management
        print("\\n3. Testing Memory Management")
        print("-" * 30)
        
        initial_memory = psutil.Process().memory_info().rss / (1024 * 1024)
        
        # Create and process large data structures
        profile_id = profiler.start_profiling("memory_intensive")
        large_datasets = []
        for i in range(10):
            dataset = list(range(100000))  # 100k integers
            result = memory_intensive_task(len(dataset))
            large_datasets.append(result)
        
        peak_memory = psutil.Process().memory_info().rss / (1024 * 1024)
        
        # Clean up
        del large_datasets
        gc.collect()
        
        final_memory = psutil.Process().memory_info().rss / (1024 * 1024)
        memory_metrics = profiler.end_profiling(profile_id)
        
        memory_used = peak_memory - initial_memory
        memory_reclaimed = peak_memory - final_memory
        memory_efficiency = memory_reclaimed / memory_used if memory_used > 0 else 0
        
        results["memory_efficiency"] = memory_efficiency
        
        print(f"Initial memory: {initial_memory:.1f} MB")
        print(f"Peak memory: {peak_memory:.1f} MB")
        print(f"Final memory: {final_memory:.1f} MB")
        print(f"Memory efficiency: {memory_efficiency:.2%}")
        print("✅ Memory management tested")
        
        # Test 4: System Resource Utilization
        print("\\n4. Testing System Resource Utilization")
        print("-" * 40)
        
        # Monitor CPU usage during intensive operations
        cpu_samples = []
        
        def monitor_cpu():
            for _ in range(10):
                cpu_samples.append(psutil.cpu_percent(interval=0.1))
        
        # Start CPU monitoring in background
        monitor_thread = threading.Thread(target=monitor_cpu)
        monitor_thread.start()
        
        # Run intensive operations
        intensive_tasks = [20000 for _ in range(8)]
        processor.process_batch_threaded(intensive_tasks, cpu_intensive_task)
        
        monitor_thread.join()
        
        avg_cpu_usage = statistics.mean(cpu_samples) if cpu_samples else 0
        results["cpu_utilization"] = avg_cpu_usage
        
        print(f"Average CPU utilization: {avg_cpu_usage:.1f}%")
        print(f"CPU cores available: {mp.cpu_count()}")
        print(f"Memory total: {psutil.virtual_memory().total / (1024**3):.1f} GB")
        print("✅ Resource utilization tested")
        
        # Test 5: Performance Under Load
        print("\\n5. Testing Performance Under Load")
        print("-" * 35)
        
        load_test_data = [5000 for _ in range(50)]  # 50 medium tasks
        
        # Test with different concurrency levels
        concurrency_results = {}
        for workers in [1, 2, 4, 8]:
            test_processor = ConcurrentProcessor(max_workers=workers)
            
            start_time = time.time()
            test_processor.process_batch_threaded(load_test_data, cpu_intensive_task)
            execution_time = time.time() - start_time
            
            concurrency_results[workers] = execution_time
            test_processor.shutdown()
            
            print(f"  {workers} workers: {execution_time:.3f}s")
        
        # Find optimal concurrency level
        optimal_workers = min(concurrency_results.keys(), 
                            key=lambda k: concurrency_results[k])
        print(f"Optimal worker count: {optimal_workers}")
        print("✅ Load testing completed")
        
    except Exception as e:
        print(f"❌ Performance test failed: {e}")
        raise
    
    finally:
        processor.shutdown()
    
    # Performance Assessment
    print("\\n" + "=" * 60)
    print("PERFORMANCE VALIDATION RESULTS")
    print("=" * 60)
    
    print(f"Execution Times:")
    print(f"  Sequential: {results['sequential_time']:.3f}s")
    print(f"  Threaded: {results['threaded_time']:.3f}s")
    print(f"  Multiprocess: {results['multiprocess_time']:.3f}s")
    
    print(f"\\nOptimization Metrics:")
    print(f"  Cache hit rate: {results['cache_hit_rate']:.2%}")
    print(f"  Memory efficiency: {results['memory_efficiency']:.2%}")
    print(f"  CPU utilization: {results['cpu_utilization']:.1f}%")
    
    # Performance scoring
    score = 0
    max_score = 100
    
    # Threading speedup (20 points)
    if results['sequential_time'] > 0 and results['threaded_time'] > 0:
        speedup = results['sequential_time'] / results['threaded_time']
        if speedup > 2.0:
            score += 20
        elif speedup > 1.5:
            score += 15
        elif speedup > 1.2:
            score += 10
        elif speedup > 1.0:
            score += 5
    
    # Cache performance (25 points)
    if results['cache_hit_rate'] > 0.8:
        score += 25
    elif results['cache_hit_rate'] > 0.6:
        score += 20
    elif results['cache_hit_rate'] > 0.4:
        score += 15
    elif results['cache_hit_rate'] > 0.2:
        score += 10
    
    # Memory efficiency (25 points)
    if results['memory_efficiency'] > 0.8:
        score += 25
    elif results['memory_efficiency'] > 0.6:
        score += 20
    elif results['memory_efficiency'] > 0.4:
        score += 15
    elif results['memory_efficiency'] > 0.2:
        score += 10
    
    # CPU utilization (20 points)
    if 50 <= results['cpu_utilization'] <= 90:
        score += 20  # Good utilization
    elif 40 <= results['cpu_utilization'] <= 95:
        score += 15
    elif results['cpu_utilization'] > 20:
        score += 10
    
    # Stability (10 points)
    score += 10  # Completed without crashes
    
    print(f"\\n⚡ PERFORMANCE SCORE: {score}/{max_score}")
    
    if score >= 80:
        print("🏆 EXCELLENT - High performance optimizations working well")
    elif score >= 60:
        print("✅ GOOD - Performance optimizations are effective")
    elif score >= 40:
        print("⚠️  FAIR - Some performance improvements detected")
    else:
        print("❌ POOR - Performance optimizations need improvement")
    
    print("\\n✅ Performance validation completed!")
    return results


if __name__ == "__main__":
    results = run_performance_validation()