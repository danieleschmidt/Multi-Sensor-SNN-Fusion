#!/usr/bin/env python3
"""
Simple Performance Validation Script

Tests performance optimization features using only built-in Python libraries.
"""

import time
import threading
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import gc
import random
import statistics
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from collections import deque


@dataclass
class PerformanceMetrics:
    """Performance metrics tracking."""
    execution_time: float
    throughput_ops_per_sec: float


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
        try:
            futures = [self.process_pool.submit(processor_func, item) for item in data_list]
            return [future.result(timeout=10) for future in futures]  # 10s timeout
        except Exception as e:
            print(f"  ⚠️  Multiprocessing failed: {e}")
            # Fallback to sequential
            return [processor_func(item) for item in data_list]
    
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


def io_simulation_task(duration: float) -> str:
    """IO simulation task."""
    time.sleep(duration)
    return f"Completed after {duration:.3f}s"


def run_simple_performance_validation():
    """Run performance validation with built-in libraries only."""
    print("⚡ Starting Simple Performance Validation...")
    print("=" * 55)
    
    cache = SimpleCache(max_size=100)
    processor = ConcurrentProcessor(max_workers=4)
    
    results = {
        "sequential_time": 0.0,
        "threaded_time": 0.0,
        "multiprocess_time": 0.0,
        "cache_hit_rate": 0.0,
        "tests_passed": 0,
        "total_tests": 6
    }
    
    try:
        # Test 1: Sequential vs Concurrent Processing
        print("1. Testing Sequential vs Concurrent Processing")
        print("-" * 45)
        
        # Prepare test data (smaller for quick testing)
        test_data = [5000 + random.randint(0, 2000) for _ in range(12)]
        
        # Sequential processing
        start_time = time.time()
        sequential_results = [cpu_intensive_task(n) for n in test_data]
        results["sequential_time"] = time.time() - start_time
        
        print(f"  Sequential: {results['sequential_time']:.3f}s")
        
        # Threaded processing
        start_time = time.time()
        threaded_results = processor.process_batch_threaded(test_data, cpu_intensive_task)
        results["threaded_time"] = time.time() - start_time
        
        print(f"  Threaded: {results['threaded_time']:.3f}s")
        
        # Multiprocess processing
        start_time = time.time()
        mp_results = processor.process_batch_multiprocess(test_data, cpu_intensive_task)
        results["multiprocess_time"] = time.time() - start_time
        
        print(f"  Multiprocess: {results['multiprocess_time']:.3f}s")
        
        # Calculate speedup
        if results["sequential_time"] > 0 and results["threaded_time"] > 0:
            threaded_speedup = results["sequential_time"] / results["threaded_time"]
            print(f"  Threading speedup: {threaded_speedup:.2f}x")
            
            if threaded_speedup > 1.1:  # At least 10% improvement
                results["tests_passed"] += 1
                print("  ✅ Threading optimization effective")
            else:
                print("  ⚠️  Threading shows minimal improvement")
        
        # Verify results consistency
        if (len(sequential_results) == len(threaded_results) == len(mp_results) and
            abs(sum(sequential_results) - sum(threaded_results)) < 1.0):
            results["tests_passed"] += 1
            print("  ✅ Results consistency verified")
        else:
            print("  ❌ Results inconsistent between processing methods")
        
        # Test 2: Caching Performance
        print("\\n2. Testing Caching Performance")
        print("-" * 30)
        
        # Populate cache with some data
        for i in range(50):
            cache.put(f"key_{i}", f"value_{i}")
        
        # Test cache performance with mixed hits/misses
        cache_test_keys = []
        for _ in range(200):
            if random.random() < 0.7:  # 70% chance of hit
                cache_test_keys.append(f"key_{random.randint(0, 49)}")
            else:  # 30% chance of miss
                cache_test_keys.append(f"key_{random.randint(50, 99)}")
        
        start_time = time.time()
        cache_results = []
        for key in cache_test_keys:
            result = cache.get(key)
            if result is None:
                # Simulate expensive computation for cache miss
                time.sleep(0.001)  # 1ms computation
                result = f"computed_{key}"
                cache.put(key, result)
            cache_results.append(result)
        
        cache_time = time.time() - start_time
        results["cache_hit_rate"] = cache.get_hit_rate()
        
        print(f"  Cache hit rate: {results['cache_hit_rate']:.2%}")
        print(f"  Cache operations time: {cache_time:.3f}s")
        
        if results["cache_hit_rate"] > 0.5:  # At least 50% hit rate
            results["tests_passed"] += 1
            print("  ✅ Caching performance good")
        else:
            print("  ⚠️  Low cache hit rate")
        
        # Test 3: Memory Management
        print("\\n3. Testing Memory Management")
        print("-" * 30)
        
        # Create large data structures and clean them up
        start_time = time.time()
        large_datasets = []
        for i in range(10):
            dataset = list(range(50000))  # 50k integers
            result = memory_intensive_task(len(dataset))
            large_datasets.append(result)
        
        processing_time = time.time() - start_time
        
        # Clean up
        del large_datasets
        gc.collect()
        
        print(f"  Memory operations time: {processing_time:.3f}s")
        
        if processing_time < 5.0:  # Reasonable time
            results["tests_passed"] += 1
            print("  ✅ Memory operations efficient")
        else:
            print("  ⚠️  Memory operations slow")
        
        # Test 4: Concurrent IO Simulation
        print("\\n4. Testing Concurrent IO Operations")
        print("-" * 37)
        
        io_tasks = [0.05 for _ in range(20)]  # 20 tasks of 50ms each
        
        # Sequential IO
        start_time = time.time()
        sequential_io = [io_simulation_task(duration) for duration in io_tasks]
        sequential_io_time = time.time() - start_time
        
        # Concurrent IO
        start_time = time.time()
        concurrent_io = processor.process_batch_threaded(io_tasks, io_simulation_task)
        concurrent_io_time = time.time() - start_time
        
        io_speedup = sequential_io_time / concurrent_io_time if concurrent_io_time > 0 else 0
        
        print(f"  Sequential IO: {sequential_io_time:.3f}s")
        print(f"  Concurrent IO: {concurrent_io_time:.3f}s")
        print(f"  IO speedup: {io_speedup:.2f}x")
        
        if io_speedup > 3.0:  # Significant improvement for IO
            results["tests_passed"] += 1
            print("  ✅ Concurrent IO optimization excellent")
        elif io_speedup > 1.5:
            results["tests_passed"] += 1
            print("  ✅ Concurrent IO optimization good")
        else:
            print("  ⚠️  Concurrent IO shows minimal improvement")
        
        # Test 5: Load Testing
        print("\\n5. Testing Performance Under Load")
        print("-" * 35)
        
        load_sizes = [100, 500, 1000, 2000]
        load_results = []
        
        for size in load_sizes:
            load_data = [1000 for _ in range(min(size // 100, 20))]  # Scale down for testing
            
            start_time = time.time()
            processor.process_batch_threaded(load_data, cpu_intensive_task)
            load_time = time.time() - start_time
            
            throughput = len(load_data) / load_time if load_time > 0 else 0
            load_results.append(throughput)
            
            print(f"  Load {size}: {throughput:.1f} ops/sec")
        
        # Check if throughput is consistent
        if len(load_results) >= 2:
            throughput_variance = statistics.variance(load_results)
            avg_throughput = statistics.mean(load_results)
            cv = (throughput_variance ** 0.5) / avg_throughput if avg_throughput > 0 else float('inf')
            
            if cv < 0.3:  # Coefficient of variation < 30%
                results["tests_passed"] += 1
                print("  ✅ Performance stable under load")
            else:
                print("  ⚠️  Performance varies significantly under load")
        
        # Test 6: Resource Cleanup
        print("\\n6. Testing Resource Cleanup")
        print("-" * 30)
        
        # Test that resources are properly cleaned up
        test_processor = ConcurrentProcessor(max_workers=2)
        test_data_small = [1000 for _ in range(5)]
        
        start_time = time.time()
        test_processor.process_batch_threaded(test_data_small, cpu_intensive_task)
        test_processor.shutdown()
        cleanup_time = time.time() - start_time
        
        print(f"  Resource cleanup time: {cleanup_time:.3f}s")
        
        if cleanup_time < 2.0:  # Quick cleanup
            results["tests_passed"] += 1
            print("  ✅ Resource cleanup efficient")
        else:
            print("  ⚠️  Resource cleanup slow")
            
    except Exception as e:
        print(f"❌ Performance test failed: {e}")
        return results
    
    finally:
        processor.shutdown()
    
    # Performance Assessment
    print("\\n" + "=" * 55)
    print("PERFORMANCE VALIDATION RESULTS")
    print("=" * 55)
    
    print(f"Tests Passed: {results['tests_passed']}/{results['total_tests']}")
    
    print(f"\\nTiming Results:")
    print(f"  Sequential processing: {results['sequential_time']:.3f}s")
    print(f"  Threaded processing: {results['threaded_time']:.3f}s")
    if results['multiprocess_time'] > 0:
        print(f"  Multiprocess processing: {results['multiprocess_time']:.3f}s")
    
    print(f"\\nOptimization Metrics:")
    if results['sequential_time'] > 0 and results['threaded_time'] > 0:
        speedup = results['sequential_time'] / results['threaded_time']
        print(f"  Threading speedup: {speedup:.2f}x")
    print(f"  Cache hit rate: {results['cache_hit_rate']:.2%}")
    
    # Overall performance score
    score_percentage = (results['tests_passed'] / results['total_tests']) * 100
    
    print(f"\\n⚡ PERFORMANCE SCORE: {score_percentage:.0f}%")
    
    if score_percentage >= 80:
        print("🏆 EXCELLENT - Performance optimizations working very well")
    elif score_percentage >= 60:
        print("✅ GOOD - Performance optimizations are effective")
    elif score_percentage >= 40:
        print("⚠️  FAIR - Some performance improvements detected")
    else:
        print("❌ NEEDS IMPROVEMENT - Performance optimizations require attention")
    
    print("\\n✅ Simple performance validation completed!")
    return results


if __name__ == "__main__":
    results = run_simple_performance_validation()